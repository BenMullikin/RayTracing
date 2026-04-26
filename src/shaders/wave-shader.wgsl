// ==========================================
// VERTEX SHADER (Unchanged)
// ==========================================
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 2u);
    let y = f32((in_vertex_index & 1u) << 1u);
    out.clip_position = vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x, y);
    return out;
}

// ==========================================
// FRAGMENT SHADER (Water Rendering)
// ==========================================
@group(0) @binding(0) var render_target: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.clip_position.xy);

    // Sample the height (Red channel) of center and neighbors
    let center = textureLoad(render_target, coords, 0).r;
    let right = textureLoad(render_target, coords + vec2<i32>(1, 0), 0).r;
    let up = textureLoad(render_target, coords + vec2<i32>(0, 1), 0).r;

    // Calculate gradients to build a 3D surface normal
    let dx = right - center;
    let dy = up - center;
    let normal = normalize(vec3<f32>(-dx, -dy, 0.02)); // Tweak 0.02 for wave steepness

    // Lighting setup
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let reflect_dir = reflect(-light_dir, normal);

    // Specular highlight for shiny water
    let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.8;

    // Water color mapping based on height
    let deep_color = vec3<f32>(0.0, 0.1, 0.3);
    let shallow_color = vec3<f32>(0.0, 0.5, 0.6);
    let crest_color = vec3<f32>(0.8, 0.9, 1.0);

    var base_color = mix(deep_color, shallow_color, center + 0.5);
    base_color = mix(base_color, crest_color, smoothstep(0.5, 1.5, center));

    let final_color = base_color + vec3<f32>(specular);

    return vec4<f32>(final_color, 1.0);
}

// ==========================================
// COMPUTE SHADER (Wave Physics)
// ==========================================
struct Uniform {
    frame: u32,
    time: f32,
    zoom: f32,
    pan: vec2<f32>,
    cursor_position: vec2<f32>,
    cursor_button: f32,
};

@group(0) @binding(1) var state_in: texture_2d<f32>;
@group(0) @binding(2) var state_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uniforms: Uniform;

fn get_state(coords: vec2<i32>, dim: vec2<i32>) -> vec2<f32> {
    // Clamp to create reflective boundaries (waves bounce off walls)
    let clamped = clamp(coords, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(state_in, clamped, 0).rg;
}

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(state_in);
    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y) { return; }

    let dim = vec2<i32>(dimensions);

    // Read current state: r = height, g = velocity
    let current = get_state(coords, dim);
    var height = current.r;
    var velocity = current.g;

    // 1. Calculate the Laplacian (Curvature of the surface)
    let up = get_state(coords + vec2<i32>(0, -1), dim).r;
    let down = get_state(coords + vec2<i32>(0, 1), dim).r;
    let left = get_state(coords + vec2<i32>(-1, 0), dim).r;
    let right = get_state(coords + vec2<i32>(1, 0), dim).r;

    // Discrete 4-way Laplacian
    let laplacian = (up + down + left + right) - (4.0 * height);

    // 2. Physics Parameters
    let wave_speed = 0.25; // How fast waves move (too high = math explodes)
    let damping = 0.995;   // How fast waves die out (1.0 = forever)

    // 3. Integration (Update velocity, then update height)
    velocity += laplacian * wave_speed;
    velocity *= damping;
    height += velocity;

    // 4. Mouse Interaction
    if uniforms.cursor_button > 0.5 {
        // Convert normalized cursor to pixel coordinates
        let pixel_cursor = vec2<f32>(
            (uniforms.cursor_position.x * 0.5 + 0.5) * f32(dimensions.x),
            (uniforms.cursor_position.y * 0.5 + 0.5) * f32(dimensions.y)
        );

        let dist = distance(vec2<f32>(coords), pixel_cursor);

        // If the mouse is pressed, push the water down at the cursor!
        if dist < 10.0 {
            height = -2.0;
        }
    }

    // Write the new height and velocity to the output texture
    textureStore(state_out, coords, vec4<f32>(height, velocity, 0.0, 1.0));
}
