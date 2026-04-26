

// Vertex shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 2u);
    let y = f32((in_vertex_index & 1u) << 1u);
    out.clip_position = vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
    out.tex_coords = vec2<f32>(x, y);
    return out;
}

// Fragment Shader

@group(0) @binding(0) var render_target: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.clip_position.xy);

    // 1. Sample the current pixel and its immediate neighbors
    let center = textureLoad(render_target, coords, 0).g;
    let right = textureLoad(render_target, coords + vec2<i32>(1, 0), 0).g;
    let up = textureLoad(render_target, coords + vec2<i32>(0, 1), 0).g;

    // 2. Calculate the slope (derivative) in X and Y
    let dx = right - center;
    let dy = up - center;

    // 3. Construct a Surface Normal vector
    // The Z value determines how "steep" or "flat" the bumps look. 
    // Tweak 0.05 to change the extrusion depth!
    let normal = normalize(vec3<f32>(-dx, -dy, 0.02));

    // 4. Define a Light Source direction
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0)); // Light coming from top-right

    // 5. Calculate Diffuse Lighting (Lambertian reflection)
    // How directly is the light hitting this slope?
    let diffuse = max(dot(normal, light_dir), 0.0);

    // 6. Calculate Specular Highlight (Shiny reflection)
    let view_dir = vec3<f32>(0.0, 0.0, 1.0); // Looking straight at the screen
    let reflect_dir = reflect(-light_dir, normal);
    let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0) * 0.5; // 16.0 is glossiness

    // 7. Base Color (Dark teal for the background, bright cyan for the chemical)
    let base_color = mix(vec3<f32>(0.02, 0.05, 0.1), vec3<f32>(0.0, 0.8, 0.6), center * 2.0);

    // Combine base color with lighting
    let final_color = (base_color * diffuse) + vec3<f32>(specular);

    return vec4<f32>(final_color, 1.0);
}

// Compute Shader 

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

fn get_laplacian(coords: vec2<i32>, dim: vec2<i32>) -> vec2<f32> {
    let center = get_state(coords, dim);

    let up = get_state(coords + vec2<i32>(0, -1), dim);
    let down = get_state(coords + vec2<i32>(0, 1), dim);
    let left = get_state(coords + vec2<i32>(-1, 0), dim);
    let right = get_state(coords + vec2<i32>(1, 0), dim);

    let ul = get_state(coords + vec2<i32>(-1, -1), dim);
    let ur = get_state(coords + vec2<i32>(1, -1), dim);
    let dl = get_state(coords + vec2<i32>(-1, 1), dim);
    let dr = get_state(coords + vec2<i32>(1, 1), dim);

    // Using the weights: adjacent = 0.2, Diagonal = 0.05, Center = -1.0
    return (up + down + left + right) * 0.2 + (ul + ur + dl + dr) * 0.05 + center * -1.0;
}

fn hash(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn get_state(coords: vec2<i32>, dim: vec2<i32>) -> vec2<f32> {
    let clamped = clamp(coords, vec2<i32>(0), dim - vec2<i32>(1));
    return textureLoad(state_in, clamped, 0).rg;
}

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(state_in);
    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y) { return; }

    if uniforms.frame == 30u {
        let resolution = vec2<f32>(f32(dimensions.x), f32(dimensions.y));
        let uv = vec2<f32>(coords) / resolution - 0.5;

        var a = 1.0;
        var b = 0.0;

        let center = vec2<f32>(f32(dimensions.x) / 2.0, f32(dimensions.y) / 2.0);
        if distance(vec2<f32>(coords), center) < 50.0 {
            a = 0.5;
            b = hash(vec2<f32>(coords)) * 0.5 + 0.5;
        }

        textureStore(state_out, coords, vec4<f32>(a, b, 0.0, 1.0));
        return;
    }

    let state = get_state(coords, vec2<i32>(dimensions));
    var a = state.r;
    var b = state.g;

    if uniforms.cursor_button > 0.5 {
        let pixel_cursor_x = (uniforms.cursor_position.x * 0.5 + 0.5) * f32(dimensions.x);
        let pixel_cursor_y = (uniforms.cursor_position.y * 0.5 + 0.5) * f32(dimensions.y);
        let pixel_cursor = vec2<f32>(pixel_cursor_x, pixel_cursor_y);
        let distance_pix = distance(vec2<f32>(coords), pixel_cursor);
        if distance_pix < 15.0 {
            b += 0.2 * (1.0 - distance_pix / 15.0);
        }
    }

    let laplacian = get_laplacian(coords, vec2<i32>(dimensions));

    let uv = vec2<f32>(coords) / vec2<f32>(f32(dimensions.x), f32(dimensions.y));

    let Da = 1.0;
    let Db = 0.5;
    let feed = 0.055; // 0.010 + (uv.x * 0.090); // 0.055;
    let kill = 0.062; // 0.045 + (uv.y * 0.025); // 0.062;

    let reaction = a * b * b;

    let dt = 1.0;

    var a_new = a + (Da * laplacian.r - reaction + feed * (1.0 - a)) * dt;
    var b_new = b + (Db * laplacian.g + reaction - (kill + feed) * b) * dt;

    a_new = clamp(a_new, 0.0, 1.0);
    b_new = clamp(b_new, 0.0, 1.0);

    textureStore(state_out, coords, vec4<f32>(a_new, b_new, 0.0, 1.0));
}
