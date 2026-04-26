

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
    let state = textureLoad(render_target, coords, 0);

    let val = clamp(state.g * 2.25, 0.0, 1.0);

    // let a = vec3<f32>(0.5, 0.5, 0.5);
    // let b = vec3<f32>(0.5, 0.5, 0.5);
    // let c = vec3<f32>(1.0, 1.0, 1.0);
    // let d = vec3<f32>(0.3, 0.2, 0.2);

    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(2.0, 1.0, 0.0);
    let d = vec3<f32>(0.5, 0.2, 0.25);

    let color = a + b * cos(6.28318 * (c * val + d));

    let final_color = mix(vec3<f32>(0.0), color, val);

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
    let feed = 0.05546; // 0.010 + (uv.x * 0.090); // 0.055;
    let kill = 0.062594; // 0.045 + (uv.y * 0.025); // 0.062;

    let reaction = a * b * b;

    let dt = 1.0;

    var a_new = a + (Da * laplacian.r - reaction + feed * (1.0 - a)) * dt;
    var b_new = b + (Db * laplacian.g + reaction - (kill + feed) * b) * dt;

    a_new = clamp(a_new, 0.0, 1.0);
    b_new = clamp(b_new, 0.0, 1.0);

    textureStore(state_out, coords, vec4<f32>(a_new, b_new, 0.0, 1.0));
}
