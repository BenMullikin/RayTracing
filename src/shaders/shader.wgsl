

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
    return textureLoad(render_target, coords, 0);
}

// Compute Shader 

//@group(0) @binding(1) var screen_output: texture_storage_2d<rgba32Float, write>;

//@compute
//@workgroup_size(16, 16)
//fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//    let dimensions = textureDimensions(screen_output);
//    let coords = vec2<i32>(global_id.xy);
//
//    if coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y) {
//        return;
//    }
//
//    let r = f32(coords.x) / f32(dimensions.x);
//    let g = f32(coords.y) / f32(dimensions.y);
//
//    let color = vec4<f32>(r, g, 0.2, 1.0);
//
//    textureStore(screen_output, coords, color);
//}

struct Uniform {
    time: f32,
    zoom: f32,
    pan: vec2<f32>,
};

@group(0) @binding(1) var screen_old: texture_2d<f32>;
@group(0) @binding(2) var screen_output: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> uniforms: Uniform;

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(screen_output);
    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y) { return; }

    // 1. Normalize coordinates from [0, width] to [-1.0, 1.0]
    let resolution = vec2<f32>(f32(dimensions.x), f32(dimensions.y));
    var uv = (vec2<f32>(coords) / resolution) * 2.0 - 1.0;

    // 2. Fix the aspect ratio so the fractal doesn't stretch
    uv.x *= resolution.x / resolution.y;

    // 3. Mandelbrot Math
    //let c = (uv * 1.5 * uniforms.zoom) - vec2<f32>(0.5, 0.0) + uniforms.pan; // Scale and pan to center it
    //var z = vec2<f32>(0.0, 0.0);
    //var iter = 0u;
    //let max_iter = u32(100.0 / uniforms.zoom);

    // 4. Julia Set Math
    var z = (uv * 1.5 * uniforms.zoom) + uniforms.pan;
    let c = vec2<f32>(
        cos(uniforms.time * 0.5) * 0.7885,
        sin(uniforms.time * 0.5) * 0.7885
    );
    var iter = 0;
    let max_iter = 150u;

    for (var i = 0u; i < max_iter; i++) {
        let x = (z.x * z.x - z.y * z.y) + c.x;
        let y = (2.0 * z.x * z.y) + c.y;
        if (x * x + y * y) > 4.0 { break; }
        z = vec2<f32>(x, y);
        iter++;
    }

    // 4. Colorize based on how many iterations it took to escape
    let t = f32(iter) / f32(max_iter);
    let color = vec4<f32>(
        sin(t * 15.0) * 0.5 + 0.5,
        sin(t * 10.0 + 2.0) * 0.5 + 0.5,
        sin(t * 5.0 + 4.0) * 0.5 + 0.5,
        1.0
    );

    textureStore(screen_output, coords, color);
}
