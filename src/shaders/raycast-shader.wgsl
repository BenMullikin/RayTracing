// ==========================================
// 1. RENDER PASS (Vertex & Fragment)
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

@group(0) @binding(0) var render_target: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.clip_position.xy);
    return textureLoad(render_target, coords, 0);
}

// ==========================================
// 2. COMPUTE PASS (Analytic Raycaster)
// ==========================================

struct Uniform {
    frame: u32,
    time: f32,
    zoom: f32,
    _pad1: u32,
    pan: vec2<f32>,
    cursor_position: vec2<f32>,
    cursor_button: f32,
};

@group(0) @binding(1) var state_in: texture_2d<f32>; // Unused here, but keeps layout happy
@group(0) @binding(2) var state_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uniforms: Uniform;

// Signed Distance Field (SDF) for a line segment. Used to draw the laser beam!
fn sd_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

// Analytic intersection of a ray and a circle using the quadratic formula
fn intersect_circle(ro: vec2<f32>, rd: vec2<f32>, center: vec2<f32>, radius: f32) -> vec3<f32> {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let h = b * b - c;

    // If h > 0, the ray hit the circle!
    if h > 0.0 {
        let t = -b - sqrt(h);
        if t > 0.001 { // Prevent self-intersection acne
            let normal = normalize((ro + rd * t) - center);
            return vec3<f32>(t, normal.x, normal.y);
        }
    }
    return vec3<f32>(-1.0, 0.0, 0.0); // Missed
}

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(state_out);
    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(dimensions.x) || coords.y >= i32(dimensions.y) { return; }

    let resolution = vec2<f32>(f32(dimensions.x), f32(dimensions.y));

    // Normalize UVs and apply camera pan/zoom
    var uv = (vec2<f32>(coords) / resolution) * 2.0 - 1.0;
    uv.x *= resolution.x / resolution.y;
    uv = (uv / uniforms.zoom) + uniforms.pan;

    var cursor_uv = uniforms.cursor_position;
    cursor_uv.x *= resolution.x / resolution.y;
    cursor_uv = (cursor_uv / uniforms.zoom) + uniforms.pan;

    // --- SCENE DEFINITION ---
    // vec3(x, y, radius)
    let m1 = vec3<f32>(0.5, 0.0, 0.2);
    let m2 = vec3<f32>(-0.6, 0.4, 0.25);
    let m3 = vec3<f32>(-0.2, -0.7, 0.15);

    // --- RAY ORIGIN & DIRECTION ---
    var ro = vec2<f32>(0.0, 0.0);
    var rd = normalize(cursor_uv - ro);

    // Interactive mode: Click to hold the laser and let it spin
    if uniforms.cursor_button > 0.5 {
        ro = cursor_uv;
        rd = vec2<f32>(cos(uniforms.time * 2.0), sin(uniforms.time * 2.0));
    }

    // --- CALCULATE BOUNCES ---
    let max_bounces = 4;
    var points = array<vec2<f32>, 5>(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
    points[0] = ro;

    for (var i = 1; i <= max_bounces; i++) {
        var min_t = 1000.0;
        var normal = vec2<f32>(0.0);
        var hit = false;

        // Check intersection against all mirrors
        let i1 = intersect_circle(ro, rd, m1.xy, m1.z);
        if i1.x > 0.0 && i1.x < min_t { min_t = i1.x; normal = i1.yz; hit = true; }

        let i2 = intersect_circle(ro, rd, m2.xy, m2.z);
        if i2.x > 0.0 && i2.x < min_t { min_t = i2.x; normal = i2.yz; hit = true; }

        let i3 = intersect_circle(ro, rd, m3.xy, m3.z);
        if i3.x > 0.0 && i3.x < min_t { min_t = i3.x; normal = i3.yz; hit = true; }

        if hit {
            // Move origin to the hit point, reflect the direction using the normal!
            ro = ro + rd * min_t;
            rd = reflect(rd, normal);
            points[i] = ro;
        } else {
            // Shoot off into infinity
            points[i] = ro + rd * 20.0;
            break;
        }
    }

    // --- DRAWING LOGIC ---
    var color = vec3<f32>(0.02, 0.02, 0.05); // Dark background

    // 1. Draw Mirrors
    let d1 = length(uv - m1.xy) - m1.z;
    let d2 = length(uv - m2.xy) - m2.z;
    let d3 = length(uv - m3.xy) - m3.z;
    let min_mirror_d = min(min(d1, d2), d3);

    if min_mirror_d < 0.0 {
        color = vec3<f32>(0.1, 0.2, 0.3); // Solid mirror fill
    }
    color += vec3<f32>(0.0, 0.6, 1.0) * (0.003 / abs(min_mirror_d)); // Neon edge glow

    // 2. Draw Laser Beam
    var beam_dist = 1000.0;
    for (var i = 0; i < max_bounces; i++) {
        let a = points[i];
        let b = points[i + 1];

        // If the point is at 0,0 and it's not the origin, the ray terminated early.
        if i > 0 && length(a) == 0.0 && length(b) == 0.0 { break; }

        // Find distance from current pixel to the laser segment
        let d = sd_segment(uv, a, b);
        beam_dist = min(beam_dist, d);
    }

    // Laser Glow Math (Inverse Square Falloff)
    let laser_color = vec3<f32>(1.0, 0.1, 0.2);
    let glow = 0.001 / (beam_dist + 0.0001);
    color += laser_color * glow * 3.0;

    // Solid white hot core
    if beam_dist < 0.003 {
        color += vec3<f32>(1.0, 1.0, 1.0);
    }

    textureStore(state_out, coords, vec4<f32>(color, 1.0));
}
