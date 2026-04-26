// ==========================================
// UNIFORMS & STRUCTURES
// ==========================================

struct SystemUniform {
    resolution: vec2<u32>,
    time: f32,
    dt: f32,
    frame: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct CameraUniform {
    position: vec3<f32>,
    _pad1: u32,
    forward: vec3<f32>,
    _pad2: u32,
    right: vec3<f32>,
    _pad3: u32,
    up: vec3<f32>,
    _pad4: u32,
};

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

// Group 0: Global System
@group(0) @binding(0) var<uniform> system: SystemUniform;

// Group 1: App Specific (Render Pass)
@group(1) @binding(0) var render_target: texture_2d<f32>;

// Group 1: App Specific (Compute Pass)
@group(1) @binding(1) var<uniform> camera: CameraUniform;
@group(1) @binding(2) var state_in: texture_2d<f32>;
@group(1) @binding(3) var state_out: texture_storage_2d<rgba16float, write>;

// ==========================================
// VERTEX & FRAGMENT (The Screen Quad)
// ==========================================

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(in_vertex_index & 2u);
    let y = f32((in_vertex_index & 1u) << 1u);
    out.clip_position = vec4<f32>(2.0 * x - 1.0, 1.0 - 2.0 * y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.clip_position.xy);
    return textureLoad(render_target, coords, 0);
}

// ==========================================
// COMPUTE (The Raymarcher)
// ==========================================

// Signed Distance Field (SDF) for our scene
fn map_scene(p: vec3<f32>) -> f32 {
    // A sphere that bounces using your system time!
    let sphere_pos = vec3<f32>(0.0, sin(system.time * 2.0) * 0.5 + 1.0, 5.0);
    let sphere = length(p - sphere_pos) - 1.0;

    // An infinite flat floor at y = -1.0
    let ground = p.y + 1.0;

    // Return the closest distance to any object
    return min(sphere, ground);
}

// Calculates the normal vector by taking a tiny sample around the hit point
fn get_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.001, 0.0);
    let n = vec3<f32>(
        map_scene(p + e.xyy) - map_scene(p - e.xyy),
        map_scene(p + e.yxy) - map_scene(p - e.yxy),
        map_scene(p + e.yyx) - map_scene(p - e.yyx)
    );
    return normalize(n);
}

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(system.resolution.x) || coords.y >= i32(system.resolution.y) { return; }

    // 1. Normalize screen coordinates to [-1, 1] and fix aspect ratio
    let res = vec2<f32>(f32(system.resolution.x), f32(system.resolution.y));
    var uv = (vec2<f32>(coords) / res) * 2.0 - 1.0;
    uv.y = -uv.y; // Flip Y so up is positive
    uv.x *= res.x / res.y;

    // 2. Generate Ray Direction using your Camera Matrix!
    let ro = camera.position;
    // The "focal length" is essentially the multiplier on the forward vector.
    let rd = normalize(camera.forward * 2.0 + camera.right * uv.x + camera.up * uv.y);

    // 3. Raymarching Loop
    var t = 0.0;
    let max_steps = 250;
    let max_dist = 100.0;
    let surf_dist = 0.001;

    var hit = false;
    var p = vec3<f32>(0.0);

    for (var i = 0; i < max_steps; i++) {
        p = ro + rd * t;
        let d = map_scene(p);
        t += d;

        if d < surf_dist {
            hit = true;
            break;
        }
        if t > max_dist {
            break;
        }
    }

    // 4. Shading & Lighting
    var color = vec3<f32>(0.05, 0.05, 0.05); // Background color

    if hit {
        let n = get_normal(p);
        let light_dir = normalize(vec3<f32>(1.0, 2.0, -1.0));
        let diffuse = max(dot(n, light_dir), 0.0);

        // Base color based on height (makes the bouncing sphere look distinct from the floor)
        let base_color = mix(vec3<f32>(0.2, 0.4, 0.8), vec3<f32>(0.8, 0.3, 0.2), clamp(p.y, 0.0, 1.0));

        color = base_color * diffuse;
    }

    textureStore(state_out, coords, vec4<f32>(color, 1.0));
}
