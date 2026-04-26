// ==========================================
// UNIFORMS & STRUCTURES
// ==========================================

struct SystemUniform {
    resolution: vec2<u32>,
    time: f32,
    dt: f32,
    frame: u32,
    tick: u32,
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

struct Material {
    material_type: u32,
    fuzz: f32,
    color: vec3<f32>,
}

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct Sphere {
    position: vec3<f32>,
    radius: f32,
    material: Material,
}

struct Scene {
    objects: array<Sphere, 5>,
}

// ==========================================
// Bindings
// ==========================================

@group(0) @binding(0) var<uniform> system: SystemUniform;

@group(1) @binding(0) var render_target: texture_2d<f32>;
@group(1) @binding(1) var<uniform> camera: CameraUniform;
@group(1) @binding(2) var state_in: texture_2d<f32>;
@group(1) @binding(3) var state_out: texture_storage_2d<rgba16float, write>;

// ==========================================
// Utility Functions
// ==========================================

fn hash(p: u32) -> u32 {
    var x = p;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = ((x >> 16u) ^ x) * 0x45d9f3bu;
    x = (x >> 16u) ^ x;
    return x;
}

fn random_f32(seed: ptr<function, u32>) -> f32 {
    *seed = hash(*seed);
    return f32(*seed) / f32(0xffffffffu);
}

fn random_f32_range(seed: ptr<function, u32>, min: f32, max: f32) -> f32 {
    return min + (max - min) * random_f32(seed);
}

fn random_vec3(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(random_f32(seed), random_f32(seed), random_f32(seed));
}

fn random_vec3_range(seed: ptr<function, u32>, min: f32, max: f32) -> vec3<f32> {
    return vec3<f32>(
        random_f32_range(seed, min, max),
        random_f32_range(seed, min, max),
        random_f32_range(seed, min, max),
    );
}

fn random_unit_vector(seed: ptr<function, u32>) -> vec3<f32> {
    let a = random_f32_range(seed, 0.0, 6.2831853); // 2 * PI
    let z = random_f32_range(seed, -1.0, 1.0);
    let r = sqrt(1.0 - z * z);
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn random_on_hemisphere(seed: ptr<function, u32>, normal: vec3<f32>) -> vec3<f32> {
    let on_unit_sphere = random_unit_vector(seed);
    if dot(on_unit_sphere, normal) > 0.0 {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

// ==========================================
// VERTEX & FRAGMENT
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
    let linear_color = textureLoad(render_target, coords, 0).rgb;

    let gamma_corrected = sqrt(linear_color);

    return vec4<f32>(gamma_corrected, 1.0);
}

// ==========================================
// COMPUTE
// ==========================================

fn hit_sphere(sphere_center: vec3<f32>, sphere_radius: f32, ray: Ray) -> f32 {
    let oc = sphere_center - ray.origin;
    var a = dot(ray.dir, ray.dir);
    var h = dot(ray.dir, oc);
    var c = dot(oc, oc) - sphere_radius * sphere_radius;
    var discriminate = h * h - a * c;
    if discriminate < 0 {
        return -1.0;
    } else {
        return ((h - sqrt(discriminate)) / a);
    }
}

@compute
@workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let SAMPLES_PER_FRAME = 20u;
    let max_bounces = 10u;

    let coords = vec2<i32>(global_id.xy);
    if coords.x >= i32(system.resolution.x) || coords.y >= i32(system.resolution.y) { return; }

    let res = vec2<f32>(f32(system.resolution.x), f32(system.resolution.y));

    var total_color = vec3<f32>(0.0);

    var seed = hash(global_id.x + global_id.y * system.resolution.x + system.tick * 12345u);

    let material_1 = Material(0, 0.0, vec3<f32>(0.1, 0.1, 0.1));
    let material_2 = Material(0, 0.0, vec3<f32>(1.0, 0.1, 0.1));
    let material_3 = Material(0, 0.0, vec3<f32>(0.1, 0.1, 1.0));
    let material_4 = Material(0, 0.0, vec3<f32>(0.1, 1.0, 0.1));
    let material_5 = Material(1, 0.01, vec3<f32>(1.0, 1.0, 1.0));

    let sphere_1 = Sphere(vec3<f32>(0.0, -100.5, -1.0), 100.0, material_1);
    let sphere_2 = Sphere(vec3<f32>(0.0, 0.0, -1.0), 0.5, material_2);
    let sphere_3 = Sphere(vec3<f32>(-1.0, 0.0, -1.0), 0.5, material_3);
    let sphere_4 = Sphere(vec3<f32>(1.0, 0.0, -1.0), 0.5, material_4);
    //let sphere_5 = Sphere(vec3<f32>(2.0, 1.0 * cos(f32(system.tick) * 0.01) + 1.5, -2.0), 1.0, mamaterial_5);
    let sphere_5 = Sphere(vec3<f32>(2.0, 1.5, -2.0), 1.0, material_5);

    let scene = Scene(array(sphere_1, sphere_2, sphere_3, sphere_4, sphere_5));

    for (var s = 0u; s < SAMPLES_PER_FRAME; s = s + 1u) {
        let offset = vec2<f32>(random_f32(&seed), random_f32(&seed)) - 0.5;
        let jittered_coords = vec2<f32>(coords) + offset;
        var uv = (jittered_coords / res) * 2.0 - 1.0;
        uv.y = -uv.y;
        uv.x *= res.x / res.y;

        let camera_position = camera.position;
        let camera_direction = normalize(camera.forward * 2.0 + camera.right * uv.x + camera.up * uv.y);

        var current_ray = Ray(camera_position, camera_direction);
        var ray_color = vec3<f32>(1.0);
        var final_color = vec3<f32>(0.0);

        for (var bounce = 0u; bounce < max_bounces; bounce++) {

            var closest_so_far = 999999.0;
            var hit_anything = false;
            var hit_index = 0u;

            for (var i: u32 = 0u; i < 5; i = i + 1u) {
                var sphere = scene.objects[i];
                let t = hit_sphere(sphere.position, sphere.radius, current_ray);

                if t > 0.001 && t < closest_so_far {
                    closest_so_far = t;
                    hit_anything = true;
                    hit_index = i;
                    //let N = normalize(ray.origin + t * ray.dir - sphere.position);
                    //color = 0.5 * vec3<f32>(N.x + 1.0, N.y + 1.0, N.z + 1.0);
                }
            }

            if hit_anything {
                let sphere = scene.objects[hit_index];

                let hit_point = current_ray.origin + closest_so_far * current_ray.dir;
                let normal = normalize(hit_point - sphere.position);

                var scatter_dir: vec3<f32>;

                if sphere.material.material_type == 0u {
                    scatter_dir = normal + random_unit_vector(&seed);

                    if length(scatter_dir) < 0.001 {
                        scatter_dir = normal;
                    }
                } else {
                    let reflected = reflect(current_ray.dir, normal);
                    scatter_dir = reflected + sphere.material.fuzz * random_unit_vector(&seed);

                    if dot(scatter_dir, normal) <= 0.0 {
                        ray_color = vec3<f32>(0.0);
                        break;
                    }
                }

                current_ray = Ray(hit_point, normalize(scatter_dir));
                ray_color *= sphere.material.color;
            } else {
                let a = 0.5 * (current_ray.dir.y + 1.0);
                //var sky_color = (1.0 - a) * vec3<f32>(1.0, 1.0, 1.0) + a * vec3<f32>(0.2, 0.5, 1.0);
                var sky_color = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.3, 0.6, 1.5), a);
                final_color = ray_color * sky_color;
                break;
            }
        }
        total_color += final_color;
    }

    total_color = total_color / f32(SAMPLES_PER_FRAME);

    // Anti-Aliasing
    if system.frame == 0u {
        textureStore(state_out, coords, vec4<f32>(total_color, 1.0));
    } else {
        let prev_color = textureLoad(state_in, coords, 0).rgb;
        let accumulated = (prev_color * f32(system.frame) + total_color) / (f32(system.frame) + 1.0);
        textureStore(state_out, coords, vec4<f32>(accumulated, 1.0));
    }
}
