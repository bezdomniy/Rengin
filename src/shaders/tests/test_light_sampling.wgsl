#import rengin::random::{init_pcg3d, next_pcg3d, rand_pcg3d, f32_zero_to_one, random_to_sphere}
#import rengin::pathtracer_math::{light_pdf_sphere, scattering_pdf, mis_pdf}

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// ---- light_pdf for sphere: input = (radius, distance, _, _) ----
@compute @workgroup_size(1)
fn test_light_pdf_sphere(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let pdf = light_pdf_sphere(inp.x, inp.y);
    outputs_4f[id.x] = vec4<f32>(pdf, 0.0, 0.0, 0.0);
}

// ---- scattering_pdf: input = (dir.x, dir.y, dir.z, normal_selector) ----
@compute @workgroup_size(1)
fn test_scattering_pdf(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let direction = vec3<f32>(inp.x, inp.y, inp.z);
    var normal = vec3<f32>(0.0, 0.0, 1.0);
    if inp.w > 0.5 && inp.w < 1.5 {
        normal = vec3<f32>(0.0, 1.0, 0.0);
    } else if inp.w > 1.5 {
        normal = vec3<f32>(1.0, 0.0, 0.0);
    }
    let pdf = scattering_pdf(direction, normal);
    outputs_4f[id.x] = vec4<f32>(pdf, 0.0, 0.0, 0.0);
}

// ---- MIS weight: input = (scattering_pdf, light_pdf, p_scatter, _) ----
@compute @workgroup_size(1)
fn test_mis_weight(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let pdf = mis_pdf(inp.x, inp.y, inp.z);
    outputs_4f[id.x] = vec4<f32>(pdf, 0.0, 0.0, 0.0);
}

// ---- random_to_sphere: inputs_4u = seeds, inputs_4f = (radius, dist_sq, _, _) ----
@group(0) @binding(2) var<storage, read> inputs_4u: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> outputs_4f_b: array<vec4<f32>>;

@compute @workgroup_size(1)
fn test_random_to_sphere(@builtin(global_invocation_id) id: vec3<u32>) {
    let seed = inputs_4u[id.x];
    let params = inputs_4f[id.x];
    init_pcg3d(vec3<u32>(seed.x, seed.y, seed.z));
    let v = random_to_sphere(params.x, params.y);
    outputs_4f_b[id.x] = vec4<f32>(v, 0.0);
}
