#import rengin::random::{
    f32_zero_to_one, f32_negone_to_one, rescale,
    init_pcg3d, next_pcg3d, rand_pcg3d,
    random_cosine_direction, random_uniform_direction,
    random_in_unit_sphere, random_uniform_on_hemisphere,
}

// All entry points share bindings 0 (input) and 1 (output).
// The Rust test provides appropriately-sized buffers per entry point.

@group(0) @binding(0) var<storage, read> inputs_4u: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// f32_zero_to_one: inputs_4u[i].x = seed
@compute @workgroup_size(1)
fn test_f32_zero_to_one(@builtin(global_invocation_id) id: vec3<u32>) {
    outputs_4f[id.x] = vec4<f32>(f32_zero_to_one(inputs_4u[id.x].x), 0.0, 0.0, 0.0);
}

// f32_negone_to_one: inputs_4u[i].x = seed
@compute @workgroup_size(1)
fn test_f32_negone_to_one(@builtin(global_invocation_id) id: vec3<u32>) {
    outputs_4f[id.x] = vec4<f32>(f32_negone_to_one(inputs_4u[id.x].x), 0.0, 0.0, 0.0);
}

// rescale: reinterpret inputs_4u as float via separate binding
// Actually we need float inputs for rescale. Use bitcast.
@compute @workgroup_size(1)
fn test_rescale(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    let value = bitcast<f32>(inp.x);
    let min_val = bitcast<f32>(inp.y);
    let max_val = bitcast<f32>(inp.z);
    outputs_4f[id.x] = vec4<f32>(rescale(value, min_val, max_val), 0.0, 0.0, 0.0);
}

// PCG3D determinism: inputs_4u[i] = (seed_x, seed_y, seed_z, _)
// outputs two consecutive states packed into 8 u32s
// We'll use outputs_4f to store two vec4<u32> bitcast as float
@compute @workgroup_size(1)
fn test_pcg3d_determinism(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(inp.x, inp.y, inp.z));
    let first = rand_pcg3d;
    next_pcg3d();
    let second = rand_pcg3d;
    let out_idx = id.x * 2u;
    outputs_4f[out_idx] = bitcast<vec4<f32>>(vec4<u32>(first, 0u));
    outputs_4f[out_idx + 1u] = bitcast<vec4<f32>>(vec4<u32>(second, 0u));
}

// random_cosine_direction: inputs_4u[i] = (seed_x, seed_y, seed_z, _)
@compute @workgroup_size(1)
fn test_random_cosine_direction(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(inp.x, inp.y, inp.z));
    let v = random_cosine_direction();
    outputs_4f[id.x] = vec4<f32>(v, 0.0);
}

// random_uniform_direction
@compute @workgroup_size(1)
fn test_random_uniform_direction(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(inp.x, inp.y, inp.z));
    let v = random_uniform_direction();
    outputs_4f[id.x] = vec4<f32>(v, 0.0);
}

// random_in_unit_sphere
@compute @workgroup_size(1)
fn test_random_in_unit_sphere(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(inp.x, inp.y, inp.z));
    let v = random_in_unit_sphere();
    outputs_4f[id.x] = vec4<f32>(v, 0.0);
}

// random_uniform_on_hemisphere
@compute @workgroup_size(1)
fn test_random_uniform_on_hemisphere(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4u[id.x];
    init_pcg3d(vec3<u32>(inp.x, inp.y, inp.z));
    let v = random_uniform_on_hemisphere();
    outputs_4f[id.x] = vec4<f32>(v, 0.0);
}
