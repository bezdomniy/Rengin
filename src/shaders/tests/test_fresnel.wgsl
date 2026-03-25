#import rengin::fresnel

struct SchlickInput {
    cos_i: f32,
    r0: f32,
    _pad0: f32,
    _pad1: f32,
}

struct SchlickFullInput {
    cos_i: f32,
    eta_t: f32,
    eta_i: f32,
    _pad: f32,
}

struct SchlickLazanyiInput {
    cos_i: f32,
    eta_t: f32,
    k: f32,
    _pad: f32,
}

struct F32Output {
    result: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// _schlick: inputs_4f[i] = (cos_i, r0, _, _)
@compute @workgroup_size(1)
fn test_schlick_base(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    outputs_4f[id.x] = vec4<f32>(fresnel::_schlick(inp.x, inp.y), 0.0, 0.0, 0.0);
}

// schlick: inputs_4f[i] = (cos_i, eta_t, eta_i, _)
@compute @workgroup_size(1)
fn test_schlick_full(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    outputs_4f[id.x] = vec4<f32>(fresnel::schlick(inp.x, inp.y, inp.z), 0.0, 0.0, 0.0);
}

// schlick_lazanyi: inputs_4f[i] = (cos_i, eta_t, k, _)
@compute @workgroup_size(1)
fn test_schlick_lazanyi(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    outputs_4f[id.x] = vec4<f32>(fresnel::schlick_lazanyi(inp.x, inp.y, inp.z), 0.0, 0.0, 0.0);
}
