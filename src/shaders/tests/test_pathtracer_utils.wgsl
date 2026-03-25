#import rengin::pathtracer_utils::{onb, surface_area}
#import rengin::types::ObjectParam

// All entry points share bindings 0 (input) and 1 (output).

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// onb: input = (nx, ny, nz, _), output = 3 vec4s (col0.xyz+pad, col1.xyz+pad, col2.xyz+pad)
@compute @workgroup_size(1)
fn test_onb(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let n = vec3<f32>(inp.x, inp.y, inp.z);
    let basis = onb(n);
    let base = id.x * 3u;
    outputs_4f[base] = vec4<f32>(basis[0], 0.0);
    outputs_4f[base + 1u] = vec4<f32>(basis[1], 0.0);
    outputs_4f[base + 2u] = vec4<f32>(basis[2], 0.0);
}

// surface_area: input is ObjectParam (large struct), output = (area, _, _, _)
// ObjectParam must be read from a separate binding since it's a different type.
// We'll use a dedicated pair of bindings for surface_area tests.
@group(0) @binding(2) var<storage, read> sa_inputs: array<ObjectParam>;
@group(0) @binding(3) var<storage, read_write> sa_outputs: array<vec4<f32>>;

@compute @workgroup_size(1)
fn test_surface_area(@builtin(global_invocation_id) id: vec3<u32>) {
    let obj = sa_inputs[id.x];
    sa_outputs[id.x] = vec4<f32>(surface_area(obj), 0.0, 0.0, 0.0);
}
