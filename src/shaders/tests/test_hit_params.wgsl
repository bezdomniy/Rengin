#import rengin::hit_params_math::{normalAt_params, getHitParams_params, HitResult, orientNormals}

@group(0) @binding(0) var<storage, read> inputs_4f: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs_4f: array<vec4<f32>>;

// ---- normalAt for unit sphere: input = (px, py, pz, type_enum) ----
// Identity transform
@compute @workgroup_size(1)
fn test_normal_at(@builtin(global_invocation_id) id: vec3<u32>) {
    let inp = inputs_4f[id.x];
    let p = vec3<f32>(inp.x, inp.y, inp.z);
    let type_enum = u32(inp.w);
    let identity = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    let n = normalAt_params(p, type_enum, identity, identity);
    outputs_4f[id.x] = vec4<f32>(n, 0.0);
}

// ---- normalAt with scale transform: input = (px, py, pz, _), scale in next vec4 ----
@compute @workgroup_size(1)
fn test_normal_at_scaled(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 2u;
    let inp = inputs_4f[base];
    let scale_inp = inputs_4f[base + 1u];
    let p = vec3<f32>(inp.x, inp.y, inp.z);
    let type_enum = u32(inp.w);
    let sx = scale_inp.x;
    let sy = scale_inp.y;
    let sz = scale_inp.z;
    let transform = mat4x4<f32>(
        vec4<f32>(sx, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, sy, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, sz, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    let inverse_transform = mat4x4<f32>(
        vec4<f32>(1.0 / sx, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0 / sy, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0 / sz, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    let n = normalAt_params(p, type_enum, transform, inverse_transform);
    outputs_4f[id.x] = vec4<f32>(n, 0.0);
}

// ---- getHitParams: input = ray_o(3f), closestT, ray_d(3f), type_enum ----
// Output: 7 vec4s per test case (p, normalv, eyev, reflectv, overPoint, underPoint, front_face)
@compute @workgroup_size(1)
fn test_get_hit_params(@builtin(global_invocation_id) id: vec3<u32>) {
    let base_in = id.x * 2u;
    let inp0 = inputs_4f[base_in];
    let inp1 = inputs_4f[base_in + 1u];
    let ray_o = inp0.xyz;
    let closest_t = inp0.w;
    let ray_d = inp1.xyz;
    let type_enum = u32(inp1.w);
    let identity = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
    );
    let hp = getHitParams_params(ray_o, ray_d, closest_t, type_enum, identity, identity);
    let base_out = id.x * 7u;
    outputs_4f[base_out] = vec4<f32>(hp.p, 0.0);
    outputs_4f[base_out + 1u] = vec4<f32>(hp.normalv, 0.0);
    outputs_4f[base_out + 2u] = vec4<f32>(hp.eyev, 0.0);
    outputs_4f[base_out + 3u] = vec4<f32>(hp.reflectv, 0.0);
    outputs_4f[base_out + 4u] = vec4<f32>(hp.overPoint, 0.0);
    outputs_4f[base_out + 5u] = vec4<f32>(hp.underPoint, 0.0);
    outputs_4f[base_out + 6u] = vec4<f32>(select(0.0, 1.0, hp.front_face), 0.0, 0.0, 0.0);
}

// ---- reflection vector test: input = (dx, dy, dz, _), (nx, ny, nz, _) ----
@compute @workgroup_size(1)
fn test_reflect(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 2u;
    let d = inputs_4f[base].xyz;
    let n = inputs_4f[base + 1u].xyz;
    let r = reflect(d, n);
    outputs_4f[id.x] = vec4<f32>(r, 0.0);
}

// ---- orientNormals: input = ray_d, geometric_n, shading_n (3 vec4s) ----
// Output: geometric, shading, front_face (3 vec4s)
@compute @workgroup_size(1)
fn test_orient_normals(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 3u;
    let oriented = orientNormals(inputs_4f[base].xyz, inputs_4f[base + 1u].xyz, inputs_4f[base + 2u].xyz);
    outputs_4f[base] = vec4<f32>(oriented.geometric, 0.0);
    outputs_4f[base + 1u] = vec4<f32>(oriented.shading, 0.0);
    outputs_4f[base + 2u] = vec4<f32>(select(0.0, 1.0, oriented.front_face), 0.0, 0.0, 0.0);
}
