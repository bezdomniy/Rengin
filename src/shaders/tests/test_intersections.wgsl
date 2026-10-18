#import rengin::types::{Ray, Intersection, NodeInner}
#import rengin::constants::{EPSILON, MAXLEN}
#import rengin::intersects_math::{intersectSphere, intersectPlane, intersectCube, intersectAABB_inv, intersectWorldAABB, intersectTriangleVertices}

// All entry points share bindings 0 (input) and 1 (output).
// Input/output interpretation differs per entry point but layout is the same.

// Ray tests: input is 12 floats (rayO.xyz, ri, rayD.xyz, pos_as_float, throughput.xyzw)
// Output is: uv.xy, id_as_float, closestT, model_id_as_float, pad, pad, pad

@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;

fn load_ray(idx: u32) -> Ray {
    let base = idx * 3u;
    let a = inputs[base];
    let b = inputs[base + 1u];
    let c = inputs[base + 2u];
    return Ray(a.xyz, a.w, b.xyz, bitcast<u32>(b.w), c);
}

fn store_intersection(idx: u32, r: Intersection) {
    let base = idx * 2u;
    outputs[base] = vec4<f32>(r.uv, bitcast<f32>(r.id), r.closestT);
    outputs[base + 1u] = vec4<f32>(bitcast<f32>(r.model_id), 0.0, 0.0, 0.0);
}

// --- intersectSphere ---
@compute @workgroup_size(1)
fn test_intersect_sphere(@builtin(global_invocation_id) id: vec3<u32>) {
    let ray = load_ray(id.x);
    let no_hit = Intersection(vec2<f32>(0.0), -1, MAXLEN, 0u);
    let result = intersectSphere(ray, no_hit, 0u);
    store_intersection(id.x, result);
}

// --- intersectPlane ---
@compute @workgroup_size(1)
fn test_intersect_plane(@builtin(global_invocation_id) id: vec3<u32>) {
    let ray = load_ray(id.x);
    let no_hit = Intersection(vec2<f32>(0.0), -1, MAXLEN, 0u);
    let result = intersectPlane(ray, no_hit, 0u);
    store_intersection(id.x, result);
}

// --- intersectCube ---
@compute @workgroup_size(1)
fn test_intersect_cube(@builtin(global_invocation_id) id: vec3<u32>) {
    let ray = load_ray(id.x);
    let no_hit = Intersection(vec2<f32>(0.0), -1, MAXLEN, 0u);
    let result = intersectCube(ray, no_hit, 0u);
    store_intersection(id.x, result);
}

// --- intersectAABB_inv ---
// Input: ray_o.xyz, pad, inv_d.xyz, closest_t, node_first.xyz, skip_ptr, node_second.xyz, prim_end
// = 4 vec4s per test case
@compute @workgroup_size(1)
fn test_intersect_aabb(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 4u;
    let a = inputs[base];
    let b = inputs[base + 1u];
    let c = inputs[base + 2u];
    let d = inputs[base + 3u];
    let node = NodeInner(c.xyz, bitcast<u32>(c.w), d.xyz, bitcast<u32>(d.w));
    let result = intersectAABB_inv(a.xyz, b.xyz, node, b.w);
    outputs[id.x] = vec4<f32>(select(0.0, 1.0, result), 0.0, 0.0, 0.0);
}

// --- intersectWorldAABB ---
// Input: rayO.xyz, pad, rayD.xyz, closest_t, aabb_min.xyz, pad, aabb_max.xyz, pad
// = 4 vec4s per test case
@compute @workgroup_size(1)
fn test_intersect_world_aabb(@builtin(global_invocation_id) id: vec3<u32>) {
    let base = id.x * 4u;
    let a = inputs[base];
    let b = inputs[base + 1u];
    let c = inputs[base + 2u];
    let d = inputs[base + 3u];
    let ray = Ray(a.xyz, 1.0, b.xyz, 0u, vec4<f32>(1.0));
    let result = intersectWorldAABB(ray, c.xyz, d.xyz, b.w);
    outputs[id.x] = vec4<f32>(select(0.0, 1.0, result), 0.0, 0.0, 0.0);
}

// --- intersectTriangleVertices on two triangles sharing an edge ---
// Input: quad corners a, b, c, d (triangles abc and acd share edge ac), then
// ray_o.xyz, ray_d.xyz per case (2 vec4s).
// Output per case: (hit abc, hit acd, t abc, t acd)
@compute @workgroup_size(64)
fn test_triangle_pair(@builtin(global_invocation_id) id: vec3<u32>) {
    let case_count = (arrayLength(&inputs) - 4u) / 2u;
    if id.x >= case_count {
        return;
    }
    let a = inputs[0].xyz;
    let b = inputs[1].xyz;
    let c = inputs[2].xyz;
    let d = inputs[3].xyz;
    let ray_o = inputs[4u + 2u * id.x].xyz;
    let ray_d = inputs[5u + 2u * id.x].xyz;
    let first = intersectTriangleVertices(ray_o, ray_d, a, b, c, MAXLEN);
    let second = intersectTriangleVertices(ray_o, ray_d, a, c, d, MAXLEN);
    outputs[id.x] = vec4<f32>(select(0.0, 1.0, first.hit), select(0.0, 1.0, second.hit), first.t, second.t);
}

// --- intersectTriangleVertices: single triangle, returns (hit, u, v, t) ---
// Input: a, b, c, ray_o, ray_d (5 vec4s)
@compute @workgroup_size(1)
fn test_intersect_triangle(@builtin(global_invocation_id) id: vec3<u32>) {
    let hit = intersectTriangleVertices(inputs[3].xyz, inputs[4].xyz, inputs[0].xyz, inputs[1].xyz, inputs[2].xyz, MAXLEN);
    outputs[0] = vec4<f32>(select(0.0, 1.0, hit.hit), hit.uv, hit.t);
}
