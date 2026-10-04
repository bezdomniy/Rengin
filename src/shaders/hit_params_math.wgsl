#define_import_path rengin::hit_params_math

// Pure math functions for surface normal and hit parameter computation.
// No bind group dependencies — suitable for both production and testing.

/// Transform a normal from object space to world space using the inverse transpose.
fn normalToWorld_mat(normal: vec3<f32>, inverse_transform: mat4x4<f32>) -> vec3<f32> {
    return normalize((transpose(inverse_transform) * vec4<f32>(normal, 0.0)).xyz);
}

/// Compute the surface normal at point `p` for primitive `typeEnum` (0=sphere, 1=plane, 2=cube).
/// Uses explicit transforms instead of buffer lookups.
fn normalAt_params(p: vec3<f32>, typeEnum: u32, transform: mat4x4<f32>, inverse_transform: mat4x4<f32>) -> vec3<f32> {
    if typeEnum == 0u { // Sphere
        let objectPoint = (inverse_transform * vec4<f32>(p, 1.0)).xyz;
        return normalToWorld_mat(objectPoint, inverse_transform);
    } else if typeEnum == 1u { // Plane
        return normalToWorld_mat(vec3<f32>(0.0, 1.0, 0.0), inverse_transform);
    } else if typeEnum == 2u { // Cube
        let objectPoint = (inverse_transform * vec4<f32>(p, 1.0)).xyz;
        let p1 = abs(objectPoint.x);
        let p2 = abs(objectPoint.y);
        let p3 = abs(objectPoint.z);
        var objectNormal = normalize(vec3<f32>(objectPoint.x, 0.0, 0.0));
        if p2 > p1 && p2 > p3 {
            objectNormal = normalize(vec3<f32>(0.0, objectPoint.y, 0.0));
        } else if p3 > p1 && p3 > p2 {
            objectNormal = normalize(vec3<f32>(0.0, 0.0, objectPoint.z));
        }
        return normalToWorld_mat(objectNormal, inverse_transform);
    }
    return vec3<f32>(0.0);
}

// A hit point computed as o + t * d carries rounding error proportional to
// |o| and t. (Error in t mostly moves the point along the ray; only its
// component along the normal matters, and that stays proportional to t even at
// grazing angles.) New rays start this far off the surface: a generous
// multiple of that bound, so it holds at any scene scale while staying far
// smaller than a fixed epsilon in small scenes.
const SELF_INTERSECTION_SCALE: f32 = 64.0 * 1.1920929e-7;

fn selfIntersectionOffset(ray_o: vec3<f32>, t: f32) -> f32 {
    let o = abs(ray_o);
    return SELF_INTERSECTION_SCALE * (max(o.x, max(o.y, o.z)) + abs(t));
}

struct OrientedNormals {
    geometric: vec3<f32>,
    shading: vec3<f32>,
    front_face: bool,
}

/// Flips both normals to the side the ray came from. The side is decided by
/// the geometric normal: an interpolated shading normal can point away from
/// the ray at grazing angles even though the ray hit the front of the
/// surface. Mesh winding need not match the vertex normals, so which side is
/// "outward" (for entering and leaving media) comes from the shading normal.
fn orientNormals(ray_d: vec3<f32>, geometric_n: vec3<f32>, shading_n: vec3<f32>) -> OrientedNormals {
    let outward = select(-geometric_n, geometric_n, dot(geometric_n, shading_n) >= 0.0);
    let front_face = dot(ray_d, outward) < 0.0;
    let side = select(-1.0, 1.0, front_face);
    return OrientedNormals(outward * side, shading_n * side, front_face);
}

struct HitResult {
    p: vec3<f32>,
    normalv: vec3<f32>,
    geometric_normalv: vec3<f32>,
    eyev: vec3<f32>,
    reflectv: vec3<f32>,
    overPoint: vec3<f32>,
    underPoint: vec3<f32>,
    front_face: bool,
}

/// Compute hit parameters from ray, intersection distance, and transforms.
/// Pure math version — no buffer lookups.
fn getHitParams_params(
    ray_o: vec3<f32>,
    ray_d: vec3<f32>,
    closestT: f32,
    typeEnum: u32,
    transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
) -> HitResult {
    var result: HitResult;
    let normal_ray_d = normalize(ray_d);
    result.p = ray_o + normal_ray_d * closestT;
    // Analytic primitives have no separate shading normal.
    let normal = normalAt_params(result.p, typeEnum, transform, inverse_transform);
    let oriented = orientNormals(normal_ray_d, normal, normal);
    result.normalv = oriented.shading;
    result.geometric_normalv = oriented.geometric;
    result.front_face = oriented.front_face;
    result.eyev = -normal_ray_d;
    result.reflectv = reflect(normal_ray_d, result.normalv);
    let offset = selfIntersectionOffset(ray_o, closestT);
    result.overPoint = result.p + result.geometric_normalv * offset;
    result.underPoint = result.p - result.geometric_normalv * offset;
    return result;
}
