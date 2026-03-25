#define_import_path rengin::hit_params_math

#import rengin::constants::EPSILON

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

struct HitResult {
    p: vec3<f32>,
    normalv: vec3<f32>,
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
    result.normalv = normalAt_params(result.p, typeEnum, transform, inverse_transform);
    result.eyev = -normal_ray_d;
    result.front_face = dot(ray_d, result.normalv) < 0.0;
    if !result.front_face {
        result.normalv = -result.normalv;
    }
    result.reflectv = reflect(normal_ray_d, result.normalv);
    result.overPoint = result.p + result.normalv * EPSILON;
    result.underPoint = result.p - result.normalv * EPSILON;
    return result;
}
