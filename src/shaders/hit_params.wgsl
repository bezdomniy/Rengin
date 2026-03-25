#define_import_path rengin::hit_params

#import rengin::types::{ObjectParam, Intersection, Normal, HitParams, Ray}
#import rengin::constants::EPSILON
#import rengin::pt_bindings::{object_params, normal_nodes}
#import rengin::hit_params_math::{normalToWorld_mat, normalAt_params}

fn normalToWorld(normal: vec3<f32>, ob_param: ObjectParam) -> vec3<f32>
{
    return normalToWorld_mat(normal, ob_param.inverse_transform);
}

fn normalAt(p: vec3<f32>, intersection: Intersection, typeEnum: u32) -> vec3<f32> {
    let ob_param = object_params.ObjectParams[intersection.model_id];
    if typeEnum <= 2u {
        // Sphere, Plane, Cube — use the pure math version
        return normalAt_params(p, typeEnum, ob_param.transform, ob_param.inverse_transform);
    }
    else { //Model — requires buffer access for per-vertex normals
        let normal: Normal = normal_nodes.Normals[intersection.id];
        return normalToWorld((normal.normal_b.xyz * intersection.uv.x + normal.normal_c.xyz * intersection.uv.y + normal.normal_a.xyz * (1.0 - intersection.uv.x - intersection.uv.y)),ob_param);
    }
    return vec3<f32>(0.0);
}


fn getHitParams(ray: Ray, intersection: Intersection, typeEnum: u32) -> HitParams
{
    var hitParams: HitParams;
    let normal_ray_d = normalize(ray.rayD);
    hitParams.p =
        ray.rayO + normal_ray_d * intersection.closestT;
    // TODO check that uv only null have using none-uv normalAt version
    hitParams.normalv =
        normalAt(hitParams.p, intersection, typeEnum);
    // hitParams.eyev = -ray.rayD;
    hitParams.eyev = -normal_ray_d;

    // hitParams.front_face = dot(hitParams.normalv, hitParams.eyev) < 0.0;
    hitParams.front_face = dot(ray.rayD, hitParams.normalv) < 0.0;
    if (!hitParams.front_face)
    {
        hitParams.normalv = -hitParams.normalv;
    }

    hitParams.reflectv =
        reflect(normal_ray_d, hitParams.normalv);
    hitParams.overPoint =
        hitParams.p + hitParams.normalv * EPSILON;
    hitParams.underPoint =
        hitParams.p - hitParams.normalv * EPSILON;

    return hitParams;
}
