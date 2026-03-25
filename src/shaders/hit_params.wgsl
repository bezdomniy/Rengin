#define_import_path rengin::hit_params

#import rengin::types::{ObjectParam, Intersection, Normal, HitParams, Ray}
#import rengin::constants::EPSILON
#import rengin::pt_bindings::{object_params, normal_nodes}

fn normalToWorld(normal: vec3<f32>, ob_param: ObjectParam) -> vec3<f32>
{
    return normalize((transpose(ob_param.inverse_transform) * vec4<f32>(normal,0.0)).xyz);
}

fn normalAt(p: vec3<f32>, intersection: Intersection, typeEnum: u32) -> vec3<f32> {
    let ob_param = object_params.ObjectParams[intersection.model_id];
    if (typeEnum == 0u) { //Sphere
        let objectPoint = (ob_param.inverse_transform * vec4<f32>(p,1.0)).xyz;
        return normalToWorld(objectPoint,ob_param);
    }
    else if (typeEnum == 1u) { //Plane
        return normalToWorld(vec3<f32>(0.0, 1.0, 0.0),ob_param);
    }
    else if (typeEnum == 2u) { //Cube
        let objectPoint = (ob_param.inverse_transform * vec4<f32>(p,1.0)).xyz;
        let p1 = abs(objectPoint.x);
        let p2 = abs(objectPoint.y);
        let p3 = abs(objectPoint.z);
        var objectNormal = normalize(vec3<f32>(objectPoint.x, 0.0, 0.0));

        if (p2 > p1 && p2 > p3) {
            objectNormal = normalize(vec3<f32>(0.0, objectPoint.y, 0.0));
        }
        else if (p3 > p1 && p3 > p2) {
            objectNormal = normalize(vec3<f32>(0.0, 0.0,objectPoint.z));
        }

        return normalToWorld(objectNormal,ob_param);
    }
    else { //Model
        let normal: Normal = normal_nodes.Normals[intersection.id];
        return normalToWorld((normal.normal_b.xyz * intersection.uv.x + normal.normal_c.xyz * intersection.uv.y + normal.normal_a.xyz * (1.0 - intersection.uv.x - intersection.uv.y)),ob_param);
        // n.w = 0.0;
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

