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
        return normalToWorld((normal.normal2.xyz * intersection.uv.x + normal.normal3.xyz * intersection.uv.y + normal.normal1.xyz * (1.0 - intersection.uv.x - intersection.uv.y)),ob_param);
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

fn _schlick(cos_i:f32, r0: f32) -> f32 {
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn schlick(cos_i:f32, eta_t: f32, eta_i: f32) -> f32 {
    // // Use Schlick's approximation for reflectance.
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn _schlick_lazanyi(cos_i:f32, eta_t: f32, eta_i: f32, a: f32, alpha:f32) -> f32 {
    let r0 = pow((eta_i-eta_t) / (eta_i+eta_t),2.0);
    return _schlick(cos_i, r0) - a * cos_i * pow(1.0 - cos_i , alpha);
}

fn schlick_lazanyi(cos_i:f32, eta_t: f32, k: f32) -> f32 {
    return (pow(eta_t - 1.0, 2.0) + 4.0 * eta_t * pow(1.0 - cos_i,5.0) + pow(k,2.0)) / (pow(eta_t+1.0,2.0) + pow(k,2.0));
}