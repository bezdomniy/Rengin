#define_import_path rengin::hit_params

#import rengin::types::{ObjectParam, Intersection, Normal, HitParams, Ray}
#import rengin::pt_bindings::{object_params, leaf_nodes, normal_nodes}
#import rengin::hit_params_math::{normalToWorld_mat, normalAt_params, orientNormals, selfIntersectionOffset}

fn normalToWorld(normal: vec3<f32>, ob_param: ObjectParam) -> vec3<f32>
{
    return normalToWorld_mat(normal, ob_param.inverse_transform);
}

fn getHitParams(ray: Ray, intersection: Intersection, typeEnum: u32) -> HitParams
{
    var hitParams: HitParams;
    let normal_ray_d = normalize(ray.rayD);
    hitParams.p =
        ray.rayO + normal_ray_d * intersection.closestT;

    let ob_param = object_params.ObjectParams[intersection.model_id];
    var geometric_normal: vec3<f32>;
    var shading_normal: vec3<f32>;
    if typeEnum <= 2u {
        // Sphere, Plane, Cube: the analytic normal is exact
        shading_normal = normalAt_params(hitParams.p, typeEnum, ob_param.transform, ob_param.inverse_transform);
        geometric_normal = shading_normal;
    } else {
        let tri = leaf_nodes.LeafNodes[intersection.id];
        geometric_normal = normalToWorld(cross(tri.vertex_b - tri.vertex_a, tri.vertex_c - tri.vertex_a), ob_param);
        let normal: Normal = normal_nodes.Normals[intersection.id];
        shading_normal = normalToWorld((normal.normal_b.xyz * intersection.uv.x + normal.normal_c.xyz * intersection.uv.y + normal.normal_a.xyz * (1.0 - intersection.uv.x - intersection.uv.y)), ob_param);
    }

    let oriented = orientNormals(normal_ray_d, geometric_normal, shading_normal);
    hitParams.normalv = oriented.shading;
    hitParams.geometric_normalv = oriented.geometric;
    hitParams.front_face = oriented.front_face;
    hitParams.eyev = -normal_ray_d;

    hitParams.reflectv =
        reflect(normal_ray_d, hitParams.normalv);
    // Offset along the geometric normal: the shading normal can point into
    // the surface.
    let offset = selfIntersectionOffset(ray.rayO, intersection.closestT);
    hitParams.overPoint =
        hitParams.p + hitParams.geometric_normalv * offset;
    hitParams.underPoint =
        hitParams.p - hitParams.geometric_normalv * offset;

    return hitParams;
}
