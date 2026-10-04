#define_import_path rengin::intersects_math

#import rengin::types::{NodeInner, Ray, Intersection}
#import rengin::constants::MAXLEN

// Hits only need t > 0: rays leaving a surface start offset from it by
// selfIntersectionOffset, which scales with the scene instead of being a fixed
// distance.

// Slab tests must be inclusive and slightly conservative. BVH nodes around
// axis-aligned coplanar triangles have zero thickness, so rays that hit them
// get t_min == t_max, and nearly flat nodes can round to t_max < t_min. Scaling
// t_max (pbrt's 1 + 2 * gamma(3), widened for WGSL's 2.5 ULP division) keeps
// those hits; a false positive only costs an extra node test.
const AABB_T_MAX_SCALE: f32 = 1.0 + 1e-5;

fn slabsOverlap(t_min: f32, t_max: f32, closest_t: f32) -> bool {
    return t_max * AABB_T_MAX_SCALE >= max(t_min, 0.0) && t_min < closest_t;
}

fn intersectAABB_inv(ray_o: vec3<f32>, inv_d: vec3<f32>, inner_node: NodeInner, closest_t: f32) -> bool {
    let t0s = (inner_node.first - ray_o) * inv_d;
    let t1s = (inner_node.second - ray_o) * inv_d;
    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);
    let t_min = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    let t_max = min(tbigger.x, min(tbigger.y, tbigger.z));
    return slabsOverlap(t_min, t_max, closest_t);
}

fn intersectSphere(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    let a = dot(ray.rayD, ray.rayD);
    let b = 2.0 * dot(ray.rayD, ray.rayO);
    let c = dot(ray.rayO, ray.rayO) - 1.0;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return inIntersection;
    }

    let t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    let t2 = (-b + sqrt(discriminant)) / (2.0 * a);

    if t1 < inIntersection.closestT || t2 < inIntersection.closestT {
        if t1 < t2 && t1 > 0.0 {
            return Intersection(vec2<f32>(0.0), 0, t1, object_id);
        }

        if t2 > 0.0 && t2 < inIntersection.closestT {
            return Intersection(vec2<f32>(0.0), 0, t2, object_id);
        }
    }
    return inIntersection;
}

fn intersectPlane(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    // No parallel check: rayD is in object space and not normalized, so a
    // threshold on it would depend on the plane's scale. A ray parallel to
    // the plane gives an infinite or NaN t, which the comparisons reject.
    let t: f32 = -ray.rayO.y / ray.rayD.y;

    if t < inIntersection.closestT && t > 0.0 {
        return Intersection(vec2<f32>(0.0), 0, t, object_id);
    }
    return inIntersection;
}

fn intersectCube(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    let inv_d = 1.0 / ray.rayD;
    let t0s = (-1.0 - ray.rayO) * inv_d;
    let t1s = (1.0 - ray.rayO) * inv_d;
    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);
    let t_min = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    let t_max = min(tbigger.x, min(tbigger.y, tbigger.z));

    if t_max <= t_min {
        return inIntersection;
    }

    if t_min < inIntersection.closestT && t_min > 0.0 {
        return Intersection(vec2<f32>(0.0), 0, t_min, object_id);
    } else if t_max < inIntersection.closestT && t_max > 0.0 {
        return Intersection(vec2<f32>(0.0), 0, t_max, object_id);
    }
    return inIntersection;
}


// World-space AABB ray test (avoids expensive object-space transform for distant objects)
fn intersectWorldAABB(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>, closest_t: f32) -> bool {
    let inv_d = 1.0 / ray.rayD;
    let t0s = (aabb_min - ray.rayO) * inv_d;
    let t1s = (aabb_max - ray.rayO) * inv_d;
    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);
    let t_min = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    let t_max = min(tbigger.x, min(tbigger.y, tbigger.z));
    return slabsOverlap(t_min, t_max, closest_t);
}
