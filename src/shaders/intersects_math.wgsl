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

struct TriangleHit {
    hit: bool,
    /// Barycentric weights of vertices b and c.
    uv: vec2<f32>,
    t: f32,
}

/// Reorders `v` so that axis `kz` becomes z, keeping the cyclic order.
fn permuteToZ(v: vec3<f32>, kz: u32) -> vec3<f32> {
    if kz == 0u {
        return v.yzx;
    } else if kz == 1u {
        return v.zxy;
    }
    return v;
}

/// 2D edge function of the directed edge p -> q. The products are always
/// evaluated with the vertices in a canonical order, so two triangles sharing
/// an edge get exactly negated values even if the compiler fuses the
/// multiply-add.
fn edgeFunction(p: vec2<f32>, q: vec2<f32>) -> f32 {
    let swap = p.x > q.x || (p.x == q.x && p.y > q.y);
    let lo = select(p, q, swap);
    let hi = select(q, p, swap);
    let e = lo.x * hi.y - lo.y * hi.x;
    return select(e, -e, swap);
}

/// Watertight ray-triangle intersection (Woop, Benthin and Wald 2013, as in
/// pbrt). The vertices are moved into a space where the ray runs along +z from
/// the origin; each vertex's transformed position depends only on that vertex
/// and the ray, so a ray through an edge shared by two triangles hits at least
/// one of them.
fn intersectTriangleVertices(ray_o: vec3<f32>, ray_d: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, closest_t: f32) -> TriangleHit {
    let miss = TriangleHit(false, vec2<f32>(0.0), 0.0);

    let abs_d = abs(ray_d);
    var kz = 2u;
    if abs_d.x > abs_d.y && abs_d.x > abs_d.z {
        kz = 0u;
    } else if abs_d.y > abs_d.z {
        kz = 1u;
    }
    let d = permuteToZ(ray_d, kz);
    var p0 = permuteToZ(a - ray_o, kz);
    var p1 = permuteToZ(b - ray_o, kz);
    var p2 = permuteToZ(c - ray_o, kz);

    // Shear so the ray direction becomes +z
    let shear = vec3<f32>(-d.x / d.z, -d.y / d.z, 1.0 / d.z);
    p0 = vec3<f32>(p0.xy + shear.xy * p0.z, p0.z);
    p1 = vec3<f32>(p1.xy + shear.xy * p1.z, p1.z);
    p2 = vec3<f32>(p2.xy + shear.xy * p2.z, p2.z);

    let e0 = edgeFunction(p1.xy, p2.xy);
    let e1 = edgeFunction(p2.xy, p0.xy);
    let e2 = edgeFunction(p0.xy, p1.xy);
    if (e0 < 0.0 || e1 < 0.0 || e2 < 0.0) && (e0 > 0.0 || e1 > 0.0 || e2 > 0.0) {
        return miss;
    }
    let det = e0 + e1 + e2;
    if det == 0.0 {
        return miss;
    }

    let t_scaled = e0 * (p0.z * shear.z) + e1 * (p1.z * shear.z) + e2 * (p2.z * shear.z);
    let t = t_scaled / det;
    if !(t > 0.0 && t < closest_t) {
        return miss;
    }
    return TriangleHit(true, vec2<f32>(e1, e2) / det, t);
}

fn intersectSphere(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    let a = dot(ray.rayD, ray.rayD);
    let b = 2.0 * dot(ray.rayD, ray.rayO);
    let c = dot(ray.rayO, ray.rayO) - 1.0;

    // b*b - 4ac cancels catastrophically when the origin is far from the
    // sphere. Computing it from the ray's closest approach to the centre, v,
    // keeps it accurate: b*b - 4ac = 4a(1 - |v|^2) = 4a(1 + |v|)(1 - |v|).
    let v = ray.rayO - (b / (2.0 * a)) * ray.rayD;
    let v_length = length(v);
    let discriminant = 4.0 * a * (1.0 + v_length) * (1.0 - v_length);
    if discriminant < 0.0 {
        return inIntersection;
    }

    // Avoid subtracting nearly equal values when choosing the roots.
    let root = sqrt(discriminant);
    let q = select(-0.5 * (b + root), -0.5 * (b - root), b < 0.0);
    let t_near = min(q / a, c / q);
    let t_far = max(q / a, c / q);

    if t_near > 0.0 && t_near < inIntersection.closestT {
        return Intersection(vec2<f32>(0.0), 0, t_near, object_id);
    }
    if t_far > 0.0 && t_far < inIntersection.closestT {
        return Intersection(vec2<f32>(0.0), 0, t_far, object_id);
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
