#define_import_path rengin::intersects

#import rengin::types::{NodeInner, NodeLeaf, Ray, Intersection, ObjectParam}
#import rengin::constants::{EPSILON, MAXLEN}
#import rengin::pt_bindings::{inner_nodes, leaf_nodes, object_params, ubo}
#import rengin::stats::{
    p_bvh_node_tests, p_bvh_node_skips,
    p_triangle_tests, p_triangle_hits,
    p_world_aabb_tests, p_world_aabb_skips,
    p_objects_tested,
}

fn intersectAABB_inv(ray_o: vec3<f32>, inv_d: vec3<f32>, inner_node: NodeInner, closest_t: f32) -> bool {
    let t0s = (inner_node.first - ray_o) * inv_d;
    let t1s = (inner_node.second - ray_o) * inv_d;
    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);
    let t_min = max(tsmaller.x, max(tsmaller.y, tsmaller.z));
    let t_max = min(tbigger.x, min(tbigger.y, tbigger.z));
    return t_max > max(t_min, 0.0) && t_min < closest_t;
}

fn intersectTriangle(ray: Ray, triangleIdx: u32, inIntersection: Intersection, object_id: u32) -> Intersection {
    let tri = leaf_nodes.LeafNodes[triangleIdx];
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = tri.vertex_b - tri.vertex_a;
    let e2: vec3<f32> = tri.vertex_c - tri.vertex_a;

    let dirCrossE2: vec3<f32> = cross(ray.rayD, e2);
    let det: f32 = dot(e1, dirCrossE2);

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - tri.vertex_a);
    uv.x = f * dot(p1ToOrigin, dirCrossE2);

    let originCrossE1: vec3<f32> = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD, originCrossE1);

    let t = f * dot(e2, originCrossE1);

    let isHit: bool = (uv.x >= 0.0) && (uv.y >= 0.0) && (uv.x + uv.y <= 1.0) && (t < inIntersection.closestT) && (t > EPSILON);

    if isHit {
        return Intersection(uv, i32(triangleIdx), t, object_id);
    }
    return inIntersection;
}

fn intersectInnerNodes(ray: Ray, inIntersection: Intersection, min_inner_node_idx: u32, max_inner_node_idx: u32, leaf_offset: u32, object_id: u32) -> Intersection {
    var ret: Intersection = inIntersection;

    // Precompute inverse direction for AABB tests
    let inv_d = 1.0 / ray.rayD;

    var idx = min_inner_node_idx;
    loop
    {
        if idx >= max_inner_node_idx {break;};

        let current_node: NodeInner = inner_nodes.InnerNodes[idx];

        let leaf_node: bool = current_node.prim_end_idx > 0u;

#ifdef PROFILE_STATS
        p_bvh_node_tests += 1u;
#endif
        if intersectAABB_inv(ray.rayO, inv_d, current_node, ret.closestT) {
            if leaf_node {
                for (var primIdx: u32 = current_node.skip_ptr_or_prim_idx + leaf_offset; primIdx < current_node.prim_end_idx + leaf_offset; primIdx = primIdx + 1u) {
#ifdef PROFILE_STATS
                    p_triangle_tests += 1u;
#endif
                    let next_intersection = intersectTriangle(ray, primIdx, ret, object_id);

                    if (next_intersection.closestT < ret.closestT) && (next_intersection.closestT > EPSILON) {
#ifdef PROFILE_STATS
                        p_triangle_hits += 1u;
#endif
                        ret = next_intersection;
                    }
                }
            }
            idx = idx + 1u;
        } else if leaf_node {
#ifdef PROFILE_STATS
            p_bvh_node_skips += 1u;
#endif
            idx = idx + 1u;
        } else {
#ifdef PROFILE_STATS
            p_bvh_node_skips += 1u;
#endif
            idx = current_node.skip_ptr_or_prim_idx + min_inner_node_idx;
        }
    }
    return ret;
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
        if t1 < t2 && t1 > EPSILON {
            return Intersection(vec2<f32>(0.0), 0, t1, object_id);
        }

        if t2 > EPSILON && t2 < inIntersection.closestT {
            return Intersection(vec2<f32>(0.0), 0, t2, object_id);
        }
    }
    return inIntersection;
}

fn intersectPlane(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    if abs(ray.rayD.y) < EPSILON {
        return inIntersection;
    }

    let t: f32 = -ray.rayO.y / ray.rayD.y;

    if t < inIntersection.closestT && t > EPSILON {
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

    if t_min < inIntersection.closestT && t_min > EPSILON {
        return Intersection(vec2<f32>(0.0), 0, t_min, object_id);
    } else if t_max < inIntersection.closestT && t_max > EPSILON {
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
    return t_max > max(t_min, 0.0) && t_min < closest_t;
}

fn intersect(ray: Ray, start: u32, immediate_ret: bool) -> Intersection {
    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, start);

    for (var i: u32 = start; i < ubo.n_objects; i = i + 1u) {
        let ob_params = object_params.ObjectParams[i];

        // Skip objects whose world-space AABB the ray misses
        // (planes have aabb_min == aabb_max == 0, so skip the test for them)
        if ob_params.model_type != 1u {
#ifdef PROFILE_STATS
            p_world_aabb_tests += 1u;
#endif
            if !intersectWorldAABB(ray, ob_params.aabb_min.xyz, ob_params.aabb_max.xyz, ret.closestT) {
#ifdef PROFILE_STATS
                p_world_aabb_skips += 1u;
#endif
                if immediate_ret {
                    return ret;
                }
                continue;
            }
        }

#ifdef PROFILE_STATS
        p_objects_tested += 1u;
#endif
        let transformed_ray = Ray((ob_params.inverse_transform * vec4<f32>(ray.rayO, 1.0)).xyz, ray.refractive_index, (ob_params.inverse_transform * vec4<f32>(ray.rayD, 0.0)).xyz, ray.pos, vec4<f32>(-1f));

        if ob_params.model_type == 0u { //Sphere
            ret = intersectSphere(transformed_ray, ret, i);
        } else if ob_params.model_type == 1u { //Plane
            ret = intersectPlane(transformed_ray, ret, i);
        } else if ob_params.model_type == 2u { //Cube
            ret = intersectCube(transformed_ray, ret, i);
        } else {
            // Triangle mesh
            ret = intersectInnerNodes(transformed_ray, ret, ob_params.offset_inner_nodes, ob_params.offset_inner_nodes + ob_params.len_inner_nodes, ob_params.offset_leaf_nodes, i);
        }

        if immediate_ret {
            return ret;
        }
    }

    return ret;
}
