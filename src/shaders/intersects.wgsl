#define_import_path rengin::intersects

#import rengin::types::{NodeInner, NodeLeaf, Ray, Intersection, ObjectParam}
#import rengin::constants::MAXLEN
#import rengin::pt_bindings::{inner_nodes, leaf_nodes, object_params, ubo}
#import rengin::stats::{
    p_bvh_node_tests, p_bvh_node_skips,
    p_triangle_tests, p_triangle_hits,
    p_world_aabb_tests, p_world_aabb_skips,
    p_objects_tested,
}
#import rengin::intersects_math::{intersectAABB_inv, intersectSphere, intersectPlane, intersectCube, intersectWorldAABB}

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

    let isHit: bool = (uv.x >= 0.0) && (uv.y >= 0.0) && (uv.x + uv.y <= 1.0) && (t < inIntersection.closestT) && (t > 0.0);

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

                    if next_intersection.closestT < ret.closestT {
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
