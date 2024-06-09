
fn intersectAABB(ray: Ray, inner_node: NodeInner) -> bool {
    // let INFINITY: f32 = 1.0 / 0.0;

    var t_min: f32 = NEG_INFINITY;
    var t_max: f32 = INFINITY;
    // var temp: f32;
    // var invD: f32;
    var t0: f32;
    var t1: f32;

    // let inner_node = inner_nodes.InnerNodes[aabbIdx];

    for (var a: i32 = 0; a < 3; a = a + 1) {
        let invD = 1.0 / ray.rayD[a];
        t0 = (inner_node.first[a] - ray.rayO[a]) * invD;
        t1 = (inner_node.second[a] - ray.rayO[a]) * invD;
        if invD < 0.0 {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if t0 > t_min {
            t_min = t0;
        }
        if t1 < t_max {
            t_max = t1;
        }
        // t_min = t0 > t_min ? t0 : t_min;
        // t_max = t1 < t_max ? t1 : t_max;
        if t_max <= t_min {
            return false;
        }
    }
    return true;
}

fn intersectTriangle(ray: Ray, triangleIdx: u32, inIntersection: Intersection, object_id: u32) -> Intersection {
    let tri = leaf_nodes.LeafNodes[triangleIdx];
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = tri.point2 - tri.point1;
    let e2: vec3<f32> = tri.point3 - tri.point1;

    let dirCrossE2: vec3<f32> = cross(ray.rayD, e2);
    let det: f32 = dot(e1, dirCrossE2);

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - tri.point1);
    uv.x = f * dot(p1ToOrigin, dirCrossE2);

    let originCrossE1: vec3<f32> = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD, originCrossE1);

    let t = f * dot(e2, originCrossE1);

    let isHit: bool = (uv.x >= 0.0) && (uv.y >= 0.0) && (uv.x + uv.y <= 1.0) && (t < inIntersection.closestT) && (t > EPSILON);

    if isHit {
        return Intersection(uv, i32(triangleIdx), t, object_id);
    }
    return inIntersection;
    // return isHit ? Intersection(uv,inIntersection.id,t) : inIntersection;
}

fn intersectInnerNodes(ray: Ray, inIntersection: Intersection, min_inner_node_idx: u32, max_inner_node_idx: u32, leaf_offset: u32, object_id: u32) -> Intersection {
    // var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN);
    var ret: Intersection = inIntersection;

    var idx = min_inner_node_idx;
    loop  
    {
        if idx >= max_inner_node_idx {break;};

        let current_node: NodeInner = inner_nodes.InnerNodes[idx];

        let leaf_node: bool = current_node.idx2 > 0u;

        if intersectAABB(ray, current_node) {
            if leaf_node {
                for (var primIdx: u32 = current_node.skip_ptr_or_prim_idx1 + leaf_offset; primIdx < current_node.idx2 + leaf_offset; primIdx = primIdx + 1u) {
                    let next_intersection = intersectTriangle(ray, primIdx, ret, object_id);

                    if (next_intersection.closestT < inIntersection.closestT) && (next_intersection.closestT > EPSILON) {
                        ret = next_intersection;
                    }
                }
            }
            idx = idx + 1u;
        } else if leaf_node {
            idx = idx + 1u;
        } else {
            idx = current_node.skip_ptr_or_prim_idx1 + min_inner_node_idx;
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

        if t2 > EPSILON {
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
    var t_min: f32 = NEG_INFINITY;
    var t_max: f32 = INFINITY;
    var t0: f32;
    var t1: f32;

    for (var a: i32 = 0; a < 3; a = a + 1) {
        let invD = 1.0 / ray.rayD[a];
        t0 = (-1.0 - ray.rayO[a]) * invD;
        t1 = (1.0 - ray.rayO[a]) * invD;
        if invD < 0.0 {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if t0 > t_min {
            t_min = t0;
        }
        if t1 < t_max {
            t_max = t1;
        }
        if t_max <= t_min {
            return inIntersection;
        }
    }

    if t_min < inIntersection.closestT && t_min > EPSILON {
        return Intersection(vec2<f32>(0.0), 0, t_min, object_id);
    } else if t_max < inIntersection.closestT && t_max > EPSILON {
        return Intersection(vec2<f32>(0.0), 0, t_max, object_id);
    }
    return inIntersection;
}


fn intersect(ray: Ray, start: u32, immediate_ret: bool) -> Intersection {
    // TODO: this will need the id of the object as input in future when we are rendering more than one model
    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, start);

    // TODO: fix loop range - get number of objects
    for (var i: u32 = start; i < ubo.n_objects; i = i + 1u) {
        let ob_params = object_params.ObjectParams[i];

        // // TODO: clean this up
        // if (ob_params.model_type == 9u) //point light from whitted rt 
        // {
        //     continue;
        // }
        let transformed_ray = Ray((ob_params.inverse_transform * vec4<f32>(ray.rayO, 1.0)).xyz, ray.refractive_index, (ob_params.inverse_transform * vec4<f32>(ray.rayD, 0.0)).xyz, ray.bounce_idx, vec4<f32>(-1f));

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
