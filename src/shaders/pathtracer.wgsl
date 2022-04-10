struct NodeLeaf {
    point1: vec3<f32>;
    object_id: u32;
    point2: vec3<f32>;
    pad1: u32;
    point3: vec3<f32>;
    pad2: u32;
};

struct Normal {
    normal1: vec4<f32>;
    normal2: vec4<f32>;
    normal3: vec4<f32>;
};

struct NodeInner {
    first: vec3<f32>;
    skip_ptr_or_prim_idx1: u32;
    second: vec3<f32>;
    idx2: u32;
};


struct HitParams {
    point: vec3<f32>;
    normalv: vec3<f32>;
    eyev: vec3<f32>;
    reflectv: vec3<f32>;
    overPoint: vec3<f32>;
    underPoint: vec3<f32>;
    front_face: bool;
};

struct Material {
    colour: vec4<f32>;
    emissiveness: vec4<f32>;
    ambient: f32;
    diffuse: f32;
    specular: f32;
    shininess: f32;
    reflective: f32;
    transparency: f32;
    refractive_index: f32;
};

struct UBO {
    _pad1: vec4<u32>;
    width: u32;
    n_objects: i32;
    subpixel_idx: u32;
    ray_bounces: u32;
};


struct InnerNodes {
    InnerNodes: [[stride(32)]] array<NodeInner>;
};


struct LeafNodes {
    LeafNodes: [[stride(48)]] array<NodeLeaf>;
};


struct Normals {
    Normals: [[stride(48)]] array<Normal>;
};

struct ObjectParam {
    inverse_transform: mat4x4<f32>;
    material: Material;
    offset_inner_nodes: i32;
    len_inner_nodes:i32;
    offset_leaf_nodes:u32;
    model_type: u32;
};


struct ObjectParams {
    ObjectParams: [[stride(144)]] array<ObjectParam>;
    // ObjectParams: [[stride(96)]] array<ObjectParam>;
};

struct Ray {
    rayO: vec3<f32>;
    x: i32;
    rayD: vec3<f32>;
    y: i32;
};

struct Rays {
    Rays: [[stride(32)]] array<Ray>;
};

struct Intersection {
    uv: vec2<f32>;
    id: i32;
    closestT: f32;
    model_id: u32;
};

[[group(0), binding(0)]]
var imageData: texture_storage_2d<rgba8unorm,read_write>;
[[group(0), binding(1)]]
var<uniform> ubo: UBO;
[[group(0), binding(2)]]
var<storage, read> inner_nodes: InnerNodes;
[[group(0), binding(3)]]
var<storage, read> leaf_nodes: LeafNodes;
[[group(0), binding(4)]]
var<storage, read> normal_nodes: Normals;
[[group(0), binding(5)]]
var<storage, read> object_params: ObjectParams;
[[group(0), binding(6)]]
var<storage, read_write> rays: Rays;

let EPSILON:f32 = 0.0001;
let MAXLEN: f32 = 10000.0;
let INFINITY: f32 = 340282346638528859811704183484516925440.0;
let NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;

let PHI: f32 = 1.61803398874989484820459;  // Φ = Golden Ratio 
// let RAYS_PER_PIXEL: u32 = 4u;  
// let RAY_BOUNCES: i32 = 50;

fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

fn rand(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(sin(dot(xy +seed,vec2<f32>(12.9898,78.233))) * 43758.5453+seed);
}

fn rand2(xy: vec2<f32>,seed:f32) -> f32
{
    let v = 0.152;
    let pos = (xy * v + f32(ubo.subpixel_idx) * 1500. + 50.0);

	var p3 = fract(vec3<f32>(pos.xyx) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn rescale(value: f32, min: f32, max: f32) -> f32 {
    return (value * (max - min)) + min;
    // return (((value + 1.0) * (max - min)) / 2.0) + min;
}

fn linearRand(min: f32, max: f32, xy: vec2<f32>, seed: f32) -> f32 {
    return rescale(rand(xy,seed),min,max);
    // return rescale(gold_noise(xy,seed),min,max);
}

fn sphericalRand(radius: f32, xyz: vec3<f32>, seed: f32) -> vec3<f32>
{
    // let xy = vec2<f32>(seed_xy);

    let theta: f32 = linearRand(0.0, 6.283185307179586476925286766559,xyz.xy,seed);
    let phi: f32 = acos(linearRand(-1.0, 1.0,xyz.yz,seed));

    let x: f32 = sin(phi) * cos(theta);
    let y: f32 = sin(phi) * sin(theta);
    let z: f32 = cos(phi);

    return vec3<f32>(x,y,z) * radius;
}

fn hemisphericalRand(radius: f32, normal: vec3<f32>, xyz: vec3<f32>, seed: f32) -> vec3<f32>
{
    let in_unit_sphere = sphericalRand(radius,xyz,seed);
    if (dot(in_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return in_unit_sphere;
    }
    return -in_unit_sphere;
}

fn intersectAABB(ray: Ray, aabbIdx: i32) -> bool {
    // let INFINITY: f32 = 1.0 / 0.0;

    var t_min: f32 = NEG_INFINITY;
    var t_max: f32 = INFINITY;
    // var temp: f32;
    // var invD: f32;
    var t0: f32;
    var t1: f32;

    let inner_node = inner_nodes.InnerNodes[aabbIdx];

    for (var a: i32 = 0; a < 3; a = a+1)
    {
        let invD = 1.0 / ray.rayD[a];
        t0 = (inner_node.first[a] - ray.rayO[a]) * invD;
        t1 = (inner_node.second[a] - ray.rayO[a]) * invD;
        if (invD < 0.0) {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if (t0 > t_min) {
            t_min = t0;
        }
        if (t1 < t_max) {
            t_max = t1;
        }
        // t_min = t0 > t_min ? t0 : t_min;
        // t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min) {
            return false;
        }
    }
    return true;
}

fn intersectTriangle(ray: Ray, triangleIdx: u32, inIntersection: Intersection, object_id: i32) -> Intersection {
    let triangle = leaf_nodes.LeafNodes[triangleIdx];
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = triangle.point2 - triangle.point1;
    let e2: vec3<f32> = triangle.point3 - triangle.point1;

    let dirCrossE2: vec3<f32> = cross(ray.rayD, e2);
    let det: f32 = dot(e1, dirCrossE2);

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - triangle.point1);
    uv.x = f * dot(p1ToOrigin, dirCrossE2);

    let originCrossE1: vec3<f32>  = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD, originCrossE1);
    
    let t = f * dot(e2, originCrossE1);

    let isHit: bool = (uv.x >= 0.0) && (uv.y >= 0.0)
                    && (uv.x + uv.y <= 1.0)
                    && (t < inIntersection.closestT)
                    && (t > EPSILON);

    if (isHit) {
        return Intersection(uv,i32(triangleIdx),t,u32(object_id));
    }
    return inIntersection;
    // return isHit ? Intersection(uv,inIntersection.id,t) : inIntersection;
}

fn intersectInnerNodes(ray: Ray, inIntersection: Intersection, min_inner_node_idx: i32, max_inner_node_idx: i32, leaf_offset: u32, object_id: i32) -> Intersection {
    // var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN);
    var ret: Intersection = inIntersection;

    var idx: i32 = min_inner_node_idx;
    loop  
    {
        if (idx >= max_inner_node_idx ) {break};

        let current_node: NodeInner = inner_nodes.InnerNodes[idx];
        let leaf_node: bool = current_node.idx2 > 0u;

        if (intersectAABB(ray, idx)) {
            idx = idx + 1;
            if (leaf_node) {
                for (var primIdx: u32 = current_node.skip_ptr_or_prim_idx1 + leaf_offset; primIdx < current_node.idx2 + leaf_offset; primIdx = primIdx + 1u) {
                    let next_intersection = intersectTriangle(ray, primIdx, ret,object_id);

                    if ((next_intersection.closestT < inIntersection.closestT)  && (next_intersection.closestT > EPSILON)) {
                        ret = next_intersection;
                    }
                }
            }
            
        }
        else if (leaf_node) {
            idx = idx + 1;
        }
        else {
            idx = i32(current_node.skip_ptr_or_prim_idx1) + min_inner_node_idx;
        }
    }
    return ret;
}

fn intersectSphere(ray: Ray, inIntersection: Intersection, object_id: i32) -> Intersection {
    // var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN);
    var ret: Intersection = inIntersection;

    let sphereToRay = ray.rayO;
    let a = dot(ray.rayD, ray.rayD);
    let b = 2.0 * dot(ray.rayD, sphereToRay);
    let c = dot(sphereToRay, sphereToRay) - 1.0;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return ret;
    }

    let t1 = (-b - sqrt(discriminant)) / (2.0 * a);
    let t2 = (-b + sqrt(discriminant)) / (2.0 * a);

    if (t1 < ret.closestT || t2 < ret.closestT) {
        if (t1 < t2 && t1 > EPSILON) {
            return Intersection(vec2<f32>(0.0),0,t1,u32(object_id));
        }
        
        if (t2 > EPSILON) {
            return Intersection(vec2<f32>(0.0),0,t2,u32(object_id));
        }
    }
    return ret;
}

fn intersectPlane(ray: Ray, inIntersection: Intersection, object_id: i32) -> Intersection {
    var ret: Intersection = inIntersection;

    if (abs(ray.rayD.y) < EPSILON) {
        return ret;
    }

    let t: f32 = -ray.rayO.y / ray.rayD.y;

    if (t < ret.closestT && t > EPSILON) {
        return Intersection(vec2<f32>(0.0),0,t,u32(object_id));
    }
    return ret;
}

fn intersectCube(ray: Ray, inIntersection: Intersection, object_id: i32) -> Intersection {
    var ret: Intersection = inIntersection;

    var t_min: f32 = NEG_INFINITY;
    var t_max: f32 = INFINITY;
    var t0: f32;
    var t1: f32;

    for (var a: i32 = 0; a < 3; a = a+1)
    {
        let invD = 1.0 / ray.rayD[a];
        t0 = (-1.0 - ray.rayO[a]) * invD;
        t1 = (1.0 - ray.rayO[a]) * invD;
        if (invD < 0.0) {
            let temp = t0;
            t0 = t1;
            t1 = temp;
        }
        if (t0 > t_min) {
            t_min = t0;
        }
        if (t1 < t_max) {
            t_max = t1;
        }
        if (t_max <= t_min) {
            return ret;
        }
    }

    if (t_min < ret.closestT && t_min > EPSILON) {
        return Intersection(vec2<f32>(0.0),0,t_min,u32(object_id));
    }
    else if (t_max < ret.closestT && t_max > EPSILON) {
        return Intersection(vec2<f32>(0.0),0,t_max,u32(object_id));
    }
    return ret;
}

fn intersect(ray: Ray) -> Intersection {
    // TODO: this will need the id of the object as input in future when we are rendering more than one model
    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));


    // TODO: fix loop range - get number of objects
    for (var i: i32 = 0; i < ubo.n_objects; i = i+1) {
        let ob_params = object_params.ObjectParams[i];
        let nRay: Ray = Ray((ob_params.inverse_transform * vec4<f32>(ray.rayO,1.0)).xyz, ray.x, (ob_params.inverse_transform * vec4<f32>(ray.rayD,0.0)).xyz, ray.y);


        if (ob_params.model_type == 0u) { //Sphere
            ret = intersectSphere(nRay,ret, i);
        }
        else if (ob_params.model_type == 1u) { //Plane
            ret = intersectPlane(nRay,ret, i);
        }
        else if (ob_params.model_type == 2u) { //Cube
            ret = intersectCube(nRay,ret, i);
        }
        else {
            // Triangle mesh
            ret = intersectInnerNodes(nRay,ret, ob_params.offset_inner_nodes, ob_params.offset_inner_nodes + ob_params.len_inner_nodes, ob_params.offset_leaf_nodes,i);
        }

    }

    return ret;
}

fn normalToWorld(normal: vec3<f32>, object_id: u32) -> vec3<f32>
{
    let ret: vec3<f32> = normalize((transpose(object_params.ObjectParams[object_id].inverse_transform) * vec4<f32>(normal,0.0)).xyz);
    // ret.w = 0.0;
    // ret = normalize(ret);


    return ret;
}

fn normalAt(point: vec3<f32>, intersection: Intersection, typeEnum: u32) -> vec3<f32> {
    if (typeEnum == 0u) { //Sphere
        let objectPoint = (object_params.ObjectParams[intersection.model_id].inverse_transform * vec4<f32>(point,1.0)).xyz;
        return objectPoint;
    }
    else if (typeEnum == 1u) { //Plane
        return normalToWorld(vec3<f32>(0.0, 1.0, 0.0),intersection.model_id);
    }
    else if (typeEnum == 2u) { //Cube
        let objectPoint = (object_params.ObjectParams[intersection.model_id].inverse_transform * vec4<f32>(point,1.0)).xyz;
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

        return normalToWorld(objectNormal,intersection.model_id);
    }
    else { //Model
        let normal: Normal = normal_nodes.Normals[intersection.id];
        return normalToWorld((normal.normal2.xyz * intersection.uv.x + normal.normal3.xyz * intersection.uv.y + normal.normal1.xyz * (1.0 - intersection.uv.x - intersection.uv.y)),intersection.model_id);
        // n.w = 0.0;
    }
    return vec3<f32>(0.0);
}


fn getHitParams(ray: Ray, intersection: Intersection, typeEnum: u32) -> HitParams
{
    var hitParams: HitParams;
    hitParams.point =
        ray.rayO + normalize(ray.rayD) * intersection.closestT;
    // TODO check that uv only null have using none-uv normalAt version
    hitParams.normalv = 
        normalAt(hitParams.point, intersection, typeEnum);
    // hitParams.eyev = -ray.rayD;
    hitParams.eyev = -normalize(ray.rayD);

    // hitParams.front_face = dot(hitParams.normalv, hitParams.eyev) < 0.0;
    hitParams.front_face = dot(ray.rayD, hitParams.normalv) < 0.0;
    if (!hitParams.front_face)
    {
        hitParams.normalv = -hitParams.normalv;
    }

    hitParams.reflectv =
        reflect(normalize(ray.rayD), hitParams.normalv);
    hitParams.overPoint =
        hitParams.point + hitParams.normalv * EPSILON;
    hitParams.underPoint =
        hitParams.point - hitParams.normalv * EPSILON;

    return hitParams;
}

fn schlick(cos_i:f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let r0 = pow((1.0-ref_idx) / (1.0+ref_idx),2.0);
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}


fn renderScene(init_ray: Ray) -> vec4<f32> {
    var radiance: vec4<f32> = vec4<f32>(0.0);
    var throughput: vec4<f32> = vec4<f32>(1.0);

    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    var new_ray = init_ray;

    var albedo = vec4<f32>(0.0);

    // var ob_params = object_params.ObjectParams[0];
    
    for (var bounce_idx: u32 = 0u; bounce_idx < ubo.ray_bounces; bounce_idx =  bounce_idx + 1u) {
        // Get intersected object ID
        let intersection = intersect(new_ray);
        
        if (intersection.id == -1 || intersection.closestT >= MAXLEN)
        {
            break;
        }

        // TODO: just hard code object type in the intersection rather than looking it up
        let ob_params = object_params.ObjectParams[intersection.model_id];

        let hitParams = getHitParams(new_ray, intersection, ob_params.model_type);

        var scatterTarget = vec3<f32>(0.0);
        var point = hitParams.overPoint;

        if (ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0) {
            albedo = ob_params.material.colour;
            scatterTarget = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * sphericalRand(1.0,new_ray.rayD.xyz,f32(bounce_idx)));
        }

        else if (ob_params.material.transparency > 0.0 || ob_params.material.reflective > 0.0) {
            // albedo = vec4<f32>(1.0);
            albedo = ob_params.material.colour;

            var eta_i = 1.0;
            var eta_t = ob_params.material.refractive_index;

            var cos_t = min(dot(hitParams.eyev, hitParams.normalv), 1.0);

            if (hitParams.front_face) {
                eta_t=eta_i/eta_t;
            }
            
            let reflectance = schlick(cos_t, eta_t);

            let sin_t = sqrt(1.0 - cos_t*cos_t);

            let cannot_refract = eta_t * sin_t > 1.0;

            if (cannot_refract || reflectance >= rand(new_ray.rayD.xz,f32(bounce_idx)))
            {
                // scatterTarget = hitParams.reflectv;
                scatterTarget = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * sphericalRand(1.0,new_ray.rayD.xyz,f32(bounce_idx)));
            }
            else {
                let r_out_perp =  eta_t * (normalize(new_ray.rayD) + cos_t * hitParams.normalv);
                let r_out_parallel = -sqrt(abs(1.0 - pow(length(r_out_perp),2.0))) * hitParams.normalv;
                scatterTarget = r_out_perp + r_out_parallel;

                point = hitParams.underPoint;
            }
        }
        else {
            albedo = ob_params.material.colour;
            scatterTarget = hitParams.normalv + sphericalRand(1.0,new_ray.rayD.xyz,f32(bounce_idx));    
            // Catch degenerate scatter direction
            if (abs(scatterTarget.x) < EPSILON && abs(scatterTarget.y) < EPSILON && abs(scatterTarget.z) < EPSILON) {
                scatterTarget = hitParams.normalv;
            }
        }
        
        radiance = radiance + (ob_params.material.emissiveness * throughput);
        throughput = throughput * albedo;

        new_ray = Ray(point, init_ray.x, scatterTarget, init_ray.y);
    }
 
    return radiance;

}


[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(local_invocation_id)]] local_invocation_id: vec3<u32>,
        [[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>,
        [[builtin(workgroup_id)]] workgroup_id: vec3<u32>
        ) 
{
    var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    let ray = rays.Rays[(global_invocation_id.y * ubo.width) + global_invocation_id.x];

    if (ray.x < 0) {
        return;
    }
    

    // let uv = vec2<u32>(ray.x ,ray.y);

    if (ubo.subpixel_idx > 0u) {
        color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
    }

    let ray_color = renderScene(ray);

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    color = mix(color,ray_color,scale);

    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}
