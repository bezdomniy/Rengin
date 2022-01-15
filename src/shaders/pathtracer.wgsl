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

struct Camera {
    inverseTransform: mat4x4<f32>;
    pixelSize: f32;
    halfWidth: f32;
    halfHeight: f32;
    _padding: u32;
};


struct UBO {
    lightPos: vec3<f32>;
    _padding: u32;
    camera: Camera;
    n_objects: i32;
    subpixel_idx: u32;
    sqrt_rays_per_pixel: u32;
    rnd_seed: f32;
    // max_inner_node_idx: i32;
    // max_leaf_node_idx: i32;
    // padding: array<u32,2>;
};

// 
// struct BVH {
//     InnerNodes: [[stride(32)]] array<NodeInner>;
//     LeafNodes: [[stride(128)]] array<NodeLeaf>;
// };


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
    x: u32;
    rayD: vec3<f32>;
    y: u32;
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

let EPSILON:f32 = 0.0001;
let MAXLEN: f32 = 10000.0;
let INFINITY: f32 = 340282346638528859811704183484516925440.0;
let NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;

let PHI: f32 = 1.61803398874989484820459;  // Φ = Golden Ratio 
// let RAYS_PER_PIXEL: u32 = 4u;  
let RAY_BOUNCES: i32 = 8;

fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

fn rand(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(sin(dot(xy +seed,vec2<f32>(12.9898,78.233))) * 43758.5453+seed);
}

fn rand2(xy: vec2<f32>,seed:f32) -> f32
{
    let v = 0.152;
    let pos = (xy * v + ubo.rnd_seed * 1500. + 50.0);

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

fn rayForPixel(p: vec2<u32>, sqrt_rays_per_pixel: u32, current_ray_index: u32, half_sub_pixel_size: f32) -> Ray {
    
    let sub_pixel_row_number: u32 = current_ray_index / sqrt_rays_per_pixel;
    let sub_pixel_col_number: u32 = current_ray_index % sqrt_rays_per_pixel;
    let sub_pixel_x_offset: f32 = half_sub_pixel_size * f32(sub_pixel_col_number);
    let sub_pixel_y_offset: f32 = half_sub_pixel_size * f32(sub_pixel_row_number);

    let xOffset: f32 = (f32(p.x) + sub_pixel_x_offset) * ubo.camera.pixelSize;
    let yOffset: f32 = (f32(p.y) + sub_pixel_y_offset) * ubo.camera.pixelSize;

    // let xOffset: f32 = (f32(p.x) / 0.5) * ubo.camera.pixelSize;
    // let yOffset: f32 = (f32(p.y) / 0.5) * ubo.camera.pixelSize;

    let worldX: f32 = ubo.camera.halfWidth - xOffset;
    let worldY: f32 = ubo.camera.halfHeight - yOffset;

    let pixel: vec4<f32> = ubo.camera.inverseTransform * vec4<f32>(worldX, worldY, -1.0, 1.0);

    let rayO: vec4<f32> = ubo.camera.inverseTransform * vec4<f32>(0.0, 0.0, 0.0, 1.0);

    return Ray(rayO.xyz, p.x, normalize(pixel - rayO).xyz,p.y);
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
    // if (abs(det) < EPSILON) {
    //     return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    // }

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - triangle.point1);
    uv.x = f * dot(p1ToOrigin, dirCrossE2);
    // if (uv.x < 0.0 || uv.x > 1.0) {
    //   return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    // }

    let originCrossE1: vec3<f32>  = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD, originCrossE1);
    // uv.y =1.0;

    // if (uv.y < 0.0 || (uv.x + uv.y) > 1.0) {
    //     return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    // }
    // return Intersection(uv,inIntersection.id,f * dot(e2, originCrossE1));

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

//TODO:
// i = 0
// while i < bvh.size():
// if bvh[i] is a primitive:
// perform and record the ray-object intersection check
// else if the ray does not hit the bounding volume of bvh[i]:
// i += bvh[i].skip_index
// else:
// i++
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

    // let nRay: Ray = Ray(object_params.ObjectParams[0].inverse_transform * ray.rayO, object_params.ObjectParams[0].inverse_transform * ray.rayD);
    // ret = intersectInnerNodes(nRay,ret);
    
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
    hitParams.eyev = -ray.rayD;

    if (dot(hitParams.normalv, hitParams.eyev) < 0.0)
    {
        // intersection.comps->inside = true;
        hitParams.normalv = -hitParams.normalv;
    }
    // else
    // {
    //   intersection.comps->inside = false;
    // }

    hitParams.reflectv =
        reflect(ray.rayD, hitParams.normalv);
    hitParams.overPoint =
        hitParams.point + hitParams.normalv * EPSILON;
    hitParams.underPoint =
        hitParams.point - hitParams.normalv * EPSILON;

    // get_refractive_index_from_to(intersections, intersection, comps);
    hitParams.front_face = dot(ray.rayD, hitParams.normalv) < 0.0;
    if (!hitParams.front_face) {
        hitParams.normalv = -hitParams.normalv;
    }

    return hitParams;
}

fn isShadowed(point: vec3<f32>, lightPos: vec3<f32>) -> bool
{
  let v: vec3<f32> = lightPos - point;
  let distance: f32 = length(v);
  let direction: vec3<f32> = normalize(v);

  let intersection: Intersection = intersect(Ray(point,0u,direction,0u));

  if (intersection.closestT > EPSILON && intersection.closestT < distance)
  {
    return true;
  }

  return false;
}

fn reflectance(cosine:f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let r0 = ((1.0-ref_idx) / (1.0+ref_idx))*2.0;
    return r0 + (1.0-r0)*pow((1.0 - cosine),5.0);
}

struct Node {
    hit_colour: vec4<f32>;
    emissiveness: vec4<f32>;
};

fn renderScene(pixel: vec2<u32>,current_ray_idx: u32,sqrt_rays_per_pixel: u32,half_sub_pixel_size: f32) -> vec4<f32> {
    // int id = 0;
    var color: vec4<f32> = vec4<f32>(0.0);
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // var ray: Ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);

    var new_ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);
    var type_enum = 0;
    // var intersection: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));

    var stack: array<Node,RAY_BOUNCES>;
    var top_stack = -1;

    
    for (var bounce_idx: i32 = 0; bounce_idx < RAY_BOUNCES; bounce_idx =  bounce_idx + 1) {
        // Get intersected object ID
        let intersection = intersect(new_ray);
        
        if (intersection.closestT >= MAXLEN || intersection.id == -1)
        {
            // let unit_direction = normalize(new_ray.rayD);
            // let t = 0.5*(unit_direction.y + 1.0);
            // top_stack = top_stack + 1;
            // stack[top_stack] = Node((1.0-t)*vec4<f32>(1.0, 1.0, 1.0, 1.0) + t*vec4<f32>(0.5, 0.7, 1.0, 1.0),vec4<f32>(0.0,0.0,0.0,0.0));

            // top_stack = top_stack - 1;
            // continue
            break;
        }

        // TODO: just hard code object type in the intersection rather than looking it up
        let ob_params = object_params.ObjectParams[intersection.model_id];

        if (ob_params.material.emissiveness.w > 0.0) {
            top_stack = top_stack + 1;
            stack[top_stack] = Node(vec4<f32>(0.0,0.0,0.0,0.0),ob_params.material.emissiveness);
            break;
        }

        let hitParams: HitParams = getHitParams(new_ray, intersection, ob_params.model_type);
        // var scatterTarget: vec4<f32> = hitParams.normalv + hemisphericalRand(1.0,hitParams.normalv.xyz,new_ray.rayD.xyz,ubo.rnd_seed);
        var scatterTarget = hitParams.normalv + sphericalRand(1.0,new_ray.rayD.xyz,f32(ubo.subpixel_idx));

        if (abs(scatterTarget.x) < EPSILON && abs(scatterTarget.y) < EPSILON && abs(scatterTarget.z) < EPSILON )
        {
            scatterTarget = hitParams.normalv;
        }

        new_ray = Ray(hitParams.overPoint, pixel.x, scatterTarget, pixel.y);

        let hit_colour = ob_params.material.colour;

        // if (bounce_idx == 0) {
        //     color = hit_colour;
        // }

        top_stack = top_stack + 1;
        stack[top_stack] = Node(hit_colour,ob_params.material.emissiveness);


    }

    if (top_stack > -1) {
        let node = stack[top_stack];
        top_stack = top_stack - 1;
        color = node.emissiveness;

        loop {
            if (top_stack == -1) {
                break;
            }
            let node = stack[top_stack];
            top_stack = top_stack - 1;
            // color = color * node.hit_colour;
            color = node.emissiveness + color * node.hit_colour;

        }
    }
 
    return color;

}


[[stage(compute), workgroup_size(16, 16)]]
fn main([[builtin(local_invocation_id)]] local_invocation_id: vec3<u32>,
        [[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>,
        [[builtin(workgroup_id)]] workgroup_id: vec3<u32>
        ) 
{

    var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    if (ubo.subpixel_idx > 0u) {
        let inUV = vec2<i32>(i32(global_invocation_id.x) ,i32(global_invocation_id.y) );
        color = textureLoad(imageData,inUV);
    }

    let half_sub_pixel_size = 1.0 / f32(ubo.sqrt_rays_per_pixel) / 2.0;
    let ray_color = renderScene(global_invocation_id.xy,ubo.subpixel_idx,ubo.sqrt_rays_per_pixel,half_sub_pixel_size);

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    color = (color * (1.0 - scale)) + (ray_color * scale);

    color.r = clamp(color.r,0.0,0.999);
    color.g = clamp(color.g,0.0,0.999);
    color.b = clamp(color.b,0.0,0.999);
    color.a = clamp(color.a,0.0,0.999);
    

    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}
