// TODO: wont work - using 0.12 wgpu syntax
struct NodeLeaf {
    point1: vec3<f32>;
    pad1: u32;
    point2: vec3<f32>;
    pad2: u32;
    point3: vec3<f32>;
    pad3: u32;
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
    lightPos: vec3<f32>;
    is_pathtracer: bool;
    resolution: vec2<u32>;
    _pad2: vec2<u32>;
    n_objects: i32;
    subpixel_idx: u32;
    ray_bounces: u32;
    _pad3: u32;
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

fn float_to_linear_rgb(x: f32) -> f32 {
    if (x > 0.04045) {
        return pow((x + 0.055) / 1.055,2.4);
    }
    return x / 12.92;
}

fn to_linear_rgb(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(float_to_linear_rgb(c.x),float_to_linear_rgb(c.y),float_to_linear_rgb(c.z),1.0);
}

const EPSILON:f32 = 0.0001;
const MAXLEN: f32 = 10000.0;
const INFINITY: f32 = 340282346638528859811704183484516925440.0;
const NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;

const PHI: f32 = 1.61803398874989484820459;  // Φ = Golden Ratio 
// const RAYS_PER_PIXEL: u32 = 4u;  
const MAX_RAY_BOUNCES: i32 = 16;

// fn jenkinsHash(x: u32) -> u32 {
//     x += x << 10u;
//     x ^= x >> 6u;
//     x += x << 3u;
//     x ^= x >> 11u;
//     x += x << 15u;
//     return x;
// }

// fn initRNG(pixel: vec2<u32> , width: u32, frame: u32) -> u32 {
//     let rngState = dot(pixel , uint2(1, width)) ^ jenkinsHash(frame);
//     return jenkinsHash(rngState);
// }


fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

fn rand(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(sin(dot(xy +seed,vec2<f32>(12.9898,78.233))) * 43758.5453+seed);
}

fn _rand2(xy: vec2<f32>,seed:f32) -> f32
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
        if (idx >= max_inner_node_idx ) {break;};

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

fn _schlick(cos_t:f32, r0: f32) -> f32 {
    return r0 + (1.0-r0)*pow((1.0 - cos_t),5.0);
}

fn schlick(cos_t:f32, eta_t: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let r0 = pow((1.0-eta_t) / (1.0+eta_t),2.0);
    return r0 + (1.0-r0)*pow((1.0 - cos_t),5.0);
}

fn _schlick_lazanyi(cos_t:f32, eta_t: f32, a: f32, alpha:f32) -> f32 {
    let r0 = pow((1.0-eta_t) / (1.0+eta_t),2.0);
    return _schlick(cos_t, r0) - a * cos_t * pow(1.0 - cos_t , alpha);
}

fn schlick_lazanyi(cos_t:f32, eta_t: f32, k: f32) -> f32 {
    return (pow(eta_t - 1.0, 2.0) + 4.0 * eta_t * pow(1.0 - cos_t,5.0) + pow(k,2.0)) / (pow(eta_t+1.0,2.0) + pow(k,2.0));
}

fn isShadowed(point: vec3<f32>, lightPos: vec3<f32>) -> bool
{
  let v: vec3<f32> = lightPos - point;
  let distance: f32 = length(v);
  let direction: vec3<f32> = normalize(v);

  let intersection: Intersection = intersect(Ray(point,0,direction,0));

  if (intersection.closestT > EPSILON && intersection.closestT < distance)
  {
    return true;
  }

  return false;
}


fn lighting(material: Material, lightPos: vec3<f32>, hitParams: HitParams, shadowed: bool) -> vec4<f32>
{
  // return material.colour;
  var diffuse: vec4<f32>;
  var specular: vec4<f32>;

  let intensity = vec4<f32>(1.0,1.0,1.0,1.0); // TODO temp placeholder

  let effectiveColour = intensity * material.colour; //* light->intensity;

  let ambient = effectiveColour * material.ambient;
  // vec4 ambient = vec4(0.3,0.0,0.0,1.0);
  if (shadowed) {
    return ambient;
  }

  let lightv: vec3<f32> = normalize(lightPos - hitParams.overPoint);

  let lightDotNormal: f32 = dot(lightv, hitParams.normalv);
  if (lightDotNormal < 0.0)
  {
    diffuse = vec4<f32>(0.0, 0.0, 0.0,0.0);
    specular = vec4<f32>(0.0, 0.0, 0.0,0.0);
  }
  else
  {
    // compute the diffuse contribution​
    diffuse = effectiveColour * material.diffuse * lightDotNormal;

    // reflect_dot_eye represents the cosine of the angle between the
    // reflection vector and the eye vector. A negative number means the
    // light reflects away from the eye.​
    let reflectv = reflect(-lightv, hitParams.normalv);
    let reflectDotEye = dot(reflectv, hitParams.eyev);

    if (reflectDotEye <= 0.0)
    {
      specular = vec4<f32>(0.0, 0.0, 0.0,0.0);
    }
    else
    {
      // compute the specular contribution​
      let factor = pow(reflectDotEye, material.shininess);
      specular = intensity * material.specular * factor;
    }
  }

  return (ambient + diffuse + specular);
}

struct RenderRay {
    ray: Ray;
    bounce_number: u32;
    reflectance: f32;
    reflective: f32;
    transparent: f32;
};


fn renderRtScene(init_ray: Ray) -> vec4<f32> {
    // int id = 0;
    var color: vec4<f32> = vec4<f32>(0.0);
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // var ray: Ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);

    // let init_ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);
    // let init_ray = rays.Rays[(pixel.y * ubo.width) + pixel.x];
    var type_enum = 0;
    // var intersection: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));

    var stack: array<RenderRay,MAX_RAY_BOUNCES>;
    var top_stack = -1;

    top_stack = top_stack + 1;
    stack[top_stack] = RenderRay (init_ray,0u,1.0,1.0,1.0);

    loop  {
        if (top_stack < 0) { break }

        let new_ray = stack[top_stack];
        top_stack = top_stack - 1;

        if (new_ray.bounce_number >= ubo.ray_bounces) { 
            continue;
        }

        let intersection = intersect(new_ray.ray);
        
        if (intersection.closestT >= MAXLEN || intersection.id == -1)
        {
            continue;
        }

        // TODO: just hard code object type in the intersection rather than looking it up
        let ob_params = object_params.ObjectParams[intersection.model_id];


        let hitParams: HitParams = getHitParams(new_ray.ray, intersection, ob_params.model_type);
        let shadowed: bool = isShadowed(hitParams.overPoint, ubo.lightPos);
        // let shadowed = false;
        
        let albedo = lighting(ob_params.material, ubo.lightPos,
                                hitParams, shadowed) 
                                * new_ray.reflectance 
                                * new_ray.reflective 
                                * new_ray.transparent
                                ;
        color = color + albedo;

        if (ob_params.material.transparency > 0.0 || ob_params.material.reflective > 0.0) {
            var eta_i = 1.0;
            var eta_t = ob_params.material.refractive_index;

            var cos_i = clamp(-1.0,1.0,dot(hitParams.eyev,hitParams.normalv));

            if (!hitParams.front_face) {
                eta_t=1.0/eta_t;
            }

            var reflectance = 1.0;
            let do_schlick = ob_params.material.transparency > 0.0 && ob_params.material.reflective > 0.0;
            if (do_schlick) {
                reflectance = schlick(cos_i, eta_t);
            }

            if (ob_params.material.reflective > 0.0) {
                top_stack = top_stack + 1;
                stack[top_stack] = RenderRay (Ray(hitParams.overPoint, new_ray.ray.x, hitParams.reflectv,new_ray.ray.y),new_ray.bounce_number + 1u,reflectance * new_ray.reflectance,new_ray.reflective * ob_params.material.reflective,1.0); 
            }
                        
            if (ob_params.material.transparency > 0.0) {
                let n_ratio = eta_i / eta_t;
                let sin_2t = (n_ratio * n_ratio) * (1.0 - (cos_i * cos_i));

                if (sin_2t <= 1.0)
                {
                    let cos_t = sqrt(1.0 - sin_2t);
                    let direction = hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                                (hitParams.eyev * n_ratio);

                    if (do_schlick) {
                        reflectance = 1.0-reflectance;
                    }

                    top_stack = top_stack + 1;
                    stack[top_stack] = RenderRay (Ray(hitParams.underPoint,new_ray.ray.x, direction,new_ray.ray.y),new_ray.bounce_number + 1u,reflectance * new_ray.reflectance,1.0,new_ray.transparent * ob_params.material.transparency); 
                }
            }
        }
    }
 
    return color;

}

fn renderPtScene(init_ray: Ray) -> vec4<f32> {
    var radiance: vec4<f32> = vec4<f32>(0.0);
    var throughput: vec4<f32> = vec4<f32>(1.0);

    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    var new_ray = init_ray;

    var albedo = vec4<f32>(0.0);

    // var ob_params = object_params.ObjectParams[0];
    // let ray_miss_colour = vec4<f32>(1.0);
    let ray_miss_colour = vec4<f32>(0.1,0.1,0.1,1.0);
    // let ray_miss_colour = vec4<f32>(0.529, 0.808, 0.922,1.0);
    
    for (var bounce_idx: u32 = 0u; bounce_idx < ubo.ray_bounces; bounce_idx =  bounce_idx + 1u) {
        // Get intersected object ID
        let intersection = intersect(new_ray);
        
        if (intersection.id == -1 || intersection.closestT >= MAXLEN)
        {
            radiance = radiance + (throughput * ray_miss_colour);
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
            // let reflectance = schlick_lazanyi(cos_t,eta_t,0.0);

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

[[stage(fragment)]]
fn main([[location(0)]] inUV: vec2<f32>) -> [[location(0)]] vec4<f32> {
    // TODO: fix a way to take the scaling into the fragment shader too.
    //       be sure to exclude the ray bounce accumulation array from PT shader too
    let xy = vec2<i32>(inUV*vec2<f32>(ubo.resolution));

    // var color = textureLoad(imageData,xy);
    // if (ubo.is_pathtracer) {
    //     color = sqrt(color);
    // }
    
    // color = clamp(color,vec4<f32>(0.0),vec4<f32>(0.999));
    // return to_linear_rgb(color);

    var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    let ray = rays.Rays[(xy.y * i32(ubo.resolution.x)) + xy.x];

    if (ray.x < 0) {
        return color;
    }
    

    // let uv = vec2<u32>(ray.x ,ray.y);

    if (ubo.subpixel_idx > 0u) {
        color = textureLoad(imageData,xy);
    }

    var ray_color = vec4<f32>(0.0);

    // ray_color = renderPtScene(ray);
    // ray_color = renderRtScene(ray);

    // TODO: this branching really slows things down - compile 2 shaders instead
    if (ubo.is_pathtracer) {
        ray_color = renderPtScene(ray);
    }
    else {
        ray_color = renderRtScene(ray);
    }

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    color = mix(color,ray_color,scale);

    textureStore(imageData, xy, color);

    if (ubo.is_pathtracer) {
        color = sqrt(color);
    }
    
    color = clamp(color,vec4<f32>(0.0),vec4<f32>(0.999));
    return to_linear_rgb(color);
}
