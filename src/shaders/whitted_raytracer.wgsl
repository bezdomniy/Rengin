// TODO: !! fix so only lights get used as lights, not emissive shapes

struct NodeLeaf {
    point1: vec3<f32>,
    pad1: u32,
    point2: vec3<f32>,
    pad2: u32,
    point3: vec3<f32>,
    pad3: u32,
};

struct Normal {
    normal1: vec4<f32>,
    normal2: vec4<f32>,
    normal3: vec4<f32>,
};

struct NodeInner {
    first: vec3<f32>,
    skip_ptr_or_prim_idx1: u32,
    second: vec3<f32>,
    idx2: u32,
};


struct HitParams {
    p: vec3<f32>,
    normalv: vec3<f32>,
    eyev: vec3<f32>,
    reflectv: vec3<f32>,
    overPoint: vec3<f32>,
    underPoint: vec3<f32>,
    front_face: bool,
};

struct Material {
    colour: vec4<f32>,
    emissiveness: vec4<f32>,
    reflective: f32,
    transparency: f32,
    refractive_index: f32,
    ambient: f32,
    diffuse: f32,
    specular: f32,
    shininess: f32,
    _pad: u32,
};


struct UBO {
    lightPos: vec3<f32>,
    is_pathtracer: u32,
    resolution: vec2<u32>,
    _pad2: vec2<u32>,
    n_objects: i32,
    lights_offset: u32,
    subpixel_idx: u32,
    ray_bounces: u32,
};

struct InnerNodes {
    InnerNodes: array<NodeInner>,
};


struct LeafNodes {
    LeafNodes: array<NodeLeaf>,
};


struct Normals {
    Normals: array<Normal>,
};

struct ObjectParam {
    transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    material: Material,
    offset_inner_nodes: i32,
    len_inner_nodes:i32,
    offset_leaf_nodes:u32,
    model_type: u32,
};


struct ObjectParams {
    ObjectParams: array<ObjectParam>,
};

struct Ray {
    rayO: vec3<f32>,
    x: i32,
    rayD: vec3<f32>,
    y: i32,
};

struct Rays {
    Rays: array<Ray>,
};

struct Intersection {
    uv: vec2<f32>,
    id: i32,
    closestT: f32,
    model_id: u32,
};

@group(0) @binding(0)
var imageData: texture_storage_2d<rgba8unorm,read_write>;
@group(0) @binding(1)
var<uniform> ubo: UBO;
@group(0) @binding(2)
var<storage, read> inner_nodes: InnerNodes;
@group(0) @binding(3)
var<storage, read> leaf_nodes: LeafNodes;
@group(0) @binding(4)
var<storage, read> normal_nodes: Normals;
@group(0) @binding(5)
var<storage, read> object_params: ObjectParams;
@group(0) @binding(6)
var<storage, read_write> rays: Rays;

let EPSILON:f32 = 0.001;
let MAXLEN: f32 = 10000.0;
let INFINITY: f32 = 340282346638528859811704183484516925440.0;
let NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;

let MAX_RAY_BOUNCE_ARRAY_SIZE: i32 = 8;

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
    let tri = leaf_nodes.LeafNodes[triangleIdx];
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = tri.point2 - tri.point1;
    let e2: vec3<f32> = tri.point3 - tri.point1;

    let dirCrossE2: vec3<f32> = cross(ray.rayD, e2);
    let det: f32 = dot(e1, dirCrossE2);

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - tri.point1);
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

fn normalAt(p: vec3<f32>, intersection: Intersection, typeEnum: u32) -> vec3<f32> {
    if (typeEnum == 0u) { //Sphere
        let objectPoint = (object_params.ObjectParams[intersection.model_id].inverse_transform * vec4<f32>(p,1.0)).xyz;
        return normalToWorld(objectPoint,intersection.model_id);
    }
    else if (typeEnum == 1u) { //Plane
        return normalToWorld(vec3<f32>(0.0, 1.0, 0.0),intersection.model_id);
    }
    else if (typeEnum == 2u) { //Cube
        let objectPoint = (object_params.ObjectParams[intersection.model_id].inverse_transform * vec4<f32>(p,1.0)).xyz;
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
    hitParams.p =
        ray.rayO + normalize(ray.rayD) * intersection.closestT;
    // TODO check that uv only null have using none-uv normalAt version
    hitParams.normalv = 
        normalAt(hitParams.p, intersection, typeEnum);
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
        hitParams.p + hitParams.normalv * EPSILON;
    hitParams.underPoint =
        hitParams.p - hitParams.normalv * EPSILON;

    return hitParams;
}

fn _schlick(cos_i:f32, r0: f32) -> f32 {
    return r0 + (1.0-r0)*pow((1.0 - cos_i),5.0);
}

fn schlick(cos_i:f32, eta_t: f32, eta_i: f32) -> f32 {
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

fn isShadowed(p: vec3<f32>, lightPos: vec3<f32>) -> bool
{
  let v: vec3<f32> = lightPos - p;
  let distance: f32 = length(v);
  let direction: vec3<f32> = normalize(v);

  let intersection: Intersection = intersect(Ray(p,0,direction,0));

  if (intersection.closestT > EPSILON && intersection.closestT < distance)
  {
    return true;
  }

  return false;
}


fn lighting(material: Material, lightPos: vec3<f32>, light_emissiveness: vec4<f32>, hitParams: HitParams, shadowed: bool) -> vec4<f32>
{
  // return material.colour;
  var diffuse: vec4<f32>;
  var specular: vec4<f32>;

//   let intensity = vec4<f32>(1.0,1.0,1.0,1.0); // TODO temp placeholder

  let effectiveColour = light_emissiveness * material.colour; //* light->intensity;

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
      specular = light_emissiveness * material.specular * factor;
    }
  }

  return (ambient + diffuse + specular);
}

struct RenderRay {
    ray: Ray,
    bounce_number: u32,
    reflectance: f32,
    reflective: f32,
    transparent: f32,
    refractive_index: f32,
};


fn renderScene(init_ray: Ray) -> vec4<f32> {
    // int id = 0;
    var color: vec4<f32> = vec4<f32>(0.0);
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // var ray: Ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);

    // let init_ray = rayForPixel(pixel,sqrt_rays_per_pixel,current_ray_idx,half_sub_pixel_size);
    // let init_ray = rays.Rays[(pixel.y * ubo.width) + pixel.x];
    var type_enum = 0;
    // var intersection: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));

    var stack: array<RenderRay,MAX_RAY_BOUNCE_ARRAY_SIZE>;
    var top_stack = -1;

    top_stack = top_stack + 1;
    stack[top_stack] = RenderRay (init_ray,0u,1.0,1.0,1.0,1.0);

    // TODO: check this is light model_type (9), currently not compatible with pathtracer scenes
    let light = object_params.ObjectParams[ubo.lights_offset];
    // inverse_transform for a light is just the transform (not inversed)
    let light_position = light.transform * vec4<f32>(0.0,0.0,0.0,1.0);
    let light_emissiveness = light.material.emissiveness;

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
        let shadowed: bool = isShadowed(hitParams.overPoint, light_position.xyz);
        // let shadowed = false;
        
        let albedo = lighting(ob_params.material,
                                light_position.xyz,
                                light_emissiveness,
                                hitParams, shadowed) 
                                * new_ray.reflectance 
                                * new_ray.reflective 
                                * new_ray.transparent
                                ;
        color = color + albedo;

        if (ob_params.material.transparency > 0.0 || ob_params.material.reflective > 0.0) {
            var eta_t = ob_params.material.refractive_index;

            var cos_i = clamp(-1.0,1.0,dot(hitParams.eyev,hitParams.normalv));

            if (!hitParams.front_face) {
                eta_t=eta_t/new_ray.refractive_index;
            }

            var reflectance = 1.0;
            let do_schlick = ob_params.material.transparency > 0.0 && ob_params.material.reflective > 0.0;

            let n_ratio = new_ray.refractive_index / eta_t;
            let sin_2t = pow(n_ratio,2.0) * (1.0 - pow(cos_i,2.0));
            let cos_t = sqrt(1.0 - sin_2t);

            if (do_schlick) {
                reflectance = schlick(cos_i, eta_t, new_ray.refractive_index);
            }

            if (ob_params.material.reflective > 0.0) {
                top_stack = top_stack + 1;
                stack[top_stack] = RenderRay (Ray(hitParams.overPoint, new_ray.ray.x, hitParams.reflectv,new_ray.ray.y),new_ray.bounce_number + 1u,reflectance * new_ray.reflectance,new_ray.reflective * ob_params.material.reflective,1.0,ob_params.material.refractive_index); 
            }
                        
            if (sin_2t <= 1.0 && ob_params.material.transparency > 0.0) {
                let direction = hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                            (hitParams.eyev * n_ratio);

                top_stack = top_stack + 1;
                stack[top_stack] = RenderRay (Ray(hitParams.underPoint,new_ray.ray.x, direction,new_ray.ray.y),new_ray.bounce_number + 1u,(1.0-reflectance) * new_ray.reflectance,1.0,new_ray.transparent * ob_params.material.transparency,ob_params.material.refractive_index); 
           
            }
        }
    }
 
    return color;
}


@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
        @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) 
{
    // var color: vec4<f32> = vec4<f32>(f32((workgroup_id.x * workgroup_id.y) % 4u) / 4.0,0.0,0.0,1.0);
    var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);

    // if (ubo.subpixel_idx > 0u) {
    //     color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
    // }
    // textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);

    let ray = rays.Rays[(global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x];

    if (ray.x < 0) {
        return;
    }

    if (ubo.subpixel_idx > 0u) {
        color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
    }

    let ray_color = renderScene(ray);

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    color = mix(color,ray_color,scale);

    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}