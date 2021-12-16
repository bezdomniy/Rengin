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
    point: vec4<f32>;
    normalv: vec4<f32>;
    eyev: vec4<f32>;
    reflectv: vec4<f32>;
    overPoint: vec4<f32>;
    underPoint: vec4<f32>;
};

struct Material {
    colour: vec4<f32>;
    emissiveness: vec4<f32>;
    ambient: f32;
    diffuse: f32;
    specular: f32;
    shininess: f32;
};

struct Camera {
    inverseTransform: mat4x4<f32>;
    pixelSize: f32;
    halfWidth: f32;
    halfHeight: f32;
    width: i32;
};

[[block]]
struct UBO {
    lightPos: vec4<f32>;
    camera: Camera;
    n_objects: i32;
    // max_inner_node_idx: i32;
    // max_leaf_node_idx: i32;
    // padding: array<u32,2>;
};

// [[block]]
// struct BVH {
//     InnerNodes: [[stride(32)]] array<NodeInner>;
//     LeafNodes: [[stride(128)]] array<NodeLeaf>;
// };

[[block]]
struct InnerNodes {
    InnerNodes: [[stride(32)]] array<NodeInner>;
};

[[block]]
struct LeafNodes {
    LeafNodes: [[stride(48)]] array<NodeLeaf>;
};

[[block]]
struct Normals {
    Normals: [[stride(48)]] array<Normal>;
};

struct ObjectParam {
    inverse_transform: mat4x4<f32>;
    material: Material;
    len_inner_nodes:i32;
    len_leaf_nodes:i32;
};

[[block]]
struct ObjectParams {
    ObjectParams: [[stride(128)]] array<ObjectParam>;
    // ObjectParams: [[stride(96)]] array<ObjectParam>;
};

struct Ray {
    rayO: vec4<f32>;
    rayD: vec4<f32>;
};

struct Intersection {
    uv: vec2<f32>;
    id: i32;
    closestT: f32;
    model_id: u32;
};

[[group(0), binding(0)]]
var imageData: texture_storage_2d<rgba8unorm,write>;
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

fn gold_noise(xy: vec2<f32>, seed: f32) -> f32 {
    return fract(tan(distance(xy*PHI, xy)*seed)*xy.x);
}

fn rand(seed_xy: vec2<f32>) -> f32 {
    return fract(sin(dot(seed_xy ,vec2<f32>(12.9898,78.233))) * 43758.5453);
}

fn rescale(value: f32, min: f32, max: f32) -> f32 {
    return (value * (max - min)) + min;
}

fn linearRand(min: f32, max: f32, seed_xy: vec2<f32>) -> f32 {
    return rescale(rand(seed_xy),min,max);
}

fn sphericalRand(radius: f32, seed_xyz: vec3<f32>) -> vec3<f32>
{
    // let xy = vec2<f32>(seed_xy);

    let theta: f32 = linearRand(0.0, 6.283185307179586476925286766559,seed_xyz.xy);
    let phi: f32 = acos(linearRand(-1.0, 1.0,seed_xyz.yz));

    let x: f32 = sin(phi) * cos(theta);
    let y: f32 = sin(phi) * sin(theta);
    let z: f32 = cos(phi);

    return vec3<f32>(x,y,z) * radius;
}

fn rayForPixel(p: vec2<u32>) -> Ray {
    let xOffset: f32 = (f32(p.x) + 0.5) * ubo.camera.pixelSize;
    let yOffset: f32 = (f32(p.y) + 0.5) * ubo.camera.pixelSize;

    let worldX: f32 = ubo.camera.halfWidth - xOffset;
    let worldY: f32 = ubo.camera.halfHeight - yOffset;

    let pixel: vec4<f32> = ubo.camera.inverseTransform * vec4<f32>(worldX, worldY, -1.0, 1.0);

    let rayO: vec4<f32> = ubo.camera.inverseTransform * vec4<f32>(0.0, 0.0, 0.0, 1.0);

    return Ray(rayO, normalize(pixel - rayO));
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

fn triangleIntersect(ray: Ray, triangleIdx: u32, inIntersection: Intersection) -> Intersection {
    let triangle = leaf_nodes.LeafNodes[triangleIdx];
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = triangle.point2 - triangle.point1;
    let e2: vec3<f32> = triangle.point3 - triangle.point1;

    let dirCrossE2: vec3<f32> = cross(ray.rayD.xyz, e2);
    let det: f32 = dot(e1, dirCrossE2);
    // if (abs(det) < EPSILON) {
    //     return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    // }

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO.xyz - triangle.point1);
    uv.x = f * dot(p1ToOrigin, dirCrossE2);
    // if (uv.x < 0.0 || uv.x > 1.0) {
    //   return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    // }

    let originCrossE1: vec3<f32>  = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD.xyz, originCrossE1);
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
        // TODO: probs dont need object_id in the leaf nodes, just in the pbject params
        return Intersection(uv,i32(triangleIdx),t,triangle.object_id);
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
fn intersectInnerNodes(ray: Ray, inIntersection: Intersection, min_inner_node_idx: i32, max_inner_node_idx: i32, leaf_offset: u32) -> Intersection {
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
                    let next_intersection = triangleIntersect(ray, primIdx, ret);

                    if ((next_intersection.closestT < inIntersection.closestT)  && (next_intersection.closestT > EPSILON)) {
                        ret = next_intersection;
                    }
                }
            }
            
        }
        elseif (leaf_node) {
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

    let sphereToRay = ray.rayO - vec4<f32>(0.0, 0.0, 0.0, 1.0);
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
        if (t1 < t2) {
            return Intersection(vec2<f32>(0.0),0,t1,u32(object_id));
        }
        
        return Intersection(vec2<f32>(0.0),0,t2,u32(object_id));
    }
    return ret;
}

fn intersect(ray: Ray) -> Intersection {
    // TODO: this will need the id of the object as input in future when we are rendering more than one model
    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));

    var min_inner_node_idx = 0;
    var max_inner_node_idx = 0;
    var leaf_offset = u32(0);

    // TODO: fix loop range - get number of objects
    for (var i: i32 = 0; i < ubo.n_objects; i = i+1) {
        let ob_params = object_params.ObjectParams[i];
        let nRay: Ray = Ray(ob_params.inverse_transform * ray.rayO, ob_params.inverse_transform * ray.rayD);

        if (ob_params.len_leaf_nodes > 0) {
            // Triangle mesh
            max_inner_node_idx = max_inner_node_idx + ob_params.len_inner_nodes;
            ret = intersectInnerNodes(nRay,ret, min_inner_node_idx, max_inner_node_idx, leaf_offset);
            min_inner_node_idx = min_inner_node_idx + ob_params.len_inner_nodes;
            leaf_offset = leaf_offset + u32(ob_params.len_leaf_nodes);
        }
        else {
            let type_enum = ob_params.len_inner_nodes;
            if (type_enum == 1) { //Sphere
                ret = intersectSphere(nRay,ret, i);
            }
        }

    }

    // let nRay: Ray = Ray(object_params.ObjectParams[0].inverse_transform * ray.rayO, object_params.ObjectParams[0].inverse_transform * ray.rayD);
    // ret = intersectInnerNodes(nRay,ret);
    
    return ret;
}

fn normalToWorld(normal: vec4<f32>, object_id: u32) -> vec4<f32>
{
    let ret: vec4<f32> = normalize(vec4<f32>((transpose(object_params.ObjectParams[object_id].inverse_transform) * normal).xyz,0.0));
    // ret.w = 0.0;
    // ret = normalize(ret);

    return ret;
}

fn normalAt(point: vec4<f32>, intersection: Intersection, typeEnum: i32) -> vec4<f32> {
    // var n: vec4<f32> = vec4<f32>(0.0);
    // let objectPoint: vec4<f32> = object_params.ObjectParams[0].inverse_transform * point; // World to object

    // if (typeEnum == 0) {
    //     n = objectPoint - vec4<f32>(0.0, 0.0, 0.0, 1.0);
    // }
    // elseif (typeEnum == 1) {
    //     n = vec4<f32>(0.0,1.0,0.0,0.0);
    // }
    if (typeEnum == 0) {
        let normal: Normal = normal_nodes.Normals[intersection.id];
        return normalToWorld((normal.normal2 * intersection.uv.x + normal.normal3 * intersection.uv.y + normal.normal1 * (1.0 - intersection.uv.x - intersection.uv.y)),intersection.model_id);
        // n.w = 0.0;
    }
     elseif (typeEnum == 1) {
        let objectPoint = object_params.ObjectParams[intersection.model_id].inverse_transform * point;
        return objectPoint - vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(0.0);
}

struct HitParams {
    point: vec4<f32>;
    normalv: vec4<f32>;
    eyev: vec4<f32>;
    reflectv: vec4<f32>;
    overPoint: vec4<f32>;
    underPoint: vec4<f32>;
};

fn getHitParams(ray: Ray, intersection: Intersection, typeEnum: i32) -> HitParams
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

    return hitParams;
}

fn lighting(material: Material, lightPos: vec4<f32>, hitParams: HitParams) -> vec4<f32>
{
  // return material.colour;
  var diffuse: vec4<f32>;
  var specular: vec4<f32>;

  let intensity: vec4<f32> = vec4<f32>(1.0,1.0,1.0,1.0); // TODO temp placeholder

  let effectiveColour: vec4<f32> = intensity * material.colour; //* light->intensity;

  let ambient: vec4<f32> = effectiveColour * material.ambient;
  // vec4 ambient = vec4(0.3,0.0,0.0,1.0);

  let lightv: vec4<f32> = normalize(lightPos - hitParams.overPoint);

  let lightDotNormal: f32 = dot(lightv, hitParams.normalv);
  if (lightDotNormal < 0.0)
  {
    diffuse = vec4<f32>(0.0, 0.0, 0.0,1.0);
    specular = vec4<f32>(0.0, 0.0, 0.0,1.0);
  }
  else
  {
    // compute the diffuse contribution​
    diffuse = effectiveColour * material.diffuse * lightDotNormal;

    // reflect_dot_eye represents the cosine of the angle between the
    // reflection vector and the eye vector. A negative number means the
    // light reflects away from the eye.​
    let reflectv: vec4<f32> = reflect(-lightv, hitParams.normalv);
    let reflectDotEye: f32 = dot(reflectv, hitParams.eyev);

    if (reflectDotEye <= 0.0)
    {
      specular = vec4<f32>(0.0, 0.0, 0.0,1.0);
    }
    else
    {
      // compute the specular contribution​
      let factor: f32 = pow(reflectDotEye, material.shininess);
      specular = intensity * material.specular * factor;
    }
  }

  return (ambient + diffuse + specular);
}


fn renderScene(ray: Ray) -> vec4<f32> {
    // int id = 0;
    var color: vec4<f32> = vec4<f32>(0.0);
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // Get intersected object ID
    var intersection: Intersection = intersect(ray);
    
    if (intersection.closestT >= MAXLEN || intersection.id == -1)
    {
        // color = vec4<f32>(0.0,0.0,1.0,1.0);
        return color;
    }
    
    // vec4 pos = rayO + t * rayD;
    // vec4 lightVec = normalize(ubo.lightPos - pos);       
    // vec3 normal;

    // if (intersection.id >= 0) {
    //   for (int i = 0; i < arrayLength(shapes); i++)
    //   {
    //     if (objectID == i)
    //     {
    //       HitParams hitParams = getHitParams(rayO, rayD, t, shapes[i].inverseTransform, shapes[i].typeEnum, shapes[i].data[3], shapes[i].data[4], shapes[i].data[5], uv);

    //       bool shadowed = isShadowed(hitParams.overPoint, ubo.lightPos);
    //       color = lighting(shapes[i].material, ubo.lightPos,
    //                               hitParams, shadowed);
    //       // color = vec4(1.0,0.0,0.0,1.0);
    //     }
    //   }
    //     color = vec4<f32>(1.0,0.0,0.0,1.0);
    // }

    // else {
    if (intersection.id != -1) {
        let ob_params = object_params.ObjectParams[intersection.model_id];
        var type_enum = 0;
        if (ob_params.len_leaf_nodes == 0) {
            type_enum = ob_params.len_inner_nodes;
        }

        let hitParams: HitParams = getHitParams(ray, intersection, type_enum);

        color = lighting(ob_params.material, ubo.lightPos,
                                hitParams);
        color.w = 1.0;

        // color.g = 0.0;
        // color.g = 1.0;
        // color.g = 0.0;
        // color = vec4<f32>(0.0,1.0,0.0,1.0);
    }
    
    return color;
}

[[stage(compute), workgroup_size(8, 8)]]
fn main([[builtin(local_invocation_id)]] local_invocation_id: vec3<u32>,
        [[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>,
        [[builtin(workgroup_id)]] workgroup_id: vec3<u32>
        ) {
    let ray: Ray = rayForPixel(global_invocation_id.xy);
    let color: vec4<f32> = renderScene(ray);

    // var color: vec4<f32> = vec4<f32>(0.0,1.0,0.0,1.0);

    // if (workgroup_id.x == u32(4) && workgroup_id.y == u32(4)) {
    //     color = vec4<f32>(vec3<f32>(local_invocation_id.xyz)/255.0,1.0);
    // }

    // var color: vec4<f32> = vec4<f32>(f32(workgroup_id.x % 2u),f32(workgroup_id.y % 2u),0.0,1.0);

    // if (workgroup_id.x == u32(4) && workgroup_id.y == u32(4)) {
    //     color = vec4<f32>(vec3<f32>(local_invocation_id.xyz)/255.0,1.0);
    // }

    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}
