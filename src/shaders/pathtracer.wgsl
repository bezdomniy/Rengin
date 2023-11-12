

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
    _pad1: vec3<u32>,
    is_pathtracer: u32,
    resolution: vec2<u32>,
    _pad2: vec2<u32>,
    n_objects: u32,
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

// TODO: include surface area here, maybe in material
struct ObjectParam {
    transform: mat4x4<f32>,
    inverse_transform: mat4x4<f32>,
    material: Material,
    offset_inner_nodes: u32,
    len_inner_nodes:u32,
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
var<storage, read> rays: Rays;

fn float_to_linear_rgb(x: f32) -> f32 {
    if (x > 0.04045) {
        return pow((x + 0.055) / 1.055,2.4);
    }
    return x / 12.92;
}

fn to_linear_rgb(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(float_to_linear_rgb(c.x),float_to_linear_rgb(c.y),float_to_linear_rgb(c.z),1.0);
}

const EPSILON:f32 = 0.001;
const MAXLEN: f32 = 10000.0;
const INFINITY: f32 = 340282346638528859811704183484516925440.0;
const NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;
const PI: f32 = 3.1415926535897932384626433832795;

var<private> rand_pcg4d: vec4<u32>;

fn init_pcg4d(v: vec4<u32>)
{
    rand_pcg4d = v * 1664525u + 1013904223u;

    rand_pcg4d.x = rand_pcg4d.x + (rand_pcg4d.y * rand_pcg4d.w);
    rand_pcg4d.y = rand_pcg4d.y + (rand_pcg4d.z * rand_pcg4d.x);
    rand_pcg4d.z = rand_pcg4d.z + (rand_pcg4d.x * rand_pcg4d.y);
    rand_pcg4d.w = rand_pcg4d.w + (rand_pcg4d.y * rand_pcg4d.z);

    let _rand_pcg4d = vec4<u32>(rand_pcg4d.x >> 16u,rand_pcg4d.y >> 16u,rand_pcg4d.z >> 16u,rand_pcg4d.w >> 16u);
    rand_pcg4d = rand_pcg4d ^ _rand_pcg4d;

    // rand_pcg4d = rand_pcg4d ^ (rand_pcg4d >> 16u);
    rand_pcg4d.x = rand_pcg4d.x + (rand_pcg4d.y * rand_pcg4d.w);
    rand_pcg4d.y = rand_pcg4d.y + (rand_pcg4d.z * rand_pcg4d.x);
    rand_pcg4d.z = rand_pcg4d.z + (rand_pcg4d.x * rand_pcg4d.y);
    rand_pcg4d.w = rand_pcg4d.w + (rand_pcg4d.y * rand_pcg4d.z);
}

fn u32_to_f32(x: u32) -> f32 {
    return bitcast<f32>(0x3f800000u | (x >> 9u)) - 1.0;
}

// alternative
fn _u32_to_f32(x: u32) -> f32 {
    return f32(x) * (1.0/f32(0xffffffffu));
}

fn rescale(value: f32, min: f32, max: f32) -> f32 {
    return (value * (max - min)) + min;
}

fn random_in_unit_sphere() -> vec3<f32> {
    let phi = 2.0 * PI * u32_to_f32(rand_pcg4d.x);
    let cos_theta = 2.0 * u32_to_f32(rand_pcg4d.y) - 1.0;
    let u = u32_to_f32(rand_pcg4d.z);

    let theta = acos(cos_theta);
    let r = pow(u, 1.0 / 3.0);

    let x = r * sin(theta) * cos(phi);
    let y = r * sin(theta) * sin(phi);
    let z = r * cos(theta);
    return vec3<f32>(x,y,z);
}

fn random_in_cube() -> vec3<f32> {
    let x = rescale(u32_to_f32(rand_pcg4d.x), -1.0, 1.0);
    let y = rescale(u32_to_f32(rand_pcg4d.y), -1.0, 1.0);
    let z = rescale(u32_to_f32(rand_pcg4d.z), -1.0, 1.0);
    return vec3<f32>(x,y,z);
}



// Probably not need this, just point in cube is fine
// from this: https://stackoverflow.com/questions/18182376/figuring-out-how-much-of-the-side-of-a-cube-is-visible
// TODO: check if the dot products of 3 negative face indeed add to -1

// var<private> FACE_NORMALS: array<vec3<f32>,6> = 
//     array<vec3<f32>,6>(  
//         vec3<f32>(-1f,0f,0f),vec3<f32>(0f,-1f,0f),vec3<f32>(0f,0f,-1f),
//         vec3<f32>(1f,0f,0f), vec3<f32>(0f,1f,0f), vec3<f32>(0f,0f,1f)
//                                     );

// fn random_to_cube_face(view_vector: vec3<f32>) -> vec3<f32> {
//     let a1 = rescale(u32_to_f32(rand_pcg4d.x), -1f, 1f);
//     let a2 = rescale(u32_to_f32(rand_pcg4d.y), -1f, 1f);

//     let choice_idx = u32_to_f32(rand_pcg4d.z);

//     var faces = array<vec3<f32>,3>(vec3<f32>(0.0),vec3<f32>(0.0),vec3<f32>(0.0));
//     var thres = array<f32,3>(0f,0f,0f);

//     let start = rand_pcg4d.z % 6u;
//     var j = 0u;
//     for (var i = start; i < start + 6u; i = i+1u) {
//         let idx = i % 6u;
//         var r = FACE_NORMALS[idx];
//         let d = dot(r,normalize(view_vector));
//         if (d < 0f) {
//             // r[idx % 3u] = -r[idx % 3u];
//             r[(idx + 1u) % 3u] = a1;
//             r[(idx + 2u) % 3u] = a2;

//             faces[j] = r;
//             thres[j] = -d;

//             if (j == 2u) {
//                 break;
//             }

//             j = j + 1u;

//             // return r;
//         }
//     }

//     var tot = 0f;
//     for (var i = 0u; i < j; i = i+1u) {
//         tot = tot + thres[i];
//         if (choice_idx < tot) {
//             return faces[i];
//         }
//     }

//     return faces[j];
// }

fn random_cosine_direction() -> vec3<f32> {
    let r1 = u32_to_f32(rand_pcg4d.x);
    let r2 = u32_to_f32(rand_pcg4d.y);
    let z = sqrt(1.0-r2);

    let phi = 2.0*PI*r1;
    let x = cos(phi)*sqrt(r2);
    let y = sin(phi)*sqrt(r2);

    return vec3<f32>(x, y, z);
}

// TODO: update so it can handle non-spherical shapes
fn random_to_sphere(radius: f32, distance_squared: f32) -> vec3<f32> {
    let r1 = u32_to_f32(rand_pcg4d.x);
    let r2 = u32_to_f32(rand_pcg4d.y);
    let z = 1.0 + r2*(sqrt(1.0-radius*radius/distance_squared) - 1.0);

    let phi = 2.0*PI*r1;
    let scale = sqrt(1.0-(z*z));
    let x = cos(phi)*scale;
    let y = sin(phi)*scale;

    return vec3<f32>(x, y, z);
}

fn random_uniform_direction() -> vec3<f32> {
    let r1 = u32_to_f32(rand_pcg4d.x);
    let r2 = u32_to_f32(rand_pcg4d.y);
    let x = cos(2.0*PI*r1)*2.0*sqrt(r2*(1.0-r2));
    let y = sin(2.0*PI*r1)*2.0*sqrt(r2*(1.0-r2));
    let z = 1.0 - 2.0*r2;

    return vec3<f32>(x, y, z);
}

fn random_uniform_on_hemisphere() -> vec3<f32> {
    let azimuthal = 2.0*PI*u32_to_f32(rand_pcg4d.x);
    let z = u32_to_f32(rand_pcg4d.y);

    let xyproj = sqrt(1.0-(z*z));

    let x = cos(azimuthal)*xyproj;
    let y = sin(azimuthal)*xyproj;

    return vec3<f32>(x, y, z);
}

// fn random_uniform_direction(radius: f32) -> vec3<f32>
// {
//     let theta: f32 = rescale(u32_to_f32(rand_pcg4d.x), 0.0, PI * 2.0);
//     let phi: f32 = acos(rescale(u32_to_f32(rand_pcg4d.y), -1.0, 1.0));

//     let x: f32 = sin(phi) * cos(theta);
//     let y: f32 = sin(phi) * sin(theta);
//     let z: f32 = cos(phi);

//     return normalize(vec3<f32>(x,y,z) * radius);
// }

fn hemisphericalRand(normal: vec3<f32>) -> vec3<f32>
{
    let in_unit_sphere = random_uniform_direction();
    if (dot(in_unit_sphere, normal) > 0.0) { // In the same hemisphere as the normal
        return in_unit_sphere;
    }
    return -in_unit_sphere;
}

fn intersectAABB(ray: Ray, inner_node: NodeInner) -> bool {
    // let INFINITY: f32 = 1.0 / 0.0;

    var t_min: f32 = NEG_INFINITY;
    var t_max: f32 = INFINITY;
    // var temp: f32;
    // var invD: f32;
    var t0: f32;
    var t1: f32;

    // let inner_node = inner_nodes.InnerNodes[aabbIdx];

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

    let originCrossE1: vec3<f32>  = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD, originCrossE1);
    
    let t = f * dot(e2, originCrossE1);

    let isHit: bool = (uv.x >= 0.0) && (uv.y >= 0.0)
                    && (uv.x + uv.y <= 1.0)
                    && (t < inIntersection.closestT)
                    && (t > EPSILON);

    if (isHit) {
        return Intersection(uv,i32(triangleIdx),t,object_id);
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
        if (idx >= max_inner_node_idx ) {break;};

        let current_node: NodeInner = inner_nodes.InnerNodes[idx];

        let leaf_node: bool = current_node.idx2 > 0u;

        if (intersectAABB(ray, current_node)) {
            if (leaf_node) {
                for (var primIdx: u32 = current_node.skip_ptr_or_prim_idx1 + leaf_offset; primIdx < current_node.idx2 + leaf_offset; primIdx = primIdx + 1u) {
                    let next_intersection = intersectTriangle(ray, primIdx, ret,object_id);

                    if ((next_intersection.closestT < inIntersection.closestT)  && (next_intersection.closestT > EPSILON)) {
                        ret = next_intersection;
                    }
                }
            }
            idx = idx + 1u;
        }
        else if (leaf_node) {
            idx = idx + 1u;
        }
        else {
            idx = current_node.skip_ptr_or_prim_idx1 + min_inner_node_idx;
        }
    }
    return ret;
}

fn intersectSphere(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
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
            return Intersection(vec2<f32>(0.0),0,t1,object_id);
        }
        
        if (t2 > EPSILON) {
            return Intersection(vec2<f32>(0.0),0,t2,object_id);
        }
    }
    return ret;
}

fn intersectPlane(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
    var ret: Intersection = inIntersection;

    if (abs(ray.rayD.y) < EPSILON) {
        return ret;
    }

    let t: f32 = -ray.rayO.y / ray.rayD.y;

    if (t < ret.closestT && t > EPSILON) {
        return Intersection(vec2<f32>(0.0),0,t,object_id);
    }
    return ret;
}

fn intersectCube(ray: Ray, inIntersection: Intersection, object_id: u32) -> Intersection {
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
        return Intersection(vec2<f32>(0.0),0,t_min,object_id);
    }
    else if (t_max < ret.closestT && t_max > EPSILON) {
        return Intersection(vec2<f32>(0.0),0,t_max,object_id);
    }
    return ret;
}

fn intersect(ray: Ray,start:u32, immediate_ret: bool) -> Intersection {
    // TODO: this will need the id of the object as input in future when we are rendering more than one model
    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN, u32(0));


    // TODO: fix loop range - get number of objects
    for (var i: u32 = start; i < ubo.n_objects; i = i+1u) {
        let ob_params = object_params.ObjectParams[i];

        // TODO: clean this up
        if (ob_params.model_type == 9u) //point light from whitted rt 
        {
            continue;
        }
            
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

        if (immediate_ret) {
            return ret;
        }
    }

    return ret;
}

fn normalToWorld(normal: vec3<f32>, ob_param: ObjectParam) -> vec3<f32>
{
    let ret: vec3<f32> = normalize((transpose(ob_param.inverse_transform) * vec4<f32>(normal,0.0)).xyz);
    // ret.w = 0.0;
    // ret = normalize(ret);


    return ret;
}

fn normalAt(p: vec3<f32>, intersection: Intersection, typeEnum: u32) -> vec3<f32> {
    let ob_param = object_params.ObjectParams[intersection.model_id];
    if (typeEnum == 0u) { //Sphere
        let objectPoint = (ob_param.inverse_transform * vec4<f32>(p,1.0)).xyz;
        return normalToWorld(objectPoint,ob_param);
    }
    else if (typeEnum == 1u) { //Plane
        return normalToWorld(vec3<f32>(0.0, 1.0, 0.0),ob_param);
    }
    else if (typeEnum == 2u) { //Cube
        let objectPoint = (ob_param.inverse_transform * vec4<f32>(p,1.0)).xyz;
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

        return normalToWorld(objectNormal,ob_param);
    }
    else { //Model
        let normal: Normal = normal_nodes.Normals[intersection.id];
        return normalToWorld((normal.normal2.xyz * intersection.uv.x + normal.normal3.xyz * intersection.uv.y + normal.normal1.xyz * (1.0 - intersection.uv.x - intersection.uv.y)),ob_param);
        // n.w = 0.0;
    }
    return vec3<f32>(0.0);
}


fn getHitParams(ray: Ray, intersection: Intersection, typeEnum: u32) -> HitParams
{
    var hitParams: HitParams;    
    let normal_ray_d = normalize(ray.rayD);
    hitParams.p =
        ray.rayO + normal_ray_d * intersection.closestT;
    // TODO check that uv only null have using none-uv normalAt version
    hitParams.normalv = 
        normalAt(hitParams.p, intersection, typeEnum);
    // hitParams.eyev = -ray.rayD;
    hitParams.eyev = -normal_ray_d;

    // hitParams.front_face = dot(hitParams.normalv, hitParams.eyev) < 0.0;
    hitParams.front_face = dot(ray.rayD, hitParams.normalv) < 0.0;
    if (!hitParams.front_face)
    {
        hitParams.normalv = -hitParams.normalv;
    }

    hitParams.reflectv =
        reflect(normal_ray_d, hitParams.normalv);
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
    // // Use Schlick's approximation for reflectance.
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

struct RenderRay {
    ray: Ray,
    refractive_index: f32,
};


// fn onb(n: vec3<f32>) -> array<vec3<f32>,3> {
//     var out: array<vec3<f32>,3>;

//     out[2] = n;

//     if ( n.z >= n.y ) {
//         let a = 1.0/(1.0 + n.z);
//         let b = -n.x * n.y * a;
//         out[0] = vec3<f32>( 1.0 - n.x*n.x*a, b, -n.x );
//         out[1] = vec3<f32>( b, 1.0 - n.y*n.y*a, -n.y );
//     } else {
//         let a = 1.0/(1.0 + n.y);
//         let b = -n.x*n.z*a;
//         out[0] = vec3<f32>( b, -n.z, 1.0 - n.z*n.z*a );
//         out[1] = vec3<f32>( 1.0 - n.x*n.x*a, -n.x, b );
//     }

//     return out;
// }


	// float sign = n.z >= 0.f ? 1.0f : -1.f;
	// float a = -1.0/(sign + n.z);
	// float b = n.x * n.y * a; 
	// b1 = vec3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
	// b2 = vec3(b, sign + n.y * n.y * a, -n.y);	


fn onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;
    out[2] = n; 

    let sign = select(-1.0,1.0, n.z >= 0.0);
    let a = -1.0 / (sign / n.z);
    let b = n.x * n.y * a;
    out[0] = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    out[1] = vec3<f32>(b, sign + n.y * n.y * a, -n.y);

    return out;
}

fn _onb(n: vec3<f32>) -> mat3x3<f32> {
    var out: mat3x3<f32>;

    out[2] = n; 
    if(n.z < -0.99995)
    {
        out[0] = vec3<f32>(0.0 , -1.0, 0.0);
        out[1] = vec3<f32>(-1.0, 0.0, 0.0);
    }
    else
    {
        let a = 1.0/(1.0 + n.z);
        let b = -n.x*n.y*a;
        out[0] = vec3<f32>(1.0 - n.x*n.x*a, b, -n.x);
        out[1] = vec3<f32>(b, 1.0 - n.y*n.y*a , -n.y);
    }

    return out;
}


// fn onb(n: vec3<f32>) -> array<vec3<f32>,3> {
//     var out: array<vec3<f32>,3>;
//     out[2] = n;

//     if (abs(out[2].x) > 0.9) {
//         let a = vec3<f32>(0.0,1.0,0.0);
//     }
//     else {
//         let a = vec3<f32>(1.0,0.0,0.0);
//     }
//     out[1] = normalize(cross(out[2], a));
//     out[0] = cross(out[2], out[1]);

//     return out;
// }

fn onb_local(v: vec3<f32>, onb: mat3x3<f32>) -> vec3<f32> {
    // return v.x*onb[0] + v.y*onb[1] + v.z*onb[2];
    return onb * v;
}

fn random_light() -> ObjectParam {
    let i = u32(rescale(u32_to_f32(rand_pcg4d.x), f32(ubo.lights_offset), f32(ubo.n_objects)));
    return object_params.ObjectParams[i];
}

fn random_to_light(light: ObjectParam, origin: vec3<f32>) -> vec3<f32> {
    let center = (light.transform * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
    let direction = center - origin;
    let distance_squared = pow(length(direction),2.0);

    let onb = onb(normalize(direction));

    let scale = vec3<f32>(length(light.transform[0].xyz),length(light.transform[1].xyz),length(light.transform[2].xyz));

    // TODO: fix the negative offset - lots of hot pixels without it, something wrong
    let radius = max(max(scale.x,scale.y),scale.z) - 0.05f;
    
    if (light.model_type == 0u) {
        let r = onb * random_to_sphere(radius,distance_squared);
        return r;
    }
    else if (light.model_type == 1u) {
        return vec3<f32>(0.0);
    }
    else if (light.model_type == 2u) {
        let p = random_in_cube();
        // let p = random_to_cube_face((light.inverse_transform * vec4<f32>(direction,0f)).xyz);
        let r = (light.transform * vec4<f32>(p,1f)).xyz;
        return normalize(r - origin);
    }

    // TODO: for model mesh, choose random triangle, then random point on it
    return vec3<f32>(1.0);
}

// TODO: check the initial size cube and sphere and centroid coordinates and adjust accordingly
fn surface_area(object: ObjectParam) -> f32 {
    let scale = vec3<f32>(length(object.transform[0].xyz),length(object.transform[1].xyz),length(object.transform[2].xyz));
    if (object.model_type == 0u) {
        return 4.0 * PI * pow((pow(scale.x * scale.z,1.6075) + pow(scale.z * scale.y,1.6075)  + pow(scale.x * scale.y,1.6075))/3.0,1.0/1.6075);
    }
    else if (object.model_type == 1u) {
        // TODO
        return 1.0;
    }
    else if (object.model_type == 2u) {
        // return (343.0 - 213.0) * (332.0 - 227.0);
        return 2.0 * ((2.0 * scale.x * scale.z) + (2.0 * scale.z * scale.y) + (2.0 * scale.x * scale.y));
    }
    else if (object.model_type == 9u) {
        return 1.0;
    }

    // TODO
    return 1.0;
}

fn light_pdf(ray: Ray, intersection: Intersection) -> f32 {
    if (intersection.id == -1 || intersection.closestT >= MAXLEN) {
        return 0f;
    }

    let light = object_params.ObjectParams[intersection.model_id];

    if (light.model_type == 0u) {
        let scale = vec3<f32>(length(light.transform[0].xyz),length(light.transform[1].xyz),length(light.transform[2].xyz));
        let radius = max(max(scale.x,scale.y),scale.z);

        // let hit_point = ray.rayO + intersection.closestT * normalize(ray.rayD);
        // let normal = normalAt(hit_point, intersection, 0u);
        // let center = hit_point - (normal * radius );
        let center = (light.transform * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;

        let cos_theta_max = sqrt(1.0 - (radius*radius/pow(length(center-ray.rayO),2.0)));
        let solid_angle = 2.0*PI*(1.0-cos_theta_max);
        return 1.0 / solid_angle;


        // let cos_theta = dot(-ray.rayD, normalize(hit_point - center));
        // let light_area = surface_area(light);
        // return pow(intersection.closestT,2.0) / (light_area * cos_theta * 0.5);


        // let light_area = surface_area(light);
        // return pow(intersection.closestT,2.0) / (cosine * light_area);

        // let light_area = surface_area(light);
        // let point = ray.rayO + normalize(ray.rayD) * intersection.closestT;
        // let n_light = normalAt(point, intersection,0u);
        // let solid_angle = (light_area * dot(n_light, ray.rayD)) / pow(intersection.closestT,2.0);
        // return 1.0 / solid_angle;
    }
    else if (light.model_type == 1u) {
        // TODO
        return 0.0;
    }
    else if (light.model_type == 2u) {
        let p = ray.rayO + normalize(ray.rayD) * intersection.closestT;
        let normal = normalAt(p, intersection, light.model_type);
        let light_area = surface_area(light);
        // return 1f - light_area;
        let distance_squared = pow(intersection.closestT,2f);
        let cosine = abs(dot(normal,-ray.rayD));

        return distance_squared / (cosine * light_area);
    }
    else if (light.model_type == 9u) {
        return 0.0;
    }

    // TODO
    return 0.0;
}

fn renderScene(init_ray: Ray, xy: vec2<u32>,light_sample: bool) -> vec4<f32> {
    init_pcg4d(vec4<u32>(xy.x, xy.y, ubo.subpixel_idx, ubo.n_objects));
    var p_scatter = 0.5;
    if (ubo.lights_offset == ubo.n_objects) {
        p_scatter = 1.0;
    }
    
    var throughput: vec4<f32> = vec4<f32>(1.0);

    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    var new_ray = RenderRay(init_ray,1.0);

    // var ob_params = object_params.ObjectParams[0];
    // let ray_miss_colour = vec4<f32>(1.0);
    // let ray_miss_colour = vec4<f32>(0.1,0.1,0.1,1.0);
    let ray_miss_colour = vec4<f32>(0.0);

    var sample_light = false;
    
    for (var bounce_idx: u32 = 0u; bounce_idx < ubo.ray_bounces; bounce_idx =  bounce_idx + 1u) {

        // Get intersected object ID
        let intersection = intersect(new_ray.ray,0u,false);
        
        if (intersection.id == -1 || intersection.closestT >= MAXLEN)
        {
            return ray_miss_colour * throughput;
        }

        // TODO: just hard code object type in the intersection rather than looking it up
        let ob_params = object_params.ObjectParams[intersection.model_id];

        
        if (ob_params.material.emissiveness.x > 0.0) {
            return ob_params.material.emissiveness * throughput;
        }

        let hitParams = getHitParams(new_ray.ray, intersection, ob_params.model_type);

        var is_specular = ob_params.material.reflective > 0.0 || ob_params.material.transparency > 0.0;
        var scattering_target =  vec3<f32>(0.0);
        var p = hitParams.overPoint;
        let albedo = ob_params.material.colour;
        

        var pdf_adj = 1.0;

        let onb = onb(hitParams.normalv);

        if (is_specular) {
            if (ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0) {
                // scattering_target = hitParams.reflectv;

                // let onb_reflect = onb(hitParams.reflectv);
                // let noise_direction = onb_local(random_cosine_direction(), onb_reflect);
                // scattering_target = ((1.0 - ob_params.material.reflective) * noise_direction);
                // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * random_uniform_direction());
                scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * onb * random_cosine_direction());
            }

            else if (ob_params.material.transparency > 0.0) {
                var eta_t = ob_params.material.refractive_index;

                var cos_i = min(dot(hitParams.eyev, hitParams.normalv), 1.0);

                if (!hitParams.front_face) {
                    eta_t=eta_t/new_ray.refractive_index;
                }

                let reflectance = schlick(cos_i, eta_t, new_ray.refractive_index);
                // let reflectance = schlick_lazanyi(cos_i,eta_t,0.0);

                let n_ratio = new_ray.refractive_index / eta_t;
                let sin_2t = pow(n_ratio, 2.0) * (1.0 - pow(cos_i, 2.0));

                if (sin_2t > 1.0 || reflectance >= u32_to_f32(rand_pcg4d.w))
                {
                    // scattering_target = hitParams.reflectv;
                    // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * random_uniform_direction());
                    scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * onb * random_cosine_direction());
                    // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * hemisphericalRand(1.0,hitParams.normalv));
                }
                else {
                    let cos_t = sqrt(1.0 - sin_2t);

                    // scattering_target = (hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                    //                 (hitParams.eyev * n_ratio)) + ((1.0 - ob_params.material.transparency) * onb_local(-random_cosine_direction(), onb));
                    scattering_target = hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                                    (hitParams.eyev * n_ratio);
                    p = hitParams.underPoint;
                }
            }
            let ray = Ray(p, init_ray.x, normalize(scattering_target), init_ray.y);
            new_ray = RenderRay(ray, ob_params.material.refractive_index);
        }
        else {

            sample_light = light_sample && u32_to_f32(rand_pcg4d.w) < p_scatter;
            if (sample_light) {
                let light = random_light();
                scattering_target = random_to_light(light, p);
                // scattering_target = on_light - p;
            }
            else {
                // scattering_target = random_cosine_direction();
                scattering_target = onb * random_cosine_direction();
            }

            let direction = normalize(scattering_target);
            let ray = Ray(p, init_ray.x, direction, init_ray.y);
            new_ray = RenderRay(ray, ob_params.material.refractive_index);   

            
            let scattering_cosine = dot(direction, onb[2]);
            // var scattering_pdf = 0f;
            // if (scattering_cosine > 0f) {
                let scattering_pdf = scattering_cosine / PI;
            // }

            // let l_intersection = intersect(ray,ubo.lights_offset,false);
            // let v_light_pdf = light_pdf(ray, l_intersection);

            var v_light_pdf = 0f;
            
            for (var i_light: u32 = ubo.lights_offset; i_light < ubo.n_objects; i_light = i_light+1u) {
                let l_intersection = intersect(ray,i_light,true);
                if (l_intersection.id == -1 || l_intersection.closestT >= MAXLEN) {
                    continue;
                }
                
                v_light_pdf = v_light_pdf + light_pdf(ray, l_intersection);
                // break;
            }

            v_light_pdf = v_light_pdf / f32(ubo.n_objects - ubo.lights_offset);
            // v_light_pdf = v_light_pdf / f32(ubo.n_objects);
            
            let pdf = (p_scatter * scattering_pdf) + ((1.0 - p_scatter) * v_light_pdf);

            if (pdf > 0.0) {
                pdf_adj = scattering_pdf / pdf;
            }
            else {
                pdf_adj = scattering_pdf;
            }
        }

        throughput = throughput * albedo * pdf_adj;
        init_pcg4d(rand_pcg4d);
    }

    return ray_miss_colour * throughput;
}



@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
        @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) 
{
    let ray = rays.Rays[(global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x];

    if (ray.x < 0) {
        return;
    }

    var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    if (ubo.subpixel_idx > 0u) {
        color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
    }

    
    var light_sample = true;
    if (ubo.lights_offset == ubo.n_objects) {
        light_sample = false;
    }
    
    var ray_color = renderScene(ray,global_invocation_id.xy,light_sample);
    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    if (ray_color.x != ray_color.x) { ray_color.x = 0.0; }
    if (ray_color.y != ray_color.y) { ray_color.y = 0.0; }
    if (ray_color.z != ray_color.z) { ray_color.z = 0.0; }
    // if (ray_color.w != ray_color.w) { ray_color.w = 1.0; }

    color = mix(color,ray_color,scale);
    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}
