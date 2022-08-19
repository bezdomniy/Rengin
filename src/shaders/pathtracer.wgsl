

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
    refractive_index: f32,
    rayD: vec3<f32>,
    bounce_idx: i32,
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
var<storage, read_write> rays: array<Ray>;
@group(0) @binding(7)
var<storage, read_write> throughput: array<vec4<f32>>;

fn float_to_linear_rgb(x: f32) -> f32 {
    if (x > 0.04045) {
        return pow((x + 0.055) / 1.055,2.4);
    }
    return x / 12.92;
}

fn to_linear_rgb(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(float_to_linear_rgb(c.x),float_to_linear_rgb(c.y),float_to_linear_rgb(c.z),1.0);
}

let EPSILON:f32 = 0.001;
let MAXLEN: f32 = 10000.0;
let INFINITY: f32 = 340282346638528859811704183484516925440.0;
let NEG_INFINITY: f32 = -340282346638528859811704183484516925440.0;
let PI: f32 = 3.1415926535897932384626433832795;

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

fn random_cosine_direction() -> vec3<f32> {
    let r1 = u32_to_f32(rand_pcg4d.x);
    let r2 = u32_to_f32(rand_pcg4d.y);
    let z = sqrt(1.0-r2);

    let phi = 2.0*PI*r1;
    let x = cos(phi)*sqrt(r2);
    let y = sin(phi)*sqrt(r2);

    return vec3<f32>(x, y, z);
}

fn random_to_sphere(radius: f32, distance_squared: f32) -> vec3<f32> {
    let r1 = u32_to_f32(rand_pcg4d.x);
    let r2 = u32_to_f32(rand_pcg4d.y);
    let z = 1.0 + r2*(sqrt(1.0-radius*radius/distance_squared) - 1.0);

    let phi = 2.0*PI*r1;
    let x = cos(phi)*sqrt(1.0-(z*z));
    let y = sin(phi)*sqrt(1.0-(z*z));

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

fn intersectAABB(ray: Ray, aabbIdx: u32) -> bool {
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
        if (idx >= max_inner_node_idx ) {break};

        let current_node: NodeInner = inner_nodes.InnerNodes[idx];
        let leaf_node: bool = current_node.idx2 > 0u;

        if (intersectAABB(ray, idx)) {
            idx = idx + 1u;
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
        if (ob_params.model_type == 9u) {
            continue;
        }
            
        let nRay: Ray = Ray((ob_params.inverse_transform * vec4<f32>(ray.rayO,1.0)).xyz, ray.refractive_index, (ob_params.inverse_transform * vec4<f32>(ray.rayD,0.0)).xyz, ray.bounce_idx);

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

fn onb(n: vec3<f32>) -> array<vec3<f32>,3> {
    var out: array<vec3<f32>,3>;

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

fn onb_local(v: vec3<f32>, onb: array<vec3<f32>,3>) -> vec3<f32> {
    return v.x*onb[0] + v.y*onb[1] + v.z*onb[2];
}

fn random_light() -> ObjectParam {
    let i = u32(rescale(u32_to_f32(rand_pcg4d.x), f32(ubo.lights_offset), f32(ubo.n_objects)));
    return object_params.ObjectParams[i];
}

fn random_point_on_light(light: ObjectParam, origin: vec3<f32>) -> vec3<f32> {
    if (light.model_type == 0u) {
        // let r = random_cosine_direction();
        // let r = random_uniform_on_hemisphere();
        let r = random_in_unit_sphere();
        
        // let center = (light.transform * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
        // let to_light = -normalize(center - origin);
        // let onb = onb(to_light);
        // let r = onb_local(random_uniform_on_hemisphere(),onb);

        return (light.transform * vec4<f32>(r,1.0)).xyz;
    }
    else if (light.model_type == 1u) {
        return vec3<f32>(0.0);
    }
    else if (light.model_type == 2u) {
        
        let r = random_in_cube();
        return (light.transform * vec4<f32>(r,1.0)).xyz; 

        // // this is random on cube surface
        // let random_side = rand_pcg4d.z % 6u;
        // let axis = random_side % 3u;

        // if (random_side > 2u) {
        //     let c = 1.0;
        // }
        // else {
        //     let c = -1.0;
        // }

        // var r = vec3<f32>(0.0);
        // r[axis] = c;
        // r[(axis + 1u) % 3u] = (u32_to_f32(rand_pcg4d.x) * 2.0) - 1.0;
        // r[(axis + 2u) % 3u] = (u32_to_f32(rand_pcg4d.y) * 2.0) - 1.0;

        // return (light.transform * vec4<f32>(r,1.0)).xyz;
    }

    // TODO: for model mesh, choose random triangle, then random point on it
    return vec3<f32>(0.0);
}

// TODO: check the initial size cube and sphere and centroid coordinates and adjust accordingly
fn surface_area(object: ObjectParam) -> f32 {
    let scale = vec3<f32>(length(object.transform[0].xyz),length(object.transform[1].xyz),length(object.transform[2].xyz));
    if (object.model_type == 0u) {
        return 4.0 * PI * pow((pow(scale.x * scale.z,1.6075) + pow(scale.x * scale.z,1.6075)  + pow(scale.x * scale.z,1.6075))/3.0,1.0/1.6075);
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

fn light_pdf(ray: Ray, intersection: Intersection, cosine: f32) -> f32 {
    let light = object_params.ObjectParams[intersection.model_id];
    if (light.model_type == 0u) {
        // let scale = vec3<f32>(length(light.transform[0].xyz),length(light.transform[1].xyz),length(light.transform[2].xyz));
        // let radius = max(max(scale.x,scale.y),scale.z);
        // let center = (light.transform * vec4<f32>(0.0,0.0,0.0,1.0)).xyz;
        // let o = ray.rayO;

        // let cos_theta_max = sqrt(1.0 - (radius*radius/pow(length(center-o),2.0)));
        // let solid_angle = 2.0*PI*(1.0-cos_theta_max);
        // return 1.0 / solid_angle;


        let light_area = surface_area(light);
        return pow(intersection.closestT,2.0) / (cosine * light_area);

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
        let light_area = surface_area(light);
        return pow(intersection.closestT,2.0) / (cosine * light_area);
    }
    else if (light.model_type == 9u) {
        return 0.0;
    }

    // TODO
    return 0.0;
}

fn renderScene(ray: Ray, offset: u32,light_sample: bool) -> bool {
    var p_scatter = 0.5;
    if (ubo.lights_offset == ubo.n_objects) {
        p_scatter = 1.0;
    }
    
    var uv: vec2<f32>;
    var t: f32 = MAXLEN;

    // var new_ray = RenderRay(init_ray,1.0);

    var albedo = vec4<f32>(0.0);

    // var ob_params = object_params.ObjectParams[0];
    // let ray_miss_colour = vec4<f32>(1.0);
    // let ray_miss_colour = vec4<f32>(0.1,0.1,0.1,1.0);
    let ray_miss_colour = vec4<f32>(0.0);
    
    

    // Get intersected object ID
    let intersection = intersect(ray,0u,false);
    
    if (intersection.id == -1 || intersection.closestT >= MAXLEN)
    {
        rays[offset] = Ray(vec3<f32>(-1f), -1f, vec3<f32>(-1f), -1);
        throughput[offset] = throughput[offset] * ray_miss_colour;
        return true;
    }

    // TODO: just hard code object type in the intersection rather than looking it up
    let ob_params = object_params.ObjectParams[intersection.model_id];

    if (ob_params.material.emissiveness.x > 0.0) {
        rays[offset] = Ray(vec3<f32>(-1f), -1f, vec3<f32>(-1f), -1);
        throughput[offset] = throughput[offset] * ob_params.material.emissiveness;
        return true;
    }

    let hitParams = getHitParams(ray, intersection, ob_params.model_type);

    var is_specular = ob_params.material.reflective > 0.0 || ob_params.material.transparency > 0.0;
    var scattering_target =  vec3<f32>(0.0);
    var p = hitParams.overPoint;
    let albedo = ob_params.material.colour;

    var pdf_adj = 1.0;

    if (is_specular) {
        if (ob_params.material.reflective > 0.0 && ob_params.material.transparency == 0.0) {
            // scattering_target = hitParams.reflectv;

            // let onb_reflect = onb(hitParams.reflectv);
            // let noise_direction = onb_local(random_cosine_direction(), onb_reflect);
            // scattering_target = ((1.0 - ob_params.material.reflective) * noise_direction);
            scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * random_uniform_direction());
        }

        else if (ob_params.material.transparency > 0.0) {
            var eta_t = ob_params.material.refractive_index;

            var cos_i = min(dot(hitParams.eyev, hitParams.normalv), 1.0);

            if (!hitParams.front_face) {
                eta_t=eta_t/ray.refractive_index;
            }

            let reflectance = schlick(cos_i, eta_t, ray.refractive_index);
            // let reflectance = schlick_lazanyi(cos_i,eta_t,0.0);

            let n_ratio = ray.refractive_index / eta_t;
            let sin_2t = pow(n_ratio, 2.0) * (1.0 - pow(cos_i, 2.0));

            if (sin_2t > 1.0 || reflectance >= u32_to_f32(rand_pcg4d.w))
            {
                // scattering_target = hitParams.reflectv;
                scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * random_uniform_direction());
                // scattering_target = hitParams.reflectv + ((1.0 - ob_params.material.reflective) * hemisphericalRand(1.0,hitParams.normalv));
            }
            else {
                let cos_t = sqrt(1.0 - sin_2t);

                scattering_target = hitParams.normalv * ((n_ratio * cos_i) - cos_t) -
                                (hitParams.eyev * n_ratio);

                p = hitParams.underPoint;
            }
        }
        rays[offset] = Ray(p, ob_params.material.refractive_index, normalize(scattering_target), ray.bounce_idx + 1);
    }
    else {
        let onb = onb(hitParams.normalv);

        if (light_sample && u32_to_f32(rand_pcg4d.w) < p_scatter) {
            let light = random_light();
            let on_light = random_point_on_light(light, p);
            scattering_target = on_light - p;
        }
        else {
            scattering_target = onb_local(random_cosine_direction(), onb);
        }

        let direction = normalize(scattering_target);
        rays[offset] = Ray(p, ob_params.material.refractive_index, direction, ray.bounce_idx + 1);
        
        let scattering_cosine = dot(direction, onb[2]);
        let scattering_pdf = scattering_cosine / PI;

        var v_light_pdf = 0.0;
        for (var i_light: u32 = ubo.lights_offset; i_light < ubo.n_objects; i_light = i_light+1u) {
            let l_intersection = intersect(ray,i_light,true);
            if (l_intersection.id == -1 || l_intersection.closestT >= MAXLEN) {
                continue;
            }
            
            v_light_pdf = v_light_pdf + light_pdf(ray, l_intersection, scattering_cosine);
        }
        
        let pdf = (p_scatter * scattering_pdf) + ((1.0 - p_scatter) * v_light_pdf);

        if (pdf > 0.0) {
            pdf_adj = scattering_pdf / pdf;
        }
        else {
            pdf_adj = scattering_pdf;
        }
    }

    // throughput = throughput * albedo * pdf_adj;
    
    throughput[offset] = throughput[offset] * albedo * pdf_adj;
    // return albedo * pdf_adj;
    return false;
}



@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) local_invocation_id: vec3<u32>,
        @builtin(global_invocation_id) global_invocation_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) 
{
    let offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    let ray = rays[offset];

    if (ray.bounce_idx < 0) {
        return;
    }
    
    var light_sample = true;
    if (ubo.lights_offset == ubo.n_objects) {
        light_sample = false;
    }
    
    init_pcg4d(vec4<u32>(global_invocation_id.x, global_invocation_id.y, ubo.subpixel_idx, u32(ray.bounce_idx)));
    var finished = renderScene(ray,offset,light_sample);

// TODO check if end of bounces too
    if (finished) { // || ubo.ray_bounces == u32(ray.bounce_idx + 1)) {
        var color: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
        if (ubo.subpixel_idx > 0u) {
            color = textureLoad(imageData,vec2<i32>(global_invocation_id.xy));
        }

        let ray_color = throughput[offset];
        let scale = 1.0 / f32(ubo.subpixel_idx + 1u);
        color = mix(color,ray_color,scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
    }
    else if (ubo.ray_bounces == u32(ray.bounce_idx + 1)) {
        let scale = 1.0 / f32(ubo.subpixel_idx + 1u);
        let color = mix(textureLoad(imageData,vec2<i32>(global_invocation_id.xy)),vec4<f32>(0.0),scale);
        textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
    }


}
