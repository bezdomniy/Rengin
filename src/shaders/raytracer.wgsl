struct NodeBLAS {
    point1: vec4<f32>;
    point2: vec4<f32>;
    point3: vec4<f32>;
    normal1: vec4<f32>;
    normal2: vec4<f32>;
    normal3: vec4<f32>;
};

struct NodeTLAS {
    first: vec4<f32>;
    second: vec4<f32>;
};

struct Node {
    level: i32;
    branch: i32;
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
    len_tlas: i32;
    len_blas: i32;
    _padding: array<u32,2>;
};

[[block]]
struct TLAS {
    TLAS: [[stride(32)]] array<NodeTLAS>;
};

[[block]]
struct BLAS {
    BLAS: [[stride(96)]] array<NodeBLAS>;
};

[[block]]
struct ObjectParams {
    inverseTransform: mat4x4<f32>;
    material: Material;
};

struct Ray {
    rayO: vec4<f32>;
    rayD: vec4<f32>;
};

struct Intersection {
    uv: vec2<f32>;
    id: i32;
    closestT: f32;
};

[[group(0), binding(0)]]
var imageData: texture_storage_2d<rgba8unorm,write>;
[[group(0), binding(1)]]
var<uniform> ubo: UBO;
[[group(0), binding(2)]]
var<storage, read> tlas: TLAS;
[[group(0), binding(3)]]
var<storage, read> blas: BLAS;
[[group(0), binding(4)]]
var<storage, read> objectParams: ObjectParams;

let EPSILON:f32 = 0.0001;
let MAXLEN: f32 = 10000.0;

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
    let INFINITY: f32 = 1.0 / 0.0;

    var t_min: f32 = -INFINITY;
    var t_max: f32 = INFINITY;
    var temp: f32;
    var invD: f32;
    var t0: f32;
    var t1: f32;

    for (var a: i32 = 0; a < 3; a = a+1)
    {
        invD = 1.0 / ray.rayD[a];
        t0 = (tlas.TLAS[aabbIdx].first[a] - ray.rayO[a]) * invD;
        t1 = (tlas.TLAS[aabbIdx].second[a] - ray.rayO[a]) * invD;
        if (invD < 0.0) {
            temp = t0;
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

fn triangleIntersect(ray: Ray, triangleIdx: i32, inIntersection: Intersection) -> Intersection {
    var uv: vec2<f32> = vec2<f32>(0.0);
    let e1: vec3<f32> = (blas.BLAS[triangleIdx].point2 - blas.BLAS[triangleIdx].point1).xyz;
    let e2: vec3<f32> = (blas.BLAS[triangleIdx].point3 - blas.BLAS[triangleIdx].point1).xyz;

    let dirCrossE2: vec3<f32> = cross(ray.rayD.xyz, e2);
    let det: f32 = dot(e1, dirCrossE2);
    if (abs(det) < EPSILON) {
        return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    }

    let f: f32 = 1.0 / det;
    let p1ToOrigin: vec3<f32> = (ray.rayO - blas.BLAS[triangleIdx].point1).xyz;
    uv.x = f * dot(p1ToOrigin, dirCrossE2);
    if (uv.x < 0.0 || uv.x > 1.0) {
      return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    }

    let originCrossE1: vec3<f32>  = cross(p1ToOrigin, e1);
    uv.y = f * dot(ray.rayD.xyz, originCrossE1);
    // uv.y =1.0;
    if (uv.y < 0.0 || (uv.x + uv.y) > 1.0) {
        return Intersection(inIntersection.uv,inIntersection.id,-1.0);
    }
    return Intersection(uv,inIntersection.id,f * dot(e2, originCrossE1));
}

struct Node {
  level: i32;
  branch: i32;
};
let MAX_STACK_SIZE:i32 = 25;
var topStack: i32 = -1;
var<private> stack: [[stride(8)]] array<Node,MAX_STACK_SIZE>;


fn push_stack(node: Node) {
  topStack = topStack + 1;
  stack[topStack] = node;
}

fn pop_stack() -> Node {
  let ret: Node = stack[topStack];
  topStack = topStack - 1;
  return ret;
}

fn intersectTLAS(ray: Ray) -> Intersection {
    // int topPrimivitiveIndices = 0;
    var nextNode: Node = Node(0, 0);
    var firstChildIdx: i32 = -1;
    var t1: Intersection = Intersection(vec2<f32>(0.0), -1, -1.0);
    var t2: Intersection = Intersection(vec2<f32>(0.0), -1, -1.0);
    var primIdx: i32 = -1;

    var ret: Intersection = Intersection(vec2<f32>(0.0), -1, MAXLEN);
    push_stack(nextNode);

    loop  
    {
        if (topStack <= -1) {break};

        nextNode = pop_stack();
        // nodeIdx = int(pow(2, nextNode.level))  - 1 + nextNode.branch;

        firstChildIdx = i32(pow(2.0, f32(nextNode.level) + 1.0)) - 1 + (nextNode.branch * 2);

        // var test: u32 = ubo.len_tlas;

        if (firstChildIdx < ubo.len_tlas) {
            if (intersectAABB(ray, firstChildIdx)) {
                push_stack(Node(nextNode.level + 1, nextNode.branch * 2));
            }

            if (intersectAABB(ray, firstChildIdx + 1)) {
                push_stack(Node(nextNode.level + 1, (nextNode.branch * 2) + 1));
            }
        }
        else {
            // primitiveIndices[topPrimivitiveIndices] = nextNode.branch * 2;
            // topPrimivitiveIndices += 1;
            t1 = Intersection(ret.uv, ret.id, -1.0);
            t2 = Intersection(ret.uv, ret.id, -1.0);
            primIdx = nextNode.branch * 2;

            t1 = triangleIntersect(ray, primIdx, ret);

            if (primIdx + 1 < ubo.len_blas && blas.BLAS[primIdx + 1].point1.w > 0.0) {
                t2 = triangleIntersect(ray, primIdx + 1, ret);
            }

            if ((t1.closestT > EPSILON) && (t1.closestT < ret.closestT) && (t2.closestT < 0.0 || (t1.closestT < t2.closestT))) {
                ret.uv = t1.uv;
                ret.id = -(primIdx + 2);
                ret.closestT = t1.closestT;
            }
            elseif ((t2.closestT > EPSILON) && (t2.closestT < ret.closestT))
            {
                ret.uv = t2.uv;
                ret.id = -(primIdx + 3);
                ret.closestT = t2.closestT;
            }
        }
    }
    return ret;
}

fn intersect(ray: Ray) -> Intersection {
    // TODO: this will need the id of the object as input in future when we are rendering more than one model
    let nRay: Ray = Ray(objectParams.inverseTransform * ray.rayO, objectParams.inverseTransform * ray.rayD);
    return intersectTLAS(nRay);
}

fn normalToWorld(normal: vec4<f32>) -> vec4<f32>
{
    var ret: vec4<f32> = transpose(objectParams.inverseTransform) * normal;
    ret.w = 0.0;
    ret = normalize(ret);

    return ret;
}

fn normalAt(point: vec4<f32>, intersection: Intersection, typeEnum: i32) -> vec4<f32> {
    var n: vec4<f32> = vec4<f32>(0.0);
    let objectPoint: vec4<f32> = objectParams.inverseTransform * point; // World to object

    if (typeEnum == 0) {
        n = objectPoint - vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    elseif (typeEnum == 1) {
        n = vec4<f32>(0.0,1.0,0.0,0.0);
    }
    elseif (typeEnum == 2) {
        let shape: NodeBLAS = blas.BLAS[-(intersection.id+2)];
        n = shape.normal2 * intersection.uv.x + shape.normal3 * intersection.uv.y + shape.normal1 * (1.0 - intersection.uv.x - intersection.uv.y);
        n.w = 0.0;
    }
    return normalToWorld(n);
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

fn isShadowed(point: vec4<f32>, lightPos: vec4<f32>) -> bool
{
  let v: vec4<f32> = lightPos - point;
  let distance: f32 = length(v);
  let direction: vec4<f32> = normalize(v);

  let intersection: Intersection = intersect(Ray(point,direction));

  if (intersection.closestT > EPSILON && intersection.closestT < distance)
  {
    return true;
  }

  return false;
}

fn lighting(material: Material, lightPos: vec4<f32>, hitParams: HitParams, shadowed: bool) -> vec4<f32>
{
  // return material.colour;
  var diffuse: vec4<f32>;
  var specular: vec4<f32>;

  let intensity: vec4<f32> = vec4<f32>(1.0,1.0,1.0,1.0); // TODO temp placeholder

  let effectiveColour: vec4<f32> = intensity * material.colour; //* light->intensity;

  let ambient: vec4<f32> = effectiveColour * material.ambient;
  // vec4 ambient = vec4(0.3,0.0,0.0,1.0);
  if (shadowed) {
    return ambient;
  }

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

    if (intersection.id >= 0) {
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
        color = vec4<f32>(1.0,0.0,0.0,1.0);
    }

    else {
        let hitParams: HitParams = getHitParams(ray, intersection, 2);

        let shadowed: bool = isShadowed(hitParams.overPoint, ubo.lightPos);
        // bool shadowed = false;
        color = lighting(objectParams.material, ubo.lightPos,
                                hitParams, shadowed);
        color.w = 1.0;
        // color = vec4<f32>(0.0,1.0,0.0,1.0);
    }
    
    return color;
}

[[stage(compute), workgroup_size(32, 32, 1)]]
fn main([[builtin(global_invocation_id)]] global_invocation_id: vec3<u32>) {
    let ray: Ray = rayForPixel(global_invocation_id.xy);
    let color: vec4<f32> = renderScene(ray);

    // let color: vec4<f32> = vec4<f32>(0.0,1.0,0.0,1.0);

    textureStore(imageData, vec2<i32>(global_invocation_id.xy), color);
}