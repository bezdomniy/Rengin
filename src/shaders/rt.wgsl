struct NodeBLAS {
    point1_: vec4<f32>;
    point2_: vec4<f32>;
    point3_: vec4<f32>;
    normal1_: vec4<f32>;
    normal2_: vec4<f32>;
    normal3_: vec4<f32>;
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
};

struct NodeTLAS1 {
    first: vec4<f32>;
    second: vec4<f32>;
};

[[block]]
struct TLAS {
    TLAS: [[stride(32)]] array<NodeTLAS1>;
};

struct NodeBLAS1 {
    point1_: vec4<f32>;
    point2_: vec4<f32>;
    point3_: vec4<f32>;
    normal1_: vec4<f32>;
    normal2_: vec4<f32>;
    normal3_: vec4<f32>;
};

[[block]]
struct BLAS {
    BLAS: [[stride(96)]] array<NodeBLAS1>;
};

struct Material1 {
    colour: vec4<f32>;
    ambient: f32;
    diffuse: f32;
    specular: f32;
    shininess: f32;
};

[[block]]
struct ObjectParams {
    inverseTransform: mat4x4<f32>;
    material: Material1;
};

[[group(0), binding(1)]]
var<uniform> ubo: UBO;
[[group(0), binding(2)]]
var<storage, read_write> tlas: TLAS;
[[group(0), binding(3)]]
var<storage, read_write> blas: BLAS;
[[group(0), binding(4)]]
var<storage, read_write> objectParams: ObjectParams;
var<private> gl_GlobalInvocationID1: vec3<u32>;
[[group(0), binding(0)]]
var imageData: texture_storage_2d<rgba8unorm,write>;

fn lightingstructMaterialvf4f1f1f1f11vf4structHitParamsvf4vf4vf4vf4vf4vf41b1_(material: ptr<function, Material>, lightPos: ptr<function, vec4<f32>>, hitParams: ptr<function, HitParams>, shadowed: ptr<function, bool>) -> vec4<f32> {
    var intensity: vec4<f32>;
    var effectiveColour: vec4<f32>;
    var ambient: vec4<f32>;
    var lightv: vec4<f32>;
    var lightDotNormal: f32;
    var diffuse: vec4<f32>;
    var specular: vec4<f32>;
    var reflectv: vec4<f32>;
    var reflectDotEye: f32;
    var factor: f32;

    intensity = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    let _e56: vec4<f32> = intensity;
    let _e58: vec4<f32> = (*material).colour;
    effectiveColour = (_e56 * _e58);
    let _e60: vec4<f32> = effectiveColour;
    let _e62: f32 = (*material).ambient;
    ambient = (_e60 * _e62);
    let _e64: bool = *shadowed;
    if (_e64) {
        let _e65: vec4<f32> = ambient;
        return _e65;
    }
    let _e66: vec4<f32> = *lightPos;
    let _e68: vec4<f32> = (*hitParams).overPoint;
    lightv = normalize((_e66 - _e68));
    let _e71: vec4<f32> = lightv;
    let _e73: vec4<f32> = (*hitParams).normalv;
    lightDotNormal = dot(_e71, _e73);
    let _e75: f32 = lightDotNormal;
    if ((_e75 < 0.0)) {
        diffuse = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        specular = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        let _e77: vec4<f32> = effectiveColour;
        let _e79: f32 = (*material).diffuse;
        let _e81: f32 = lightDotNormal;
        diffuse = ((_e77 * _e79) * _e81);
        let _e83: vec4<f32> = lightv;
        let _e86: vec4<f32> = (*hitParams).normalv;
        reflectv = reflect(-(_e83), _e86);
        let _e88: vec4<f32> = reflectv;
        let _e90: vec4<f32> = (*hitParams).eyev;
        reflectDotEye = dot(_e88, _e90);
        let _e92: f32 = reflectDotEye;
        if ((_e92 <= 0.0)) {
            specular = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            let _e94: f32 = reflectDotEye;
            let _e96: f32 = (*material).shininess;
            factor = pow(_e94, _e96);
            let _e98: vec4<f32> = intensity;
            let _e100: f32 = (*material).specular;
            let _e102: f32 = factor;
            specular = ((_e98 * _e100) * _e102);
        }
    }
    let _e104: vec4<f32> = ambient;
    let _e105: vec4<f32> = diffuse;
    let _e107: vec4<f32> = specular;
    return ((_e104 + _e105) + _e107);
}

fn triangleIntersectvf4vf4structNodeBLASvf4vf4vf4vf4vf4vf41vf2_(rayO: ptr<function, vec4<f32>>, rayD: ptr<function, vec4<f32>>, triangle: ptr<function, NodeBLAS>, uv: ptr<function, vec2<f32>>) -> f32 {
    var e1_: vec3<f32>;
    var e2_: vec3<f32>;
    var dirCrossE2_: vec3<f32>;
    var det: f32;
    var f: f32;
    var p1ToOrigin: vec3<f32>;
    var originCrossE1_: vec3<f32>;
    var phi_258_: bool;
    var phi_288_: bool;

    let _e54: vec4<f32> = (*triangle).point2_;
    let _e56: vec4<f32> = (*triangle).point1_;
    let _e57: vec4<f32> = (_e54 - _e56);
    e1_ = vec3<f32>(_e57.x, _e57.y, _e57.z);
    let _e63: vec4<f32> = (*triangle).point3_;
    let _e65: vec4<f32> = (*triangle).point1_;
    let _e66: vec4<f32> = (_e63 - _e65);
    e2_ = vec3<f32>(_e66.x, _e66.y, _e66.z);
    let _e71: vec4<f32> = *rayD;
    let _e76: vec3<f32> = e2_;
    dirCrossE2_ = cross(vec3<f32>(_e71.x, _e71.y, _e71.z), _e76);
    let _e78: vec3<f32> = e1_;
    let _e79: vec3<f32> = dirCrossE2_;
    det = dot(_e78, _e79);
    let _e81: f32 = det;
    if ((abs(_e81) < 0.00009999999747378752)) {
        return -1.0;
    }
    let _e84: f32 = det;
    f = (1.0 / _e84);
    let _e86: vec4<f32> = *rayO;
    let _e88: vec4<f32> = (*triangle).point1_;
    let _e89: vec4<f32> = (_e86 - _e88);
    p1ToOrigin = vec3<f32>(_e89.x, _e89.y, _e89.z);
    let _e94: f32 = f;
    let _e95: vec3<f32> = p1ToOrigin;
    let _e96: vec3<f32> = dirCrossE2_;
    uv[0u] = (_e94 * dot(_e95, _e96));
    let _e101: f32 = uv[0u];
    let _e102: bool = (_e101 < 0.0);
    phi_258_ = _e102;
    if (!(_e102)) {
        let _e105: f32 = uv[0u];
        phi_258_ = (_e105 > 1.0);
    }
    let _e108: bool = phi_258_;
    if (_e108) {
        return -1.0;
    }
    let _e109: vec3<f32> = p1ToOrigin;
    let _e110: vec3<f32> = e1_;
    originCrossE1_ = cross(_e109, _e110);
    let _e112: f32 = f;
    let _e113: vec4<f32> = *rayD;
    let _e118: vec3<f32> = originCrossE1_;
    uv[1u] = (_e112 * dot(vec3<f32>(_e113.x, _e113.y, _e113.z), _e118));
    let _e123: f32 = uv[1u];
    let _e124: bool = (_e123 < 0.0);
    phi_288_ = _e124;
    if (!(_e124)) {
        let _e127: f32 = uv[0u];
        let _e129: f32 = uv[1u];
        phi_288_ = ((_e127 + _e129) > 1.0);
    }
    let _e133: bool = phi_288_;
    if (_e133) {
        return -1.0;
    }
    let _e134: f32 = f;
    let _e135: vec3<f32> = e2_;
    let _e136: vec3<f32> = originCrossE1_;
    return (_e134 * dot(_e135, _e136));
}

fn intersectAABBvf4vf4structNodeTLASvf4vf41_(rayO1: ptr<function, vec4<f32>>, rayD1: ptr<function, vec4<f32>>, aabb: ptr<function, NodeTLAS>) -> bool {
    var t_min: f32;
    var t_max: f32;
    var a: i32;
    var invD: f32;
    var t0_: f32;
    var t1_: f32;
    var temp: f32;

    t_min = -inf;
    t_max = inf;
    a = 0;
    loop {
        let _e52: i32 = a;
        if ((_e52 < 3)) {
        } else {
            break;
        }
        let _e54: i32 = a;
        let _e56: f32 = rayD1[_e54];
        invD = (1.0 / _e56);
        let _e58: i32 = a;
        let _e61: f32 = (*aabb).first[_e58];
        let _e62: i32 = a;
        let _e64: f32 = rayO1[_e62];
        let _e66: f32 = invD;
        t0_ = ((_e61 - _e64) * _e66);
        let _e68: i32 = a;
        let _e71: f32 = (*aabb).second[_e68];
        let _e72: i32 = a;
        let _e74: f32 = rayO1[_e72];
        let _e76: f32 = invD;
        t1_ = ((_e71 - _e74) * _e76);
        let _e78: f32 = invD;
        if ((_e78 < 0.0)) {
            let _e80: f32 = t0_;
            temp = _e80;
            let _e81: f32 = t1_;
            t0_ = _e81;
            let _e82: f32 = temp;
            t1_ = _e82;
        }
        let _e83: f32 = t0_;
        let _e84: f32 = t_min;
        let _e86: f32 = t0_;
        let _e87: f32 = t_min;
        t_min = select(_e87, _e86, (_e83 > _e84));
        let _e89: f32 = t1_;
        let _e90: f32 = t_max;
        let _e92: f32 = t1_;
        let _e93: f32 = t_max;
        t_max = select(_e93, _e92, (_e89 < _e90));
        let _e95: f32 = t_max;
        let _e96: f32 = t_min;
        if ((_e95 <= _e96)) {
            return false;
        }
        continuing {
            let _e98: i32 = a;
            a = (_e98 + 1);
        }
    }
    return true;
}

fn pop_stackstructNodei1i1125i1_(stack: ptr<function, array<Node,25u>>, top: ptr<function, i32>) -> Node {
    var ret: Node;

    let _e45: i32 = *top;
    let _e47: Node = stack[_e45];
    ret = _e47;
    let _e48: i32 = *top;
    top = (_e48 - 1);
    let _e50: Node = ret;
    return _e50;
}

fn push_stackstructNodei1i11structNodei1i1125i1_(node: ptr<function, Node>, stack1: ptr<function, array<Node,25u>>, top1: ptr<function, i32>) {
    let _e45: i32 = *top1;
    top1 = (_e45 + 1);
    let _e47: i32 = *top1;
    let _e48: Node = *node;
    stack1[_e47] = _e48;
    return;
}

fn intersectTLASvf4vf4vf2f1i1_(rayO2: ptr<function, vec4<f32>>, rayD2: ptr<function, vec4<f32>>, uv1: ptr<function, vec2<f32>>, resT: ptr<function, f32>, id: ptr<function, i32>) {
    var topStack: i32;
    var nextNode: Node;
    var stack2: array<Node,25u>;
    var param: Node;
    var param1: array<Node,25u>;
    var param2: i32;
    var param3: array<Node,25u>;
    var param4: i32;
    var firstChildIdx: i32;
    var param5: vec4<f32>;
    var param6: vec4<f32>;
    var param7: NodeTLAS;
    var newNode: Node;
    var param8: Node;
    var param9: array<Node,25u>;
    var param10: i32;
    var param11: vec4<f32>;
    var param12: vec4<f32>;
    var param13: NodeTLAS;
    var newNode1: Node;
    var param14: Node;
    var param15: array<Node,25u>;
    var param16: i32;
    var t1_1: f32;
    var t2_: f32;
    var primIdx: i32;
    var param17: vec4<f32>;
    var param18: vec4<f32>;
    var param19: NodeBLAS;
    var param20: vec2<f32>;
    var param21: vec4<f32>;
    var param22: vec4<f32>;
    var param23: NodeBLAS;
    var param24: vec2<f32>;
    var phi_552_: bool;
    var phi_593_: bool;

    topStack = -1;
    nextNode = Node(0, 0);
    let _e81: Node = nextNode;
    param = _e81;
    let _e82: array<Node,25u> = stack2;
    param1 = _e82;
    let _e83: i32 = topStack;
    param2 = _e83;
    push_stackstructNodei1i11structNodei1i1125i1_(param, param1, param2);
    let _e84: array<Node,25u> = param1;
    stack2 = _e84;
    let _e85: i32 = param2;
    topStack = _e85;
    loop {
        let _e86: i32 = topStack;
        if ((_e86 > -1)) {
        } else {
            break;
        }
        let _e88: array<Node,25u> = stack2;
        param3 = _e88;
        let _e89: i32 = topStack;
        param4 = _e89;
        let _e90: Node = pop_stackstructNodei1i1125i1_(param3, param4);
        let _e91: array<Node,25u> = param3;
        stack2 = _e91;
        let _e92: i32 = param4;
        topStack = _e92;
        nextNode = _e90;
        let _e94: i32 = nextNode.level;
        let _e101: i32 = nextNode.branch;
        firstChildIdx = ((i32(pow(2.0, f32((_e94 + 1)))) - 1) + (_e101 * 2));
        let _e104: i32 = firstChildIdx;
        if ((_e104 < i32(arrayLength(&tlas.TLAS)))) {
            let _e109: i32 = firstChildIdx;
            let _e110: vec4<f32> = *rayO2;
            param5 = _e110;
            let _e111: vec4<f32> = *rayD2;
            param6 = _e111;
            let _e114: NodeTLAS1 = tlas.TLAS[_e109];
            param7.first = _e114.first;
            param7.second = _e114.second;
            let _e119: bool = intersectAABBvf4vf4structNodeTLASvf4vf41_(param5, param6, param7);
            if (_e119) {
                let _e121: i32 = nextNode.level;
                let _e124: i32 = nextNode.branch;
                newNode = Node((_e121 + 1), (_e124 * 2));
                let _e127: Node = newNode;
                param8 = _e127;
                let _e128: array<Node,25u> = stack2;
                param9 = _e128;
                let _e129: i32 = topStack;
                param10 = _e129;
                push_stackstructNodei1i11structNodei1i1125i1_(param8, param9, param10);
                let _e130: array<Node,25u> = param9;
                stack2 = _e130;
                let _e131: i32 = param10;
                topStack = _e131;
            }
            let _e132: i32 = firstChildIdx;
            let _e134: vec4<f32> = *rayO2;
            param11 = _e134;
            let _e135: vec4<f32> = *rayD2;
            param12 = _e135;
            let _e138: NodeTLAS1 = tlas.TLAS[(_e132 + 1)];
            param13.first = _e138.first;
            param13.second = _e138.second;
            let _e143: bool = intersectAABBvf4vf4structNodeTLASvf4vf41_(param11, param12, param13);
            if (_e143) {
                let _e145: i32 = nextNode.level;
                let _e148: i32 = nextNode.branch;
                newNode1 = Node((_e145 + 1), ((_e148 * 2) + 1));
                let _e152: Node = newNode1;
                param14 = _e152;
                let _e153: array<Node,25u> = stack2;
                param15 = _e153;
                let _e154: i32 = topStack;
                param16 = _e154;
                push_stackstructNodei1i11structNodei1i1125i1_(param14, param15, param16);
                let _e155: array<Node,25u> = param15;
                stack2 = _e155;
                let _e156: i32 = param16;
                topStack = _e156;
            }
        } else {
            t1_1 = -1.0;
            t2_ = -1.0;
            let _e158: i32 = nextNode.branch;
            primIdx = (_e158 * 2);
            let _e160: i32 = primIdx;
            let _e161: vec4<f32> = *rayO2;
            param17 = _e161;
            let _e162: vec4<f32> = *rayD2;
            param18 = _e162;
            let _e165: NodeBLAS1 = blas.BLAS[_e160];
            param19.point1_ = _e165.point1_;
            param19.point2_ = _e165.point2_;
            param19.point3_ = _e165.point3_;
            param19.normal1_ = _e165.normal1_;
            param19.normal2_ = _e165.normal2_;
            param19.normal3_ = _e165.normal3_;
            let _e178: f32 = triangleIntersectvf4vf4structNodeBLASvf4vf4vf4vf4vf4vf41vf2_(param17, param18, param19, param20);
            let _e179: vec2<f32> = param20;
            uv1 = _e179;
            t1_1 = _e178;
            let _e180: i32 = primIdx;
            let _e185: bool = ((_e180 + 1) < i32(arrayLength(&blas.BLAS)));
            phi_552_ = _e185;
            if (_e185) {
                let _e186: i32 = primIdx;
                let _e192: f32 = blas.BLAS[(_e186 + 1)].point1_[3u];
                phi_552_ = (_e192 > 0.0);
            }
            let _e195: bool = phi_552_;
            if (_e195) {
                let _e196: i32 = primIdx;
                let _e198: vec4<f32> = *rayO2;
                param21 = _e198;
                let _e199: vec4<f32> = *rayD2;
                param22 = _e199;
                let _e202: NodeBLAS1 = blas.BLAS[(_e196 + 1)];
                param23.point1_ = _e202.point1_;
                param23.point2_ = _e202.point2_;
                param23.point3_ = _e202.point3_;
                param23.normal1_ = _e202.normal1_;
                param23.normal2_ = _e202.normal2_;
                param23.normal3_ = _e202.normal3_;
                let _e215: f32 = triangleIntersectvf4vf4structNodeBLASvf4vf4vf4vf4vf4vf41vf2_(param21, param22, param23, param24);
                let _e216: vec2<f32> = param24;
                uv1 = _e216;
                t2_ = _e215;
            }
            let _e217: f32 = t1_1;
            let _e219: f32 = t1_1;
            let _e220: f32 = *resT;
            let _e222: bool = ((_e217 > 0.00009999999747378752) && (_e219 < _e220));
            phi_593_ = _e222;
            if (_e222) {
                let _e223: f32 = t2_;
                let _e225: f32 = t1_1;
                let _e226: f32 = t2_;
                phi_593_ = ((_e223 < 0.0) || (_e225 < _e226));
            }
            let _e230: bool = phi_593_;
            if (_e230) {
                let _e231: i32 = primIdx;
                id = -((_e231 + 2));
                let _e234: f32 = t1_1;
                resT = _e234;
            } else {
                let _e235: f32 = t2_;
                let _e237: f32 = t2_;
                let _e238: f32 = *resT;
                if (((_e235 > 0.00009999999747378752) && (_e237 < _e238))) {
                    let _e241: i32 = primIdx;
                    id = -((_e241 + 3));
                    let _e244: f32 = t2_;
                    resT = _e244;
                }
            }
        }
    }
    return;
}

fn transformRaymf44vf4vf4vf4vf4_(m: ptr<function, mat4x4<f32>>, rayO3: ptr<function, vec4<f32>>, rayD3: ptr<function, vec4<f32>>, nRayO: ptr<function, vec4<f32>>, nRayD: ptr<function, vec4<f32>>) {
    let _e47: mat4x4<f32> = *m;
    let _e48: vec4<f32> = *rayO3;
    nRayO = (_e47 * _e48);
    let _e50: mat4x4<f32> = *m;
    let _e51: vec4<f32> = *rayD3;
    nRayD = (_e50 * _e51);
    return;
}

fn intersectvf4vf4f1vf2_(rayO4: ptr<function, vec4<f32>>, rayD4: ptr<function, vec4<f32>>, resT1: ptr<function, f32>, uv2: ptr<function, vec2<f32>>) -> i32 {
    var id1: i32;
    var t: f32;
    var nRayO1: vec4<f32>;
    var nRayD1: vec4<f32>;
    var param25: mat4x4<f32>;
    var param26: vec4<f32>;
    var param27: vec4<f32>;
    var param28: vec4<f32>;
    var param29: vec4<f32>;
    var param30: vec4<f32>;
    var param31: vec4<f32>;
    var param32: vec2<f32>;
    var param33: f32;
    var param34: i32;

    id1 = -1;
    t = -1.0;
    let _e61: mat4x4<f32> = objectParams.inverseTransform;
    param25 = _e61;
    let _e62: vec4<f32> = *rayO4;
    param26 = _e62;
    let _e63: vec4<f32> = *rayD4;
    param27 = _e63;
    transformRaymf44vf4vf4vf4vf4_(param25, param26, param27, param28, param29);
    let _e64: vec4<f32> = param28;
    nRayO1 = _e64;
    let _e65: vec4<f32> = param29;
    nRayD1 = _e65;
    let _e66: vec4<f32> = nRayO1;
    param30 = _e66;
    let _e67: vec4<f32> = nRayD1;
    param31 = _e67;
    let _e68: f32 = *resT1;
    param33 = _e68;
    let _e69: i32 = id1;
    param34 = _e69;
    intersectTLASvf4vf4vf2f1i1_(param30, param31, param32, param33, param34);
    let _e70: vec2<f32> = param32;
    uv2 = _e70;
    let _e71: f32 = param33;
    resT1 = _e71;
    let _e72: i32 = param34;
    id1 = _e72;
    let _e73: i32 = id1;
    return _e73;
}

fn isShadowedvf4vf4_(point: ptr<function, vec4<f32>>, lightPos1: ptr<function, vec4<f32>>) -> bool {
    var v: vec4<f32>;
    var distance: f32;
    var direction: vec4<f32>;
    var t1: f32;
    var uv3: vec2<f32>;
    var param35: vec4<f32>;
    var param36: vec4<f32>;
    var param37: f32;
    var param38: vec2<f32>;

    let _e53: vec4<f32> = *lightPos1;
    let _e54: vec4<f32> = *point;
    v = (_e53 - _e54);
    let _e56: vec4<f32> = v;
    distance = length(_e56);
    let _e58: vec4<f32> = v;
    direction = normalize(_e58);
    t1 = 10000.0;
    let _e60: vec4<f32> = *point;
    param35 = _e60;
    let _e61: vec4<f32> = direction;
    param36 = _e61;
    let _e62: f32 = t1;
    param37 = _e62;
    let _e63: i32 = intersectvf4vf4f1vf2_(param35, param36, param37, param38);
    let _e64: f32 = param37;
    t1 = _e64;
    let _e65: vec2<f32> = param38;
    uv3 = _e65;
    let _e66: f32 = t1;
    let _e68: f32 = t1;
    let _e69: f32 = distance;
    if (((_e66 > 0.00009999999747378752) && (_e68 < _e69))) {
        return true;
    }
    return false;
}

fn normalToWorldvf4mf44_(normal: ptr<function, vec4<f32>>, inverseTransform: ptr<function, mat4x4<f32>>) -> vec4<f32> {
    var ret1: vec4<f32>;

    let _e45: mat4x4<f32> = *inverseTransform;
    let _e47: vec4<f32> = *normal;
    ret1 = (transpose(_e45) * _e47);
    ret1[3u] = 0.0;
    let _e50: vec4<f32> = ret1;
    ret1 = normalize(_e50);
    let _e52: vec4<f32> = ret1;
    return _e52;
}

fn normalAtvf4mf44i1vf4vf4vf4vf2_(point1: ptr<function, vec4<f32>>, inverseTransform1: ptr<function, mat4x4<f32>>, typeEnum: ptr<function, i32>, n1_: ptr<function, vec4<f32>>, n2_: ptr<function, vec4<f32>>, n3_: ptr<function, vec4<f32>>, uv4: ptr<function, vec2<f32>>) -> vec4<f32> {
    var n: vec4<f32>;
    var objectPoint: vec4<f32>;
    var param39: vec4<f32>;
    var param40: mat4x4<f32>;

    n = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let _e53: mat4x4<f32> = *inverseTransform1;
    let _e54: vec4<f32> = *point1;
    objectPoint = (_e53 * _e54);
    let _e56: i32 = *typeEnum;
    if ((_e56 == 0)) {
        let _e58: vec4<f32> = objectPoint;
        n = (_e58 - vec4<f32>(0.0, 0.0, 0.0, 1.0));
    } else {
        let _e60: i32 = *typeEnum;
        if ((_e60 == 1)) {
            n = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        } else {
            let _e62: i32 = *typeEnum;
            if ((_e62 == 2)) {
                let _e64: vec4<f32> = *n2_;
                let _e66: f32 = uv4[0u];
                let _e68: vec4<f32> = *n3_;
                let _e70: f32 = uv4[1u];
                let _e73: vec4<f32> = *n1_;
                let _e75: f32 = uv4[0u];
                let _e78: f32 = uv4[1u];
                n = (((_e64 * _e66) + (_e68 * _e70)) + (_e73 * ((1.0 - _e75) - _e78)));
                n[3u] = 0.0;
            }
        }
    }
    let _e83: vec4<f32> = n;
    param39 = _e83;
    let _e84: mat4x4<f32> = *inverseTransform1;
    param40 = _e84;
    let _e85: vec4<f32> = normalToWorldvf4mf44_(param39, param40);
    return _e85;
}

fn getHitParamsvf4vf4f1mf44i1vf4vf4vf4vf2_(rayO5: ptr<function, vec4<f32>>, rayD5: ptr<function, vec4<f32>>, t2: ptr<function, f32>, inverseTransform2: ptr<function, mat4x4<f32>>, typeEnum1: ptr<function, i32>, n1_1: ptr<function, vec4<f32>>, n2_1: ptr<function, vec4<f32>>, n3_1: ptr<function, vec4<f32>>, uv5: ptr<function, vec2<f32>>) -> HitParams {
    var hitParams1: HitParams;
    var param41: vec4<f32>;
    var param42: mat4x4<f32>;
    var param43: i32;
    var param44: vec4<f32>;
    var param45: vec4<f32>;
    var param46: vec4<f32>;
    var param47: vec2<f32>;

    let _e59: vec4<f32> = *rayO5;
    let _e60: vec4<f32> = *rayD5;
    let _e62: f32 = *t2;
    hitParams1.point = (_e59 + (normalize(_e60) * _e62));
    let _e67: vec4<f32> = hitParams1.point;
    param41 = _e67;
    let _e68: mat4x4<f32> = *inverseTransform2;
    param42 = _e68;
    let _e69: i32 = *typeEnum1;
    param43 = _e69;
    let _e70: vec4<f32> = *n1_1;
    param44 = _e70;
    let _e71: vec4<f32> = *n2_1;
    param45 = _e71;
    let _e72: vec4<f32> = *n3_1;
    param46 = _e72;
    let _e73: vec2<f32> = *uv5;
    param47 = _e73;
    let _e74: vec4<f32> = normalAtvf4mf44i1vf4vf4vf4vf2_(param41, param42, param43, param44, param45, param46, param47);
    hitParams1.normalv = _e74;
    let _e76: vec4<f32> = *rayD5;
    hitParams1.eyev = -(_e76);
    let _e80: vec4<f32> = hitParams1.normalv;
    let _e82: vec4<f32> = hitParams1.eyev;
    if ((dot(_e80, _e82) < 0.0)) {
        let _e86: vec4<f32> = hitParams1.normalv;
        hitParams1.normalv = -(_e86);
    }
    let _e89: vec4<f32> = *rayD5;
    let _e91: vec4<f32> = hitParams1.normalv;
    hitParams1.reflectv = reflect(_e89, _e91);
    let _e95: vec4<f32> = hitParams1.point;
    let _e97: vec4<f32> = hitParams1.normalv;
    hitParams1.overPoint = (_e95 + (_e97 * 0.00009999999747378752));
    let _e102: vec4<f32> = hitParams1.point;
    let _e104: vec4<f32> = hitParams1.normalv;
    hitParams1.underPoint = (_e102 - (_e104 * 0.00009999999747378752));
    let _e108: HitParams = hitParams1;
    return _e108;
}

fn renderScenevf4vf4_(rayO6: ptr<function, vec4<f32>>, rayD6: ptr<function, vec4<f32>>) -> vec4<f32> {
    var color: vec4<f32>;
    var t3: f32;
    var objectID: i32;
    var uv6: vec2<f32>;
    var param48: vec4<f32>;
    var param49: vec4<f32>;
    var param50: f32;
    var param51: vec2<f32>;
    var hitParams2: HitParams;
    var param52: vec4<f32>;
    var param53: vec4<f32>;
    var param54: f32;
    var param55: mat4x4<f32>;
    var param56: i32;
    var param57: vec4<f32>;
    var param58: vec4<f32>;
    var param59: vec4<f32>;
    var param60: vec2<f32>;
    var shadowed1: bool;
    var param61: vec4<f32>;
    var param62: vec4<f32>;
    var param63: Material;
    var param64: vec4<f32>;
    var param65: HitParams;
    var param66: bool;

    color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    t3 = 10000.0;
    let _e69: vec4<f32> = *rayO6;
    param48 = _e69;
    let _e70: vec4<f32> = *rayD6;
    param49 = _e70;
    let _e71: f32 = t3;
    param50 = _e71;
    let _e72: i32 = intersectvf4vf4f1vf2_(param48, param49, param50, param51);
    let _e73: f32 = param50;
    t3 = _e73;
    let _e74: vec2<f32> = param51;
    uv6 = _e74;
    objectID = _e72;
    let _e75: f32 = t3;
    let _e77: i32 = objectID;
    if (((_e75 >= 10000.0) || (_e77 == -1))) {
        let _e80: vec4<f32> = color;
        return _e80;
    }
    let _e81: i32 = objectID;
    if ((_e81 >= 0)) {
        color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    } else {
        let _e83: i32 = objectID;
        let _e86: i32 = objectID;
        let _e89: i32 = objectID;
        let _e92: vec4<f32> = *rayO6;
        param52 = _e92;
        let _e93: vec4<f32> = *rayD6;
        param53 = _e93;
        let _e94: f32 = t3;
        param54 = _e94;
        let _e96: mat4x4<f32> = objectParams.inverseTransform;
        param55 = _e96;
        param56 = 2;
        let _e100: vec4<f32> = blas.BLAS[-((_e83 + 2))].normal1_;
        param57 = _e100;
        let _e104: vec4<f32> = blas.BLAS[-((_e86 + 2))].normal2_;
        param58 = _e104;
        let _e108: vec4<f32> = blas.BLAS[-((_e89 + 2))].normal3_;
        param59 = _e108;
        let _e109: vec2<f32> = uv6;
        param60 = _e109;
        let _e110: HitParams = getHitParamsvf4vf4f1mf44i1vf4vf4vf4vf2_(param52, param53, param54, param55, param56, param57, param58, param59, param60);
        hitParams2 = _e110;
        let _e112: vec4<f32> = hitParams2.overPoint;
        param61 = _e112;
        let _e114: vec4<f32> = ubo.lightPos;
        param62 = _e114;
        let _e115: bool = isShadowedvf4vf4_(param61, param62);
        shadowed1 = _e115;
        let _e117: Material1 = objectParams.material;
        param63.colour = _e117.colour;
        param63.ambient = _e117.ambient;
        param63.diffuse = _e117.diffuse;
        param63.specular = _e117.specular;
        param63.shininess = _e117.shininess;
        let _e129: vec4<f32> = ubo.lightPos;
        param64 = _e129;
        let _e130: HitParams = hitParams2;
        param65 = _e130;
        let _e131: bool = shadowed1;
        param66 = _e131;
        let _e132: vec4<f32> = lightingstructMaterialvf4f1f1f1f11vf4structHitParamsvf4vf4vf4vf4vf4vf41b1_(param63, param64, param65, param66);
        color = _e132;
        color[3u] = 1.0;
    }
    let _e134: vec4<f32> = color;
    return _e134;
}

fn rayForPixelvf2vf4vf4_(p: ptr<function, vec2<f32>>, rayO7: ptr<function, vec4<f32>>, rayD7: ptr<function, vec4<f32>>) {
    var xOffset: f32;
    var yOffset: f32;
    var worldX: f32;
    var worldY: f32;
    var pixel: vec4<f32>;

    let _e51: f32 = p[0u];
    let _e55: f32 = ubo.camera.pixelSize;
    xOffset = ((_e51 + 0.5) * _e55);
    let _e58: f32 = p[1u];
    let _e62: f32 = ubo.camera.pixelSize;
    yOffset = ((_e58 + 0.5) * _e62);
    let _e66: f32 = ubo.camera.halfWidth;
    let _e67: f32 = xOffset;
    worldX = (_e66 - _e67);
    let _e71: f32 = ubo.camera.halfHeight;
    let _e72: f32 = yOffset;
    worldY = (_e71 - _e72);
    let _e76: mat4x4<f32> = ubo.camera.inverseTransform;
    let _e77: f32 = worldX;
    let _e78: f32 = worldY;
    pixel = (_e76 * vec4<f32>(_e77, _e78, -1.0, 1.0));
    let _e83: mat4x4<f32> = ubo.camera.inverseTransform;
    rayO7 = (_e83 * vec4<f32>(0.0, 0.0, 0.0, 1.0));
    let _e85: vec4<f32> = pixel;
    let _e86: vec4<f32> = *rayO7;
    rayD7 = normalize((_e85 - _e86));
    return;
}

fn main1() {
    var rayO8: vec4<f32>;
    var rayD8: vec4<f32>;
    var param67: vec2<f32>;
    var param68: vec4<f32>;
    var param69: vec4<f32>;
    var color1: vec4<f32>;
    var param70: vec4<f32>;
    var param71: vec4<f32>;
    var phi_994_: bool;

    let _e51: u32 = gl_GlobalInvocationID1[0u];
    let _e54: i32 = ubo.camera.width;
    let _e56: bool = (_e51 >= u32(_e54));
    phi_994_ = _e56;
    if (!(_e56)) {
        let _e59: u32 = gl_GlobalInvocationID1[1u];
        phi_994_ = (_e59 >= 360u);
    }
    let _e62: bool = phi_994_;
    if (_e62) {
        return;
    }
    let _e63: vec3<u32> = gl_GlobalInvocationID1;
    param67 = vec2<f32>(_e63.xy);
    rayForPixelvf2vf4vf4_(param67, param68, param69);
    let _e66: vec4<f32> = param68;
    rayO8 = _e66;
    let _e67: vec4<f32> = param69;
    rayD8 = _e67;
    let _e68: vec4<f32> = rayO8;
    param70 = _e68;
    let _e69: vec4<f32> = rayD8;
    param71 = _e69;
    let _e70: vec4<f32> = renderScenevf4vf4_(param70, param71);
    color1 = _e70;
    let _e71: vec3<u32> = gl_GlobalInvocationID1;
    let _e74: vec4<f32> = color1;
    textureStore(imageData, vec2<i32>(_e71.xy), _e74);
    return;
}

[[stage(compute), workgroup_size(32, 32, 1)]]
fn main([[builtin(global_invocation_id)]] gl_GlobalInvocationID: vec3<u32>) {
    gl_GlobalInvocationID1 = gl_GlobalInvocationID;
    main1();
}
