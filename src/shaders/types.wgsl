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
    throughput: vec4<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    id: i32,
    closestT: f32,
    model_id: u32,
};
