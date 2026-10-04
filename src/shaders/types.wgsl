#define_import_path rengin::types

struct NodeLeaf {
    vertex_a: vec3<f32>,
    pad_a: u32,
    vertex_b: vec3<f32>,
    pad_b: u32,
    vertex_c: vec3<f32>,
    pad_c: u32,
};

struct Normal {
    normal_a: vec4<f32>,
    normal_b: vec4<f32>,
    normal_c: vec4<f32>,
};

struct NodeInner {
    first: vec3<f32>,
    skip_ptr_or_prim_idx: u32,
    second: vec3<f32>,
    prim_end_idx: u32,
};


struct HitParams {
    p: vec3<f32>,
    normalv: vec3<f32>,
    geometric_normalv: vec3<f32>,
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
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
};


struct ObjectParams {
    ObjectParams: array<ObjectParam>,
};

struct Ray {
    rayO: vec3<f32>,
    refractive_index: f32,
    rayD: vec3<f32>,
    pos: u32,
    throughput: vec4<f32>,
};

struct Intersection {
    uv: vec2<f32>,
    id: i32,
    closestT: f32,
    model_id: u32,
};
