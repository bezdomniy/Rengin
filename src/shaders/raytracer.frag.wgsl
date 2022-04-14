

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
    _pad1: vec3<u32>;
    is_pathtracer: bool;
    resolution: vec2<u32>;
    _pad2: vec2<u32>;
    n_objects: i32;
    subpixel_idx: u32;
    bounce_idx: u32;
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
    // ObjectParams: [[stride(96)]] array<ObjectParam>;
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

[[stage(fragment)]]
fn main([[location(0)]] inUV: vec2<f32>) -> [[location(0)]] vec4<f32> {
    // // TODO: fix a way to take the scaling into the fragment shader too.
    let xy = vec2<i32>(inUV*vec2<f32>(ubo.resolution));

    var colour = textureLoad(imageData,xy);
    if (ubo.is_pathtracer) {
        colour = sqrt(colour);
    }
    
    colour = clamp(colour,vec4<f32>(0.0),vec4<f32>(0.999));
    return to_linear_rgb(colour);
}
