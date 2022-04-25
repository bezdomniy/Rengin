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

[[group(0), binding(0)]]
var imageData: texture_storage_2d<rgba8unorm,read_write>;
[[group(0), binding(1)]]
var<uniform> ubo: UBO;

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

    var color = textureLoad(imageData,xy);
    if (ubo.is_pathtracer) {
        color = sqrt(color);
    }
    
    color = clamp(color,vec4<f32>(0.0),vec4<f32>(0.999));
    return to_linear_rgb(color);
}