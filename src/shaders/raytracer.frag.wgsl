struct UBO {
    _pad: vec3<u32>;
    is_pathtracer: bool;
    width: u32;
    n_objects: i32;
    subpixel_idx: u32;
    bounce_idx: u32;
};

[[group(0), binding(0)]]
var u_Textures: texture_2d<f32>;
[[group(0), binding(1)]]
var u_Sampler: sampler;
[[group(0), binding(2)]]
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
    var colour = textureSample(u_Textures, u_Sampler, inUV);
    if (ubo.is_pathtracer) {
        colour = sqrt(colour);
    }
    
    colour = clamp(colour,vec4<f32>(0.0),vec4<f32>(0.999));
    return to_linear_rgb(colour);
}
