struct UBO {
    _pad1: vec4<u32>;
    width: u32;
    n_objects: i32;
    subpixel_idx: u32;
    bounce_idx: u32;
};

[[group(0), binding(0)]]
var u_Textures: texture_storage_2d<rgba8unorm,read>;
// var u_Textures: texture_2d<f32>;
[[group(0), binding(1)]]
var u_Sampler: sampler;
[[group(0), binding(2)]]
var u_PreviousT: texture_storage_2d<rgba8unorm,read_write>;
[[group(0), binding(3)]]
var<uniform> ubo: UBO;

[[stage(fragment)]]
fn main([[location(0)]] inUV: vec2<f32>) -> [[location(0)]] vec4<f32> {

    let position = vec2<i32>(i32(inUV.x * (800.0 - 1.0)), i32(inUV.y * (600.0 - 1.0)));

    // let ray_color = textureSample(u_Textures, u_Sampler, inUV);
    let ray_color = textureLoad(u_Textures,position);


    if (ubo.subpixel_idx == 0u) {
        textureStore(u_PreviousT, position, ray_color);
        return ray_color;
    }

    let previous_color = textureLoad(u_PreviousT,position); 

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    var color = (previous_color * (1.0 - scale)) + (ray_color * scale);

    color.r = clamp(color.r,0.0,0.999);
    color.g = clamp(color.g,0.0,0.999);
    color.b = clamp(color.b,0.0,0.999);
    color.a = clamp(color.a,0.0,0.999);

    textureStore(u_PreviousT, position, color);

    return color;
}
