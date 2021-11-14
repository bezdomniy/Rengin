[[group(0), binding(0)]]
var u_Textures: texture_2d<f32>;
[[group(0), binding(1)]]
var u_Sampler: sampler;

[[stage(fragment)]]
fn main([[location(0)]] inUV: vec2<f32>) -> [[location(0)]] vec4<f32> {
    return textureSample(u_Textures, u_Sampler, inUV);
}
