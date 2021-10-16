var<private> outFragColor: vec4<f32>;
[[group(0), binding(0)]]
var u_Textures: texture_2d<f32>;
[[group(0), binding(1)]]
var u_Sampler: sampler;
var<private> inUV1: vec2<f32>;

fn main1() {
    let e9: vec2<f32> = inUV1;
    let e10: vec4<f32> = textureSample(u_Textures, u_Sampler, e9);
    let e11: vec3<f32> = e10.xyz;
    outFragColor = vec4<f32>(e11.x, e11.y, e11.z, 1.0);
    return;
}

[[stage(fragment)]]
fn main([[location(0)]] inUV: vec2<f32>) -> [[location(0)]] vec4<f32> {
    inUV1 = inUV;
    main1();
    let e3: vec4<f32> = outFragColor;
    return e3;
}
