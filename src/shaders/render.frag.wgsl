#import rengin::ubo::UBO

@group(0) @binding(0)
var<storage, read> imageData: array<vec4<f32>>;
@group(0) @binding(1)
var<uniform> ubo: UBO;

// Reinhard tone mapping: maps [0, ∞) → [0, 1) preserving relative brightness
fn reinhard(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3<f32>(1.0) + c);
}

@fragment
fn main(@location(0) inUV: vec2<f32>) -> @location(0) vec4<f32> {
    let xy = vec2<i32>(inUV*vec2<f32>(ubo.resolution));
    let idx = u32(xy.y) * ubo.resolution.x + u32(xy.x);

    let hdr = max(imageData[idx].xyz, vec3<f32>(0.0));
    let ldr = reinhard(hdr);
    return vec4<f32>(ldr, 1.0);
}
