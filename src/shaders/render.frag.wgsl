struct UBO {
    lightPos: vec3<f32>,
    is_pathtracer: u32,
    resolution: vec2<u32>,
    _pad2: vec2<u32>,
    n_objects: i32,
    lights_offset: u32,
    subpixel_idx: u32,
    _pad3: u32,
};

struct Ray {
    rayO: vec3<f32>,
    x: i32,
    rayD: vec3<f32>,
    refractive_index: f32,
};

struct Rays {
    Rays: array<Ray>,
};

@group(0) @binding(0)
var imageData: texture_storage_2d<rgba8unorm,read_write>;
@group(0) @binding(1)
var<uniform> ubo: UBO;
@group(0) @binding(2)
var<storage, read_write> rays: Rays;
@group(0) @binding(3)
var<storage, read_write> throughputs: array<vec4<f32>>;

fn float_to_linear_rgb(x: f32) -> f32 {
    if (x > 0.04045) {
        return pow((x + 0.055) / 1.055,2.4);
    }
    return x / 12.92;
}

fn to_linear_rgb(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(float_to_linear_rgb(c.x),float_to_linear_rgb(c.y),float_to_linear_rgb(c.z),1.0);
}

@fragment
fn main(@location(0) inUV: vec2<f32>) -> @location(0) vec4<f32> {
    // // TODO: fix a way to take the scaling into the fragment shader too.
    let xy = vec2<u32>(inUV*vec2<f32>(ubo.resolution));

    var accum: vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);

    let ray = rays.Rays[(xy.y * ubo.resolution.x) + xy.x];

    if (ubo.subpixel_idx > 0u) {
        accum = textureLoad(imageData, xy);
    }

    var ray_color = vec4<f32>(0.0);
    if (ray.x == -1) {
        ray_color = throughputs[(xy.y * ubo.resolution.x) + xy.x];
    }

    // var ray_color = throughputs[(xy.y * ubo.resolution.x) + xy.x];

    let scale = 1.0 / f32(ubo.subpixel_idx + 1u);

    if (ray_color.x != ray_color.x) { ray_color.x = 0.0; }
    if (ray_color.y != ray_color.y) { ray_color.y = 0.0; }
    if (ray_color.z != ray_color.z) { ray_color.z = 0.0; }
    // if (ray_color.w != ray_color.w) { ray_color.w = 1.0; }

    accum = mix(accum,ray_color,scale);
    // accum = ray_color;
    
    textureStore(imageData, xy, accum);

    if (ubo.is_pathtracer == 1u) {
        accum = sqrt(accum);
    }

    accum = clamp(accum,vec4<f32>(0.0),vec4<f32>(0.999));
    return to_linear_rgb(accum);
}