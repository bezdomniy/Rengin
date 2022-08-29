struct UBO {
    inverse_camera_transform: mat4x4<f32>,
    half_width_height: vec2<f32>,
    pixel_size: f32,
    sqrt_rays_per_pixel: u32,
    resolution: vec2<u32>,
    n_objects: u32,
    lights_offset: u32,
    specular_offset: u32,
    subpixel_idx: u32,
    ray_bounces: u32,
    is_pathtracer: u32,
};


struct Ray {
    rayO: vec3<f32>,
    refractive_index: f32,
    rayD: vec3<f32>,
    bounce_idx: i32,
    throughput: vec4<f32>,
    radiance: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> ubo: UBO;
@group(0) @binding(1)
var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) 
{
    let sub_pixel = vec2<u32>(
        ubo.subpixel_idx / ubo.sqrt_rays_per_pixel,
        ubo.subpixel_idx % ubo.sqrt_rays_per_pixel,
    );

    let half_sub_pixel_size = 1f / f32(ubo.sqrt_rays_per_pixel) / 2f;
    let sub_pixel_offset = half_sub_pixel_size * vec2<f32>(sub_pixel);

    let offset = (vec2<f32>(global_invocation_id.xy) + sub_pixel_offset) * ubo.pixel_size;
    let world = ubo.half_width_height - offset;
    let pixel = ubo.inverse_camera_transform * vec4<f32>(world.x, world.y, -1.0, 1.0);
    let origin = (ubo.inverse_camera_transform * vec4<f32>(0f,0f,0f,1f));
    let direction = normalize((pixel - origin)).xyz;

    let rays_offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    rays[rays_offset] = Ray(origin.xyz, 1f, direction, 0, vec4<f32>(1f), vec4<f32>(0f));
}