@group(0) @binding(0)
var<uniform> ubo: UBO;
@group(0) @binding(1)
var<storage, read_write> rays: array<Ray>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) 
{
    var sub_pixel_offset = vec2<f32>(0f);

    if RANDOM_SUBPIXEL {
        init_pcg3d(vec3<u32>(global_invocation_id.x, global_invocation_id.y, ubo.subpixel_idx));
        sub_pixel_offset = random_in_square();
    }
    else {
        let half_sub_pixel_size = 1f / f32(ubo.sqrt_rays_per_pixel) / 2f;

        let sub_pixel = vec2<u32>(
            ubo.subpixel_idx / ubo.sqrt_rays_per_pixel,
            ubo.subpixel_idx % ubo.sqrt_rays_per_pixel,
        );

        sub_pixel_offset = half_sub_pixel_size * vec2<f32>(sub_pixel);
    }

    let offset = (vec2<f32>(global_invocation_id.xy) + sub_pixel_offset) * ubo.pixel_size;
    let world = ubo.half_width_height - offset;
    let pixel = ubo.inverse_camera_transform * vec4<f32>(world.x, world.y, -1.0, 1.0);
    let origin = (ubo.inverse_camera_transform * vec4<f32>(0f,0f,0f,1f));
    let direction = normalize((pixel - origin)).xyz;

    let rays_offset = (global_invocation_id.y * ubo.resolution.x) + global_invocation_id.x;
    rays[rays_offset] = Ray(origin.xyz, 1f, direction, rays_offset, vec4<f32>(1f));
}