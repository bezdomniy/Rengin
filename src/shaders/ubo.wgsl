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
    bounce_idx: u32,
};
