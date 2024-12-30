use std::f32::consts::FRAC_PI_2;

use glam::{Mat4, Vec2, Vec3, Vec4};
// use rand::Rng;
use wgpu::SurfaceConfiguration;
use winit::dpi::PhysicalSize;

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[repr(C)]
#[derive(Debug)]
pub struct Ray {
    origin: Vec3,
    refractive_index: f32,
    direction: Vec3,
    pos: u32,
    throughput: Vec4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub colour: Vec4,
    pub emissiveness: Vec4,
    pub reflective: f32,
    pub transparency: f32,
    pub refractive_index: f32,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub texture_index: u32,
}

impl Material {
    #![allow(dead_code)]
    pub fn new(
        colour: Vec4,
        texture_index: Option<u32>,
        emissiveness: Vec4,
        ambient: f32,
        diffuse: f32,
        specular: f32,
        shininess: f32,
        reflective: f32,
        transparency: f32,
        refractive_index: f32,
    ) -> Self {
        let texture_index = match texture_index {
            Some(idx) => idx,
            None => u32::MAX,
        };

        Material {
            colour,
            emissiveness,
            ambient,
            diffuse,
            specular,
            shininess,
            reflective,
            transparency,
            refractive_index,
            texture_index,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            colour: Vec4::new(0.0, 0.0, 0.0, 0.0),
            emissiveness: Vec4::new(0.0, 0.0, 0.0, 0.0),
            reflective: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
            texture_index: u32::MAX,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectParam {
    pub transform: Mat4,
    pub inverse_transform: Mat4,
    pub material: Material,
    pub offset_inner_nodes: u32,
    pub len_inner_nodes: u32,
    pub offset_leaf_nodes: u32,
    pub model_type: u32,
}

impl ObjectParam {
    #![allow(dead_code)]
    pub fn new(
        transform: Mat4,
        material: Material,
        offset_inner_nodes: u32,
        len_inner_nodes: u32,
        offset_leaf_nodes: u32,
        model_type: u32,
    ) -> Self {
        ObjectParam {
            transform,
            inverse_transform: transform.inverse(),
            material,
            offset_inner_nodes,
            len_inner_nodes,
            offset_leaf_nodes,
            model_type,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub centre: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    move_speed: f32,
}

impl Camera {
    pub fn new(p: [f32; 3], c: [f32; 3], u: [f32; 3]) -> Self {
        let position = Vec3::from_array(p);
        let centre = Vec3::from_array(c);
        let up = Vec3::from_array(u);

        let (yaw, pitch) = Camera::get_yaw_pitch(position, centre);

        let forward = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();

        Camera {
            position,
            centre,
            up,
            forward,
            yaw,
            pitch,
            move_speed: 0.2,
        }
    }

    pub fn get_transform(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, -self.up)
    }

    pub fn get_inverse_transform(&self) -> Mat4 {
        self.get_transform().inverse()
    }

    pub fn rotate(&mut self, d_x: f32, d_y: f32) {
        self.yaw += d_x.to_radians() * self.move_speed;
        self.pitch += d_y.to_radians() * self.move_speed;

        if self.pitch < -SAFE_FRAC_PI_2 {
            self.pitch = -SAFE_FRAC_PI_2;
        } else if self.pitch > SAFE_FRAC_PI_2 {
            self.pitch = SAFE_FRAC_PI_2;
        }

        let forward_mag = self.position.distance(self.centre);

        // println!("{} {}\n{} {}", d_x, d_y, self.yaw, self.pitch);

        self.forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();

        self.centre = self.forward * forward_mag;

        // println!("{:?}", self.forward);
    }

    #[allow(dead_code)]
    pub fn orbit_centre(&mut self, d_x: f32, d_y: f32, config: &SurfaceConfiguration) {
        self.yaw += d_x as f32;
        self.pitch += d_y as f32;
        log::debug!("{} {}\n{} {}", d_x, d_y, self.yaw, self.pitch);

        let norm_x = self.yaw / config.width as f32;
        let norm_y = self.pitch / config.height as f32;
        let angle_y = norm_x * 5.0;
        let angle_xz = -norm_y * 2.0;

        let camera_dist = self.position.distance(self.centre);

        let new_position = [
            angle_xz.cos() * angle_y.sin() * camera_dist,
            angle_xz.sin() * camera_dist + self.centre[1],
            angle_xz.cos() * angle_y.cos() * camera_dist,
        ];

        self.position = Vec3::from_array(new_position);
    }

    pub fn move_forward(&mut self, d: f32) {
        self.position -= self.forward * d * self.move_speed;
    }

    fn get_yaw_pitch(position: Vec3, centre: Vec3) -> (f32, f32) {
        let v = (centre - position).normalize();

        let pitch = v.y.asin();
        let yaw = v.z.atan2(v.x);

        (yaw, pitch)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ubo {
    inverse_camera_transform: [f32; 16],
    half_width_height: [f32; 2],
    pixel_size: f32,
    sqrt_rays_per_pixel: u32,
    width: u32,
    height: u32,
    n_objects: u32,
    lights_offset: u32,
    specular_offset: u32,
    subpixel_idx: u32,
    ray_bounces: u32,
    _pad: u32,
}

#[derive(Debug)]
pub struct ScreenData {
    // Compute shader uniform block object
    pub size: PhysicalSize<u32>,
    pub resolution: PhysicalSize<u32>,
    pub inverse_camera_transform: Mat4,
    pixel_size: f32,
    half_width_height: Vec2,
    fov: f32,
    specular_offset: u32,
    lights_offset: u32,
    n_objects: u32,
    pub subpixel_idx: u32,
    sqrt_rays_per_pixel: u32,
    pub ray_bounces: u32,
}

impl ScreenData {
    pub fn new(
        inverse_camera_transform: Mat4,
        n_objects: u32,
        specular_offset: u32,
        lights_offset: u32,
        ray_bounces: u32,
        size: PhysicalSize<u32>,
        resolution: PhysicalSize<u32>,
        fov: f32,
        sqrt_rays_per_pixel: u32,
    ) -> ScreenData {
        let half_view = (fov / 2f32).tan();
        let aspect = size.width as f32 / size.height as f32;

        let mut half_width_height = Vec2::new(half_view, half_view / aspect);

        if aspect < 1f32 {
            half_width_height.x = half_view;
            half_width_height.y = half_view / aspect;
        }
        let pixel_size = (half_width_height.x * 2f32) / size.width as f32;

        log::info!("Window size: {:?}", size);

        ScreenData {
            size,
            resolution,
            inverse_camera_transform,
            pixel_size,
            half_width_height,
            fov,
            n_objects,
            lights_offset,
            specular_offset,
            subpixel_idx: 0,
            sqrt_rays_per_pixel,
            ray_bounces,
        }
    }

    pub fn update_dims(&mut self, size: &PhysicalSize<u32>) {
        let half_view = (self.fov / 2f32).tan();
        let aspect = size.width as f32 / size.height as f32;

        self.size = *size;

        self.half_width_height.x = half_view;
        self.half_width_height.y = half_view / aspect;

        if aspect < 1f32 {
            self.half_width_height.x = half_view;
            self.half_width_height.y = half_view / aspect;
        }

        self.pixel_size = (self.half_width_height.x * 2f32) / size.width as f32;
    }

    pub fn generate_ubo(&self) -> Ubo {
        Ubo {
            inverse_camera_transform: self.inverse_camera_transform.to_cols_array(),
            half_width_height: self.half_width_height.to_array(),
            pixel_size: self.pixel_size,
            sqrt_rays_per_pixel: self.sqrt_rays_per_pixel,
            _pad: 0,
            width: self.size.width,
            height: self.size.height,
            n_objects: self.n_objects,
            subpixel_idx: self.subpixel_idx,
            ray_bounces: self.ray_bounces,
            lights_offset: self.lights_offset,
            specular_offset: self.specular_offset,
        }
    }
}
