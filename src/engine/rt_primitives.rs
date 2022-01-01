use std::f32::consts::FRAC_PI_2;

use glam::{const_mat4, const_vec3, const_vec4, EulerRot, Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};
use itertools::Itertools;
use rand::Rng;
use wgpu::SurfaceConfiguration;

// pub const OPENGL_TO_WGPU_MATRIX: Mat4 = const_mat4!(
//     [1.0, 0.0, 0.0, 0.0],
//     [0.0, 1.0, 0.0, 0.0],
//     [0.0, 0.0, 0.5, 0.0],
//     [0.0, 0.0, 0.5, 1.0,]
// );
const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub colour: Vec4,
    pub emissiveness: Vec4,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
    pub reflective: f32,
    pub transparency: f32,
    pub refractive_index: f32,
    _padding: u32,
}

impl Material {
    pub fn new(
        colour: Vec4,
        emissiveness: Vec4,
        ambient: f32,
        diffuse: f32,
        specular: f32,
        shininess: f32,
        reflective: f32,
        transparency: f32,
        refractive_index: f32,
    ) -> Self {
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
            _padding: 0,
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            colour: Vec4::new(0.0, 0.0, 0.0, 0.0),
            emissiveness: Vec4::new(0.0, 0.0, 0.0, 0.0),
            ambient: 0.1,
            diffuse: 0.9,
            specular: 0.9,
            shininess: 200.0,
            reflective: 0.0,
            transparency: 0.0,
            refractive_index: 0.0,
            _padding: 0,
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct Primitive {
    pub points: Mat3,
    pub normals: Mat3,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectParams {
    pub inverse_transform: Mat4,
    pub material: Material,
    pub offset_inner_nodes: u32,
    pub len_inner_nodes: u32,
    pub offset_leaf_nodes: u32,
    pub model_type: u32,
}

impl ObjectParams {
    pub fn new(
        transform: Mat4,
        material: Material,
        offset_inner_nodes: u32,
        len_inner_nodes: u32,
        offset_leaf_nodes: u32,
        model_type: u32,
    ) -> Self {
        ObjectParams {
            inverse_transform: transform.inverse(),
            material,
            offset_inner_nodes,
            len_inner_nodes,
            offset_leaf_nodes,
            model_type,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeLeaf {
    pub point1: Vec3,
    pub object_id: u32,
    pub point2: Vec3,
    pub pad1: u32,
    pub point3: Vec3,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeNormal {
    pub normals: [Vec4; 3],
}

#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct NodeInner {
    pub first: Vec3,
    pub skip_ptr_or_prim_idx1: u32,
    pub second: Vec3,
    pub prim_idx2: u32,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct BVH {
    pub inner_nodes: Vec<NodeInner>,
    pub leaf_nodes: Vec<NodeLeaf>,
    pub normal_nodes: Vec<NodeNormal>,
    pub offset_inner_nodes: Vec<u32>,
    pub len_inner_nodes: Vec<u32>,
    pub offset_leaf_nodes: Vec<u32>,
    pub model_tags: Vec<String>,
}

impl BVH {
    pub fn empty() -> Self {
        BVH {
            inner_nodes: vec![NodeInner::default()],
            leaf_nodes: vec![NodeLeaf::default()],
            normal_nodes: vec![NodeNormal::default()],
            offset_inner_nodes: vec![],
            len_inner_nodes: vec![],
            offset_leaf_nodes: vec![],
            model_tags: vec![],
        }
    }

    pub fn new(
        inner_nodes: Vec<Vec<NodeInner>>,
        leaf_nodes: Vec<Vec<NodeLeaf>>,
        normal_nodes: Vec<Vec<NodeNormal>>,
        model_tags: Vec<String>,
    ) -> Self {
        if inner_nodes.len() == 0 {
            return BVH::empty();
        }

        let len_inner_nodes: Vec<u32> = inner_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        let mut offset_inner_nodes: Vec<u32> = len_inner_nodes
            .iter()
            .scan(0, |acc, next_len| {
                *acc = *acc + next_len;
                Some(*acc)
            })
            .collect();

        offset_inner_nodes.pop();
        offset_inner_nodes.splice(0..0, [0u32]);

        let mut offset_leaf_nodes: Vec<u32> = leaf_nodes
            .iter()
            .scan(0, |acc, next_vec| {
                *acc = *acc + next_vec.len() as u32;
                Some(*acc)
            })
            .collect();

        offset_leaf_nodes.pop();
        offset_leaf_nodes.splice(0..0, [0u32]);

        let n_objects = inner_nodes.len() as u32;

        BVH {
            inner_nodes: inner_nodes.into_iter().flatten().collect::<Vec<_>>(),
            leaf_nodes: leaf_nodes.into_iter().flatten().collect::<Vec<_>>(),
            normal_nodes: normal_nodes.into_iter().flatten().collect::<Vec<_>>(),
            offset_inner_nodes,
            len_inner_nodes,
            offset_leaf_nodes,
            model_tags,
        }
    }

    pub fn find_model_locations(&self, tag: &String) -> (u32, u32, u32) {
        // println!("### TAG {:?}", tag);
        let index = self.model_tags.iter().position(|r| r == tag).unwrap();

        (
            self.offset_inner_nodes[index],
            self.len_inner_nodes[index],
            self.offset_leaf_nodes[index],
        )
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Ray {
    origin: Vec3,
    x: u32,
    direction: Vec3,
    y: u32,
}

impl Ray {
    pub fn new(x: u32, y: u32, ray_index: u32, ubo: &UBO) -> Self {
        let half_sub_pixel_size = 1.0 / (ubo.sqrt_rays_per_pixel as f32) / 2.0;

        let sub_pixel_row_number: u32 = ray_index / ubo.sqrt_rays_per_pixel;
        let sub_pixel_col_number: u32 = ray_index % ubo.sqrt_rays_per_pixel;
        let sub_pixel_x_offset: f32 = half_sub_pixel_size * (sub_pixel_col_number as f32);
        let sub_pixel_y_offset: f32 = half_sub_pixel_size * (sub_pixel_row_number as f32);

        let x_offset: f32 = ((x as f32) + sub_pixel_x_offset) * ubo.pixel_size;
        let y_offset: f32 = ((y as f32) + sub_pixel_y_offset) * ubo.pixel_size;

        let world_x: f32 = ubo.half_width - x_offset;
        let world_y: f32 = ubo.half_height - y_offset;

        let pixel = ubo.inverse_camera_transform * Vec4::new(world_x, world_y, -1.0, 1.0);

        let ray_o = ubo.inverse_camera_transform * Vec4::new(0.0, 0.0, 0.0, 1.0);

        Ray {
            origin: ray_o.xyz(),
            x,
            direction: (pixel - ray_o).xyz(),
            y,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default)]
pub struct Rays {
    pub data: Vec<Ray>,
}

// TODO: implement sorting before output to gpu buffer
impl Rays {
    pub fn new(width: u32, height: u32) -> Self {
        let mut rays = Rays {
            data: vec![
                Ray {
                    direction: Vec3::new(0.0, 0.0, 0.0),
                    x: 0,
                    origin: Vec3::new(0.0, 0.0, 0.0),
                    y: 0
                };
                (width * height) as usize
            ],
        };

        // for x in 0..width {
        //     for y in 0..height {
        //         data.
        //     }
        // }

        rays
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
        let position = const_vec3!(p);
        let centre = const_vec3!(c);
        let up = const_vec3!(u);

        let (yaw, pitch) = Camera::get_yaw_pitch(position, centre);

        // let forward = (centre - position).normalize();

        let forward = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize();

        // direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        // direction.y = sin(glm::radians(pitch));
        // direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        // cameraFront = glm::normalize(direction);

        // // Keep the camera's angle from going too high/low.
        // if pitch < -SAFE_FRAC_PI_2 {
        //     pitch = -SAFE_FRAC_PI_2;
        // } else if pitch > SAFE_FRAC_PI_2 {
        //     pitch = SAFE_FRAC_PI_2;
        // }

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

    pub fn orbit(&mut self, d_x: f32, d_y: f32, config: &SurfaceConfiguration) {
        self.yaw += d_x as f32;
        self.pitch += d_y as f32;
        println!("{} {}\n{} {}", d_x, d_y, self.yaw, self.pitch);

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

        self.position = const_vec3!(new_position);
    }

    pub fn move_forward(&mut self, d: f32) {
        self.position -= self.forward * d * self.move_speed;
    }

    fn get_yaw_pitch(position: Vec3, centre: Vec3) -> (f32, f32) {
        let v = (centre - position).normalize();

        let pitch = -v.y.asin();
        let yaw = v.z.atan2(v.x);

        (yaw, pitch)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UBO {
    // Compute shader uniform block object
    light_pos: Vec4,
    pub inverse_camera_transform: Mat4,
    pixel_size: f32,
    half_width: f32,
    half_height: f32,
    _padding: u32,
    n_objects: u32,
    pub subpixel_idx: u32,
    sqrt_rays_per_pixel: u32,
    rnd_seed: f32,
    // _padding: [u32; 1],
}

impl UBO {
    pub fn new(
        light_pos: [f32; 4],
        inverse_camera_transform: Mat4,
        n_objects: u32,
        width: u32,
        height: u32,
        fov: f32,
        sqrt_rays_per_pixel: u32,
    ) -> UBO {
        let half_view = (fov / 2f32).tan();
        let aspect = width as f32 / height as f32;

        let mut half_width = half_view;
        let mut half_height = half_view / aspect;

        if aspect < 1f32 {
            half_height = half_view;
            half_width = half_view / aspect;
        }
        let pixel_size = (half_width * 2f32) / width as f32;

        UBO {
            light_pos: const_vec4!(light_pos),
            inverse_camera_transform,
            pixel_size,
            half_width,
            half_height,
            _padding: 0,
            n_objects,
            subpixel_idx: 0,
            sqrt_rays_per_pixel,
            rnd_seed: rand::thread_rng().gen_range(0.0..1.0),
        }
    }

    pub fn update_random_seed(&mut self) {
        self.rnd_seed = rand::thread_rng().gen_range(0.0..1.0);
    }
}

impl NodeLeaf {
    pub fn new(v: [f32; 9], object_id: u32) -> Self {
        NodeLeaf {
            point1: const_vec3!([v[0], v[1], v[2]]),
            point2: const_vec3!([v[3], v[4], v[5]]),
            point3: const_vec3!([v[6], v[7], v[8]]),
            object_id,
            pad1: 0,
            pad2: 0,
        }
    }
}

impl NodeNormal {
    pub fn new(v: [f32; 9]) -> Self {
        NodeNormal {
            normals: [
                const_vec4!([v[0], v[1], v[2], 0f32]),
                const_vec4!([v[3], v[4], v[5], 0f32]),
                const_vec4!([v[6], v[7], v[8], 0f32]),
            ],
        }
    }
}

impl Primitive {
    pub fn bounds(&self) -> NodeInner {
        self.points
            .to_cols_array_2d()
            .iter()
            .fold(NodeInner::empty(), |aabb, p| {
                aabb.add_point(&const_vec3!(*p))
            })
        // .add_point(&Vec3::new(0f32, 0f32, 0f32)) // This slows it down massively, but makes cube work for some reason...
        // this is an issue with bounding boxes around axis aligned triangles - TODO, figure it out
    }

    pub fn bounds_centroid(&self) -> Vec3 {
        let bounds = self.bounds();
        0.5 * bounds.first + 0.5 * bounds.second
    }
}

impl NodeInner {
    pub fn empty() -> Self {
        NodeInner {
            first: const_vec3!([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: const_vec3!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            prim_idx2: 0,
        }
    }

    pub fn merge(&self, other: &NodeInner) -> Self {
        let min: Vec3 = const_vec3!([
            f32::min(self.first.x, other.first.x),
            f32::min(self.first.y, other.first.y),
            f32::min(self.first.z, other.first.z),
        ]);

        let max: Vec3 = const_vec3!([
            f32::max(self.second.x, other.second.x),
            f32::max(self.second.y, other.second.y),
            f32::max(self.second.z, other.second.z),
        ]);

        NodeInner {
            first: min,
            skip_ptr_or_prim_idx1: other.skip_ptr_or_prim_idx1,
            second: max,
            prim_idx2: other.prim_idx2,
        }
    }

    pub fn add_point(&self, point: &Vec3) -> Self {
        NodeInner {
            first: self.first.min(*point),
            skip_ptr_or_prim_idx1: self.skip_ptr_or_prim_idx1,
            second: self.second.max(*point),
            prim_idx2: self.prim_idx2,
        }
    }

    pub fn diagonal(&self) -> Vec3 {
        self.second - self.first
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        2 as f32 * (d.x * d.y + d.x * d.z + d.y * d.z)
    }

    pub fn offset(&self, point: &Vec3) -> Vec3 {
        let mut o = *point - self.first;
        if self.second.x > self.first.x {
            o.x /= self.second.x - self.first.x
        }
        if self.second.y > self.first.y {
            o.y /= self.second.y - self.first.y
        }
        if self.second.z > self.first.z {
            o.z /= self.second.z - self.first.z
        };
        return o;
    }
}
