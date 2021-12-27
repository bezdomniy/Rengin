use glam::{const_vec3, const_vec4, Mat3, Mat4, Vec3, Vec4, Vec4Swizzles};
use rand::Rng;

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
            refractive_index: 1.0,
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
    pub len_inner_nodes: u32,
    pub len_leaf_nodes: u32,
    pub is_light: u32,
    pub model_type: u32,
}

impl ObjectParams {
    pub fn new(
        transform: Mat4,
        material: Material,
        len_inner_nodes: u32,
        len_leaf_nodes: u32,
        is_light: u32,
        model_type: u32,
    ) -> Self {
        ObjectParams {
            inverse_transform: transform.inverse(),
            material,
            len_inner_nodes,
            len_leaf_nodes,
            is_light,
            model_type,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeLeaf {
    pub point1: Vec3,
    pub object_id: u32,
    pub point2: Vec3,
    pub pad1: u32,
    pub point3: Vec3,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeNormal {
    pub normals: [Vec4; 3],
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
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
    pub len_inner_nodes: Vec<u32>,
    pub len_leaf_nodes: Vec<u32>,
}

impl BVH {
    pub fn new(
        inner_nodes: Vec<Vec<NodeInner>>,
        leaf_nodes: Vec<Vec<NodeLeaf>>,
        normal_nodes: Vec<Vec<NodeNormal>>,
    ) -> Self {
        let len_inner_nodes: Vec<u32> = inner_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        let len_leaf_nodes: Vec<u32> = leaf_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        let n_objects = inner_nodes.len() as u32;

        BVH {
            inner_nodes: inner_nodes.into_iter().flatten().collect::<Vec<_>>(),
            leaf_nodes: leaf_nodes.into_iter().flatten().collect::<Vec<_>>(),
            normal_nodes: normal_nodes.into_iter().flatten().collect::<Vec<_>>(),
            len_inner_nodes,
            len_leaf_nodes,
        }
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
    pub fn new(x: u32, y: u32, ray_index: u32, sqrt_rays_per_pixel: u32, camera: &Camera) -> Self {
        let half_sub_pixel_size = 1.0 / (sqrt_rays_per_pixel as f32) / 2.0;

        let sub_pixel_row_number: u32 = ray_index / sqrt_rays_per_pixel;
        let sub_pixel_col_number: u32 = ray_index % sqrt_rays_per_pixel;
        let sub_pixel_x_offset: f32 = half_sub_pixel_size * (sub_pixel_col_number as f32);
        let sub_pixel_y_offset: f32 = half_sub_pixel_size * (sub_pixel_row_number as f32);

        let x_offset: f32 = ((x as f32) + sub_pixel_x_offset) * camera.pixel_size;
        let y_offset: f32 = ((y as f32) + sub_pixel_y_offset) * camera.pixel_size;

        let world_x: f32 = camera.half_width - x_offset;
        let world_y: f32 = camera.half_height - y_offset;

        let pixel = camera.inverse_transform * Vec4::new(world_x, world_y, -1.0, 1.0);

        let ray_o = camera.inverse_transform * Vec4::new(0.0, 0.0, 0.0, 1.0);

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

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub inverse_transform: Mat4,
    pub pixel_size: f32,
    pub half_width: f32,
    pub half_height: f32,
    pub width: u32,
    // pub height: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UBO {
    // Compute shader uniform block object
    light_pos: Vec4,
    pub camera: Camera,
    n_objects: u32,
    pub subpixel_idx: u32,
    sqrt_rays_per_pixel: u32,
    rnd_seed: f32,
    // _padding: [u32; 1],
}

impl UBO {
    pub fn new(
        light_pos: [f32; 4],
        n_objects: u32,
        sqrt_rays_per_pixel: u32,
        camera: Camera,
    ) -> UBO {
        UBO {
            light_pos: const_vec4!(light_pos),
            camera,
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

impl Camera {
    pub fn new(
        position: [f32; 3],
        centre: [f32; 3],
        up: [f32; 3],
        hsize: u32,
        vsize: u32,
        fov: f32,
    ) -> Camera {
        let inverse_transform =
            Mat4::look_at_rh(const_vec3!(position), const_vec3!(centre), const_vec3!(up)).inverse();
        let half_view = (fov / 2f32).tan();
        let aspect = hsize as f32 / vsize as f32;

        let mut half_width = half_view;
        let mut half_height = half_view / aspect;

        if aspect < 1f32 {
            half_height = half_view;
            half_width = half_view / aspect;
        }
        let pixel_size = (half_width * 2f32) / hsize as f32;

        Camera {
            inverse_transform,
            half_width: half_width,
            half_height: half_height,
            width: hsize,
            // height: vsize,
            pixel_size: pixel_size,
        }
    }

    pub fn update_position(&mut self, new_position: [f32; 3], centre: [f32; 3], up: [f32; 3]) {
        self.inverse_transform = Mat4::look_at_rh(
            const_vec3!(new_position),
            const_vec3!(centre),
            const_vec3!(up),
        )
        .inverse();
        // println!("{}", self.inverse_transform);
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
