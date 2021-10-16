use glam::{const_mat4, const_vec3, const_vec4, Mat3, Mat4, Vec3, Vec4};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Material {
    pub colour: Vec4,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ObjectParams {
    pub inverse_transform: Mat4,
    pub material: Material,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct BoundingBox {
    pub first: Vec4,
    pub second: Vec4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeLeaf {
    pub points: Mat4,
    pub normals: Mat4,
    // pub points: [Vec4; 3],
    // pub normals: [Vec4; 3],
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct NodeInner {
    pub first: Vec3,
    pub skip_ptr_or_prim_idx1: u32,
    pub second: Vec3,
    pub prim_idx2: u32,
}

#[derive(Debug)]
#[repr(C)]
pub struct BVH {
    pub inner_nodes: Vec<NodeInner>,
    pub leaf_nodes: Vec<NodeLeaf>,
}

// #[repr(C)]
// pub struct Shape {
//     tlas_offset: u32,
//     blas_offset: u32,
//     type_enum: u32,
// }

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
    len_inner_nodes: i32,
    len_leaf_nodes: i32,
    _padding: [u32; 2],
}

impl UBO {
    pub fn new(
        light_pos: [f32; 4],
        len_inner_nodes: i32,
        len_leaf_nodes: i32,
        camera: Camera,
    ) -> UBO {
        UBO {
            light_pos: const_vec4!(light_pos),
            camera,
            len_inner_nodes,
            len_leaf_nodes,
            _padding: [0, 0],
        }
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
    pub fn empty() -> NodeLeaf {
        NodeLeaf {
            points: const_mat4!([f32::NEG_INFINITY; 16]),
            normals: const_mat4!([f32::NEG_INFINITY; 16]),
        }
    }
    pub fn bounds(&self) -> BoundingBox {
        self.points
            .to_cols_array_2d()
            .iter()
            .fold(BoundingBox::empty(), |aabb, p| {
                aabb.add_point(&Vec4::new(p[0], p[1], p[2], 1f32))
            })
    }

    pub fn bounds_centroid(&self) -> Vec4 {
        let bounds = self.bounds();
        0.5 * bounds.first + 0.5 * bounds.second
    }
}

impl BoundingBox {
    pub fn empty() -> BoundingBox {
        BoundingBox {
            first: const_vec4!([
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]),
            second: const_vec4!([
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY
            ]),
        }
    }

    pub fn merge(&self, other: &BoundingBox) -> BoundingBox {
        let min: Vec4 = const_vec4!([
            f32::min(self.first.x, other.first.x),
            f32::min(self.first.y, other.first.y),
            f32::min(self.first.z, other.first.z),
            1.
        ]);

        let max: Vec4 = const_vec4!([
            f32::max(self.second.x, other.second.x),
            f32::max(self.second.y, other.second.y),
            f32::max(self.second.z, other.second.z),
            1.
        ]);

        BoundingBox {
            first: min,
            second: max,
        }
    }

    pub fn add_point(&self, point: &Vec4) -> Self {
        BoundingBox {
            first: self.first.min(*point),
            second: self.second.max(*point),
        }
    }
}

// TODO: need buffers containing: TLASes, BLASes, Materials, u32 pairs for shape offsets into tlas and blas buffers,
// u32 for enum of shape types to allow NodeBLAS to store different types of shapes
