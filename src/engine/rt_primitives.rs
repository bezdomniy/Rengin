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

#[derive(Debug, Default)]
#[repr(C)]
pub struct BVH {
    pub inner_nodes: Vec<NodeInner>,
    pub leaf_nodes: Vec<NodeLeaf>,
    pub max_inner_node_idx: Vec<u32>,
    pub max_leaf_node_idx: Vec<u32>,
}

impl BVH {
    pub fn new(inner_nodes: Vec<Vec<NodeInner>>, leaf_nodes: Vec<Vec<NodeLeaf>>) -> Self {
        let max_inner_node_idx: Vec<u32> = inner_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        let max_leaf_node_idx: Vec<u32> = leaf_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        BVH {
            inner_nodes: inner_nodes.into_iter().flatten().collect::<Vec<_>>(),
            leaf_nodes: leaf_nodes.into_iter().flatten().collect::<Vec<_>>(),
            max_inner_node_idx,
            max_leaf_node_idx,
        }
    }
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
    max_inner_node_idx: u32,
    max_leaf_node_idx: u32,
    _padding: [u32; 2],
}

impl UBO {
    pub fn new(
        light_pos: [f32; 4],
        max_inner_node_idx: u32,
        max_leaf_node_idx: u32,
        camera: Camera,
    ) -> UBO {
        UBO {
            light_pos: const_vec4!(light_pos),
            camera,
            max_inner_node_idx,
            max_leaf_node_idx,
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
    pub fn bounds(&self) -> NodeInner {
        self.points
            .to_cols_array_2d()
            .iter()
            .fold(NodeInner::empty(), |aabb, p| {
                aabb.add_point(&Vec3::new(p[0], p[1], p[2]))
            })
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

    pub fn total() -> Self {
        NodeInner {
            first: const_vec3!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: const_vec3!([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
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
}

// TODO: need buffers containing: TLASes, BLASes, Materials, u32 pairs for shape offsets into tlas and blas buffers,
// u32 for enum of shape types to allow NodeBLAS to store different types of shapes
