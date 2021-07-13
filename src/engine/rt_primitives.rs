use glam::{const_vec3, const_vec4, Mat4, Vec3, Vec4};
use serde::{Deserialize, Serialize};

#[repr(C)]
pub struct Material {
    pub colour: Vec4,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    pub shininess: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NodeTLAS {
    pub first: Vec4,
    pub second: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NodeBLAS {
    pub point1: Vec4,
    pub point2: Vec4,
    pub point3: Vec4,
    pub normal1: Vec4,
    pub normal2: Vec4,
    pub normal3: Vec4,
}

#[repr(C)]
pub struct Shape {
    tlas_offset: u32,
    blas_offset: u32,
    type_enum: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct Camera {
    pub inverse_transform: Mat4,
    pub pixel_size: f32,
    pub half_width: f32,
    pub half_height: f32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct UBO {
    // Compute shader uniform block object
    light_pos: Vec4,
    camera: Camera,
}

impl UBO {
    pub fn new(light_pos: [f32; 4], camera: Camera) -> UBO {
        UBO {
            light_pos: const_vec4!(light_pos),
            camera: camera,
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
            let half_height = half_view;
            let half_width = half_view / aspect;
        }
        let pixel_size = (half_width * 2f32) / hsize as f32;

        Camera {
            inverse_transform,
            half_width: half_width,
            half_height: half_height,
            width: hsize,
            height: vsize,
            pixel_size: pixel_size,
        }
    }
}

impl NodeBLAS {
    pub fn bounds(&self) -> NodeTLAS {
        let x = [self.point1.x, self.point2.x, self.point3.x];
        let y = [self.point1.y, self.point2.y, self.point3.y];
        let z = [self.point1.z, self.point2.z, self.point3.z];

        let min: Vec4 = const_vec4!([
            x.iter().copied().fold(f32::INFINITY, f32::min),
            y.iter().copied().fold(f32::INFINITY, f32::min),
            z.iter().copied().fold(f32::INFINITY, f32::min),
            1.
        ]);

        let max: Vec4 = const_vec4!([
            x.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            y.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            z.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            1.
        ]);

        NodeTLAS {
            first: min,
            second: max,
        }
    }

    pub fn bounds_centroid(&self) -> Vec4 {
        let bounds = self.bounds();
        0.5 * bounds.first + 0.5 * bounds.second
    }
}

impl NodeTLAS {
    pub fn merge(&self, other: &NodeTLAS) -> NodeTLAS {
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

        NodeTLAS {
            first: min,
            second: max,
        }
    }
}

// TODO: need buffers containing: TLASes, BLASes, Materials, u32 pairs for shape offsets into tlas and blas buffers,
// u32 for enum of shape types to allow NodeBLAS to store different types of shapes

// std::nth_element - try use: https://doc.rust-lang.org/std/primitive.slice.html#method.select_nth_unstable
