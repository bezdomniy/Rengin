use std::convert::TryInto;

use crate::engine::rt_primitives::{NodeBLAS, NodeTLAS};
use glam::{const_vec4, Vec4};
use tobj;

pub fn import_obj(path: &str) -> Vec<(Vec<NodeTLAS>, Vec<NodeBLAS>)> {
    let (models, materials) =
        tobj::load_obj(path, &tobj::LoadOptions::default()).expect("Failed to OBJ load file");

    let mut ret: Vec<(Vec<NodeTLAS>, Vec<NodeBLAS>)> = vec![];

    for model in models.iter() {
        let triangles: Vec<NodeBLAS> = model
            .mesh
            .indices
            .chunks_exact(3)
            .map(|pos| {
                let triangle_indices: [u32; 3] = pos.try_into().unwrap();
                NodeBLAS {
                    point1: const_vec4!([
                        model.mesh.positions[triangle_indices[0] as usize],
                        model.mesh.positions[(triangle_indices[0] + 1) as usize],
                        model.mesh.positions[(triangle_indices[0] + 2) as usize],
                        1.0
                    ]),
                    point2: const_vec4!([
                        model.mesh.positions[triangle_indices[1] as usize],
                        model.mesh.positions[(triangle_indices[1] + 1) as usize],
                        model.mesh.positions[(triangle_indices[1] + 2) as usize],
                        1.0
                    ]),
                    point3: const_vec4!([
                        model.mesh.positions[triangle_indices[2] as usize],
                        model.mesh.positions[(triangle_indices[2] + 1) as usize],
                        model.mesh.positions[(triangle_indices[2] + 2) as usize],
                        1.0
                    ]),
                    normal1: const_vec4!([
                        model.mesh.normals[triangle_indices[0] as usize],
                        model.mesh.normals[(triangle_indices[0] + 1) as usize],
                        model.mesh.normals[(triangle_indices[0] + 2) as usize],
                        1.0
                    ]),
                    normal2: const_vec4!([
                        model.mesh.normals[triangle_indices[1] as usize],
                        model.mesh.normals[(triangle_indices[1] + 1) as usize],
                        model.mesh.normals[(triangle_indices[1] + 2) as usize],
                        1.0
                    ]),
                    normal3: const_vec4!([
                        model.mesh.normals[triangle_indices[2] as usize],
                        model.mesh.normals[(triangle_indices[2] + 1) as usize],
                        model.mesh.normals[(triangle_indices[2] + 2) as usize],
                        1.0
                    ]),
                }
            })
            .collect();

        // TODO: probably wont be triangles vec directly, but the sorted version resulting from build_tlas
        ret.push((build_tlas(&triangles), triangles));
    }

    ret
}

fn build_tlas(triangles: &Vec<NodeBLAS>) -> Vec<NodeTLAS> {
    vec![NodeTLAS {
        first: Vec4::new(1.0, 2.0, 3.0, 4.0),
        second: Vec4::new(1.0, 2.0, 3.0, 4.0),
    }]
}
