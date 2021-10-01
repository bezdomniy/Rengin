use crate::engine::rt_primitives::{NodeBLAS, NodeTLAS};
use glam::{const_mat4, const_vec4};
use tobj;

fn empty_bounds() -> NodeTLAS {
    NodeTLAS {
        first: const_vec4!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]),
        second: const_vec4!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]),
        // first: const_vec4!([-1.0, -1.0, -1.0, -1.0]),
        // second: const_vec4!([-1.0, -1.0, -1.0, -1.0]),
    }
}

fn total_bounds() -> NodeTLAS {
    NodeTLAS {
        first: const_vec4!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]),
        second: const_vec4!([f32::INFINITY, f32::INFINITY, f32::INFINITY, 1.0]),
    }
}

pub fn import_obj(path: &str) -> Option<Vec<(Vec<NodeTLAS>, Vec<NodeBLAS>)>> {
    let (models, materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            // triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .expect("Failed to OBJ load file");

    let mut ret: Vec<(Vec<NodeTLAS>, Vec<NodeBLAS>)> = vec![];

    for model in models.iter() {
        // println!("{:?}",model.mesh.indices);
        let mut triangles: Vec<NodeBLAS> = model
            .mesh
            .indices
            .chunks_exact(3)
            .into_iter()
            .map(|triangle_indices| {
                // println!("{:?}", triangle_indices);
                NodeBLAS {
                    points: const_mat4!(
                        [
                            model.mesh.positions[3 * triangle_indices[0] as usize],
                            model.mesh.positions[(3 * triangle_indices[0] + 1) as usize],
                            model.mesh.positions[(3 * triangle_indices[0] + 2) as usize],
                            1.0
                        ],
                        [
                            model.mesh.positions[3 * triangle_indices[1] as usize],
                            model.mesh.positions[(3 * triangle_indices[1] + 1) as usize],
                            model.mesh.positions[(3 * triangle_indices[1] + 2) as usize],
                            1.0
                        ],
                        [
                            model.mesh.positions[3 * triangle_indices[2] as usize],
                            model.mesh.positions[(3 * triangle_indices[2] + 1) as usize],
                            model.mesh.positions[(3 * triangle_indices[2] + 2) as usize],
                            1.0
                        ],
                        [0.0; 4]
                    ),
                    normals: const_mat4!(
                        [
                            model.mesh.normals[3 * triangle_indices[0] as usize],
                            model.mesh.normals[(3 * triangle_indices[0] + 1) as usize],
                            model.mesh.normals[(3 * triangle_indices[0] + 2) as usize],
                            0.0
                        ],
                        [
                            model.mesh.normals[3 * triangle_indices[1] as usize],
                            model.mesh.normals[(3 * triangle_indices[1] + 1) as usize],
                            model.mesh.normals[(3 * triangle_indices[1] + 2) as usize],
                            0.0
                        ],
                        [
                            model.mesh.normals[3 * triangle_indices[2] as usize],
                            model.mesh.normals[(3 * triangle_indices[2] + 1) as usize],
                            model.mesh.normals[(3 * triangle_indices[2] + 2) as usize],
                            0.0
                        ],
                        [0.0; 4]
                    ),
                }
            })
            .collect();

        let (tlas, blas) = build(&mut triangles);
        ret.push((tlas, blas));
    }

    Some(ret)
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log_2(x: usize) -> usize {
    num_bits::<usize>() - x.leading_zeros() as usize - 1
}

fn build(triangles: &mut Vec<NodeBLAS>) -> (Vec<NodeTLAS>, Vec<NodeBLAS>) {
    let mut tlas: Vec<NodeTLAS> = Vec::new();
    tlas.resize(triangles.len().next_power_of_two(), empty_bounds());
    let tlas_height = log_2(triangles.len());

    let mut blas: Vec<NodeBLAS> = Vec::with_capacity(triangles.len().next_power_of_two());

    recursive_build(
        &mut tlas,
        &mut blas,
        triangles,
        0,
        0,
        0,
        triangles.len(),
        tlas_height,
    );

    (tlas, blas)
}

fn recursive_build(
    tlas: &mut Vec<NodeTLAS>,
    blas: &mut Vec<NodeBLAS>,
    triangle_params_unsorted: &mut Vec<NodeBLAS>,
    level: usize,
    branch: usize,
    start: usize,
    end: usize,
    tlas_height: usize,
) -> () {
    let centroid_bounds = triangle_params_unsorted[start..end].iter().fold(
        NodeTLAS {
            first: const_vec4!([f32::INFINITY, f32::INFINITY, f32::INFINITY, 1.0]),
            second: const_vec4!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]),
        },
        |acc, new| acc.merge(&new.bounds()),
    );

    println!("cb: {:?}", centroid_bounds);

    let diagonal = centroid_bounds.second - centroid_bounds.first;

    let split_dimension = if diagonal.x > diagonal.y && diagonal.x > diagonal.z {
        0
    } else if diagonal.y > diagonal.z {
        1
    } else {
        2
    };

    if centroid_bounds.first[split_dimension] == centroid_bounds.second[split_dimension] {
        ()
    } else {
        let mid = (start + end) / 2;

        triangle_params_unsorted[start..end].select_nth_unstable_by(mid - start, |a, b| {
            b.bounds_centroid()[split_dimension]
                .partial_cmp(&a.bounds_centroid()[split_dimension])
                .unwrap()
        });

        let node: usize = 2usize.pow(level as u32) + branch - 1;

        // println!("adding: {:?}",centroid_bounds);
        tlas[node] = centroid_bounds;

        let n_shapes = end - start;

        if n_shapes > 2 {
            recursive_build(
                tlas,
                blas,
                triangle_params_unsorted,
                level + 1,
                branch * 2,
                start,
                mid,
                tlas_height,
            );
            recursive_build(
                tlas,
                blas,
                triangle_params_unsorted,
                level + 1,
                (branch * 2) + 1,
                mid,
                end,
                tlas_height,
            );
        } else {
            blas.push(triangle_params_unsorted[start]);

            if n_shapes == 2 {
                blas.push(triangle_params_unsorted[start + 1]);
            } else {
                blas.push(NodeBLAS::empty());
            }
            if level < tlas_height {
                // println!("level {}, tlas_height {}", level, tlas_height);
                let dummy_node = 2usize.pow((level + 1) as u32) + (branch * 2) - 1;

                // tlas[dummy_node] = total_bounds();
                // tlas.swap(dummy_node, node);
                // println!("adding dummy: {:?}",centroid_bounds);
                tlas[dummy_node] = centroid_bounds;
                tlas[dummy_node + 1] = empty_bounds();

                blas.push(NodeBLAS::empty());
                blas.push(NodeBLAS::empty());
            }
            ()
        }
    }
    ()
}
