use crate::engine::rt_primitives::{NodeInner, NodeLeaf, BVH};
use glam::{const_mat4, const_vec3, const_vec4, Vec4Swizzles};
use tobj;

pub fn import_obj(path: &str) -> Option<BVH> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            // triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .expect("Failed to OBJ load file");

    let mut object_inner_nodes: Vec<Vec<NodeInner>> = vec![];
    let mut object_leaf_nodes: Vec<Vec<NodeLeaf>> = vec![];

    for model in models.iter() {
        // println!("{:?}",model.mesh.indices);
        let mut triangles: Vec<NodeLeaf> = model
            .mesh
            .indices
            .chunks_exact(3)
            .into_iter()
            .map(|triangle_indices| {
                // println!("{:?}", triangle_indices);
                NodeLeaf {
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

        let bounding_boxes = build(&mut triangles);

        object_inner_nodes.push(bounding_boxes);
        object_leaf_nodes.push(triangles);
    }

    Some(BVH::new(object_inner_nodes, object_leaf_nodes))
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log_2(x: usize) -> usize {
    num_bits::<usize>() - x.leading_zeros() as usize - 1
}

fn build(triangles: &mut Vec<NodeLeaf>) -> Vec<NodeInner> {
    let mut bounding_boxes: Vec<NodeInner> =
        Vec::with_capacity(triangles.len().next_power_of_two());

    let bvh_height = log_2(triangles.len());

    recursive_build(
        &mut bounding_boxes,
        triangles,
        0,
        0,
        0,
        triangles.len(),
        bvh_height,
    );

    bounding_boxes
}

fn recursive_build(
    bounding_boxes: &mut Vec<NodeInner>,
    triangle_params_unsorted: &mut Vec<NodeLeaf>,
    level: usize,
    branch: usize,
    start: usize,
    end: usize,
    bvh_height: usize,
) -> () {
    let mut centroid_bounds = triangle_params_unsorted[start..end].iter().fold(
        NodeInner {
            first: const_vec3!([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: const_vec3!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            prim_idx2: 0,
        },
        |acc, new| {
            // println!("adding: {:?}", new.bounds_centroid());
            // acc.add_point(&new.bounds_centroid())
            acc.merge(&new.bounds())
        },
    );

    // println!("cb: {:?}", centroid_bounds);

    let diagonal = centroid_bounds.second - centroid_bounds.first;

    let split_dimension = if diagonal.x > diagonal.y && diagonal.x > diagonal.z {
        0
    } else if diagonal.y > diagonal.z {
        1
    } else {
        2
    };

    let n_shapes = end - start;
    let mid = (start + end) / 2;

    let is_leaf: bool = centroid_bounds.first[split_dimension]
        == centroid_bounds.second[split_dimension]
        || n_shapes <= 2;

    if is_leaf {
        centroid_bounds.skip_ptr_or_prim_idx1 = start as u32;
        centroid_bounds.prim_idx2 = end as u32;
    } else {
        centroid_bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
        centroid_bounds.prim_idx2 = 0;

        triangle_params_unsorted[start..end].select_nth_unstable_by(mid - start, |a, b| {
            a.bounds_centroid()[split_dimension]
                .partial_cmp(&b.bounds_centroid()[split_dimension])
                .unwrap()
        });
    }
    bounding_boxes.push(centroid_bounds);

    if !is_leaf {
        recursive_build(
            bounding_boxes,
            triangle_params_unsorted,
            level + 1,
            branch * 2,
            start,
            mid,
            bvh_height,
        );
        recursive_build(
            bounding_boxes,
            triangle_params_unsorted,
            level + 1,
            (branch * 2) + 1,
            mid,
            end,
            bvh_height,
        );
    }
    ()
}
