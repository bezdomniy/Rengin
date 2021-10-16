use crate::engine::rt_primitives::{BoundingBox, BoundingBoxes, NodeInner, NodeLeaf, BVH};
use glam::{const_mat4, const_vec4, Vec4Swizzles};
use tobj;

pub fn import_obj(path: &str) -> Option<Vec<BVH>> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            // triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .expect("Failed to OBJ load file");

    let mut ret: Vec<BVH> = vec![];

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

        let (bounding_boxes, leaf_nodes) = build(&mut triangles);
        let inner_nodes = flatten(&bounding_boxes);

        ret.push(BVH::new(inner_nodes, leaf_nodes));
    }

    Some(ret)
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log_2(x: usize) -> usize {
    num_bits::<usize>() - x.leading_zeros() as usize - 1
}

// TODO: placeholder for now
fn flatten(bounding_boxes: &BoundingBoxes) -> Vec<NodeInner> {
    bounding_boxes
        .items
        .iter()
        .map(|bounding_box| NodeInner {
            first: bounding_box.first.xyz(),
            skip_ptr_or_prim_idx1: 1,
            second: bounding_box.second.xyz(),
            prim_idx2: 1,
        })
        .collect()
}

fn build(triangles: &mut Vec<NodeLeaf>) -> (BoundingBoxes, Vec<NodeLeaf>) {
    let mut bounding_boxes: BoundingBoxes = BoundingBoxes::new(triangles.len().next_power_of_two());

    let bvh_height = log_2(triangles.len());

    let mut primitives: Vec<NodeLeaf> = Vec::with_capacity(triangles.len().next_power_of_two());

    recursive_build(
        &mut bounding_boxes,
        &mut primitives,
        triangles,
        0,
        0,
        0,
        triangles.len(),
        bvh_height,
    );

    (bounding_boxes, primitives)
}

fn recursive_build(
    bounding_boxes: &mut BoundingBoxes,
    primitives: &mut Vec<NodeLeaf>,
    triangle_params_unsorted: &mut Vec<NodeLeaf>,
    level: usize,
    branch: usize,
    start: usize,
    end: usize,
    bvh_height: usize,
) -> () {
    let centroid_bounds = triangle_params_unsorted[start..end].iter().fold(
        BoundingBox {
            first: const_vec4!([f32::INFINITY, f32::INFINITY, f32::INFINITY, 1.0]),
            second: const_vec4!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, 1.0]),
        },
        |acc, new| acc.merge(&new.bounds()),
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
        bounding_boxes[node] = centroid_bounds;

        let n_shapes = end - start;

        if n_shapes > 2 {
            recursive_build(
                bounding_boxes,
                primitives,
                triangle_params_unsorted,
                level + 1,
                branch * 2,
                start,
                mid,
                bvh_height,
            );
            recursive_build(
                bounding_boxes,
                primitives,
                triangle_params_unsorted,
                level + 1,
                (branch * 2) + 1,
                mid,
                end,
                bvh_height,
            );
        } else {
            primitives.push(triangle_params_unsorted[start]);

            if n_shapes == 2 {
                primitives.push(triangle_params_unsorted[start + 1]);
            } else {
                primitives.push(NodeLeaf::empty());
            }
            if level < bvh_height {
                // println!("level {}, bvh_height {}", level, bvh_height);
                let dummy_node = 2usize.pow((level + 1) as u32) + (branch * 2) - 1;

                // tlas[dummy_node] = BoundingBox::total();
                // tlas.swap(dummy_node, node);
                // println!("adding dummy: {:?}",centroid_bounds);
                bounding_boxes[dummy_node] = centroid_bounds;
                bounding_boxes[dummy_node + 1] = BoundingBox::empty();

                primitives.push(NodeLeaf::empty());
                primitives.push(NodeLeaf::empty());
            }
            ()
        }
    }
    ()
}
