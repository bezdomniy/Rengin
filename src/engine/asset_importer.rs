use crate::engine::rt_primitives::{NodeInner, NodeLeaf, BVH};
use glam::{const_mat4, const_vec3, const_vec4, Vec4Swizzles};
use tobj;

static MAX_SHAPES_IN_NODE: usize = 4;

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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum SplitMethod {
    Middle,
    EqualCounts,
    SAH,
}

fn build(triangles: &mut Vec<NodeLeaf>) -> Vec<NodeInner> {
    let mut bounding_boxes: Vec<NodeInner> =
        Vec::with_capacity(triangles.len().next_power_of_two());

    let split_method = SplitMethod::SAH;

    recursive_build(
        &mut bounding_boxes,
        triangles,
        0,
        triangles.len(),
        split_method,
    );

    bounding_boxes
}

// TODO: make this return the skip pointer so it can bubble up
fn recursive_build(
    bounding_boxes: &mut Vec<NodeInner>,
    triangle_params_unsorted: &mut Vec<NodeLeaf>,
    start: usize,
    end: usize,
    split_method: SplitMethod,
) -> u32 {
    // println!("start end: {:?} {:?}", start, end);
    let centroid_bounds = triangle_params_unsorted[start..end].iter().fold(
        NodeInner {
            first: const_vec3!([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: const_vec3!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            prim_idx2: 0,
        },
        |acc, new| {
            // println!("adding: {:?}", new.bounds_centroid());
            acc.add_point(&new.bounds_centroid())
            // acc.merge(&new.bounds())
        },
    );

    let mut bounds = triangle_params_unsorted[start..end].iter().fold(
        NodeInner {
            first: const_vec3!([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: const_vec3!([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            prim_idx2: 0,
        },
        |acc, new| acc.merge(&new.bounds()),
    );

    // println!("cb: {:?}", centroid_bounds);

    let diagonal = centroid_bounds.diagonal();

    let split_dimension = if diagonal.x > diagonal.y && diagonal.x > diagonal.z {
        0
    } else if diagonal.y > diagonal.z {
        1
    } else {
        2
    };

    let n_shapes = end - start;
    // let mid = (start + end) / 2;

    let is_leaf: bool = centroid_bounds.first[split_dimension]
        == centroid_bounds.second[split_dimension]
        || n_shapes <= 2;

    if is_leaf {
        // println!("leaf");
        bounds.skip_ptr_or_prim_idx1 = start as u32;
        bounds.prim_idx2 = end as u32;
        bounding_boxes.push(bounds);
    } else {
        // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
        bounds.prim_idx2 = 0;

        let mut fallthrough = false;
        let mut mid = (start + end) / 2;

        if matches!(split_method, SplitMethod::EqualCounts) {
            let pmid = (centroid_bounds.first[split_dimension]
                + centroid_bounds.second[split_dimension])
                / 2f32;

            mid = triangle_params_unsorted[start..end]
                .iter_mut()
                .partition_in_place(|&n| n.bounds_centroid()[split_dimension] < pmid)
                + start;

            if mid != start && mid != end {
                fallthrough = true;
            }
        }

        if fallthrough || matches!(split_method, SplitMethod::Middle) {
            mid = (start + end) / 2;
            triangle_params_unsorted[start..end].select_nth_unstable_by(mid - start, |a, b| {
                a.bounds_centroid()[split_dimension]
                    .partial_cmp(&b.bounds_centroid()[split_dimension])
                    .unwrap()
            });
        }

        if matches!(split_method, SplitMethod::SAH) {
            if n_shapes <= 2 {
                mid = (start + end) / 2;
                triangle_params_unsorted[start..end].select_nth_unstable_by(mid - start, |a, b| {
                    a.bounds_centroid()[split_dimension]
                        .partial_cmp(&b.bounds_centroid()[split_dimension])
                        .unwrap()
                });
            } else {
                let n_buckets: usize = 12;
                let mut buckets = vec![NodeInner::empty(); n_buckets];

                for i in start..end {
                    let mut b: usize = n_buckets
                        * centroid_bounds.offset(&triangle_params_unsorted[i].bounds_centroid())
                            [split_dimension]
                            .round() as usize;

                    if b == n_buckets {
                        b = n_buckets - 1;
                    };

                    buckets[b].skip_ptr_or_prim_idx1 += 1; // Using this for the count variable
                    buckets[b].merge(&triangle_params_unsorted[i].bounds());
                }

                let mut cost = vec![0f32; n_buckets - 1];

                for i in 0..n_buckets - 1 {
                    let mut b0 = NodeInner::empty();
                    let mut b1 = NodeInner::empty();
                    let mut count0: u32 = 0;
                    let mut count1: u32 = 0;

                    for j in 0..i + 1 {
                        b0 = b1.merge(&buckets[j]);
                        count0 += buckets[j].skip_ptr_or_prim_idx1;
                    }

                    for j in i + 1..n_buckets {
                        b1 = b1.merge(&buckets[j]);
                        count1 += buckets[j].skip_ptr_or_prim_idx1;
                    }

                    cost[i] = 1f32
                        + (count0 as f32 * b0.surface_area() + count1 as f32 * b1.surface_area())
                            / bounds.surface_area();
                }

                let mut min_cost = cost[0];
                let mut min_cost_split_bucket: usize = 0;
                for i in 1..n_buckets - 1 {
                    if cost[i] < min_cost {
                        min_cost = cost[i];
                        min_cost_split_bucket = i;
                    }
                }

                let leaf_cost = n_shapes as f32;
                if n_shapes > MAX_SHAPES_IN_NODE || min_cost < leaf_cost {
                    mid = triangle_params_unsorted[start..end]
                        .iter_mut()
                        .partition_in_place(|&n| {
                            let mut b: usize = n_buckets
                                * centroid_bounds.offset(&n.bounds_centroid())[split_dimension]
                                    .round() as usize;
                            if b == n_buckets {
                                b = n_buckets - 1;
                            };
                            b <= min_cost_split_bucket
                        })
                        + start;
                } else {
                    // println!("leaf");
                    bounds.skip_ptr_or_prim_idx1 = start as u32;
                    bounds.prim_idx2 = end as u32;
                    bounding_boxes.push(bounds);
                    return bounding_boxes.len() as u32;
                }
            }
        }

        let curr_idx = bounding_boxes.len();
        bounding_boxes.push(bounds);

        recursive_build(
            bounding_boxes,
            triangle_params_unsorted,
            start,
            mid,
            split_method,
        );
        let skip_ptr = recursive_build(
            bounding_boxes,
            triangle_params_unsorted,
            mid,
            end,
            split_method,
        );

        bounding_boxes[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr + 1;

        // println!("{:?}", bounds);

        // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
        // bounds.skip_ptr_or_prim_idx1 = 1;
    }
    return bounding_boxes.len() as u32;
}
