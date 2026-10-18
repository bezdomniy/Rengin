use itertools::partition;
static MAX_SHAPES_IN_NODE: usize = 4;

use glam::{Mat3, Vec3, Vec4};

use super::obj_loader::ModelError;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeLeaf {
    pub point1: Vec3,
    pub pad1: u32,
    pub point2: Vec3,
    pub pad2: u32,
    pub point3: Vec3,
    pub pad3: u32,
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

#[derive(Debug, Default, Copy, Clone)]
pub struct TrianglePrimitive {
    pub points: Mat3,
    pub normals: Mat3,
}

#[derive(Debug, Default)]
#[repr(C)]
pub struct Bvh {
    pub inner_nodes: Vec<NodeInner>,
    pub leaf_nodes: Vec<NodeLeaf>,
    pub normal_nodes: Vec<NodeNormal>,
    pub offset_inner_nodes: Vec<u32>,
    pub len_inner_nodes: Vec<u32>,
    pub offset_leaf_nodes: Vec<u32>,
    pub model_tags: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum SplitMethod {
    Middle,
    EqualCounts,
    Sah,
}

impl Bvh {
    fn sah_bucket_index(centroid_bounds: &NodeInner, point: &Vec3, dimension: usize) -> usize {
        const N_BUCKETS: usize = 12;
        let normalized = centroid_bounds.offset(point)[dimension].clamp(0.0, 1.0);
        ((normalized * N_BUCKETS as f32) as usize).min(N_BUCKETS - 1)
    }

    pub fn empty() -> Self {
        Bvh {
            inner_nodes: vec![NodeInner::default()],
            leaf_nodes: vec![NodeLeaf::default()],
            normal_nodes: vec![NodeNormal::default()],
            offset_inner_nodes: vec![],
            len_inner_nodes: vec![],
            offset_leaf_nodes: vec![],
            model_tags: vec![],
        }
    }

    /// Loads each model in `paths` with `load` and builds one BVH per model,
    /// in order.
    pub fn new(
        paths: &[String],
        mut load: impl FnMut(&str) -> Result<Vec<TrianglePrimitive>, ModelError>,
    ) -> Result<Self, ModelError> {
        let mut object_inner_nodes: Vec<Vec<NodeInner>> = vec![];
        let mut object_leaf_nodes: Vec<Vec<NodeLeaf>> = vec![];
        let mut object_normal_nodes: Vec<Vec<NodeNormal>> = vec![];

        for path in paths {
            let mut triangles = load(path)?;
            let bounding_boxes = Bvh::build(&mut triangles);

            object_inner_nodes.push(bounding_boxes);
            object_leaf_nodes.push(
                triangles
                    .iter()
                    .map(|triangle| NodeLeaf::new(triangle.points.to_cols_array()))
                    .collect(),
            );
            object_normal_nodes.push(
                triangles
                    .iter()
                    .map(|triangle| NodeNormal::new(triangle.normals.to_cols_array()))
                    .collect(),
            );
        }

        if object_inner_nodes.is_empty() {
            return Ok(Bvh::empty());
        }

        let len_inner_nodes: Vec<u32> = object_inner_nodes
            .iter()
            .map(|next_vec| next_vec.len() as u32)
            .collect();

        let mut offset_inner_nodes: Vec<u32> = len_inner_nodes
            .iter()
            .scan(0, |acc, next_len| {
                *acc += next_len;
                Some(*acc)
            })
            .collect();

        offset_inner_nodes.pop();
        offset_inner_nodes.splice(0..0, [0u32]);

        let mut offset_leaf_nodes: Vec<u32> = object_leaf_nodes
            .iter()
            .scan(0, |acc, next_vec| {
                *acc += next_vec.len() as u32;
                Some(*acc)
            })
            .collect();

        offset_leaf_nodes.pop();
        offset_leaf_nodes.splice(0..0, [0u32]);

        // let n_objects = object_inner_nodes.len() as u32;

        Ok(Bvh {
            inner_nodes: object_inner_nodes.into_iter().flatten().collect::<Vec<_>>(),
            leaf_nodes: object_leaf_nodes.into_iter().flatten().collect::<Vec<_>>(),
            normal_nodes: object_normal_nodes
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
            offset_inner_nodes,
            len_inner_nodes,
            offset_leaf_nodes,
            model_tags: paths.to_vec(),
        })
    }

    fn build(triangles: &mut [TrianglePrimitive]) -> Vec<NodeInner> {
        let mut bounding_boxes: Vec<NodeInner> =
            Vec::with_capacity(triangles.len().next_power_of_two());

        let split_method = SplitMethod::Sah;

        Bvh::recursive_build(
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
        triangle_params_unsorted: &mut [TrianglePrimitive],
        start: usize,
        end: usize,
        split_method: SplitMethod,
    ) -> u32 {
        // println!("start end: {:?} {:?}", start, end);
        let centroid_bounds = triangle_params_unsorted[start..end].iter().fold(
            NodeInner {
                first: Vec3::from_array([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
                skip_ptr_or_prim_idx1: 0,
                second: Vec3::from_array([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
                prim_idx2: 0,
            },
            |acc, new| acc.add_point(&new.bounds_centroid()),
        );

        let mut bounds = triangle_params_unsorted[start..end].iter().fold(
            NodeInner {
                first: Vec3::from_array([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
                skip_ptr_or_prim_idx1: 0,
                second: Vec3::from_array([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
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

            if matches!(split_method, SplitMethod::Middle) {
                let pmid = (centroid_bounds.first[split_dimension]
                    + centroid_bounds.second[split_dimension])
                    / 2f32;

                mid = partition(triangle_params_unsorted[start..end].iter_mut(), |n| {
                    n.bounds_centroid()[split_dimension] < pmid
                }) + start;

                if mid != start && mid != end {
                    fallthrough = true;
                }
            }

            if fallthrough || matches!(split_method, SplitMethod::EqualCounts) {
                mid = (start + end) / 2;
                triangle_params_unsorted[start..end].select_nth_unstable_by(mid - start, |a, b| {
                    a.bounds_centroid()[split_dimension]
                        .partial_cmp(&b.bounds_centroid()[split_dimension])
                        .unwrap()
                });
            }

            if matches!(split_method, SplitMethod::Sah) {
                if n_shapes <= 2 {
                    mid = (start + end) / 2;
                    triangle_params_unsorted[start..end].select_nth_unstable_by(
                        mid - start,
                        |a, b| {
                            a.bounds_centroid()[split_dimension]
                                .partial_cmp(&b.bounds_centroid()[split_dimension])
                                .unwrap()
                        },
                    );
                    // if start != end - 1
                    //     && triangle_params_unsorted[end - 1].bounds_centroid()[split_dimension]
                    //         < triangle_params_unsorted[start].bounds_centroid()[split_dimension]
                    // {
                    //     triangle_params_unsorted.swap(start, end - 1);
                    // }
                } else {
                    let n_buckets: usize = 12;
                    let mut buckets = vec![NodeInner::empty(); n_buckets];
                    let mut bucket_counts = vec![0u32; n_buckets];

                    for triangle in triangle_params_unsorted.iter().take(end).skip(start) {
                        let b = Self::sah_bucket_index(
                            &centroid_bounds,
                            &triangle.bounds_centroid(),
                            split_dimension,
                        );
                        bucket_counts[b] += 1;
                        buckets[b] = buckets[b].merge(&triangle.bounds());
                    }

                    let mut cost = vec![0f32; n_buckets - 1];

                    for (i, c) in cost.iter_mut().enumerate().take(n_buckets - 1) {
                        let mut b0 = NodeInner::empty();
                        let mut b1 = NodeInner::empty();
                        let mut count0: u32 = 0;
                        let mut count1: u32 = 0;

                        for (node, count) in buckets
                            .iter()
                            .zip(bucket_counts.iter())
                            .take(i + 1)
                        {
                            b0 = b0.merge(node);
                            count0 += count;
                        }

                        for (node, count) in buckets
                            .iter()
                            .zip(bucket_counts.iter())
                            .take(n_buckets)
                            .skip(i + 1)
                        {
                            b1 = b1.merge(node);
                            count1 += count;
                        }

                        let cost0 = if count0 > 0 {
                            count0 as f32 * b0.surface_area()
                        } else {
                            0.0
                        };
                        let cost1 = if count1 > 0 {
                            count1 as f32 * b1.surface_area()
                        } else {
                            0.0
                        };
                        *c = 1f32 + (cost0 + cost1) / bounds.surface_area();
                    }

                    let mut min_cost = cost[0];
                    let mut min_cost_split_bucket: usize = 0;

                    for (i, c) in cost.iter().enumerate().take(n_buckets - 1).skip(1) {
                        if *c < min_cost {
                            min_cost = *c;
                            min_cost_split_bucket = i;
                        }
                    }

                    let leaf_cost = n_shapes as f32;
                    if n_shapes > MAX_SHAPES_IN_NODE || min_cost < leaf_cost {
                        mid = partition(triangle_params_unsorted[start..end].iter_mut(), |n| {
                            let b = Self::sah_bucket_index(
                                &centroid_bounds,
                                &n.bounds_centroid(),
                                split_dimension,
                            );
                            b <= min_cost_split_bucket
                        }) + start;

                        // A bucket boundary can still be empty for clustered
                        // centroids. Fall back to an equal-count split rather
                        // than creating an empty recursive range.
                        if mid == start || mid == end {
                            mid = (start + end) / 2;
                            triangle_params_unsorted[start..end].select_nth_unstable_by(
                                mid - start,
                                |a, b| {
                                    a.bounds_centroid()[split_dimension]
                                        .partial_cmp(&b.bounds_centroid()[split_dimension])
                                        .unwrap()
                                },
                            );
                        }
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

            Bvh::recursive_build(
                bounding_boxes,
                triangle_params_unsorted,
                start,
                mid,
                split_method,
            );
            let skip_ptr = Bvh::recursive_build(
                bounding_boxes,
                triangle_params_unsorted,
                mid,
                end,
                split_method,
            );

            bounding_boxes[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr;

            // println!("{:?}", bounds);

            // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
            // bounds.skip_ptr_or_prim_idx1 = 1;
        }
        bounding_boxes.len() as u32
    }

    /// Returns the inner-node offset, inner-node count and leaf offset of the
    /// model at `index` in the `paths` passed to [`Bvh::new`].
    pub fn model_location(&self, index: usize) -> (u32, u32, u32) {
        (
            self.offset_inner_nodes[index],
            self.len_inner_nodes[index],
            self.offset_leaf_nodes[index],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_at(x: f32) -> TrianglePrimitive {
        TrianglePrimitive {
            points: Mat3::from_cols(
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(x + 0.1, 1.0, 0.0),
                Vec3::new(x + 0.1, 0.0, 1.0),
            ),
            normals: Mat3::ZERO,
        }
    }

    #[test]
    fn sah_root_split_is_not_pathologically_unbalanced() {
        let mut primitives: Vec<TrianglePrimitive> =
            (0..96).map(|index| triangle_at(index as f32)).collect();
        let nodes = Bvh::build(&mut primitives);
        let right_subtree_start = nodes[1].skip_ptr_or_prim_idx1 as usize;
        let left_primitive_count: usize = nodes[1..right_subtree_start]
            .iter()
            .filter(|node| node.prim_idx2 > 0)
            .map(|node| (node.prim_idx2 - node.skip_ptr_or_prim_idx1) as usize)
            .sum();

        assert!(
            (24..=72).contains(&left_primitive_count),
            "SAH root split placed {left_primitive_count}/96 primitives on the left"
        );
    }

    #[test]
    fn sah_leaf_ranges_cover_every_primitive_once() {
        let primitive_count = 97usize;
        let mut primitives: Vec<TrianglePrimitive> = (0..primitive_count)
            .map(|index| triangle_at((index * 17 % primitive_count) as f32))
            .collect();
        let nodes = Bvh::build(&mut primitives);
        let mut coverage = vec![0u8; primitive_count];

        for node in nodes.iter().filter(|node| node.prim_idx2 > 0) {
            for primitive_index in node.skip_ptr_or_prim_idx1..node.prim_idx2 {
                coverage[primitive_index as usize] += 1;
            }
        }

        assert!(coverage.iter().all(|count| *count == 1));
    }
}

impl NodeLeaf {
    pub fn new(v: [f32; 9]) -> Self {
        NodeLeaf {
            point1: Vec3::from_array([v[0], v[1], v[2]]),
            point2: Vec3::from_array([v[3], v[4], v[5]]),
            point3: Vec3::from_array([v[6], v[7], v[8]]),
            pad1: 0,
            pad2: 0,
            pad3: 0,
        }
    }
}

impl NodeNormal {
    pub fn new(v: [f32; 9]) -> Self {
        NodeNormal {
            normals: [
                Vec4::from_array([v[0], v[1], v[2], 0f32]),
                Vec4::from_array([v[3], v[4], v[5], 0f32]),
                Vec4::from_array([v[6], v[7], v[8], 0f32]),
            ],
        }
    }
}

trait Bounded {
    fn bounds(&self) -> NodeInner;
    fn bounds_centroid(&self) -> Vec3 {
        let bounds = self.bounds();
        0.5 * bounds.first + 0.5 * bounds.second
    }
}

impl Bounded for TrianglePrimitive {
    fn bounds(&self) -> NodeInner {
        self.points
            .to_cols_array_2d()
            .iter()
            .fold(NodeInner::empty(), |aabb, p| {
                aabb.add_point(&Vec3::from_array(*p))
            })
    }
}

impl NodeInner {
    pub fn empty() -> Self {
        NodeInner {
            first: Vec3::from_array([f32::INFINITY, f32::INFINITY, f32::INFINITY]),
            skip_ptr_or_prim_idx1: 0,
            second: Vec3::from_array([f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY]),
            prim_idx2: 0,
        }
    }

    pub fn merge(&self, other: &NodeInner) -> Self {
        let min: Vec3 = Vec3::from_array([
            f32::min(self.first.x, other.first.x),
            f32::min(self.first.y, other.first.y),
            f32::min(self.first.z, other.first.z),
        ]);

        let max: Vec3 = Vec3::from_array([
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
        2_f32 * (d.x * d.y + d.x * d.z + d.y * d.z)
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
        o
    }
}
