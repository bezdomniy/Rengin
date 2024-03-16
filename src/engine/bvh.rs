use itertools::partition;
use std::any::Any;
static MAX_SHAPES_IN_NODE: usize = 4;

use glam::{Mat3, Vec3, Vec4};

use super::rt_primitives::ObjectParam;

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

#[derive(Debug, Default, Copy, Clone)]
pub struct UnitPrimitive {} // Sphere or cube

#[derive(Debug, Default, Copy, Clone)]
pub struct PlanePrimitive {}

pub struct Primitives(Vec<Vec<Box<dyn Bounded>>>);

impl Primitives {
    pub fn new() -> Self {
        Primitives(vec![])
    }

    #[allow(dead_code)]
    pub fn extend_from_object_params(&mut self, _object_params: &[ObjectParam]) {
        // let primitives = vec![Primitive::default()];
        // self.0.push(primitives);
        todo!()
    }

    pub fn extend_from_models(&mut self, paths: &[String]) {
        let (models, _materials): (Vec<_>, Vec<_>) = paths
            .iter()
            .map(|path| {
                log::info!("Loading {:?}", path);
                tobj::load_obj(
                    path,
                    &tobj::LoadOptions {
                        triangulate: true,
                        single_index: true,
                        ..Default::default()
                    },
                )
                .expect("Failed to OBJ load file")
            })
            .unzip();

        // TODO: take Vec<Primitives> out to have it's own constructor which is then
        //       fed in as the constructor for bvh. That way can can add other shapes to it.
        for model in models.iter() {
            self.0.push(vec![]);

            for part in model.iter() {
                // println!("{:?}",model.mesh.indices);
                let primitives_iter =
                    part.mesh
                        .indices
                        .chunks_exact(3)
                        .into_iter()
                        .map(|triangle_indices| {
                            // println!("{:?}", triangle_indices);
                            Box::new(TrianglePrimitive {
                                points: Mat3::from_cols_array_2d(&[
                                    [
                                        part.mesh.positions[3 * triangle_indices[0] as usize],
                                        part.mesh.positions[(3 * triangle_indices[0] + 1) as usize],
                                        part.mesh.positions[(3 * triangle_indices[0] + 2) as usize],
                                    ],
                                    [
                                        part.mesh.positions[3 * triangle_indices[1] as usize],
                                        part.mesh.positions[(3 * triangle_indices[1] + 1) as usize],
                                        part.mesh.positions[(3 * triangle_indices[1] + 2) as usize],
                                    ],
                                    [
                                        part.mesh.positions[3 * triangle_indices[2] as usize],
                                        part.mesh.positions[(3 * triangle_indices[2] + 1) as usize],
                                        part.mesh.positions[(3 * triangle_indices[2] + 2) as usize],
                                    ],
                                ]),
                                normals: Mat3::from_cols_array_2d(&[
                                    [
                                        part.mesh.normals[3 * triangle_indices[0] as usize],
                                        part.mesh.normals[(3 * triangle_indices[0] + 1) as usize],
                                        part.mesh.normals[(3 * triangle_indices[0] + 2) as usize],
                                    ],
                                    [
                                        part.mesh.normals[3 * triangle_indices[1] as usize],
                                        part.mesh.normals[(3 * triangle_indices[1] + 1) as usize],
                                        part.mesh.normals[(3 * triangle_indices[1] + 2) as usize],
                                    ],
                                    [
                                        part.mesh.normals[3 * triangle_indices[2] as usize],
                                        part.mesh.normals[(3 * triangle_indices[2] + 1) as usize],
                                        part.mesh.normals[(3 * triangle_indices[2] + 2) as usize],
                                    ],
                                ]),
                            }) as Box<dyn Bounded>
                        });
                self.0.last_mut().unwrap().extend(primitives_iter);
            }
        }
    }
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

    pub fn new(paths: &[String]) -> Self {
        let mut object_inner_nodes: Vec<Vec<NodeInner>> = vec![];
        let mut object_leaf_nodes: Vec<Vec<NodeLeaf>> = vec![];
        let mut object_normal_nodes: Vec<Vec<NodeNormal>> = vec![];

        let mut primitives = Primitives::new();
        primitives.extend_from_models(paths);

        // TODO: take Vec<Primitives> out to have it's own constructor which is then
        //       fed in as the constructor for bvh. That way can can add other shapes to it.
        for next_primitives in primitives.0.iter_mut() {
            let bounding_boxes = Bvh::build(next_primitives);

            let (triangles, normals): (Vec<NodeLeaf>, Vec<NodeNormal>) = next_primitives
                .iter()
                .map(|primitive| {
                    let (points, normals) = if let Some(triangle) =
                        primitive.as_any().downcast_ref::<TrianglePrimitive>()
                    {
                        (triangle.points, triangle.normals)
                    } else if primitive.as_any().downcast_ref::<UnitPrimitive>().is_some()
                        || primitive
                            .as_any()
                            .downcast_ref::<PlanePrimitive>()
                            .is_some()
                    {
                        (Mat3::default(), Mat3::default())
                    } else {
                        panic!("Unknown primitive type.")
                    };
                    (
                        NodeLeaf::new(points.to_cols_array()),
                        NodeNormal::new(normals.to_cols_array()),
                    )
                })
                .unzip();

            object_inner_nodes.push(bounding_boxes);
            object_leaf_nodes.push(triangles);
            object_normal_nodes.push(normals);
        }

        if object_inner_nodes.is_empty() {
            return Bvh::empty();
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

        Bvh {
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
        }
    }

    fn build(triangles: &mut [Box<dyn Bounded>]) -> Vec<NodeInner> {
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
        triangle_params_unsorted: &mut [Box<dyn Bounded>],
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

                    for triangle in triangle_params_unsorted.iter().take(end).skip(start) {
                        let mut b: usize = n_buckets
                            * centroid_bounds.offset(&triangle.bounds_centroid())[split_dimension]
                                .round() as usize;

                        if b == n_buckets {
                            b = n_buckets - 1;
                        };

                        buckets[b].skip_ptr_or_prim_idx1 += 1; // Using this for the count variable
                        buckets[b].merge(&triangle.bounds());
                    }

                    let mut cost = vec![0f32; n_buckets - 1];

                    for (i, c) in cost.iter_mut().enumerate().take(n_buckets - 1) {
                        let mut b0 = NodeInner::empty();
                        let mut b1 = NodeInner::empty();
                        let mut count0: u32 = 0;
                        let mut count1: u32 = 0;

                        for node in buckets.iter().take(i + 1) {
                            b0 = b1.merge(node);
                            count0 += node.skip_ptr_or_prim_idx1;
                        }

                        for node in buckets.iter().take(n_buckets).skip(i + 1) {
                            b1 = b1.merge(node);
                            count1 += node.skip_ptr_or_prim_idx1;
                        }

                        *c = 1f32
                            + (count0 as f32 * b0.surface_area()
                                + count1 as f32 * b1.surface_area())
                                / bounds.surface_area();
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
                            let mut b: usize = n_buckets
                                * centroid_bounds.offset(&n.bounds_centroid())[split_dimension]
                                    .round() as usize;
                            if b == n_buckets {
                                b = n_buckets - 1;
                            };
                            b <= min_cost_split_bucket
                        }) + start;
                    } else {
                        // println!("leaf");
                        bounds.skip_ptr_or_prim_idx1 = start as u32;
                        bounds.prim_idx2 = end as u32;
                        bounding_boxes.push(bounds);
                        return bounding_boxes.len() as u32 - 1;
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

            bounding_boxes[curr_idx].skip_ptr_or_prim_idx1 = skip_ptr + 1;

            // println!("{:?}", bounds);

            // bounds.skip_ptr_or_prim_idx1 = 2u32.pow((bvh_height - level) as u32) - 1;
            // bounds.skip_ptr_or_prim_idx1 = 1;
        }
        bounding_boxes.len() as u32 - 1
    }

    pub fn find_model_locations(&self, tag: &String) -> (u32, u32, u32) {
        // println!("### TAG {:?}", tag);
        let index = self.model_tags.iter().position(|r| r == tag).unwrap();

        (
            self.offset_inner_nodes[index],
            self.len_inner_nodes[index],
            self.offset_leaf_nodes[index],
        )
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
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Bounded for UnitPrimitive {
    fn bounds(&self) -> NodeInner {
        NodeInner {
            first: Vec3::new(-1f32, -1f32, -1f32),
            second: Vec3::new(1f32, 1f32, 1f32),
            ..Default::default()
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Bounded for PlanePrimitive {
    fn bounds(&self) -> NodeInner {
        NodeInner {
            first: Vec3::new(-f32::INFINITY, 0f32, -f32::INFINITY),
            second: Vec3::new(f32::INFINITY, 0f32, f32::INFINITY),
            ..Default::default()
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
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
