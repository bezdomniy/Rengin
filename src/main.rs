#![feature(iter_partition_in_place)]

mod engine;
mod renderer;
mod shaders;

// use image::io::Reader;
// use renderdoc::RenderDoc;
use wgpu::util::{DeviceExt, StagingBelt};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, RenderPipeline, Sampler, ShaderModule,
    Texture,
};
use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};

// TODO: set $env:RUST_LOG = 'WARN' when running

// use image::codecs::pnm;

use std::borrow::Cow;
// use std::pin::Pin;
// use std::task::{Context, Poll};
// use std::thread;
use std::time::{Duration, Instant};

use std::collections::HashMap;
// use std::borrow::Cow;
// use std::collections::HashMap;
use std::env;
// use std::fs::File;
// use std::io::Write;
use std::mem;

// use core::num;

// use wgpu::BufferUsage;
use glam::{Mat4, Vec3, Vec4};

use engine::asset_importer::import_objs;
use engine::scene_importer::Scene;

use engine::rt_primitives::{Camera, NodeInner, NodeLeaf, NodeNormal, ObjectParams, BVH, UBO};

use crate::engine::rt_primitives::PtMaterial;
use crate::renderer::wgpu_utils::RenginWgpu;

static WIDTH: u32 = 800;
static HEIGHT: u32 = 600;
static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;
static RAYS_PER_PIXEL: u32 = 16;

//TODO: try doing passes over parts of the image instead of whole at a time
//      that way you can maintain framerate

struct GameState {
    pub camera_angle_y: f32,
    pub camera_angle_xz: f32,
    pub camera_dist: f32,
    pub camera_centre: [f32; 3],
    pub camera_up: [f32; 3],
}

struct RenderApp {
    renderer: RenginWgpu,
    shaders: Option<HashMap<&'static str, ShaderModule>>,
    compute_pipeline: Option<ComputePipeline>,
    compute_bind_group_layout: Option<BindGroupLayout>,
    compute_bind_group: Option<BindGroup>,
    render_pipeline: Option<RenderPipeline>,
    render_bind_group_layout: Option<BindGroupLayout>,
    render_bind_group: Option<BindGroup>,
    sampler: Option<Sampler>,
    scene: Option<Scene>,
    buffers: Option<HashMap<&'static str, Buffer>>,
    texture: Option<Texture>,
    object_params: Option<Vec<ObjectParams>>,
    ubo: Option<UBO>,
    game_state: Option<GameState>,
}

impl RenderApp {
    pub fn new(event_loop: &EventLoop<()>, continous_motion: bool) -> Self {
        let renderer = futures::executor::block_on(RenginWgpu::new(
            WIDTH,
            HEIGHT,
            WORKGROUP_SIZE,
            event_loop,
            continous_motion,
        ));

        Self {
            renderer,
            shaders: None,
            compute_pipeline: None,
            compute_bind_group_layout: None,
            compute_bind_group: None,
            render_pipeline: None,
            render_bind_group_layout: None,
            render_bind_group: None,
            sampler: None,
            scene: None,
            buffers: None,
            texture: None,
            object_params: None,
            ubo: None,
            game_state: None,
        }
        // RenderApp::init(&self, model_path)
    }

    fn init(&mut self, scene_path: &str) {
        let mut now = Instant::now();
        log::info!("Loading models...");
        // self.objects = import_objs(vec![model_path, model_path]);
        // self.objects = import_objs(vec![
        //     "./assets/models/suzanne.obj".to_string(),
        //     model_path.to_string(),
        //     "./assets/models/lucy.obj".to_string(),
        // ]);

        self.scene = Some(Scene::new(scene_path));
        log::info!(
            "Finished loading models in {} millis.",
            now.elapsed().as_millis()
        );

        // let transform1 = Mat4::from_cols_array_2d(&[
        //     [1f32, 0f32, 0f32, 2f32],
        //     [0f32, 1f32, 0f32, 0f32],
        //     [0f32, 0f32, 1f32, 0f32],
        //     [0f32, 0f32, 0f32, 1f32],
        // ])
        // .transpose();

        let rotate90_x = Mat4::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 1.5708);
        let rotate90_z = Mat4::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.5708);

        let transform0 = Mat4::from_translation(Vec3::new(-2f32, 0f32, 0f32));
        let transform1 = Mat4::from_translation(Vec3::new(3f32, -1f32, 0f32));
        let transform2 = Mat4::from_scale(Vec3::new(0.005, 0.005, 0.005));
        let transform3 = Mat4::from_translation(Vec3::new(-3f32, 3f32, 1f32))
            * Mat4::from_scale(Vec3::new(0.2, 1.0, 1.0));
        let transform4 = Mat4::from_translation(Vec3::new(0f32, -1.5f32, 0f32));
        let transform5 = Mat4::from_translation(Vec3::new(0f32, 0f32, -3f32)) * rotate90_x;
        let transform6 = Mat4::from_translation(Vec3::new(10f32, 0f32, 0f32)) * rotate90_z;
        let transform7 = Mat4::from_translation(Vec3::new(-10f32, 0f32, 0f32)) * rotate90_z;
        let transform8 = Mat4::from_translation(Vec3::new(0f32, 10f32, 0f32));
        // let transform4 = Mat4::IDENTITY;
        // let transform2 = Mat4::IDENTITY;

        let object_param0 = ObjectParams::new(
            transform0,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.831, 0.537, 0.214, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_inner_nodes
                .get(0)
                .unwrap(),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_leaf_nodes
                .get(0)
                .unwrap(),
            0,
        );

        let object_param1 = ObjectParams::new(
            transform1,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.537, 0.831, 0.914, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_inner_nodes
                .get(1)
                .unwrap(),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_leaf_nodes
                .get(1)
                .unwrap(),
            0,
        );

        let object_param2 = ObjectParams::new(
            // transform1,
            transform2,
            PtMaterial::new(
                Vec4::new(0.837, 0.131, 0.114, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                1.0,
                0.0,
                0.0,
            ),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_inner_nodes
                .get(2)
                .unwrap(),
            *self
                .scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_leaf_nodes
                .get(2)
                .unwrap(),
            0,
        );

        let object_param3 = ObjectParams::new(
            transform3,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.831, 0.537, 0.214, 1.0),
                Vec4::new(7.0, 7.0, 7.0, 7.0),
                0.1,
                0.7,
                0.3,
                200.0,
                1.0,
                1.0,
                1.5,
            ),
            1,
            0,
            1,
        );

        let object_param4 = ObjectParams::new(
            transform4,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.831, 0.537, 0.214, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            2,
            0,
            0,
        );

        let object_param5 = ObjectParams::new(
            transform5,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.231, 0.537, 0.831, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            2,
            0,
            0,
        );

        let object_param6 = ObjectParams::new(
            transform6,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.231, 0.537, 0.831, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            2,
            0,
            0,
        );

        let object_param7 = ObjectParams::new(
            transform7,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.231, 0.537, 0.831, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            2,
            0,
            0,
        );

        let object_param8 = ObjectParams::new(
            transform8,
            // inverse_transform: Mat4::from_scale(Vec3::new(0.004, 0.004, 0.004)).inverse(),
            PtMaterial::new(
                Vec4::new(0.231, 0.537, 0.831, 1.0),
                Vec4::new(0.0, 0.0, 0.0, 0.0),
                0.1,
                0.7,
                0.3,
                200.0,
                0.0,
                0.0,
                0.0,
            ),
            2,
            0,
            0,
        );

        self.object_params = Some(vec![
            object_param0,
            object_param1,
            object_param2,
            object_param3,
            object_param4,
            object_param5,
            object_param6,
            object_param7,
            object_param8,
        ]);

        let n_primitives = 6;

        // let x = &mut self.objects.as_ref().unwrap().n_objects;
        // *x += 1;

        // log::info!("tlas:{:?}, blas{:?}", dragon_tlas.len(), dragon_blas.len());
        // log::info!(
        //     "tlas:{:?}, blas{:?}",
        //     mem::size_of::<NodeTLAS>(),
        //     mem::size_of::<NodeBLAS>()
        // );

        // let camera_position = [-4f32, 2f32, -3f32];
        // let camera_centre = [0f32, 1f32, 0f32];

        let camera_angle_y = 0.0;
        let camera_angle_xz = 0.0;
        let camera_dist = 9.0;
        // TODO: find out why models are appearing upside-down
        let camera_centre = [0.0, 1.0, 0.0];
        let camera_up = [0.0, -1.0, 0.0];

        self.game_state = Some(GameState {
            camera_angle_xz,
            camera_angle_y,
            camera_dist,
            camera_centre,
            camera_up,
        });

        let camera_position = [
            camera_angle_xz.cos() * camera_angle_y.sin() * camera_dist,
            camera_angle_xz.sin() * camera_dist + camera_centre[1],
            camera_angle_xz.cos() * camera_angle_y.cos() * camera_dist,
        ];

        println!("{} {}", camera_angle_y, camera_angle_xz);

        let camera = Camera::new(
            camera_position,
            camera_centre,
            camera_up,
            WIDTH as u32,
            HEIGHT as u32,
            std::f32::consts::FRAC_PI_3,
            // 1.0472f32,
        );

        self.ubo = Some(UBO::new(
            [-4f32, 2f32, 3f32, 1f32],
            self.scene
                .as_ref()
                .unwrap()
                .bvh
                .as_ref()
                .unwrap()
                .len_inner_nodes
                .len() as u32
                + n_primitives,
            (RAYS_PER_PIXEL as f32).sqrt() as u32,
            camera,
        ));

        println!("ubo: {:?}", self.ubo);

        // log::info!("ubo:{:?},", ubo);
        now = Instant::now();
        log::info!("Building shaders...");
        self.shaders = Some(self.create_shaders());
        log::info!(
            "Finshed building shaders in {} millis",
            now.elapsed().as_millis()
        );

        // TODO: update texture size on resize
        let texture_extent = wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        };

        // The render pipeline renders data into this texture
        self.texture = Some(
            self.renderer
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    size: texture_extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    label: None,
                }),
        );

        self.buffers = Some(self.create_buffers());
        self.create_pipelines();
    }

    fn create_shaders(&mut self) -> HashMap<&'static str, ShaderModule> {
        let cs_module = self
            .renderer
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    // "shaders/pathtracer.wgsl"
                    "shaders/raytracer.wgsl"
                ))),
            });

        let vt_module = self
            .renderer
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "shaders/raytracer.vert.wgsl"
                ))),
            });

        let fg_module = self
            .renderer
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "shaders/raytracer.frag.wgsl"
                ))),
            });

        let mut shaders: HashMap<&'static str, ShaderModule> = HashMap::new();
        shaders.insert("comp", cs_module);
        shaders.insert("vert", vt_module);
        shaders.insert("frag", fg_module);

        shaders
    }

    fn create_buffers(&mut self) -> HashMap<&'static str, Buffer> {
        let now = Instant::now();
        log::info!("Creating buffers...");

        // let bvh = self.objects.as_ref().unwrap().get(0).unwrap();

        // for node in dragon_blas {
        //     println!("{:?}", node.points);
        // }
        // for node in dragon_tlas {
        //     println!("{:?}", node);
        // }

        let buf_ubo = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UBO Buffer"),
                contents: bytemuck::bytes_of(&self.ubo.unwrap()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let buf_tlas = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TLAS storage Buffer"),
                contents: bytemuck::cast_slice(
                    &self
                        .scene
                        .as_ref()
                        .unwrap()
                        .bvh
                        .as_ref()
                        .unwrap()
                        .inner_nodes,
                ),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_blas = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BLAS storage Buffer"),
                contents: bytemuck::cast_slice(
                    &self
                        .scene
                        .as_ref()
                        .unwrap()
                        .bvh
                        .as_ref()
                        .unwrap()
                        .leaf_nodes,
                ),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_normals =
            self.renderer
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("normals storage Buffer"),
                    contents: bytemuck::cast_slice(
                        &self
                            .scene
                            .as_ref()
                            .unwrap()
                            .bvh
                            .as_ref()
                            .unwrap()
                            .normal_nodes,
                    ),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let buf_op = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material storage Buffer"),
                contents: bytemuck::cast_slice(&self.object_params.as_ref().unwrap()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // let rays = ..self.render.camera

        // let buf_rays = self
        //     .renderer
        //     .device
        //     .create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //         label: Some("Ray Buffer"),
        //         contents: bytemuck::cast_slice(&self.rays.as_ref().unwrap()),
        //         usage: wgpu::BufferUsages::STORAGE,
        //     });

        log::info!(
            "Finshed loading buffers in {} millis",
            now.elapsed().as_millis()
        );

        let mut buffers: HashMap<&'static str, Buffer> = HashMap::new();
        buffers.insert("ubo", buf_ubo);
        buffers.insert("tlas", buf_tlas);
        buffers.insert("blas", buf_blas);
        buffers.insert("normals", buf_normals);
        buffers.insert("object_params", buf_op);

        buffers
    }

    fn create_pipelines(&mut self) {
        self.compute_bind_group_layout = Some(
            self.renderer
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::ReadWrite,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(mem::size_of::<UBO>() as _),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    (self
                                        .scene
                                        .as_ref()
                                        .unwrap()
                                        .bvh
                                        .as_ref()
                                        .unwrap()
                                        .inner_nodes
                                        .len()
                                        * mem::size_of::<NodeInner>())
                                        as _,
                                ),
                                // min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    (self
                                        .scene
                                        .as_ref()
                                        .unwrap()
                                        .bvh
                                        .as_ref()
                                        .unwrap()
                                        .leaf_nodes
                                        .len()
                                        * mem::size_of::<NodeLeaf>())
                                        as _,
                                ),
                                // min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    (self
                                        .scene
                                        .as_ref()
                                        .unwrap()
                                        .bvh
                                        .as_ref()
                                        .unwrap()
                                        .normal_nodes
                                        .len()
                                        * mem::size_of::<NodeNormal>())
                                        as _,
                                ),
                                // min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                // min_binding_size: None,
                                min_binding_size: wgpu::BufferSize::new(
                                    mem::size_of::<ObjectParams>() as _,
                                ),
                            },
                            count: None,
                        },
                    ],
                    label: None,
                }),
        );
        let compute_pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute"),
                    bind_group_layouts: &[self.compute_bind_group_layout.as_ref().unwrap()],
                    push_constant_ranges: &[],
                });

        // create render pipeline

        self.sampler = Some(
            self.renderer
                .device
                .create_sampler(&wgpu::SamplerDescriptor::default()),
        );

        self.render_bind_group_layout = Some(self.renderer.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: core::num::NonZeroU32::new(1),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            },
        ));

        let texture_view = self
            .texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_bind_group = Some(self.renderer.device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(self.sampler.as_ref().unwrap()),
                    },
                ],
                layout: self.render_bind_group_layout.as_ref().unwrap(),
                label: Some("bind group"),
            },
        ));

        let render_pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render"),
                    bind_group_layouts: &[self.render_bind_group_layout.as_ref().unwrap()],
                    push_constant_ranges: &[],
                });

        self.render_pipeline = Some(self.renderer.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&render_pipeline_layout),
                multiview: None,
                vertex: wgpu::VertexState {
                    module: self.shaders.as_ref().unwrap().get("vert").as_ref().unwrap(),
                    entry_point: "main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: self.shaders.as_ref().unwrap().get("frag").as_ref().unwrap(),
                    entry_point: "main",
                    targets: &[wgpu::ColorTargetState {
                        format: self.renderer.config.format,
                        // TODO: change subpixel to blending, rather than doing it in shader
                        blend: Some(wgpu::BlendState::REPLACE),
                        // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }],
                    // targets: &[self.renderer.config.format.into()],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
            },
        ));

        self.compute_pipeline = Some(self.renderer.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: self.shaders.as_ref().unwrap().get("comp").as_ref().unwrap(),
                entry_point: "main",
            },
        ));

        self.compute_bind_group = Some(
            self.renderer
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: self.compute_bind_group_layout.as_ref().unwrap(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self
                                .buffers
                                .as_ref()
                                .unwrap()
                                .get("ubo")
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self
                                .buffers
                                .as_ref()
                                .unwrap()
                                .get("tlas")
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self
                                .buffers
                                .as_ref()
                                .unwrap()
                                .get("blas")
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self
                                .buffers
                                .as_ref()
                                .unwrap()
                                .get("normals")
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self
                                .buffers
                                .as_ref()
                                .unwrap()
                                .get("object_params")
                                .as_ref()
                                .unwrap()
                                .as_entire_binding(),
                        },
                    ],
                }),
        );
    }

    fn update_device_event(
        &mut self,
        event: DeviceEvent,
        left_mouse_down: &mut bool,
        something_changed: &mut bool,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                // println!("x:{}, y:{}", position.x, position.y);
                if *left_mouse_down {
                    let game_state: &mut GameState = self.game_state.as_mut().unwrap();
                    // println!(
                    //     "{} {}",
                    //     game_state.camera_angle_y, game_state.camera_angle_xz
                    // );

                    game_state.camera_angle_y += delta.0 as f32;
                    game_state.camera_angle_xz += delta.1 as f32;

                    let norm_x = game_state.camera_angle_y / self.renderer.config.width as f32;
                    let norm_y = game_state.camera_angle_xz / self.renderer.config.height as f32;
                    let angle_y = norm_x * 5.0;
                    let angle_xz = -norm_y * 2.0;

                    let new_position = [
                        angle_xz.cos() * angle_y.sin() * game_state.camera_dist,
                        angle_xz.sin() * game_state.camera_dist + game_state.camera_centre[1],
                        angle_xz.cos() * angle_y.cos() * game_state.camera_dist,
                    ];

                    if let Some(ref mut ubo) = self.ubo {
                        // no reference before Some
                        ubo.camera.update_position(
                            new_position,
                            game_state.camera_centre,
                            game_state.camera_up,
                        );
                        ubo.subpixel_idx = 0;
                        ubo.update_random_seed();
                        *something_changed = true;
                    }
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    // println!("{} {}", x, y);
                    let game_state: &mut GameState = self.game_state.as_mut().unwrap();
                    game_state.camera_dist -= (y as f32) / 3.;

                    let norm_x = game_state.camera_angle_y / self.renderer.config.width as f32;
                    let norm_y = game_state.camera_angle_xz / self.renderer.config.height as f32;
                    let angle_y = norm_x * 5.0;
                    let angle_xz = -norm_y * 2.0;

                    let new_position = [
                        angle_xz.cos() * angle_y.sin() * game_state.camera_dist,
                        angle_xz.sin() * game_state.camera_dist + game_state.camera_centre[1],
                        angle_xz.cos() * angle_y.cos() * game_state.camera_dist,
                    ];

                    if let Some(ref mut ubo) = self.ubo {
                        // no reference before Some
                        ubo.camera.update_position(
                            new_position,
                            game_state.camera_centre,
                            game_state.camera_up,
                        );
                        ubo.subpixel_idx = 0;
                        ubo.update_random_seed();
                        *something_changed = true;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn update_window_event(&mut self, event: WindowEvent, left_mouse_down: &mut bool) {
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                *left_mouse_down = true;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                *left_mouse_down = false;
            }
            _ => {}
        }
    }

    pub fn update(&mut self) {
        self.renderer.queue.write_buffer(
            self.buffers.as_ref().unwrap().get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.ubo.unwrap()),
        );
    }

    pub fn render(mut self, event_loop: EventLoop<()>) {
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;
        let mut something_changed = false;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::MainEventsCleared => {
                    // // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                    //     let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    //     let time_since_last_frame = last_update_inst.elapsed();

                    //     if (something_changed
                    //         || self.ubo.as_ref().unwrap().subpixel_idx < RAYS_PER_PIXEL)
                    //         && time_since_last_frame >= target_frametime
                    //         && (!left_mouse_down || self.renderer.continous_motion)
                    //     {
                    //         println!("Drawing ray index: {}", self.ubo.unwrap().subpixel_idx);

                    //         self.renderer.window.request_redraw();

                    //         // if let Some(ref mut x) = self.ubo {
                    //         //     x.subpixel_idx += 1;
                    //         // }

                    //         println!("render time: {:?}", time_since_last_frame);
                    //         something_changed = false;

                    //         last_update_inst = Instant::now();
                    //     } else {
                    //         *control_flow = ControlFlow::WaitUntil(
                    //             Instant::now() + target_frametime - time_since_last_frame,
                    //         );
                    //     }
                }
                Event::RedrawEventsCleared => {
                    let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    let time_since_last_frame = last_update_inst.elapsed();

                    if (something_changed
                        || self.ubo.as_ref().unwrap().subpixel_idx < RAYS_PER_PIXEL)
                        && time_since_last_frame >= target_frametime
                        && (!left_mouse_down || self.renderer.continous_motion)
                    {
                        println!("Drawing ray index: {}", self.ubo.unwrap().subpixel_idx);

                        // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                        // self.renderer.instance.poll_all(true);
                        self.renderer.device.poll(wgpu::Maintain::Wait);

                        self.renderer.window.request_redraw();

                        // if let Some(ref mut x) = self.ubo {
                        //     x.subpixel_idx += 1;
                        // }

                        println!("render time: {:?}", time_since_last_frame);
                        something_changed = false;

                        last_update_inst = Instant::now();
                    } else {
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::Resized(size)
                        | WindowEvent::ScaleFactorChanged {
                            new_inner_size: &mut size,
                            ..
                        },
                    ..
                } => {
                    // println!("p: {} {}", size.width, size.height);
                    // Reconfigure the surface with the new size
                    self.renderer.config.width = size.width.max(1);
                    self.renderer.config.height = size.height.max(1);

                    let logical_size: LogicalSize<u32> =
                        winit::dpi::PhysicalSize::new(size.width, size.height)
                            .to_logical(self.renderer.scale_factor);

                    // println!("l: {} {}", logical_size.width, logical_size.height);

                    let texture_extent = wgpu::Extent3d {
                        width: logical_size.width,
                        height: logical_size.height,
                        depth_or_array_layers: 1,
                    };

                    // The render pipeline renders data into this texture
                    self.texture = Some(self.renderer.device.create_texture(
                        &wgpu::TextureDescriptor {
                            size: texture_extent,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            usage: wgpu::TextureUsages::STORAGE_BINDING
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                            label: None,
                        },
                    ));

                    let texture_view = self
                        .texture
                        .as_ref()
                        .unwrap()
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    self.compute_bind_group = Some(
                        self.renderer
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: self.compute_bind_group_layout.as_ref().unwrap(),
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(&texture_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: self
                                            .buffers
                                            .as_ref()
                                            .unwrap()
                                            .get("ubo")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: self
                                            .buffers
                                            .as_ref()
                                            .unwrap()
                                            .get("tlas")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: self
                                            .buffers
                                            .as_ref()
                                            .unwrap()
                                            .get("blas")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: self
                                            .buffers
                                            .as_ref()
                                            .unwrap()
                                            .get("normals")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 5,
                                        resource: self
                                            .buffers
                                            .as_ref()
                                            .unwrap()
                                            .get("object_params")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                ],
                            }),
                    );

                    self.render_bind_group = Some(self.renderer.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&texture_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        self.sampler.as_ref().unwrap(),
                                    ),
                                },
                            ],
                            layout: self.render_bind_group_layout.as_ref().unwrap(),
                            label: Some("bind group"),
                        },
                    ));

                    // println!("{} {}", size.width, size.height);
                    self.renderer
                        .window_surface
                        .configure(&self.renderer.device, &self.renderer.config);
                    something_changed = true;
                }
                Event::DeviceEvent { event, .. } => match event {
                    _ => {
                        self.update_device_event(
                            event,
                            &mut left_mouse_down,
                            &mut something_changed,
                        );
                    }
                },
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    // #[cfg(not(target_arch = "wasm32"))]
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::R),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        println!("{:#?}", self.renderer.instance.generate_report());
                    }
                    _ => {
                        self.update_window_event(event, &mut left_mouse_down);
                    }
                },
                Event::RedrawRequested(_) => {
                    // self.renderer.queue.submit(None);
                    // println!("blocking");
                    // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                    // println!("done");
                    // println!("redrawing");
                    self.update();
                    let frame = match self.renderer.window_surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(_) => {
                            self.renderer
                                .window_surface
                                .configure(&self.renderer.device, &self.renderer.config);
                            self.renderer
                                .window_surface
                                .get_current_texture()
                                .expect("Failed to acquire next surface texture!")
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // create render pass descriptor and its color attachments
                    let color_attachments = [wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }];
                    let render_pass_descriptor = wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: None,
                    };

                    let mut command_encoder = self
                        .renderer
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    command_encoder.push_debug_group("compute ray trace");
                    {
                        // compute pass
                        let mut cpass = command_encoder
                            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                        cpass.set_pipeline(self.compute_pipeline.as_ref().unwrap());
                        cpass.set_bind_group(0, self.compute_bind_group.as_ref().unwrap(), &[]);
                        cpass.dispatch(
                            self.renderer.config.width / WORKGROUP_SIZE[0],
                            self.renderer.config.height / WORKGROUP_SIZE[1],
                            WORKGROUP_SIZE[2],
                        );
                    }
                    command_encoder.pop_debug_group();

                    command_encoder.push_debug_group("render texture");
                    {
                        // render pass
                        let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
                        rpass.set_pipeline(self.render_pipeline.as_ref().unwrap());
                        rpass.set_bind_group(0, self.render_bind_group.as_ref().unwrap(), &[]);
                        // rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
                        // rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
                        rpass.draw(0..3, 0..1);
                    }
                    command_encoder.pop_debug_group();

                    self.renderer
                        .queue
                        .submit(std::iter::once(command_encoder.finish()));

                    frame.present();

                    if let Some(ref mut x) = self.ubo {
                        x.subpixel_idx += 1;
                    }
                }
                _ => {}
            }
        });
    }
}

fn main() {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    let scene_path = &args[1];
    // let objects = import_obj(model_path);
    // let (tlas, blas) = objects.as_ref().unwrap().get(0).unwrap();

    // let inverseTransform = Mat4::IDENTITY;

    // let camera_angle_y = 0.0;
    // let camera_angle_xz = 0.0;
    // let camera_dist = 9.0;
    // // TODO: find out why models are appearing upside-down
    // let camera_centre = [0.0, 1.0, 0.0];
    // let camera_up = [0.0, -1.0, 0.0];

    // let game_state = Some(GameState {
    //     camera_angle_xz,
    //     camera_angle_y,
    //     camera_dist,
    //     camera_centre,
    //     camera_up,
    // });

    // let camera_position = [
    //     camera_angle_xz.cos() * camera_angle_y.sin() * camera_dist,
    //     camera_angle_xz.sin() * camera_dist + camera_centre[1],
    //     -camera_angle_xz.cos() * camera_angle_y.cos() * camera_dist,
    // ];

    // // println!("{} {}", camera_angle_y, camera_angle_xz);

    // let camera = Camera::new(
    //     camera_position,
    //     camera_centre,
    //     camera_up,
    //     WIDTH as u32,
    //     HEIGHT as u32,
    //     std::f32::consts::FRAC_PI_3,
    //     // 1.0472f32,
    // );

    // for x in 0..WIDTH {
    //     for y in 0..HEIGHT {
    //         let ray: Ray = rayForPixel(x, y, &camera);
    //         intersect(ray, inverseTransform, tlas);
    //     }
    // }

    let event_loop = EventLoop::new();
    let mut app = RenderApp::new(&event_loop, args[2].parse::<bool>().unwrap());
    app.init(scene_path);

    // let mut renderdoc_api: RenderDoc<renderdoc::V100> = RenderDoc::new().unwrap();
    // renderdoc_api.start_frame_capture(std::ptr::null(), std::ptr::null());
    app.render(event_loop);

    // drop(app);

    // log::info!("sleeping...");
    // thread::sleep(Duration::from_millis(4000));
    // log::info!("waking.");
    // renderdoc_api.end_frame_capture(std::ptr::null(), std::ptr::null());
}
