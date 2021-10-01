mod engine;
mod renderer;
mod shaders;

// use image::io::Reader;
// use renderdoc::RenderDoc;
use wgpu::util::{DeviceExt, StagingBelt};
use wgpu::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, RenderPipeline, Sampler, ShaderModule,
    Texture, TextureView,
};
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
use std::thread;
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
use glam::{Mat4, Vec4};

use engine::asset_importer::import_obj;

use engine::rt_primitives::{Camera, Material, NodeBLAS, NodeTLAS, ObjectParams, UBO};

use crate::renderer::wgpu_utils::RenginWgpu;

static WIDTH: u32 = 800;
static HEIGHT: u32 = 600;
static WORKGROUP_SIZE: [u32; 3] = [32, 32, 1];

static FRAMERATE: f64 = 10.0;

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
    objects: Option<Vec<(Vec<NodeTLAS>, Vec<NodeBLAS>)>>,
    buffers: Option<HashMap<&'static str, Buffer>>,
    texture: Option<Texture>,
    object_params: Option<ObjectParams>,
    ubo: Option<UBO>,
    game_state: Option<GameState>,
}

impl RenderApp {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        let renderer =
            futures::executor::block_on(RenginWgpu::new(WIDTH, HEIGHT, WORKGROUP_SIZE, event_loop));

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
            objects: None,
            buffers: None,
            texture: None,
            object_params: None,
            ubo: None,
            game_state: None,
        }
        // RenderApp::init(&self, model_path)
    }

    fn init(&mut self, model_path: &str) {
        let mut now = Instant::now();
        log::info!("Loading models...");
        self.objects = import_obj(model_path);
        // let (dragon_tlas, dragon_blas) = &objects[0];
        log::info!(
            "Finished loading models in {} millis.",
            now.elapsed().as_millis()
        );

        self.object_params = Some(ObjectParams {
            inverse_transform: Mat4::IDENTITY,
            material: Material {
                colour: Vec4::new(0.537, 0.831, 0.914, 1.0),
                ambient: 0.1,
                diffuse: 0.7,
                specular: 0.3,
                shininess: 200.0,
            },
        });

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
            -camera_angle_xz.cos() * camera_angle_y.cos() * camera_dist,
        ];

        println!("{} {}", camera_angle_y, camera_angle_xz);

        let camera = Camera::new(
            camera_position,
            camera_centre,
            camera_up,
            WIDTH as u32,
            HEIGHT as u32,
            1.0472f32,
        );

        self.ubo = Some(UBO::new(
            [-4f32, 2f32, -3f32, 1f32],
            self.objects.as_ref().unwrap().get(0).unwrap().0.len() as i32,
            self.objects.as_ref().unwrap().get(0).unwrap().1.len() as i32,
            camera,
        ));

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

        // self.staging_belt = Some(StagingBelt::new(0x100));

        // let texture_view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));

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
        let mut now = Instant::now();
        log::info!("Creating buffers...");

        let (dragon_tlas, dragon_blas) = self.objects.as_ref().unwrap().get(0).unwrap();

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
                contents: bytemuck::cast_slice(&dragon_tlas),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_blas = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BLAS storage Buffer"),
                contents: bytemuck::cast_slice(&dragon_blas),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_op = self
            .renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material storage Buffer"),
                contents: bytemuck::bytes_of(&self.object_params.unwrap()),
                usage: wgpu::BufferUsages::STORAGE,
            });

        log::info!(
            "Finshed loading buffers in {} millis",
            now.elapsed().as_millis()
        );

        let mut buffers: HashMap<&'static str, Buffer> = HashMap::new();
        buffers.insert("ubo", buf_ubo);
        buffers.insert("tlas", buf_tlas);
        buffers.insert("blas", buf_blas);
        buffers.insert("object_params", buf_op);

        buffers
    }

    fn create_pipelines(&mut self) {
        self.compute_bind_group_layout = Some(self.renderer.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
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
                            // min_binding_size: wgpu::BufferSize::new(
                            //     (dragon_tlas.len() * mem::size_of::<NodeTLAS>()) as _,
                            // ),
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            // min_binding_size: wgpu::BufferSize::new(
                            //     (dragon_blas.len() * mem::size_of::<NodeBLAS>()) as _,
                            // ),
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            // min_binding_size: None,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<ObjectParams>() as _
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            },
        ));
        let compute_pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("compute"),
                    bind_group_layouts: &[&self.compute_bind_group_layout.as_ref().unwrap()],
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
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
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
                        resource: wgpu::BindingResource::Sampler(&self.sampler.as_ref().unwrap()),
                    },
                ],
                layout: &self.render_bind_group_layout.as_ref().unwrap(),
                label: Some("bind group"),
            },
        ));

        let render_pipeline_layout =
            self.renderer
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render"),
                    bind_group_layouts: &[&self.render_bind_group_layout.as_ref().unwrap()],
                    push_constant_ranges: &[],
                });

        self.render_pipeline = Some(
            self.renderer
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: self.shaders.as_ref().unwrap().get("vert").as_ref().unwrap(),
                        entry_point: "main",
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: self.shaders.as_ref().unwrap().get("frag").as_ref().unwrap(),
                        entry_point: "main",
                        targets: &[self
                            .renderer
                            .window_surface
                            .get_preferred_format(&self.renderer.adapter)
                            .unwrap()
                            .into()],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                }),
        );

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
                    layout: &self.compute_bind_group_layout.as_ref().unwrap(),
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

                    game_state.camera_angle_y = game_state.camera_angle_y + (delta.0 as f32);
                    game_state.camera_angle_xz = game_state.camera_angle_xz + (delta.1 as f32);

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
                        *something_changed = true;
                    }
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(x, y) => {
                    // println!("{} {}", x, y);
                    let game_state: &mut GameState = self.game_state.as_mut().unwrap();
                    game_state.camera_dist = game_state.camera_dist - ((y as f32) / 3.0);

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

    pub fn render(mut self, event_loop: EventLoop<()>) {
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;
        let mut something_changed = false;

        event_loop.run(move |event, _, control_flow| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            // let _ = (&self.renderer.instance, &self.adapter, &compute_pipeline); //, &self.device, &self.config);

            *control_flow = ControlFlow::Wait;
            match event {
                Event::RedrawEventsCleared => {
                    let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    let time_since_last_frame = last_update_inst.elapsed();

                    if something_changed && time_since_last_frame >= target_frametime {
                        self.renderer.window.request_redraw();
                        last_update_inst = Instant::now();
                        something_changed = false;
                    } else {
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    let texture_extent = wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
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
                                layout: &self.compute_bind_group_layout.as_ref().unwrap(),
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
                                        &self.sampler.as_ref().unwrap(),
                                    ),
                                },
                            ],
                            layout: &self.render_bind_group_layout.as_ref().unwrap(),
                            label: Some("bind group"),
                        },
                    ));

                    // Reconfigure the surface with the new size
                    self.renderer.config.width = size.width;
                    self.renderer.config.height = size.height;
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
                    // println!("blocking");
                    // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                    // println!("done");
                    println!("redrawing");
                    let frame = match self.renderer.window_surface.get_current_frame() {
                        Ok(frame) => frame,
                        Err(_) => {
                            self.renderer
                                .window_surface
                                .configure(&self.renderer.device, &self.renderer.config);
                            self.renderer
                                .window_surface
                                .get_current_frame()
                                .expect("Failed to acquire next surface texture!")
                        }
                    };
                    let view = frame
                        .output
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

                    // wgpu::BufferSize::new(mem::size_of::<UBO>() as _)

                    // println!("{}", self.ubo.unwrap().camera.inverse_transform);
                    let mut staging_belt = StagingBelt::new(0x100);
                    staging_belt
                        .write_buffer(
                            &mut command_encoder,
                            self.buffers.as_ref().unwrap().get("ubo").unwrap(),
                            0,
                            wgpu::BufferSize::new(mem::size_of::<UBO>() as wgpu::BufferAddress)
                                .unwrap(),
                            &self.renderer.device,
                        )
                        .copy_from_slice(bytemuck::bytes_of(&self.ubo.unwrap()));

                    staging_belt.finish();

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

                    self.renderer.queue.submit(Some(command_encoder.finish()));
                }
                _ => {}
            }
        });
    }
}

fn main() {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    let model_path = &args[1];

    let event_loop = EventLoop::new();
    let mut app = RenderApp::new(&event_loop);
    app.init(&model_path);

    // let mut renderdoc_api: RenderDoc<renderdoc::V100> = RenderDoc::new().unwrap();
    // renderdoc_api.start_frame_capture(std::ptr::null(), std::ptr::null());
    app.render(event_loop);

    // drop(app);

    // log::info!("sleeping...");
    // thread::sleep(Duration::from_millis(4000));
    // log::info!("waking.");
    // renderdoc_api.end_frame_capture(std::ptr::null(), std::ptr::null());
}
