#[cfg(target_arch = "wasm32")]
use wgpu_gecko as wgpu;

use std::{borrow::Cow, collections::HashMap, mem, time::Instant};

use crate::{
    engine::rt_primitives::{ObjectParams, Ray, ScreenData, UBO},
    engine::{
        bvh::{NodeInner, NodeLeaf, NodeNormal, BVH},
        rt_primitives::Rays,
    },
    RendererType,
};

use wgpu::{
    util::DeviceExt, Adapter, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device,
    Instance, Queue, RenderPipeline, Sampler, ShaderModule, Surface, Texture,
};
use winit::{dpi::LogicalSize, dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

pub struct RenginWgpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub render_pipeline: Option<RenderPipeline>,
    pub render_bind_group_layout: Option<BindGroupLayout>,
    pub render_bind_group: Option<BindGroup>,
    pub buffers: Option<HashMap<&'static str, Buffer>>,
    pub compute_target_texture: Option<Texture>,
    pub window: winit::window::Window,
    pub window_surface: Surface,
    pub config: wgpu::SurfaceConfiguration,
    pub physical_size: PhysicalSize<u32>,
    pub logical_size: LogicalSize<u32>,
    pub workgroup_size: [u32; 3],
    pub continous_motion: bool,
    pub rays_per_pixel: u32,
    pub ray_bounces: u32,
    pub scale_factor: f64,
    pub resolution: PhysicalSize<u32>,
}

impl RenginWgpu {
    pub fn update_window_size(&mut self, width: u32, height: u32) {
        self.physical_size = winit::dpi::PhysicalSize::new(width, height);
        self.logical_size = self.physical_size.to_logical(self.scale_factor);

        self.config.width = self.physical_size.width;
        self.config.height = self.physical_size.height;
    }

    pub async fn new(
        width: u32,
        height: u32,
        workgroup_size: [u32; 3],
        event_loop: &EventLoop<()>,
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
    ) -> Self {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        // let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::DX12);
        let instance = wgpu::Instance::new(backend);

        // TODO: window might not be on primary monitor
        let resolution = event_loop.primary_monitor().unwrap().size();

        let scale_factor: f64 = event_loop.primary_monitor().unwrap().scale_factor();
        let logical_size: LogicalSize<u32> = winit::dpi::LogicalSize::new(width, height);
        let physical_size: PhysicalSize<u32> = logical_size.to_physical(scale_factor);

        let window = WindowBuilder::new()
            .with_title("Rengin")
            .with_resizable(true)
            .with_inner_size(physical_size)
            .build(event_loop)
            .unwrap();

        let size = window.inner_size();

        let window_surface = unsafe { instance.create_surface(&window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
        println!("{:?}\n{:?}", adapter.features(), wgpu::Features::default());

        let trace_dir = std::env::var("WGPU_TRACE");

        let optional_features = {
            wgpu::Features::UNSIZED_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::PUSH_CONSTANTS
        };
        // let required_features = { wgpu::Features::TEXTURE_BINDING_ARRAY };
        let required_features = { wgpu::Features::empty() };
        let required_limits = {
            wgpu::Limits {
                max_push_constant_size: 0,
                max_storage_buffer_binding_size: 1024 << 20,
                ..wgpu::Limits::default()
            }
        }
        .using_resolution(adapter.limits());
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this example: {:?}",
            required_features - adapter_features
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (optional_features & adapter_features) | required_features,
                    limits: required_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: window_surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        // println!("{} {}", width, height);
        window_surface.configure(&device, &config);

        // let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let physical_size: PhysicalSize<u32> = winit::dpi::PhysicalSize::new(width, height);

        let logical_size: LogicalSize<u32> = physical_size.to_logical(scale_factor);

        RenginWgpu {
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,
            render_bind_group: None,
            render_bind_group_layout: None,
            render_pipeline: None,
            buffers: None,
            compute_target_texture: None,
            window: window,
            window_surface,
            config,
            physical_size,
            logical_size,
            workgroup_size,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
            scale_factor,
            resolution,
        }
    }

    pub fn create_target_textures(&mut self) {
        let texture_extent = wgpu::Extent3d {
            width: self.physical_size.width,
            height: self.physical_size.height,
            depth_or_array_layers: 1,
        };

        self.compute_target_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
        }));
    }

    pub fn create_shaders(
        &self,
        renderer_type: RendererType,
    ) -> HashMap<&'static str, ShaderModule> {
        let vt_module = self
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/raytracer.vert.wgsl"
                ))),
            });

        let fg_module = self
            .device
            .create_shader_module(&wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/raytracer.frag.wgsl"
                ))),
            });

        let mut shaders: HashMap<&'static str, ShaderModule> = HashMap::new();
        shaders.insert("vert", vt_module);
        shaders.insert("frag", fg_module);

        shaders
    }

    pub fn create_buffers(
        &mut self,
        bvh: &BVH,
        screen_data: &ScreenData,
        rays: &Rays,
        object_params: &Vec<ObjectParams>,
    ) {
        let now = Instant::now();
        log::info!("Creating buffers...");

        let buf_ubo = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UBO Buffer"),
                contents: bytemuck::bytes_of(&screen_data.generate_ubo()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let buf_tlas = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TLAS storage Buffer"),
                contents: bytemuck::cast_slice(&bvh.inner_nodes),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_blas = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BLAS storage Buffer"),
                contents: bytemuck::cast_slice(&bvh.leaf_nodes),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_normals = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("normals storage Buffer"),
                contents: bytemuck::cast_slice(&bvh.normal_nodes),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_op = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Material storage Buffer"),
                contents: bytemuck::cast_slice(object_params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_rays = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray Buffer"),
                contents: bytemuck::cast_slice(&rays.data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        log::info!(
            "Finshed loading buffers in {} millis",
            now.elapsed().as_millis()
        );

        self.buffers = Some(HashMap::from([
            ("ubo", buf_ubo),
            ("tlas", buf_tlas),
            ("blas", buf_blas),
            ("normals", buf_normals),
            ("object_params", buf_op),
            ("rays", buf_rays),
        ]));
    }

    pub fn create_pipelines(
        &mut self,
        shaders: &HashMap<&'static str, ShaderModule>,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &BVH,
        rays: &Rays,
    ) {
        // create render pipeline
        self.render_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("render bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadWrite,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(mem::size_of::<UBO>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (bvh.inner_nodes.len() * mem::size_of::<NodeInner>()) as _,
                            ),
                            // min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (bvh.leaf_nodes.len() * mem::size_of::<NodeLeaf>()) as _,
                            ),
                            // min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (bvh.normal_nodes.len() * mem::size_of::<NodeNormal>()) as _,
                            ),
                            // min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (rays.data.len() * mem::size_of::<Ray>()) as _,
                            ),
                            // min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        let render_pipeline_layout = Some(self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[self.render_bind_group_layout.as_ref().unwrap()],
                push_constant_ranges: &[],
            },
        ));

        self.render_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: None,
                layout: render_pipeline_layout.as_ref(),
                multiview: None,
                vertex: wgpu::VertexState {
                    module: shaders.get("vert").as_ref().unwrap(),
                    entry_point: "main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: shaders.get("frag").as_ref().unwrap(),
                    entry_point: "main",
                    targets: &[wgpu::ColorTargetState {
                        format: self.config.format,
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

        // TODO: remove buffers as arg and move into RenginWgpu state
        self.create_bind_groups();
    }

    pub fn create_bind_groups(&mut self) {
        self.create_target_textures();

        let compute_target_texture_view = self
            .compute_target_texture
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.render_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&compute_target_texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ubo")
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
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("rays")
                            .unwrap()
                            .as_entire_binding(),
                    },
                ],
                layout: self.render_bind_group_layout.as_ref().unwrap(),
                label: Some("bind group"),
            }),
        );
    }
}
