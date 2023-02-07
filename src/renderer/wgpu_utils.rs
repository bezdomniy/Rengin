#[cfg(target_arch = "wasm32")]
use wgpu_gecko as wgpu;

use std::{borrow::Cow, collections::HashMap, mem, time::Instant};

use crate::{
    engine::rt_primitives::{ObjectParam, Ray, ScreenData, Ubo},
    engine::{
        bvh::{Bvh, NodeInner, NodeLeaf, NodeNormal},
        rt_primitives::Rays,
    },
    RendererType,
};

use wgpu::{
    util::DeviceExt, Adapter, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device,
    Instance, Queue, RenderPipeline, Surface, Texture,
};
use winit::{dpi::PhysicalSize, window::Window};

use super::{RenginRenderer, RenginShaderModule};

pub struct RenginWgpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub compute_pipeline: Option<ComputePipeline>,
    pub compute_bind_group_layout: Option<BindGroupLayout>,
    pub compute_bind_group: Option<BindGroup>,
    pub render_pipeline: Option<RenderPipeline>,
    pub render_bind_group_layout: Option<BindGroupLayout>,
    pub render_bind_group: Option<BindGroup>,
    pub buffers: Option<HashMap<&'static str, Buffer>>,
    pub compute_target_texture: Option<Texture>,
    pub surface: Surface,
    pub config: wgpu::SurfaceConfiguration,
    pub shaders: Option<HashMap<&'static str, RenginShaderModule>>,
    // pub physical_size: PhysicalSize<u32>,
    // pub logical_size: LogicalSize<u32>,
    // pub workgroup_size: [u32; 3],
    pub continous_motion: bool,
    pub rays_per_pixel: u32,
    pub ray_bounces: u32,
    // pub scale_factor: f64,
    // pub resolution: PhysicalSize<u32>,
}

impl RenginWgpu {
    pub async fn new(
        window: &Window,
        // workgroup_size: [u32; 3],
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
    ) -> Self {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::all());

        log::info!("backend: {:?}", backend);
        // let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::DX12);
        let instance = wgpu::Instance::new(backend);
        log::info!("instance: {:?}", instance);

        let window_surface = unsafe { instance.create_surface(&window) };
        log::info!("window_surface: {:?}", window_surface);

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found on the system!");

        log::info!("adapter: {:?}", adapter);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapter_info = adapter.get_info();
            log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
            log::info!("{:?}\n{:?}", adapter.features(), wgpu::Features::default());
        }

        let trace_dir = std::env::var("WGPU_TRACE");

        let optional_features = {
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::PUSH_CONSTANTS
        };
        // let required_features = { wgpu::Features::TEXTURE_BINDING_ARRAY };
        let required_features = { wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES };
        // let required_features = { wgpu::Features::empty() };

        let required_limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits {
                max_push_constant_size: 0,
                max_storage_buffer_binding_size: 1024 << 20,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            }
        } else {
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
            format: window_surface.get_supported_formats(&adapter)[0],
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: window_surface.get_supported_alpha_modes(&adapter)[0],
        };
        // println!("{} {}", width, height);
        window_surface.configure(&device, &config);

        RenginWgpu {
            instance,
            adapter,
            device,
            queue,
            compute_bind_group: None,
            compute_bind_group_layout: None,
            compute_pipeline: None,
            render_bind_group: None,
            render_bind_group_layout: None,
            render_pipeline: None,
            buffers: None,
            compute_target_texture: None,
            surface: window_surface,
            config,
            shaders: None,
            // physical_size,
            // logical_size,
            // workgroup_size,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
            // scale_factor,
            // resolution,
        }
    }
}

impl RenginRenderer for RenginWgpu {
    fn update_window_size(&mut self, physical_size: &PhysicalSize<u32>) {
        // self.physical_size = winit::dpi::PhysicalSize::new(width, height);
        // self.logical_size = self.physical_size.to_logical(self.scale_factor);

        self.config.width = physical_size.width;
        self.config.height = physical_size.height;

        self.surface.configure(&self.device, &self.config);

        self.create_bind_groups(physical_size);
    }

    fn create_target_textures(&mut self, physical_size: &PhysicalSize<u32>) {
        let texture_extent = wgpu::Extent3d {
            width: physical_size.width,
            height: physical_size.height,
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

    fn create_shaders(&mut self, renderer_type: RendererType) {
        let cs_module = match renderer_type {
            RendererType::PathTracer => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                            "../shaders/pathtracer.wgsl"
                        ))),
                    })
            }
            RendererType::RayTracer => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                            "../shaders/whitted_raytracer.wgsl"
                        ))),
                    })
            }
        };

        let vt_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/render.vert.wgsl"
                ))),
            });

        let fg_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/render.frag.wgsl"
                ))),
            });

        let mut shaders = HashMap::new();
        shaders.insert("comp", RenginShaderModule::WgpuShaderModule(cs_module));
        shaders.insert("vert", RenginShaderModule::WgpuShaderModule(vt_module));
        shaders.insert("frag", RenginShaderModule::WgpuShaderModule(fg_module));

        self.shaders = Some(shaders);
    }

    fn create_buffers(
        &mut self,
        bvh: &Bvh,
        screen_data: &ScreenData,
        rays: &Rays,
        object_params: &[ObjectParam],
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

    fn create_pipelines(
        &mut self,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &Bvh,
        rays: &Rays,
        object_params: &[ObjectParam],
    ) {
        // create compute pipeline
        self.compute_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
                            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Ubo>() as _),
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
                                (bvh.inner_nodes.len() * mem::size_of::<NodeInner>()) as _,
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
                                (bvh.leaf_nodes.len() * mem::size_of::<NodeLeaf>()) as _,
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
                                (bvh.normal_nodes.len() * mem::size_of::<NodeNormal>()) as _,
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
                                (object_params.len() * mem::size_of::<ObjectParam>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
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
                label: None,
            },
        ));

        let compute_pipeline_layout = Some(self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[self.compute_bind_group_layout.as_ref().unwrap()],
                push_constant_ranges: &[],
            },
        ));

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
                            min_binding_size: wgpu::BufferSize::new(mem::size_of::<Ubo>() as _),
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
                    module: match self.shaders.as_ref().unwrap().get("vert") {
                        Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                        _ => panic!("Invalid WGPU vertex shader passed to render pipeline."),
                    },
                    entry_point: "main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: match self.shaders.as_ref().unwrap().get("frag") {
                        Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                        _ => panic!("Invalid WGPU fragment shader passed to render pipeline."),
                    },
                    entry_point: "main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: self.config.format,
                        // TODO: change subpixel to blending, rather than doing it in shader
                        blend: Some(wgpu::BlendState::REPLACE),
                        // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    // targets: &[self.renderer.config.format.into()],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
            },
        ));

        self.compute_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline"),
                layout: compute_pipeline_layout.as_ref(),
                module: match self.shaders.as_ref().unwrap().get("comp") {
                    Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                    _ => panic!("Invalid WGPU compute shader passed to compute pipeline."),
                },
                entry_point: "main",
            },
        ));
    }

    fn create_bind_groups(&mut self, physical_size: &PhysicalSize<u32>) {
        self.create_target_textures(physical_size);

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
                ],
                layout: self.render_bind_group_layout.as_ref().unwrap(),
                label: Some("render bind group"),
            }),
        );

        self.compute_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bind group"),
                layout: self.compute_bind_group_layout.as_ref().unwrap(),
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
            }),
        );
    }
}
