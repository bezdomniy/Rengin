use std::{borrow::Cow, collections::HashMap, mem, time::Instant};

use crate::{
    RendererType,
    engine::bvh::{Bvh, NodeInner, NodeLeaf, NodeNormal},
    engine::rt_primitives::{ObjectParam, Ray, ScreenData, Ubo},
};

use wgpu::{
    Adapter, BindGroup, Buffer, ComputePipeline, Device, Instance, Queue, RenderPipeline, Surface,
    Texture, util::DeviceExt,
};
use winit::{dpi::PhysicalSize, window::Window};

use super::{RenginRenderer, RenginShaderModule};

pub struct RenginWgpu<'a> {
    pub window: &'a Window,
    pub instance: Instance,
    pub _adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub raysort_pipeline: Option<ComputePipeline>,
    pub raygen_pipeline: Option<ComputePipeline>,
    pub compute_pipeline: Option<ComputePipeline>,
    pub render_pipeline: Option<RenderPipeline>,
    pub bind_groups: HashMap<String, BindGroup>,
    pub buffers: HashMap<String, Buffer>,
    pub compute_target_texture: Option<Texture>,
    pub compute_target_texture_view: Option<wgpu::TextureView>,
    pub surface: Surface<'a>,
    pub config: wgpu::SurfaceConfiguration,
    pub shaders: HashMap<String, RenginShaderModule>,
    // pub physical_size: PhysicalSize<u32>,
    // pub logical_size: LogicalSize<u32>,
    // pub workgroup_size: [u32; 3],
    pub continous_motion: bool,
    pub rays_per_pixel: u32,
    pub ray_bounces: u32,
    // pub scale_factor: f64,
    // pub resolution: PhysicalSize<u32>,
}

impl<'a> RenginWgpu<'a> {
    pub async fn new(
        window: &'a Window,
        // workgroup_size: [u32; 3],
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
    ) -> Self {
        let width = window.inner_size().width;
        let height = window.inner_size().height;

        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());

        log::info!("backend: {:?}", backends);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        log::info!("instance: {:?}", instance);

        #[cfg(not(target_arch = "wasm32"))]
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            log::debug!("Found adapter {:?}", adapter)
        }

        let window_surface = instance.create_surface(window).unwrap();
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
                    required_features: (optional_features & adapter_features) | required_features,
                    required_limits,
                    ..Default::default()
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .unwrap();

        let surface_caps = window_surface.get_capabilities(&adapter);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_caps.formats[0],
            alpha_mode: surface_caps.alpha_modes[0],
            present_mode: wgpu::PresentMode::Fifo,
            width,
            height,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        // println!("{} {}", width, height);
        window_surface.configure(&device, &config);

        RenginWgpu {
            window,
            instance,
            _adapter: adapter,
            device,
            queue,
            raysort_pipeline: None,
            // raysort_bind_group: None,
            // raygen_bind_group: None,
            raygen_pipeline: None,
            // compute_bind_group: None,
            compute_pipeline: None,
            // render_bind_group: None,
            render_pipeline: None,
            buffers: HashMap::new(),
            compute_target_texture: None,
            compute_target_texture_view: None,
            surface: window_surface,
            config,
            shaders: HashMap::new(),
            bind_groups: HashMap::new(),
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

impl<'a> RenginRenderer for RenginWgpu<'a> {
    fn update_window_size(&mut self, physical_size: &PhysicalSize<u32>) {
        // self.physical_size = winit::dpi::PhysicalSize::new(width, height);
        // self.logical_size = self.physical_size.to_logical(self.scale_factor);

        self.config.width = physical_size.width;
        self.config.height = physical_size.height;

        self.surface.configure(&self.device, &self.config);

        self.create_target_textures(physical_size);
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
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            label: None,
        }));

        self.compute_target_texture_view = Some(
            self.compute_target_texture
                .as_ref()
                .unwrap()
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
    }

    fn create_shaders(&mut self, renderer_type: RendererType) {
        let pt_main_str = include_str!("../shaders/pathtracer.wgsl");
        let ubo_str = include_str!("../shaders/ubo.wgsl");
        let types_str = include_str!("../shaders/types.wgsl");
        let rand_shader_str = include_str!("../shaders/random.wgsl");
        let constants_shader_str = include_str!("../shaders/constants.wgsl");
        let hit_params_str = include_str!("../shaders/hit_params.wgsl");
        let intersects_shader_str = include_str!("../shaders/intersects.wgsl");

        let raygen_shader_str = [
            ubo_str,
            types_str,
            constants_shader_str,
            rand_shader_str,
            include_str!("../shaders/raygen.wgsl"),
        ]
        .join("\n");

        let raysort_shader_str = [types_str, include_str!("../shaders/raysort.wgsl")].join("\n");

        let frag_shader_str = [ubo_str, include_str!("../shaders/render.frag.wgsl")].join("\n");

        let pt_shader_str = [
            ubo_str,
            types_str,
            constants_shader_str,
            rand_shader_str,
            hit_params_str,
            intersects_shader_str,
            pt_main_str,
        ]
        .join("\n");

        let rt_shader_str = [
            ubo_str,
            types_str,
            constants_shader_str,
            hit_params_str,
            intersects_shader_str,
        ]
        .join("\n");

        let cs_module = match renderer_type {
            RendererType::PathTracer => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(pt_shader_str.as_str())),
                    })
            }
            RendererType::RayTracer => {
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: None,
                        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(rt_shader_str.as_str())),
                    })
            }
        };

        let raygen_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(raygen_shader_str.as_str())),
            });

        let raysort_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(raysort_shader_str.as_str())),
            });

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
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(frag_shader_str.as_str())),
            });

        self.shaders.insert(
            "raygen".to_string(),
            RenginShaderModule::WgpuShaderModule(raygen_module),
        );
        self.shaders.insert(
            "raysort".to_string(),
            RenginShaderModule::WgpuShaderModule(raysort_module),
        );
        self.shaders.insert(
            "compute".to_string(),
            RenginShaderModule::WgpuShaderModule(cs_module),
        );
        self.shaders.insert(
            "vert".to_string(),
            RenginShaderModule::WgpuShaderModule(vt_module),
        );
        self.shaders.insert(
            "frag".to_string(),
            RenginShaderModule::WgpuShaderModule(fg_module),
        );
    }

    fn create_buffers(
        &mut self,
        bvh: &Bvh,
        screen_data: &ScreenData,
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
                contents: &vec![
                    0xff;
                    (screen_data.resolution.width * screen_data.resolution.height)
                        as usize
                        * mem::size_of::<Ray>()
                ],
                usage: wgpu::BufferUsages::STORAGE,
            });

        log::info!(
            "Finshed loading buffers in {} millis",
            now.elapsed().as_millis()
        );

        self.buffers.insert("ubo".to_string(), buf_ubo);
        self.buffers.insert("tlas".to_string(), buf_tlas);
        self.buffers.insert("blas".to_string(), buf_blas);
        self.buffers.insert("normals".to_string(), buf_normals);
        self.buffers.insert("object_params".to_string(), buf_op);
        self.buffers.insert("rays".to_string(), buf_rays);
    }

    fn create_compute_pipeline(
        &mut self,
        pipeline_name: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        buffer_names: &[&str],
    ) -> Option<wgpu::ComputePipeline> {
        let pipeline_layout =
            self.create_pipeline_layout(pipeline_name, bind_group_layout_entries, buffer_names);
        return Some(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(format!("{} pipeline", pipeline_name).as_str()),
                    layout: pipeline_layout.as_ref(),
                    module: match self.shaders.get(pipeline_name) {
                        Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                        _ => panic!("Invalid WGPU compute shader passed to compute pipeline."),
                    },
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );
    }

    fn create_render_pipeline(
        &mut self,
        pipeline_name: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        buffer_names: &[&str],
    ) -> Option<wgpu::RenderPipeline> {
        let pipeline_layout =
            self.create_pipeline_layout(pipeline_name, bind_group_layout_entries, buffer_names);

        return Some(
            self.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: None,
                    layout: pipeline_layout.as_ref(),
                    multiview: None,
                    vertex: wgpu::VertexState {
                        module: match self.shaders.get("vert") {
                            Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                            _ => panic!("Invalid WGPU vertex shader passed to render pipeline."),
                        },
                        entry_point: Some("main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: match self.shaders.get("frag") {
                            Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                            _ => panic!("Invalid WGPU fragment shader passed to render pipeline."),
                        },
                        entry_point: Some("main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: self.config.format,
                            // TODO: change subpixel to blending, rather than doing it in shader
                            blend: Some(wgpu::BlendState::REPLACE),
                            // blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                        // targets: &[self.renderer.config.format.into()],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    cache: None,
                }),
        );
    }

    fn create_pipeline_layout(
        &mut self,
        pipeline_name: &str,
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        buffer_names: &[&str],
    ) -> Option<wgpu::PipelineLayout> {
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: bind_group_layout_entries,
                    label: None,
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            entries: buffer_names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    if *name == "texture" {
                        return wgpu::BindGroupEntry {
                            binding: i as u32,
                            resource: wgpu::BindingResource::TextureView(
                                self.compute_target_texture_view.as_ref().unwrap(),
                            ),
                        };
                    }
                    wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: self
                            .buffers
                            .get(*name)
                            .expect("Buffer not found")
                            .as_entire_binding(),
                    }
                })
                .collect::<Vec<wgpu::BindGroupEntry>>()
                .as_slice(),
            layout: &bind_group_layout,
            label: Some(format!("{} bind group", pipeline_name).as_str()),
        });

        self.bind_groups
            .insert(pipeline_name.to_string(), bind_group);

        return Some(
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(pipeline_name),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                }),
        );
    }

    fn create_pipelines(
        &mut self,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &Bvh,
        screen_data: &ScreenData,
        object_params: &[ObjectParam],
    ) {
        self.raygen_pipeline = self.create_compute_pipeline(
            "raygen",
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Ubo>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            ((screen_data.resolution.width * screen_data.resolution.height)
                                as usize
                                * mem::size_of::<Ray>()) as _,
                        ),
                    },
                    count: None,
                },
            ],
            &["ubo", "rays"],
        );

        self.raysort_pipeline = self.create_compute_pipeline(
            "raysort",
            &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        ((screen_data.resolution.width * screen_data.resolution.height) as usize
                            * mem::size_of::<Ray>()) as _,
                    ),
                },
                count: None,
            }],
            &["rays"],
        );

        self.compute_pipeline = self.create_compute_pipeline(
            "compute",
            &[
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
                            ((screen_data.resolution.width * screen_data.resolution.height)
                                as usize
                                * mem::size_of::<Ray>()) as _,
                        ),
                        // min_binding_size: None,
                    },
                    count: None,
                },
            ],
            &[
                "texture",
                "ubo",
                "tlas",
                "blas",
                "normals",
                "object_params",
                "rays",
            ],
        );

        self.render_pipeline = self.create_render_pipeline(
            "render",
            &[
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
            &["texture", "ubo"],
        );
    }
}
