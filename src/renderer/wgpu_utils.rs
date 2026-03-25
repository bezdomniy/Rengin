use std::{borrow::Cow, collections::HashMap, mem, sync::Arc, time::Instant};

use crate::{
    RendererType,
    engine::bvh::{Bvh, NodeInner, NodeLeaf, NodeNormal},
    engine::rt_primitives::{ObjectParam, ScreenData, Ubo},
};

use naga_oil::compose::{
    ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue,
};
use wgpu::{
    Adapter, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, Instance, Queue,
    RenderPipeline, Surface, Texture, util::DeviceExt,
};
use winit::{dpi::PhysicalSize, window::Window};

use super::{RenginRenderer, RenginShaderModule};

pub struct RenginWgpu {
    pub window: Arc<Window>,
    pub instance: Instance,
    pub _adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub raygen_pipeline: Option<ComputePipeline>,
    pub raygen_bind_group_layout: Option<BindGroupLayout>,
    pub raygen_bind_group: Option<BindGroup>,
    pub compute_pipeline: Option<ComputePipeline>,
    pub compute_compact_pipeline: Option<ComputePipeline>,
    pub prepare_dispatch_pipeline: Option<ComputePipeline>,
    pub compute_bind_group_layout: Option<BindGroupLayout>,
    pub compute_bind_group: Option<BindGroup>,
    // Ping-pong bind groups for compaction (swap active_rays_in/out)
    pub compute_bind_group_b: Option<BindGroup>,
    pub prepare_bind_group_layout: Option<BindGroupLayout>,
    pub prepare_bind_group: Option<BindGroup>,
    pub render_pipeline: Option<RenderPipeline>,
    pub render_bind_group_layout: Option<BindGroupLayout>,
    pub render_bind_group: Option<BindGroup>,
    pub buffers: Option<HashMap<&'static str, Buffer>>,
    pub compute_target_texture: Option<Texture>,
    pub surface: Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub shaders: Option<HashMap<&'static str, RenginShaderModule>>,
    pub continous_motion: bool,
    pub rays_per_pixel: u32,
    pub ray_bounces: u32,
    pub profile: bool,
    // GPU timestamp query support
    pub timestamp_query_set: Option<wgpu::QuerySet>,
    pub timestamp_resolve_buffer: Option<Buffer>,
    pub timestamp_read_buffer: Option<Buffer>,
    pub timestamp_period: f32,
}

impl RenginWgpu {
    pub async fn new(
        window: Arc<Window>,
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
        profile: bool,
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

        for adapter in instance.enumerate_adapters(wgpu::Backends::all()).await {
            log::debug!("Found adapter {:?}", adapter)
        }

        let window_surface = instance.create_surface(window.clone()).unwrap();
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

        {
            let adapter_info = adapter.get_info();
            log::info!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
            log::info!("{:?}\n{:?}", adapter.features(), wgpu::Features::default());
        }

        let optional_features =
            wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let required_features = { wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES };

        let required_limits = {
            wgpu::Limits {
                max_storage_buffer_binding_size: 1024 << 20,
                max_buffer_size: 1024 << 20,
                max_storage_buffers_per_shader_stage: 12,
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
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: (optional_features & adapter_features) | required_features,
                required_limits,
                ..Default::default()
            })
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

        let has_timestamps = profile && device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let timestamp_period = if has_timestamps {
            queue.get_timestamp_period()
        } else {
            0.0
        };

        // Create timestamp query set: 2 per bounce + 2 for raygen + 2 for render
        let (timestamp_query_set, timestamp_resolve_buffer, timestamp_read_buffer) = if has_timestamps {
            let query_count = 2 + ray_bounces * 2 + 2;
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp query set"),
                ty: wgpu::QueryType::Timestamp,
                count: query_count,
            });
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve buffer"),
                size: (query_count as u64) * 8, // u64 per timestamp
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp read buffer"),
                size: (query_count as u64) * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(query_set), Some(resolve_buf), Some(read_buf))
        } else {
            (None, None, None)
        };

        log::info!("GPU timestamp queries: {}, period: {}ns", has_timestamps, timestamp_period);

        RenginWgpu {
            window,
            instance,
            _adapter: adapter,
            device,
            queue,
            raygen_bind_group: None,
            raygen_bind_group_layout: None,
            raygen_pipeline: None,
            compute_bind_group: None,
            compute_bind_group_b: None,
            compute_bind_group_layout: None,
            compute_pipeline: None,
            compute_compact_pipeline: None,
            prepare_dispatch_pipeline: None,
            prepare_bind_group_layout: None,
            prepare_bind_group: None,
            render_bind_group: None,
            render_bind_group_layout: None,
            render_pipeline: None,
            buffers: None,
            compute_target_texture: None,
            surface: window_surface,
            config,
            shaders: None,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
            profile,
            timestamp_query_set,
            timestamp_resolve_buffer,
            timestamp_read_buffer,
            timestamp_period,
        }
    }
}

impl<'a> RenginRenderer for RenginWgpu {
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
            format: wgpu::TextureFormat::Rgba16Float,
            view_formats: &[wgpu::TextureFormat::Rgba16Float],
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            label: None,
        }));
    }

    fn create_shaders(&mut self, renderer_type: RendererType) {
        let mut composer = Composer::default();

        // Add composable modules in dependency order (leaves first)
        let mut load_composable = |source: &str, file_path: &str| {
            match composer.add_composable_module(ComposableModuleDescriptor {
                source,
                file_path,
                ..Default::default()
            }) {
                Ok(_) => {}
                Err(e) => {
                    let msg = e.emit_to_string(&composer);
                    panic!("Failed to add shader module {file_path}:\n{msg}");
                }
            }
        };

        load_composable(include_str!("../shaders/types.wgsl"), "shaders/types.wgsl");
        load_composable(include_str!("../shaders/ubo.wgsl"), "shaders/ubo.wgsl");
        load_composable(include_str!("../shaders/constants.wgsl"), "shaders/constants.wgsl");
        load_composable(include_str!("../shaders/random.wgsl"), "shaders/random.wgsl");
        load_composable(
            include_str!("../shaders/pathtracer_bindings.wgsl"),
            "shaders/pathtracer_bindings.wgsl",
        );
        load_composable(include_str!("../shaders/stats.wgsl"), "shaders/stats.wgsl");
        load_composable(include_str!("../shaders/hit_params.wgsl"), "shaders/hit_params.wgsl");
        load_composable(include_str!("../shaders/intersects.wgsl"), "shaders/intersects.wgsl");

        // Shader defs for compile-time profiling toggle
        let mut shader_defs = HashMap::new();
        if self.profile {
            shader_defs.insert("PROFILE_STATS".to_owned(), ShaderDefValue::Bool(true));
        }

        let cs_module = match renderer_type {
            RendererType::PathTracer => {
                let module = match composer.make_naga_module(NagaModuleDescriptor {
                    source: include_str!("../shaders/pathtracer.wgsl"),
                    file_path: "shaders/pathtracer.wgsl",
                    shader_defs,
                    ..Default::default()
                }) {
                    Ok(m) => m,
                    Err(e) => {
                        let msg = e.emit_to_string(&composer);
                        panic!("Shader composition error for pathtracer:\n{msg}");
                    }
                };
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("pathtracer"),
                        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                    })
            }
            RendererType::RayTracer => {
                // TODO: create a raytracer entry-point shader with naga_oil imports
                panic!("RayTracer not yet migrated to naga_oil");
            }
        };

        let raygen_naga = match composer.make_naga_module(NagaModuleDescriptor {
            source: include_str!("../shaders/raygen.wgsl"),
            file_path: "shaders/raygen.wgsl",
            ..Default::default()
        }) {
            Ok(m) => m,
            Err(e) => {
                let msg = e.emit_to_string(&composer);
                panic!("Shader composition error for raygen:\n{msg}");
            }
        };
        let raygen_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raygen"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(raygen_naga)),
            });

        let frag_naga = match composer.make_naga_module(NagaModuleDescriptor {
            source: include_str!("../shaders/render.frag.wgsl"),
            file_path: "shaders/render.frag.wgsl",
            ..Default::default()
        }) {
            Ok(m) => m,
            Err(e) => {
                let msg = e.emit_to_string(&composer);
                panic!("Shader composition error for fragment:\n{msg}");
            }
        };
        let fg_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fragment"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(frag_naga)),
            });

        // These shaders have no imports, pass through as plain WGSL
        let vt_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vertex"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/render.vert.wgsl"
                ))),
            });

        let prepare_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("prepare dispatch"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/prepare_dispatch.wgsl"
                ))),
            });

        let mut shaders = HashMap::new();
        shaders.insert(
            "raygen",
            RenginShaderModule::WgpuShaderModule(raygen_module),
        );
        shaders.insert("comp", RenginShaderModule::WgpuShaderModule(cs_module));
        shaders.insert("vert", RenginShaderModule::WgpuShaderModule(vt_module));
        shaders.insert("frag", RenginShaderModule::WgpuShaderModule(fg_module));
        shaders.insert(
            "prepare",
            RenginShaderModule::WgpuShaderModule(prepare_module),
        );

        self.shaders = Some(shaders);
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

        // SOA ray buffers: 3 separate vec4<f32> arrays for coalesced GPU access
        let ray_buf_size = (screen_data.resolution.width * screen_data.resolution.height)
            as usize
            * mem::size_of::<glam::Vec4>();
        let buf_ray_origins = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray Origins Buffer"),
                contents: &vec![0xff; ray_buf_size],
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_ray_directions = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray Directions Buffer"),
                contents: &vec![0xff; ray_buf_size],
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_ray_throughputs = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray Throughputs Buffer"),
                contents: &vec![0xff; ray_buf_size],
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Active ray compaction buffers (ping-pong A/B)
        let total_pixels = (screen_data.resolution.width * screen_data.resolution.height) as usize;
        let active_buf_size = total_pixels * mem::size_of::<u32>();
        let buf_active_rays_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Active Rays A"),
                contents: &vec![0u8; active_buf_size],
                usage: wgpu::BufferUsages::STORAGE,
            });
        let buf_active_rays_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Active Rays B"),
                contents: &vec![0u8; active_buf_size],
                usage: wgpu::BufferUsages::STORAGE,
            });
        // ray_counts: [input_count, output_count] as atomic u32
        let buf_ray_counts = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Ray Counts"),
                contents: bytemuck::cast_slice(&[0u32; 2]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        // Indirect dispatch args: [workgroups_x, 1, 1]
        let buf_indirect_dispatch = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Indirect Dispatch Args"),
                contents: bytemuck::cast_slice(&[0u32; 3]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            });

        // Stats buffer: 8 atomic u32 counters + read-back buffer for profiling
        let stats_size = 8 * mem::size_of::<u32>();
        let buf_stats = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Shader Stats Buffer"),
                contents: &vec![0u8; stats_size],
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let buf_stats_read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shader Stats Read Buffer"),
            size: stats_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            ("ray_origins", buf_ray_origins),
            ("ray_directions", buf_ray_directions),
            ("ray_throughputs", buf_ray_throughputs),
            ("stats", buf_stats),
            ("stats_read", buf_stats_read),
            ("active_rays_a", buf_active_rays_a),
            ("active_rays_b", buf_active_rays_b),
            ("ray_counts", buf_ray_counts),
            ("indirect_dispatch", buf_indirect_dispatch),
        ]));
    }

    fn create_pipelines(
        &mut self,
        // TODO: bvh is only needed to get lengths, is there a better way to pass these?
        bvh: &Bvh,
        screen_data: &ScreenData,
        object_params: &[ObjectParam],
    ) {
        // create raygen pipeline
        self.raygen_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
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
                    // SOA ray buffers (bindings 1-3)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            },
        ));

        let raygen_pipeline_layout = Some(self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("raygen"),
                bind_group_layouts: &[self.raygen_bind_group_layout.as_ref().unwrap()],
                immediate_size: 0,
            },
        ));

        // create compute pipeline
        self.compute_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadWrite,
                            format: wgpu::TextureFormat::Rgba16Float,
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
                    // SOA ray buffers (bindings 6-8)
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                ((screen_data.resolution.width * screen_data.resolution.height)
                                    as usize
                                    * mem::size_of::<glam::Vec4>()) as _,
                            ),
                        },
                        count: None,
                    },
                    // Stats buffer (binding 9): 8 atomic u32 counters
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (8 * mem::size_of::<u32>()) as _,
                            ),
                        },
                        count: None,
                    },
                    // Active rays in (binding 10)
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(4),
                        },
                        count: None,
                    },
                    // Active rays out (binding 11)
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(4),
                        },
                        count: None,
                    },
                    // Ray counts (binding 12)
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(8),
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
                immediate_size: 0,
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
                            format: wgpu::TextureFormat::Rgba16Float,
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
                immediate_size: 0,
            },
        ));

        self.render_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: None,
                layout: render_pipeline_layout.as_ref(),
                multiview_mask: None,
                vertex: wgpu::VertexState {
                    module: match self.shaders.as_ref().unwrap().get("vert") {
                        Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                        _ => panic!("Invalid WGPU vertex shader passed to render pipeline."),
                    },
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: match self.shaders.as_ref().unwrap().get("frag") {
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
            },
        ));

        self.raygen_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Raygen pipeline"),
                layout: raygen_pipeline_layout.as_ref(),
                module: match self.shaders.as_ref().unwrap().get("raygen") {
                    Some(RenginShaderModule::WgpuShaderModule(m)) => m,
                    _ => panic!("Invalid WGPU compute shader passed to compute pipeline."),
                },
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        let comp_module = match self.shaders.as_ref().unwrap().get("comp") {
            Some(RenginShaderModule::WgpuShaderModule(m)) => m,
            _ => panic!("Invalid WGPU compute shader passed to compute pipeline."),
        };

        self.compute_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline (2D)"),
                layout: compute_pipeline_layout.as_ref(),
                module: comp_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        self.compute_compact_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute pipeline (1D compact)"),
                layout: compute_pipeline_layout.as_ref(),
                module: comp_module,
                entry_point: Some("main_compact"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        // Prepare dispatch pipeline
        self.prepare_bind_group_layout = Some(self.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("prepare dispatch layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(8),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(12),
                        },
                        count: None,
                    },
                ],
            },
        ));

        let prepare_pipeline_layout = self.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("prepare dispatch"),
                bind_group_layouts: &[self.prepare_bind_group_layout.as_ref().unwrap()],
                immediate_size: 0,
            },
        );

        let prepare_module = match self.shaders.as_ref().unwrap().get("prepare") {
            Some(RenginShaderModule::WgpuShaderModule(m)) => m,
            _ => panic!("Missing prepare dispatch shader"),
        };

        self.prepare_dispatch_pipeline = Some(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Prepare dispatch pipeline"),
                layout: Some(&prepare_pipeline_layout),
                module: prepare_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
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

        self.raygen_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raygen bind group"),
                layout: self.raygen_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ubo")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_origins")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_directions")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_throughputs")
                            .unwrap()
                            .as_entire_binding(),
                    },
                ],
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
                            .get("ray_origins")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_directions")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_throughputs")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("stats")
                            .unwrap()
                            .as_entire_binding(),
                    },
                    // Bind group A: in=A, out=B
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.buffers.as_ref().unwrap().get("active_rays_a").unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.buffers.as_ref().unwrap().get("active_rays_b").unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: self.buffers.as_ref().unwrap().get("ray_counts").unwrap().as_entire_binding(),
                    },
                ],
            }),
        );

        // Bind group B: in=B, out=A (swapped for ping-pong)
        self.compute_bind_group_b = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bind group B"),
                layout: self.compute_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&compute_target_texture_view) },
                    wgpu::BindGroupEntry { binding: 1, resource: self.buffers.as_ref().unwrap().get("ubo").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.buffers.as_ref().unwrap().get("tlas").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.buffers.as_ref().unwrap().get("blas").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.buffers.as_ref().unwrap().get("normals").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.buffers.as_ref().unwrap().get("object_params").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.buffers.as_ref().unwrap().get("ray_origins").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.buffers.as_ref().unwrap().get("ray_directions").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.buffers.as_ref().unwrap().get("ray_throughputs").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: self.buffers.as_ref().unwrap().get("stats").unwrap().as_entire_binding() },
                    // Swapped: in=B, out=A
                    wgpu::BindGroupEntry { binding: 10, resource: self.buffers.as_ref().unwrap().get("active_rays_b").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: self.buffers.as_ref().unwrap().get("active_rays_a").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: self.buffers.as_ref().unwrap().get("ray_counts").unwrap().as_entire_binding() },
                ],
            }),
        );

        // Prepare dispatch bind group
        self.prepare_bind_group = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("prepare dispatch bind group"),
                layout: self.prepare_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.buffers.as_ref().unwrap().get("ray_counts").unwrap().as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.buffers.as_ref().unwrap().get("indirect_dispatch").unwrap().as_entire_binding() },
                ],
            }),
        );
    }
}
