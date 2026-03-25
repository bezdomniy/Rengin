use std::borrow::Cow;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use naga_oil::compose::{ComposableModuleDescriptor, Composer, NagaModuleDescriptor, ShaderDefValue};
use wgpu::naga;

mod test_fresnel;
mod test_intersections;
mod test_pathtracer_utils;
mod test_prepare_dispatch;
mod test_random;

// ---------------------------------------------------------------------------
// GPU test context -- headless wgpu device
// ---------------------------------------------------------------------------

pub struct GpuTestContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuTestContext {
    pub fn try_new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).ok()?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpu_test_device"),
                ..Default::default()
            },
        ))
        .ok()?;

        Some(Self { device, queue })
    }
}

macro_rules! require_gpu {
    ($ctx:ident) => {
        let Some($ctx) = $crate::gpu::GpuTestContext::try_new() else {
            eprintln!("Skipping GPU test: no adapter available");
            return;
        };
    };
}
pub(crate) use require_gpu;

// ---------------------------------------------------------------------------
// Shader compiler -- naga_oil composer with all library modules loaded
// ---------------------------------------------------------------------------

pub struct ShaderCompiler {
    composer: Composer,
}

impl ShaderCompiler {
    pub fn new() -> Self {
        let mut composer = Composer::default();

        macro_rules! load {
            ($composer:expr, $source:expr, $file_path:expr) => {
                match $composer.add_composable_module(ComposableModuleDescriptor {
                    source: $source,
                    file_path: $file_path,
                    ..Default::default()
                }) {
                    Ok(_) => {}
                    Err(e) => {
                        let msg = e.emit_to_string(&$composer);
                        panic!("Failed to add shader module {}:\n{}", $file_path, msg);
                    }
                }
            };
        }

        // Load in dependency order (mirrors wgpu_utils.rs create_shaders)
        load!(composer, include_str!("../../src/shaders/types.wgsl"), "shaders/types.wgsl");
        load!(composer, include_str!("../../src/shaders/ubo.wgsl"), "shaders/ubo.wgsl");
        load!(composer, include_str!("../../src/shaders/constants.wgsl"), "shaders/constants.wgsl");
        load!(composer, include_str!("../../src/shaders/random.wgsl"), "shaders/random.wgsl");
        load!(composer, include_str!("../../src/shaders/fresnel.wgsl"), "shaders/fresnel.wgsl");
        load!(composer, include_str!("../../src/shaders/intersects_math.wgsl"), "shaders/intersects_math.wgsl");
        load!(composer, include_str!("../../src/shaders/pathtracer_utils.wgsl"), "shaders/pathtracer_utils.wgsl");

        Self { composer }
    }

    pub fn compile(&mut self, source: &str, file_path: &str) -> naga::Module {
        self.compile_with_defs(source, file_path, HashMap::new())
    }

    pub fn compile_with_defs(
        &mut self,
        source: &str,
        file_path: &str,
        shader_defs: HashMap<String, ShaderDefValue>,
    ) -> naga::Module {
        self.composer
            .make_naga_module(NagaModuleDescriptor {
                source,
                file_path,
                shader_defs,
                ..Default::default()
            })
            .unwrap_or_else(|e| {
                let msg = e.emit_to_string(&self.composer);
                panic!("Shader compilation error for {file_path}:\n{msg}");
            })
    }
}

// ---------------------------------------------------------------------------
// Buffer spec for dispatch_and_readback
// ---------------------------------------------------------------------------

pub struct BufferSpec {
    pub binding: u32,
    pub data: Vec<u8>,
    pub usage: wgpu::BufferUsages,
    pub read_back: bool,
}

impl BufferSpec {
    pub fn storage_readonly(binding: u32, data: Vec<u8>) -> Self {
        Self {
            binding,
            data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            read_back: false,
        }
    }

    pub fn storage_readwrite(binding: u32, data: Vec<u8>) -> Self {
        Self {
            binding,
            data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            read_back: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch and readback
// ---------------------------------------------------------------------------

pub fn dispatch_and_readback(
    ctx: &GpuTestContext,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    workgroups: [u32; 3],
    buffers: &[BufferSpec],
) -> Vec<Vec<u8>> {
    use wgpu::util::DeviceExt;

    // Create GPU buffers
    let gpu_buffers: Vec<wgpu::Buffer> = buffers
        .iter()
        .map(|spec| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("test_buffer_binding_{}", spec.binding)),
                    contents: &spec.data,
                    usage: spec.usage | wgpu::BufferUsages::COPY_SRC,
                })
        })
        .collect();

    // Build bind group layout entries
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = buffers
        .iter()
        .map(|spec| wgpu::BindGroupLayoutEntry {
            binding: spec.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if spec.usage.contains(wgpu::BufferUsages::UNIFORM) {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: !spec.read_back,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_bind_group_layout"),
            entries: &layout_entries,
        });

    let bind_group_entries: Vec<wgpu::BindGroupEntry> = gpu_buffers
        .iter()
        .enumerate()
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: buffers[i].binding,
            resource: buf.as_entire_binding(),
        })
        .collect();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_bind_group"),
        layout: &bind_group_layout,
        entries: &bind_group_entries,
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("test_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test_pipeline"),
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

    // Dispatch
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test_encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test_compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    // Create staging buffers for readback
    let staging_buffers: Vec<Option<wgpu::Buffer>> = buffers
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            if spec.read_back {
                let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("staging_{}", spec.binding)),
                    size: spec.data.len() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(
                    &gpu_buffers[i],
                    0,
                    &staging,
                    0,
                    spec.data.len() as u64,
                );
                Some(staging)
            } else {
                None
            }
        })
        .collect();

    ctx.queue.submit(std::iter::once(encoder.finish()));
    let _ = ctx.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });

    // Read back results
    let mut results = Vec::new();
    for staging in &staging_buffers {
        if let Some(buf) = staging {
            let slice = buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            let _ = ctx.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            results.push(data.to_vec());
            drop(data);
            buf.unmap();
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Convenience helper for simple input -> output test shaders
// ---------------------------------------------------------------------------

pub fn run_test_shader<I: Pod, O: Pod + Zeroable>(
    ctx: &GpuTestContext,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    inputs: &[I],
    output_count: usize,
) -> Vec<O> {
    let input_bytes = bytemuck::cast_slice(inputs).to_vec();
    let output_bytes = vec![0u8; output_count * std::mem::size_of::<O>()];

    let buffers = vec![
        BufferSpec::storage_readonly(0, input_bytes),
        BufferSpec::storage_readwrite(1, output_bytes),
    ];

    let results = dispatch_and_readback(ctx, module, entry_point, [inputs.len() as u32, 1, 1], &buffers);

    bytemuck::cast_slice(&results[0]).to_vec()
}

// ---------------------------------------------------------------------------
// Float comparison utilities
// ---------------------------------------------------------------------------

pub const GPU_TOLERANCE: f32 = 1e-3;

pub fn assert_approx_eq(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

pub fn create_shader_module(ctx: &GpuTestContext, compiler: &mut ShaderCompiler, source: &str, file_path: &str) -> wgpu::ShaderModule {
    let naga_module = compiler.compile(source, file_path);
    ctx.device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(file_path),
            source: wgpu::ShaderSource::Naga(Cow::Owned(naga_module)),
        })
}
