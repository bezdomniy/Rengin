mod engine;
mod renderer;
mod shaders;

use wgpu::util::DeviceExt;
use winit::event_loop::EventLoop;

// use vulkano_win::VkSurfaceBuild;

use image::ImageBuffer;
use image::Rgba;

use std::sync::Arc;
use std::{future, mem};

// use wgpu::BufferUsage;

use engine::asset_importer::import_obj;
use std::slice;

use engine::rt_primitives::{Camera, NodeBLAS, NodeTLAS, UBO};

use crate::renderer::wgpu_utils::RenginWgpu;

use bincode;

static WIDTH: u32 = 2400;
static HEIGHT: u32 = 1800;
static WORKGROUP_SIZE: u32 = 32;

fn main() {
    let objects = import_obj("assets/models/suzanne.obj");
    let (dragon_tlas, dragon_blas) = &objects[0];

    println!("tlas:{:?}, blas{:?}", dragon_tlas.len(), dragon_blas.len());
    println!(
        "tlas:{:?}, blas{:?}",
        mem::size_of::<NodeTLAS>(),
        mem::size_of::<NodeBLAS>()
    );

    let camera = Camera::new(
        [1f32, 3f32, -5f32],
        [0f32, 1f32, 0f32],
        [0f32, 1f32, 0f32],
        WIDTH,
        HEIGHT,
        1.0472f32,
    );

    let ubo = UBO::new([10f32, 10f32, -10f32, 1f32], camera);

    // let buffer_content = buf_image.read().unwrap();
    // let image = ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, &buffer_content[..]).unwrap();
    // image.save("image.png").unwrap();

    let event_loop = EventLoop::new();

    let wgpu = futures::executor::block_on(RenginWgpu::new(&event_loop));

    let cs_module = wgpu
        .device
        .create_shader_module(&wgpu::include_spirv!("shaders/raytracer.spv"));

    let output_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (WIDTH * mem::size_of::<f32>() as u32 * HEIGHT) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_extent = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };

    // The render pipeline renders data into this texture
    let texture = wgpu.device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    // let p_ubo: *const UBO = &ubo; // the same operator is used as with references
    // let p_ubo: *const u8 = p_ubo as *const u8;

    let buf_ubo = wgpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UBO Buffer"),
            // contents: unsafe { slice::from_raw_parts(p_ubo, mem::size_of::<UBO>()) },
            contents: &bincode::serialize(&ubo).unwrap(),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

    //TODO tlas and blas buffers

    let compute_bind_group_layout =
        wgpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(mem::size_of::<UBO>() as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (dragon_tlas.len() * mem::size_of::<NodeTLAS>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (dragon_blas.len() * mem::size_of::<NodeBLAS>()) as _,
                            ),
                        },
                        count: None,
                    },
                ],
                label: None,
            });
    let compute_pipeline_layout =
        wgpu.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

    // event_loop.run(move |event, _, control_flow| {
    //     *control_flow = ControlFlow::Wait;

    //     match event {
    //         Event::WindowEvent {
    //             event: WindowEvent::CloseRequested,
    //             ..
    //         } => {
    //             *control_flow = ControlFlow::Exit;
    //         }
    //         _ => (),
    //     }
    // });
}
