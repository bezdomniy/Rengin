mod engine;
mod renderer;
mod shaders;

use renderdoc::RenderDoc;
use wgpu::util::DeviceExt;
use winit::event_loop::EventLoop;

// TODO: set $env:RUST_LOG = 'WARN' when running

// use image::codecs::pnm;

use std::fs::File;
use std::io::Write;
use std::mem;

// use wgpu::BufferUsage;
use glam::{Mat4, Vec4};

use engine::asset_importer::import_obj;

use engine::rt_primitives::{Camera, Material, NodeBLAS, NodeTLAS, ObjectParams, UBO};

use crate::renderer::wgpu_utils::RenginWgpu;

static WIDTH: usize = 480;
static HEIGHT: usize = 360;
static WORKGROUP_SIZE: u32 = 32;

struct BufferDimensions {
    width: usize,
    height: usize,
    unpadded_bytes_per_row: usize,
    padded_bytes_per_row: usize,
}

impl BufferDimensions {
    fn new(width: usize, height: usize) -> Self {
        // let bytes_per_pixel = mem::size_of::<f32>();
        let bytes_per_pixel = mem::size_of::<u32>();
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;
        println!("{:?}", padded_bytes_per_row);
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

fn main() {
    env_logger::init();
    let objects = import_obj("assets/models/suzanne.obj");
    let (dragon_tlas, dragon_blas) = &objects[0];

    let object_params = ObjectParams {
        inverse_transform: Mat4::IDENTITY,
        material: Material {
            colour: Vec4::new(0.537, 0.831, 0.914, 1.0),
            ambient: 0.1,
            diffuse: 0.7,
            specular: 0.3,
            shininess: 200.0,
        },
    };

    // for t in dragon_blas.iter() {
    //     println!(
    //         "point1: {:?}, point2: {:?}, point3: {:?}",
    //         t.point1, t.point2, t.point3
    //     );
    // }

    // let v = bytemuck::cast_slice(&dragon_tlas);
    // // println!("tlas:{:?},", v);

    // let (head, body, _tail) = unsafe { v[32..64].align_to::<NodeTLAS>() };
    // assert!(head.is_empty(), "Data was not aligned");
    // let my_struct = &body[0];

    // println!("{:?}", my_struct);

    println!("tlas:{:?}, blas{:?}", dragon_tlas.len(), dragon_blas.len());
    println!(
        "tlas:{:?}, blas{:?}",
        mem::size_of::<NodeTLAS>(),
        mem::size_of::<NodeBLAS>()
    );

    let buffer_dimensions = BufferDimensions::new(WIDTH, HEIGHT);

    let camera = Camera::new(
        [-4f32, 2f32, -3f32],
        [0f32, 1f32, 0f32],
        [0f32, 1f32, 0f32],
        WIDTH as u32,
        HEIGHT as u32,
        1.0472f32,
    );

    let ubo = UBO::new([-4f32, 2f32, -3f32, 1f32], camera);

    println!("ubo:{:?},", ubo);

    // let v = bytemuck::bytes_of(&ubo);
    // // println!("tlas:{:?},", v);

    // let (head, body, _tail) = unsafe { v.align_to::<UBO>() };
    // assert!(head.is_empty(), "Data was not aligned");
    // let my_struct = &body[0];

    // println!("{:?}", my_struct);

    // let buffer_content = buf_image.read().unwrap();
    // let image = ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, &buffer_content[..]).unwrap();
    // image.save("image.png").unwrap();

    let event_loop = EventLoop::new();

    let wgpu = futures::executor::block_on(RenginWgpu::new(&event_loop));

    let mut renderdoc_api: RenderDoc<renderdoc::V100> = RenderDoc::new().unwrap();
    renderdoc_api.start_frame_capture(std::ptr::null(), std::ptr::null());

    let cs_module = wgpu
        .device
        .create_shader_module(&wgpu::include_spirv!("shaders/raytracer.spv"));

    let output_buffer = wgpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (buffer_dimensions.padded_bytes_per_row * buffer_dimensions.height) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_extent = wgpu::Extent3d {
        width: buffer_dimensions.width as u32,
        height: buffer_dimensions.height as u32,
        depth_or_array_layers: 1,
    };

    // The render pipeline renders data into this texture
    let texture = wgpu.device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // format: wgpu::TextureFormat::Rgba32Float,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsage::STORAGE | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // let p_ubo: *const UBO = &ubo; // the same operator is used as with references
    // let p_ubo: *const u8 = p_ubo as *const u8;

    let buf_ubo = wgpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UBO Buffer"),
            contents: bytemuck::bytes_of(&ubo),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

    let buf_tlas = wgpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TLAS storage Buffer"),
            contents: bytemuck::cast_slice(&dragon_tlas),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

    let buf_blas = wgpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BLAS storage Buffer"),
            contents: bytemuck::cast_slice(&dragon_blas),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

    let buf_op = wgpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material storage Buffer"),
            contents: bytemuck::bytes_of(&object_params),
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_DST,
        });

    let compute_bind_group_layout =
        wgpu.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
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
                            // min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStage::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            // min_binding_size: None,
                            min_binding_size: wgpu::BufferSize::new(
                                (dragon_blas.len() * mem::size_of::<NodeBLAS>()) as _,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStage::COMPUTE,
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
            });
    let compute_pipeline_layout =
        wgpu.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

    let compute_pipeline = wgpu
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: "main",
        });

    let compute_bind_group = wgpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_ubo.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_tlas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_blas.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_op.as_entire_binding(),
            },
        ],
    });

    let mut command_encoder = wgpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    command_encoder.push_debug_group("compute ray trace");
    {
        // compute pass
        let mut cpass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &compute_bind_group, &[]);
        cpass.dispatch(WORKGROUP_SIZE, WORKGROUP_SIZE, 1);
    }
    command_encoder.pop_debug_group();

    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(
                    std::num::NonZeroU32::new(buffer_dimensions.padded_bytes_per_row as u32)
                        .unwrap(),
                ),
                rows_per_image: None,
                // rows_per_image: std::num::NonZeroU32::new(HEIGHT as u32),
            },
        },
        texture_extent,
    );

    wgpu.queue.submit(Some(command_encoder.finish()));

    futures::executor::block_on(create_png(
        "./image.png",
        wgpu.device,
        output_buffer,
        &buffer_dimensions,
    ));

    renderdoc_api.end_frame_capture(std::ptr::null(), std::ptr::null());

    // futures::executor::block_on(create_ppm(
    //     "./image.ppm",
    //     wgpu.device,
    //     output_buffer,
    //     &buffer_dimensions,
    // ));

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

async fn create_png(
    png_output_path: &str,
    device: wgpu::Device,
    output_buffer: wgpu::Buffer,
    buffer_dimensions: &BufferDimensions,
) {
    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);
    // If a file system is available, write the buffer as a PNG
    let has_file_system_available = cfg!(not(target_arch = "wasm32"));
    if !has_file_system_available {
        return;
    }

    if let Ok(()) = buffer_future.await {
        let padded_buffer = buffer_slice.get_mapped_range();

        let mut png_encoder = png::Encoder::new(
            File::create(png_output_path).unwrap(),
            buffer_dimensions.width as u32,
            buffer_dimensions.height as u32,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(buffer_dimensions.unpadded_bytes_per_row);

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(buffer_dimensions.padded_bytes_per_row) {
            png_writer
                .write_all(&chunk[..buffer_dimensions.unpadded_bytes_per_row])
                .unwrap();
        }
        png_writer.finish().unwrap();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);

        output_buffer.unmap();
    }
}

// async fn create_ppm(
//     ppm_output_path: &str,
//     device: wgpu::Device,
//     output_buffer: wgpu::Buffer,
//     buffer_dimensions: &BufferDimensions,
// ) {
//     // Note that we're not calling `.await` here.
//     let buffer_slice = output_buffer.slice(..);
//     let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

//     // Poll the device in a blocking manner so that our future resolves.
//     // In an actual application, `device.poll(...)` should
//     // be called in an event loop or on another thread.
//     device.poll(wgpu::Maintain::Wait);
//     // If a file system is available, write the buffer as a PNG
//     let has_file_system_available = cfg!(not(target_arch = "wasm32"));
//     if !has_file_system_available {
//         return;
//     }

//     if let Ok(()) = buffer_future.await {
//         let padded_buffer = buffer_slice.get_mapped_range();

//         let mut pnm_encoder = pnm::PnmEncoder::new(File::create(ppm_output_path).unwrap());
//         pnm_encoder.encode(
//             buffer_slice,
//             buffer_dimensions.padded_bytes_per_row as u32,
//             480,
//             image::ColorType::Rgb8,
//         );
//         output_buffer.unmap();
//     }
// }
