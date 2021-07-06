mod engine;
mod renderer;
mod shaders;

// use winit::{
//     event::{Event, WindowEvent},
//     event_loop::{ControlFlow, EventLoop},
//     window::WindowBuilder,
// };

// use vulkano_win::VkSurfaceBuild;

use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::ImageDimensions;
use vulkano::image::StorageImage;

use image::ImageBuffer;
use image::Rgba;

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::sync::{self, GpuFuture};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};

use std::mem;
use std::sync::Arc;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::ComputePipelineAbstract;

use engine::asset_importer::import_obj;
use renderer::vulkan_utils::RenginVulkan;
// use shaders::mandelbrot;
use shaders::raytracer;

use engine::rt_primitives::{Camera, NodeBLAS, NodeTLAS, UBO};

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

    let extensions = vulkano::instance::InstanceExtensions {
        ext_debug_report: true,
        ..vulkano_win::required_extensions()
    };
    let vulkan = RenginVulkan::new(&extensions);

    let image = StorageImage::new(
        vulkan.device.clone(),
        ImageDimensions::Dim2d {
            width: WIDTH,
            height: HEIGHT,
            array_layers: 1,
        },
        Format::R8G8B8A8Unorm,
        Some(vulkan.compute_queue.family()),
    )
    .unwrap();

    let shader =
        raytracer::cs::Shader::load(vulkan.device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(vulkan.device.clone(), &shader.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    // println!("{:?}", dragon_tlas);

    let buf_image = CpuAccessibleBuffer::from_iter(
        vulkan.device.clone(),
        BufferUsage::all(),
        false,
        (0..WIDTH * HEIGHT * 4).map(|_| 0u8),
    )
    .expect("failed to create output buffer");

    let buf_ubo = CpuAccessibleBuffer::from_data(
        vulkan.device.clone(),
        BufferUsage {
            uniform_buffer: true,
            ..BufferUsage::none()
        },
        false,
        ubo,
    )
    .expect("failed to create ubo buffer");

    let (buf_tlas, buf_blas) = {
        let buf_tlas_staging = CpuAccessibleBuffer::from_iter(
            vulkan.device.clone(),
            BufferUsage {
                transfer_source: true,
                ..BufferUsage::none()
            },
            false,
            dragon_tlas.iter().cloned(),
        )
        .expect("failed to create tlas buffer");

        let buf_blas_staging = CpuAccessibleBuffer::from_iter(
            vulkan.device.clone(),
            BufferUsage {
                transfer_source: true,
                ..BufferUsage::none()
            },
            false,
            dragon_blas.iter().cloned(),
        )
        .expect("failed to create blas buffer");

        let buf_tlas = unsafe {
            DeviceLocalBuffer::<[NodeTLAS]>::raw(
                vulkan.device.clone(),
                dragon_tlas.len() * mem::size_of::<NodeTLAS>(),
                BufferUsage {
                    storage_buffer: true,
                    transfer_destination: true,
                    ..BufferUsage::none()
                },
                vec![vulkan.transfer_queue.family()],
            )
            .unwrap()
        };

        let buf_blas = unsafe {
            DeviceLocalBuffer::<[NodeBLAS]>::raw(
                vulkan.device.clone(),
                dragon_blas.len() * mem::size_of::<NodeBLAS>(),
                BufferUsage {
                    storage_buffer: true,
                    transfer_destination: true,
                    ..BufferUsage::none()
                },
                vec![vulkan.transfer_queue.family()],
            )
            .unwrap()
        };

        // Build command buffer which initialize our buffer.
        let mut builder = AutoCommandBufferBuilder::primary(
            vulkan.device.clone(),
            vulkan.transfer_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        println!(
            "## {:?} {:?}",
            buf_tlas_staging.read().unwrap().len() * mem::size_of::<NodeTLAS>(),
            buf_blas_staging.read().unwrap().len() * mem::size_of::<NodeBLAS>(),
        );

        builder
            .copy_buffer(buf_blas_staging, buf_blas.clone())
            .unwrap()
            .copy_buffer(buf_tlas_staging, buf_tlas.clone())
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(vulkan.device.clone())
            .then_execute(vulkan.transfer_queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        (buf_blas, buf_tlas)
    };

    let set_image = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap()
                .clone(),
        )
        .add_image(ImageView::new(image.clone()).unwrap())
        .unwrap()
        .build()
        .unwrap(),
    );

    let set_uniform = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(1)
                .unwrap()
                .clone(),
        )
        .add_buffer(buf_ubo.clone())
        .unwrap()
        .build()
        .unwrap(),
    );

    let set_data = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(2)
                .unwrap()
                .clone(),
        )
        .add_buffer(buf_blas.clone())
        .unwrap()
        .add_buffer(buf_tlas.clone())
        .unwrap()
        .build()
        .unwrap(),
    );

    // let mut pool = FixedSizeDescriptorSetsPool::new(
    //     compute_pipeline
    //         .layout()
    //         .descriptor_set_layout(2)
    //         .unwrap()
    //         .clone(),
    // );

    // let set_data = Arc::new(
    //     pool.next()
    //         .add_buffer(buf_tlas.clone())
    //         .unwrap()
    //         .add_buffer(buf_blas.clone())
    //         .unwrap()
    //         .build()
    //         .unwrap(),
    // );

    let mut builder = AutoCommandBufferBuilder::primary(
        vulkan.device.clone(),
        vulkan.compute_queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .dispatch(
            [WIDTH / WORKGROUP_SIZE, HEIGHT / WORKGROUP_SIZE, 1],
            compute_pipeline.clone(),
            (set_image.clone(), set_uniform.clone(), set_data.clone()),
            (),
            vec![],
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf_image.clone())
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(vulkan.device.clone())
        .then_execute(vulkan.compute_queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf_image.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(WIDTH, HEIGHT, &buffer_content[..]).unwrap();
    image.save("image.png").unwrap();

    // let event_loop = EventLoop::new();
    // let surface = WindowBuilder::new()
    //     .build_vk_surface(&event_loop, vulkan.instance.clone())
    //     .unwrap();

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
