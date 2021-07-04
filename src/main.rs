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
use vulkano::image::view::ImageViewAbstract;
use vulkano::image::ImageDimensions;
use vulkano::image::StorageImage;

use image::ImageBuffer;
use image::Rgba;

use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::sync::{self, GpuFuture};

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;

use std::sync::Arc;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::ComputePipelineAbstract;

use engine::asset_importer::import_obj;
use renderer::vulkan_utils::RenginVulkan;
// use shaders::mandelbrot;
use shaders::raytracer;

use engine::rt_primitives::{Camera, UBO};

static WIDTH: u32 = 2400;
static HEIGHT: u32 = 1800;
static WORKGROUP_SIZE: u32 = 32;

fn main() {
    let objects = import_obj("assets/models/cube.obj");
    let (dragon_tlas, dragon_blas) = &objects[0];

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
        Some(vulkan.graphics_queue.family()),
    )
    .unwrap();

    let shader =
        raytracer::cs::Shader::load(vulkan.device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(vulkan.device.clone(), &shader.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

    // println!("{:?}", dragon_tlas);

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

    let buf_tlas = CpuAccessibleBuffer::from_iter(
        vulkan.device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        false,
        dragon_tlas.iter().cloned(),
    )
    .expect("failed to create tlas buffer");

    let buf_blas = CpuAccessibleBuffer::from_iter(
        vulkan.device.clone(),
        BufferUsage {
            storage_buffer: true,
            ..BufferUsage::none()
        },
        false,
        dragon_blas.iter().cloned(),
    )
    .expect("failed to create blas buffer");

    let set = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap()
                .clone(),
        )
        .add_image(ImageView::new(image.clone()).unwrap())
        .unwrap()
        .add_buffer(buf_ubo.clone())
        .unwrap()
        .add_buffer(buf_tlas.clone())
        .unwrap()
        .add_buffer(buf_blas.clone())
        .unwrap()
        .build()
        .unwrap(),
    );

    let buf = CpuAccessibleBuffer::from_iter(
        vulkan.device.clone(),
        BufferUsage::all(),
        false,
        (0..WIDTH * HEIGHT * 4).map(|_| 0u8),
    )
    .expect("failed to create output buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        vulkan.device.clone(),
        vulkan.graphics_queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .dispatch(
            [WIDTH / WORKGROUP_SIZE, HEIGHT / WORKGROUP_SIZE, 1],
            compute_pipeline.clone(),
            set.clone(),
            (),
            vec![],
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(vulkan.device.clone())
        .then_execute(vulkan.graphics_queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
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
