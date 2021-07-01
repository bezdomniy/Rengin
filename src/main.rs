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
use shaders::mandelbrot;

fn main() {
    let objects = import_obj("assets/models/dragon.obj");
    let extensions = vulkano_win::required_extensions();
    let vulkan = RenginVulkan::new(&extensions);

    let image = StorageImage::new(
        vulkan.device.clone(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8Unorm,
        Some(vulkan.graphics_queue.family()),
    )
    .unwrap();

    let shader = mandelbrot::cs::Shader::load(vulkan.device.clone())
        .expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(vulkan.device.clone(), &shader.main_entry_point(), &(), None)
            .expect("failed to create compute pipeline"),
    );

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
        .build()
        .unwrap(),
    );

    let buf = CpuAccessibleBuffer::from_iter(
        vulkan.device.clone(),
        BufferUsage::all(),
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let mut builder = AutoCommandBufferBuilder::primary(
        vulkan.device.clone(),
        vulkan.graphics_queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .dispatch(
            [1024 / 8, 1024 / 8, 1],
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
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
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
