mod renderer;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use vulkano_win::VkSurfaceBuild;

use renderer::vulkan_utils::RenginVulkan;


fn main() {
    let extensions = vulkano_win::required_extensions();
    let vulkan = RenginVulkan::new(&extensions);
    
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, vulkan.instance.clone()).unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            _ => (),
        }
    });
}
