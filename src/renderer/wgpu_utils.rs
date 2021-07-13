use futures::io::Window;
use wgpu::{Adapter, Backend, Buffer, Device, Instance, QuerySet, Queue, Surface};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct RenginWgpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub window: winit::window::Window,
    pub window_surface: Surface,
}

impl RenginWgpu {
    pub async fn new(event_loop: &EventLoop<()>) -> RenginWgpu {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

        let window = WindowBuilder::new()
            .with_title("Blub")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(1980, 1080))
            .build(&event_loop)
            .unwrap();

        let window_surface = unsafe { instance.create_surface(&window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        RenginWgpu {
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,
            window: window,
            window_surface,
        }
    }
}
