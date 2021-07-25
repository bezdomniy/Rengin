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
            .with_title("Rengin")
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

        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

        let trace_dir = std::env::var("WGPU_TRACE");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: adapter.features() | wgpu::Features::default(),
                    limits: wgpu::Limits::default(),
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
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
