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
    pub config: wgpu::SurfaceConfiguration,
}

impl RenginWgpu {
    pub async fn new(event_loop: &EventLoop<()>, width: u32, height: u32) -> RenginWgpu {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(backend);

        let window = WindowBuilder::new()
            .with_title("Rengin")
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
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

        // let adapter = instance
        //     .request_adapter(&wgpu::RequestAdapterOptions::default())
        //     .await
        //     .unwrap();

        let adapter_info = adapter.get_info();
        println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);
        println!("{:?}\n{:?}", adapter.features(), wgpu::Features::default());

        let trace_dir = std::env::var("WGPU_TRACE");

        let optional_features = {
            wgpu::Features::UNSIZED_BINDING_ARRAY
                | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
                | wgpu::Features::PUSH_CONSTANTS
        };
        let required_features =
            { wgpu::Features::TEXTURE_BINDING_ARRAY | wgpu::Features::SPIRV_SHADER_PASSTHROUGH };
        let required_limits = {
            wgpu::Limits {
                max_push_constant_size: 4,
                ..wgpu::Limits::default()
            }
        };
        let adapter_features = adapter.features();
        assert!(
            adapter_features.contains(required_features),
            "Adapter does not support required features for this example: {:?}",
            required_features - adapter_features
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: (optional_features & adapter_features) | required_features,
                    limits: required_limits,
                },
                trace_dir.ok().as_ref().map(std::path::Path::new),
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: window_surface.get_preferred_format(&adapter).unwrap(),
            width: width,
            height: height,
            present_mode: wgpu::PresentMode::Mailbox,
        };
        window_surface.configure(&device, &config);

        RenginWgpu {
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,
            window: window,
            window_surface,
            config,
        }
    }
}
