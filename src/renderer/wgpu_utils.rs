use wgpu::{Adapter, Device, Instance, Queue, Surface};
use winit::{dpi::PhysicalSize, event_loop::EventLoop, window::WindowBuilder};

pub struct RenginWgpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub window: winit::window::Window,
    pub window_surface: Surface,
    pub config: wgpu::SurfaceConfiguration,
    pub height: u32,
    pub width: u32,
    pub workgroup_size: [u32; 3],
    pub continous_motion: bool,
    pub scale_factor: f64,
}

impl RenginWgpu {
    pub async fn new(
        width: u32,
        height: u32,
        workgroup_size: [u32; 3],
        event_loop: &EventLoop<()>,
        continous_motion: bool,
    ) -> RenginWgpu {
        let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
        let instance = wgpu::Instance::new(backend);

        let scale_factor: f64 = event_loop.primary_monitor().unwrap().scale_factor();
        let physical_size: PhysicalSize<f64> =
            winit::dpi::LogicalSize::new(width, height).to_physical(scale_factor);

        let window = WindowBuilder::new()
            .with_title("Rengin")
            .with_resizable(true)
            .with_inner_size(physical_size)
            .build(event_loop)
            .unwrap();

        let size = window.inner_size();

        let window_surface = unsafe { instance.create_surface(&window) };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&window_surface),
                force_fallback_adapter: false,
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
        let required_features = { wgpu::Features::TEXTURE_BINDING_ARRAY };
        let required_limits = {
            wgpu::Limits {
                max_push_constant_size: 4,
                ..wgpu::Limits::default()
            }
        }
        .using_resolution(adapter.limits());
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
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        // println!("{} {}", width, height);
        window_surface.configure(&device, &config);

        RenginWgpu {
            instance: instance,
            adapter: adapter,
            device: device,
            queue: queue,
            window: window,
            window_surface,
            config,
            width,
            height,
            workgroup_size,
            continous_motion,
            scale_factor,
        }
    }

    // pub fn render(
    //     mut self,
    //     event_loop: EventLoop<()>,
    //     compute_pipeline: &'static ComputePipeline,
    //     compute_bind_group: &'static BindGroup,
    //     render_pipeline: &'static RenderPipeline,
    //     render_bind_group: &'static BindGroup,
    // ) {
    //     // let Self {
    //     //     instance,
    //     //     adapter,
    //     //     device,
    //     //     queue,
    //     //     window,
    //     //     window_surface,
    //     //     mut event_loop,
    //     //     mut config,
    //     //     height,
    //     //     width,
    //     //     workgroup_size,
    //     // } = self;

    //     event_loop.run(move |event, _, control_flow| {
    //         // Have the closure take ownership of the resources.
    //         // `event_loop.run` never returns, therefore we must do this to ensure
    //         // the resources are properly cleaned up.
    //         let _ = (&self.instance, &self.adapter, &compute_pipeline); //, &self.device, &self.config);

    //         *control_flow = ControlFlow::Wait;
    //         match event {
    //             Event::WindowEvent {
    //                 event: WindowEvent::Resized(size),
    //                 ..
    //             } => {
    //                 // // Reconfigure the surface with the new size
    //                 // config.width = size.width;
    //                 // config.height = size.height;
    //                 // surface.configure(&device, &config);
    //             }
    //             Event::RedrawRequested(_) => {
    //                 let frame = match self.window_surface.get_current_frame() {
    //                     Ok(frame) => frame,
    //                     Err(_) => {
    //                         self.window_surface.configure(&&self.device, &&self.config);
    //                         self.window_surface
    //                             .get_current_frame()
    //                             .expect("Failed to acquire next surface texture!")
    //                     }
    //                 };
    //                 let view = frame
    //                     .output
    //                     .texture
    //                     .create_view(&wgpu::TextureViewDescriptor::default());

    //                 // create render pass descriptor and its color attachments
    //                 let color_attachments = [wgpu::RenderPassColorAttachment {
    //                     view: &view,
    //                     resolve_target: None,
    //                     ops: wgpu::Operations {
    //                         load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    //                         store: true,
    //                     },
    //                 }];
    //                 let render_pass_descriptor = wgpu::RenderPassDescriptor {
    //                     label: None,
    //                     color_attachments: &color_attachments,
    //                     depth_stencil_attachment: None,
    //                 };

    //                 let mut command_encoder = self
    //                     .device
    //                     .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    //                 command_encoder.push_debug_group("compute ray trace");
    //                 {
    //                     // compute pass
    //                     let mut cpass = command_encoder
    //                         .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
    //                     cpass.set_pipeline(compute_pipeline);
    //                     cpass.set_bind_group(0, compute_bind_group, &[]);
    //                     cpass.dispatch(
    //                         self.width / self.workgroup_size[0],
    //                         self.height / self.workgroup_size[1],
    //                         self.workgroup_size[2],
    //                     );
    //                 }
    //                 command_encoder.pop_debug_group();

    //                 command_encoder.push_debug_group("render texture");
    //                 {
    //                     // render pass
    //                     let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
    //                     rpass.set_pipeline(render_pipeline);
    //                     rpass.set_bind_group(0, render_bind_group, &[]);
    //                     // rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
    //                     // rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
    //                     rpass.draw(0..3, 0..1);
    //                 }
    //                 command_encoder.pop_debug_group();

    //                 self.queue.submit(Some(command_encoder.finish()));
    //             }
    //             Event::WindowEvent {
    //                 event: WindowEvent::CloseRequested,
    //                 ..
    //             } => *control_flow = ControlFlow::Exit,
    //             _ => {}
    //         }
    //     });

    //     // let frame = match self.window_surface.get_current_frame() {
    //     //     Ok(frame) => frame,
    //     //     Err(_) => {
    //     //         self.window_surface.configure(&self.device, &self.config);
    //     //         self.window_surface
    //     //             .get_current_frame()
    //     //             .expect("Failed to acquire next surface texture!")
    //     //     }
    //     // };
    //     // let view = frame
    //     //     .output
    //     //     .texture
    //     //     .create_view(&wgpu::TextureViewDescriptor::default());

    //     // // create render pass descriptor and its color attachments
    //     // let color_attachments = [wgpu::RenderPassColorAttachment {
    //     //     view: &view,
    //     //     resolve_target: None,
    //     //     ops: wgpu::Operations {
    //     //         load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
    //     //         store: true,
    //     //     },
    //     // }];
    //     // let render_pass_descriptor = wgpu::RenderPassDescriptor {
    //     //     label: None,
    //     //     color_attachments: &color_attachments,
    //     //     depth_stencil_attachment: None,
    //     // };

    //     // let mut command_encoder = self
    //     //     .device
    //     //     .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    //     // command_encoder.push_debug_group("compute ray trace");
    //     // {
    //     //     // compute pass
    //     //     let mut cpass =
    //     //         command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
    //     //     cpass.set_pipeline(compute_pipeline);
    //     //     cpass.set_bind_group(0, compute_bind_group, &[]);
    //     //     cpass.dispatch(
    //     //         self.width / self.workgroup_size[0],
    //     //         self.height / self.workgroup_size[1],
    //     //         self.workgroup_size[2],
    //     //     );
    //     // }
    //     // command_encoder.pop_debug_group();

    //     // command_encoder.push_debug_group("render texture");
    //     // {
    //     //     // render pass
    //     //     let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
    //     //     rpass.set_pipeline(render_pipeline);
    //     //     rpass.set_bind_group(0, render_bind_group, &[]);
    //     //     // rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
    //     //     // rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
    //     //     rpass.draw(0..3, 0..1);
    //     // }
    //     // command_encoder.pop_debug_group();

    //     // self.queue.submit(Some(command_encoder.finish()));
    // }
}
