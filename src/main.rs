mod engine;
mod renderer;
mod shaders;

use wgpu::{Buffer, ShaderModule, Texture};

use winit::dpi::LogicalSize;
use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};

use std::time::{Duration, Instant};

use clap::Parser;
use engine::scene_importer::Scene;
use std::collections::HashMap;

use crate::engine::rt_primitives::Rays;
use crate::renderer::wgpu_utils::RenginWgpu;
use engine::rt_primitives::{Camera, UBO};

static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;

//TODO: try doing passes over parts of the image instead of whole at a time
//      that way you can maintain framerate

pub enum RendererType {
    PathTracer,
    RayTracer,
}

struct GameState {
    pub camera: Camera,
}

struct RenderApp {
    renderer: RenginWgpu,
    shaders: HashMap<&'static str, ShaderModule>,
    scene: Scene,
    buffers: HashMap<&'static str, Buffer>,
    texture: Texture,
    ubo: UBO,
    game_state: GameState,
}

impl RenderApp {
    pub fn new(
        scene_path: &str,
        event_loop: &EventLoop<()>,
        continous_motion: bool,
        rays_per_pixel: u32,
        renderer_type: RendererType,
    ) -> Self {
        let mut now = Instant::now();
        log::info!("Loading models...");

        let scene = Scene::new(scene_path);
        log::info!(
            "Finished loading models in {} millis.",
            now.elapsed().as_millis()
        );

        let mut renderer = futures::executor::block_on(RenginWgpu::new(
            scene.camera.as_ref().unwrap().width,
            scene.camera.as_ref().unwrap().height,
            WORKGROUP_SIZE,
            event_loop,
            continous_motion,
            rays_per_pixel,
        ));

        let game_state = GameState {
            camera: Camera::new(
                scene.camera.as_ref().unwrap().from,
                scene.camera.as_ref().unwrap().to,
                scene.camera.as_ref().unwrap().up,
            ),
        };

        let light_value = scene.lights.as_ref().unwrap()[0].at;
        let ubo = UBO::new(
            // [-4f32, 2f32, 3f32, 1f32],
            [light_value[0], light_value[1], light_value[2]],
            game_state.camera.get_inverse_transform(),
            scene.object_params.as_ref().unwrap().len() as u32,
            scene.camera.as_ref().unwrap().width,
            scene.camera.as_ref().unwrap().height,
            scene.camera.as_ref().unwrap().field_of_view,
            (renderer.rays_per_pixel as f32).sqrt() as u32,
        );

        let rays = Rays::empty(&renderer.resolution);

        println!("ubo: {:?}", ubo);

        // log::info!("ubo:{:?},", ubo);
        now = Instant::now();
        log::info!("Building shaders...");
        let shaders = renderer.create_shaders(renderer_type);
        log::info!(
            "Finshed building shaders in {} millis",
            now.elapsed().as_millis()
        );

        // TODO: update texture size on resize
        let texture_extent = wgpu::Extent3d {
            width: scene.camera.as_ref().unwrap().width,
            height: scene.camera.as_ref().unwrap().height,
            depth_or_array_layers: 1,
        };

        // The render pipeline renders data into this texture
        let texture = renderer.device.create_texture(&wgpu::TextureDescriptor {
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            label: None,
        });

        let buffers = renderer.create_buffers(
            scene.bvh.as_ref().unwrap(),
            &ubo,
            &rays,
            scene.object_params.as_ref().unwrap(),
        );
        renderer.create_pipelines(
            &buffers,
            &shaders,
            &texture,
            scene.bvh.as_ref().unwrap(),
            &rays,
        );

        Self {
            renderer,
            shaders,
            scene,
            buffers,
            texture,
            ubo,
            game_state,
        }
    }

    fn update_device_event(&mut self, event: DeviceEvent, left_mouse_down: &mut bool) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                // println!("x:{}, y:{}", position.x, position.y);
                if *left_mouse_down {
                    self.game_state
                        .camera
                        .rotate(delta.0 as f32, delta.1 as f32);

                    self.ubo.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.ubo.subpixel_idx = 0;
                    self.ubo.update_random_seed();
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    // println!("y: {}", y);
                    self.game_state.camera.move_forward(y as f32);

                    self.ubo.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.ubo.subpixel_idx = 0;
                    self.ubo.update_random_seed();
                }
                MouseScrollDelta::PixelDelta(xy) => {
                    // println!("pix xy: {:?}", xy);
                    self.game_state.camera.move_forward(xy.y as f32);

                    self.ubo.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.ubo.subpixel_idx = 0;
                    self.ubo.update_random_seed();
                } // _ => {}
            },
            _ => {}
        }
    }

    fn update_window_event(&mut self, event: WindowEvent, left_mouse_down: &mut bool) {
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                *left_mouse_down = true;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                *left_mouse_down = false;
            }
            _ => {}
        }
    }

    pub fn update(&mut self) {
        let rays = Rays::new(
            self.renderer.logical_size.width,
            self.renderer.logical_size.height,
            &self.renderer.resolution,
            &self.ubo,
        );

        let (ray_origins, ray_directions) = rays.make_buffers();

        self.renderer.queue.write_buffer(
            self.buffers.get("ray_origins").unwrap(),
            0,
            bytemuck::cast_slice(&ray_origins),
        );

        self.renderer.queue.write_buffer(
            self.buffers.get("ray_directions").unwrap(),
            0,
            bytemuck::cast_slice(&ray_directions),
        );

        self.renderer.queue.write_buffer(
            self.buffers.get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.ubo),
        );

        self.ubo.subpixel_idx += 1;
    }

    pub fn render(mut self, event_loop: EventLoop<()>) {
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;

        event_loop.run(move |event, _, control_flow| {
            // *control_flow = ControlFlow::Wait;
            match event {
                Event::MainEventsCleared => {
                    if self.ubo.subpixel_idx < self.renderer.rays_per_pixel {
                        self.update();

                        let frame = match self.renderer.window_surface.get_current_texture() {
                            Ok(frame) => frame,
                            Err(_) => {
                                self.renderer
                                    .window_surface
                                    .configure(&self.renderer.device, &self.renderer.config);
                                self.renderer
                                    .window_surface
                                    .get_current_texture()
                                    .expect("Failed to acquire next surface texture!")
                            }
                        };
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        // create render pass descriptor and its color attachments
                        let color_attachments = [wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        }];
                        let render_pass_descriptor = wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: None,
                        };

                        let mut command_encoder = self.renderer.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: None },
                        );

                        command_encoder.push_debug_group("compute ray trace");
                        {
                            // compute pass
                            let mut cpass = command_encoder
                                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                            cpass.set_pipeline(self.renderer.compute_pipeline.as_ref().unwrap());
                            cpass.set_bind_group(
                                0,
                                self.renderer.compute_bind_group.as_ref().unwrap(),
                                &[],
                            );

                            cpass.dispatch(
                                (self.renderer.logical_size.width / WORKGROUP_SIZE[0])
                                    + WORKGROUP_SIZE[0],
                                (self.renderer.logical_size.height / WORKGROUP_SIZE[1])
                                    + WORKGROUP_SIZE[1],
                                WORKGROUP_SIZE[2],
                            );
                        }
                        command_encoder.pop_debug_group();

                        command_encoder.push_debug_group("render texture");
                        {
                            // render pass
                            let mut rpass =
                                command_encoder.begin_render_pass(&render_pass_descriptor);
                            rpass.set_pipeline(self.renderer.render_pipeline.as_ref().unwrap());
                            rpass.set_bind_group(
                                0,
                                self.renderer.render_bind_group.as_ref().unwrap(),
                                &[],
                            );
                            // rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
                            // rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
                            rpass.draw(0..3, 0..1);
                        }
                        command_encoder.pop_debug_group();

                        self.renderer
                            .queue
                            .submit(std::iter::once(command_encoder.finish()));

                        self.renderer.device.poll(wgpu::Maintain::Wait);
                        frame.present();
                    }
                }
                Event::RedrawEventsCleared => {
                    let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    let time_since_last_frame = last_update_inst.elapsed();

                    if (!left_mouse_down || self.renderer.continous_motion)
                        && ((self.ubo.subpixel_idx < self.renderer.rays_per_pixel)
                            || (self.ubo.subpixel_idx == 0
                                && time_since_last_frame >= target_frametime))
                    {
                        println!("Drawing ray index: {}", self.ubo.subpixel_idx);

                        last_update_inst = Instant::now();
                    } else {
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }
                }
                Event::WindowEvent {
                    event:
                        WindowEvent::Resized(size)
                        | WindowEvent::ScaleFactorChanged {
                            new_inner_size: &mut size,
                            ..
                        },
                    ..
                } => {
                    // // println!("p: {} {}", size.width, size.height);
                    // // Reconfigure the surface with the new size
                    // self.renderer.config.width = size.width.max(1);
                    // self.renderer.config.height = size.height.max(1);
                    self.ubo.subpixel_idx = 0;

                    // let logical_size: LogicalSize<u32> =
                    //     winit::dpi::PhysicalSize::new(size.width, size.height)
                    //         .to_logical(self.renderer.scale_factor);

                    self.renderer.update_window_size(size.width, size.height);

                    self.ubo.update_dims(&self.renderer.logical_size);

                    // println!("l: {} {}", logical_size.width, logical_size.height);

                    let texture_extent = wgpu::Extent3d {
                        width: self.renderer.logical_size.width,
                        height: self.renderer.logical_size.height,
                        depth_or_array_layers: 1,
                    };

                    // The render pipeline renders data into this texture
                    self.texture = self
                        .renderer
                        .device
                        .create_texture(&wgpu::TextureDescriptor {
                            size: texture_extent,
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            usage: wgpu::TextureUsages::STORAGE_BINDING
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                            label: None,
                        });

                    let texture_view = self
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // TODO: move this into wgpu_utils function
                    self.renderer.compute_bind_group = Some(
                        self.renderer
                            .device
                            .create_bind_group(&wgpu::BindGroupDescriptor {
                                label: None,
                                layout: self.renderer.compute_bind_group_layout.as_ref().unwrap(),
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(&texture_view),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: self
                                            .buffers
                                            .get("ubo")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: self
                                            .buffers
                                            .get("tlas")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: self
                                            .buffers
                                            .get("blas")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: self
                                            .buffers
                                            .get("normals")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 5,
                                        resource: self
                                            .buffers
                                            .get("object_params")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 6,
                                        resource: self
                                            .buffers
                                            .get("ray_origins")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 7,
                                        resource: self
                                            .buffers
                                            .get("ray_directions")
                                            .as_ref()
                                            .unwrap()
                                            .as_entire_binding(),
                                    },
                                ],
                            }),
                    );

                    // TODO: change this to function in wgpu_utils
                    self.renderer.render_bind_group = Some(self.renderer.device.create_bind_group(
                        &wgpu::BindGroupDescriptor {
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&texture_view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(
                                        &self.renderer.sampler,
                                    ),
                                },
                            ],
                            layout: self.renderer.render_bind_group_layout.as_ref().unwrap(),
                            label: Some("bind group"),
                        },
                    ));

                    // println!("{} {}", size.width, size.height);
                    self.renderer
                        .window_surface
                        .configure(&self.renderer.device, &self.renderer.config);
                }
                Event::DeviceEvent { event, .. } => match event {
                    _ => {
                        self.update_device_event(event, &mut left_mouse_down);
                    }
                },
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    // #[cfg(not(target_arch = "wasm32"))]
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::R),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        println!("{:#?}", self.renderer.instance.generate_report());
                    }
                    _ => {
                        self.update_window_event(event, &mut left_mouse_down);
                    }
                },
                Event::RedrawRequested(_) => {}
                _ => {}
            }
        });
    }
}

/// A wgpu ray tracer
#[derive(Parser, Debug)]
#[clap(about, author)]
struct Args {
    /// Path to scene definition YAML
    #[clap(short, long)]
    scene: String,

    /// Only redraw on mouse up
    #[clap(short, long)]
    draw_on_mouseup: bool,

    /// Use Path tracer renderer
    #[clap(short, long)]
    pathtracer: bool,
    /// Number of rays per pixel
    #[clap(short, long, default_value_t = 8)]
    rays_per_pixel: u32,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let renderer_type = if args.pathtracer {
        RendererType::PathTracer
    } else {
        RendererType::RayTracer
    };

    let scene_path = &args.scene;
    let event_loop = EventLoop::new();
    let app = RenderApp::new(
        scene_path,
        &event_loop,
        !args.draw_on_mouseup,
        args.rays_per_pixel,
        renderer_type,
    );
    app.render(event_loop);
}
