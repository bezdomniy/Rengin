mod engine;
mod renderer;
mod shaders;

use winit::event::{
    DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
    WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop};

use std::process::exit;
use std::time::{Duration, Instant};

use clap::Parser;
use engine::scene_importer::Scene;

use crate::engine::rt_primitives::Rays;
use crate::renderer::wgpu_utils::RenginWgpu;
use engine::rt_primitives::{Camera, ScreenData};

static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;

pub enum RendererType {
    PathTracer,
    RayTracer,
}

struct GameState {
    pub camera: Camera,
}

struct RenderApp {
    renderer: RenginWgpu,
    screen_data: ScreenData,
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
        let screen_data = ScreenData::new(
            [light_value[0], light_value[1], light_value[2]],
            game_state.camera.get_inverse_transform(),
            scene.object_params.as_ref().unwrap().len() as u32,
            scene.camera.as_ref().unwrap().width,
            scene.camera.as_ref().unwrap().height,
            scene.camera.as_ref().unwrap().field_of_view,
            (renderer.rays_per_pixel as f32).sqrt() as u32,
        );

        let rays = Rays::empty(&renderer.resolution);

        println!("screen_data: {:?}", screen_data);

        now = Instant::now();
        log::info!("Building shaders...");
        let shaders = renderer.create_shaders(renderer_type);
        log::info!(
            "Finshed building shaders in {} millis",
            now.elapsed().as_millis()
        );

        renderer.create_buffers(
            scene.bvh.as_ref().unwrap(),
            &screen_data,
            &rays,
            scene.object_params.as_ref().unwrap(),
        );
        renderer.create_pipelines(
            // &buffers,
            &shaders,
            // &renderer.target_texture,
            scene.bvh.as_ref().unwrap(),
            &rays,
        );

        Self {
            renderer,
            screen_data,
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
                        .rotate(-delta.0 as f32, delta.1 as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    self.screen_data.update_random_seed();
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    // println!("y: {}", y);
                    self.game_state.camera.move_forward(y as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    self.screen_data.update_random_seed();
                }
                MouseScrollDelta::PixelDelta(xy) => {
                    // println!("pix xy: {:?}", xy);
                    self.game_state.camera.move_forward(xy.y as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    self.screen_data.update_random_seed();
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
            &self.renderer.physical_size,
            &self.renderer.resolution,
            &self.screen_data,
        );

        self.renderer.queue.write_buffer(
            self.renderer.buffers.as_ref().unwrap().get("rays").unwrap(),
            0,
            bytemuck::cast_slice(&rays.data),
        );
        self.renderer.queue.write_buffer(
            self.renderer.buffers.as_ref().unwrap().get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.screen_data.generate_ubo()),
        );

        self.screen_data.subpixel_idx += 1;
    }

    pub fn render(mut self, event_loop: EventLoop<()>) {
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;

        event_loop.run(move |event, _, control_flow| {
            // *control_flow = ControlFlow::Wait;
            match event {
                Event::MainEventsCleared => {
                    if self.screen_data.subpixel_idx < self.renderer.rays_per_pixel {
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
                                (self.renderer.physical_size.width / WORKGROUP_SIZE[0])
                                    + WORKGROUP_SIZE[0],
                                (self.renderer.physical_size.height / WORKGROUP_SIZE[1])
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
                        && ((self.screen_data.subpixel_idx < self.renderer.rays_per_pixel)
                            || (self.screen_data.subpixel_idx == 0
                                && time_since_last_frame >= target_frametime))
                    {
                        println!("Drawing ray index: {}", self.screen_data.subpixel_idx);

                        last_update_inst = Instant::now();
                    } else {
                        // exit(0);
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
                    self.screen_data.subpixel_idx = 0;

                    self.renderer.update_window_size(size.width, size.height);

                    self.screen_data.update_dims(&self.renderer.physical_size);

                    // println!("l: {} {}", logical_size.width, logical_size.height);

                    self.renderer.create_bind_groups();

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
