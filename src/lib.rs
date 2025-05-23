mod engine;
mod renderer;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta,
    WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use engine::rt_primitives::{Camera, ScreenData};
use engine::scene_importer::Scene;
use renderer::RenginRenderer;
use renderer::wgpu_utils::RenginWgpu;

static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;

pub enum RendererType {
    PathTracer,
    RayTracer,
}

struct GameState {
    pub camera: Camera,
}

struct State {
    renderer: RenginWgpu,
    screen_data: ScreenData,
    game_state: GameState,
    left_mouse_down: bool,
    last_update_inst: Instant,
}

#[derive(Default)]
pub struct RenderApp {
    state: Option<State>,
}

impl State {
    pub async fn new(
        window: Arc<Window>,
        resolution: &PhysicalSize<u32>,
        scene: &Scene,
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
        renderer_type: RendererType,
    ) -> Self {
        let physical_size = window.inner_size();

        let mut renderer = RenginWgpu::new(
            window,
            // WORKGROUP_SIZE,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
        )
        .await;

        let game_state = GameState {
            camera: Camera::new(
                scene.camera.as_ref().unwrap().from,
                scene.camera.as_ref().unwrap().to,
                scene.camera.as_ref().unwrap().up,
            ),
        };

        let screen_data = ScreenData::new(
            game_state.camera.get_inverse_transform(),
            scene.object_params.as_ref().unwrap().len() as u32,
            scene.specular_offset as u32,
            scene.lights_offset as u32,
            ray_bounces,
            physical_size,
            *resolution,
            scene.camera.as_ref().unwrap().field_of_view,
            (renderer.rays_per_pixel as f32).sqrt() as u32,
        );

        log::debug!("screen_data: {:?}", screen_data);

        let now = Instant::now();
        log::info!("Building shaders...");
        renderer.create_shaders(renderer_type);
        log::info!(
            "Finshed building shaders in {} millis",
            now.elapsed().as_millis()
        );

        renderer.create_buffers(
            scene.bvh.as_ref().unwrap(),
            &screen_data,
            scene.object_params.as_ref().unwrap(),
        );
        renderer.create_pipelines(
            scene.bvh.as_ref().unwrap(),
            &screen_data,
            scene.object_params.as_ref().unwrap(),
        );

        // TODO: remove buffers as arg and move into RenginWgpu state
        renderer.create_bind_groups(&physical_size);

        Self {
            renderer,
            screen_data,
            game_state,
            left_mouse_down: false,
            last_update_inst: Instant::now(),
        }
    }

    fn update_device_event(&mut self, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                // println!("x:{}, y:{}", position.x, position.y);
                if self.left_mouse_down {
                    self.game_state
                        .camera
                        .rotate(-delta.0 as f32, delta.1 as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    // self.screen_data.update_random_seed();
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    // println!("y: {}", y);
                    self.game_state.camera.move_forward(y as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    // self.screen_data.update_random_seed();
                }
                MouseScrollDelta::PixelDelta(xy) => {
                    // println!("pix xy: {:?}", xy);
                    self.game_state.camera.move_forward(xy.y as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                    // self.screen_data.update_random_seed();
                } // _ => {}
            },
            _ => {}
        }
    }

    fn update_window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.left_mouse_down = true;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.left_mouse_down = false;
            }
            _ => {}
        }
    }

    pub fn update(&mut self) {
        self.renderer.queue.write_buffer(
            self.renderer.buffers.as_ref().unwrap().get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.screen_data.generate_ubo()),
        );

        self.screen_data.subpixel_idx += 1;

        self.generate_primary_rays();
    }

    fn generate_primary_rays(&self) {
        let mut command_encoder = self
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        command_encoder.push_debug_group("raygen");
        {
            // compute pass
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Default::default(),
            });
            cpass.set_pipeline(self.renderer.raygen_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, self.renderer.raygen_bind_group.as_ref().unwrap(), &[]);

            cpass.dispatch_workgroups(
                self.screen_data.size.width / WORKGROUP_SIZE[0],
                // + (self.screen_data.size.width % WORKGROUP_SIZE[0]),
                self.screen_data.size.height / WORKGROUP_SIZE[1],
                // + (self.screen_data.size.height % WORKGROUP_SIZE[1]),
                WORKGROUP_SIZE[2],
            );
        }
        command_encoder.pop_debug_group();

        self.renderer
            .queue
            .submit(std::iter::once(command_encoder.finish()));

        // self.renderer.device.poll(wgpu::Maintain::Wait);
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let monitor = event_loop.available_monitors().next().unwrap();
        let monitor_scale_factor = monitor.scale_factor();
        let resolution = monitor.size();

        let args = Args::parse();
        let renderer_type = if args.whitted {
            panic!("Whitted Ray Tracer has been deprecated and will no longer work.");
        } else {
            RendererType::PathTracer
        };

        let now = Instant::now();
        log::info!("Loading models...{}", args.scene);

        let scene = Scene::new(&args.scene, &renderer_type);
        log::info!(
            "Finished loading models in {} millis.",
            now.elapsed().as_millis()
        );

        let logical_size: LogicalSize<u32> = winit::dpi::LogicalSize::new(
            scene.camera.as_ref().unwrap().width,
            scene.camera.as_ref().unwrap().height,
        );
        let physical_size: PhysicalSize<u32> = logical_size.to_physical(monitor_scale_factor);

        let window_attributes = WindowAttributes::default()
            .with_title("Fantastic window number one!")
            .with_resizable(true)
            .with_inner_size(physical_size);

        // Create window object
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let state = pollster::block_on(State::new(
            window.clone(),
            &resolution,
            &scene,
            !args.draw_on_mouseup,
            args.rays_per_pixel,
            args.bounces,
            renderer_type,
        ));
        self.state = Some(state);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();

        match event {
            WindowEvent::Resized(size) => {
                state.screen_data.update_dims(&size);
                state.screen_data.subpixel_idx = 0;
                state.renderer.update_window_size(&size);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested => event_loop.exit(),

            // #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(s),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if s == "R" => {
                println!("{:#?}", state.renderer.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {}
            _ => {
                state.update_window_event(event);
            }
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let state = self.state.as_mut().unwrap();
        state.update_device_event(event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let state = self.state.as_mut().unwrap();
        let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
        // let mut last_update_inst = Instant::now();
        let time_since_last_frame = state.last_update_inst.elapsed();

        if (!state.left_mouse_down || state.renderer.continous_motion)
            && ((state.screen_data.subpixel_idx < state.renderer.rays_per_pixel)
                || (state.screen_data.subpixel_idx == 0
                    && time_since_last_frame >= target_frametime))
        {
            log::info!(
                "Drawing ray index: {}, framerate: {}",
                state.screen_data.subpixel_idx,
                1000u128 / time_since_last_frame.as_millis()
            );

            state.last_update_inst = Instant::now();
            if state.screen_data.subpixel_idx < state.renderer.rays_per_pixel {
                state.update();

                let frame = match state.renderer.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        state.renderer
                            .surface
                            .configure(&state.renderer.device, &state.renderer.config);
                        state.renderer
                            .surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // TODO: create send texture to use to keep track of previous frame

                // create render pass descriptor and its color attachments
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // load: wgpu::LoadOp::Load,
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let render_pass_descriptor = wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: None,
                    occlusion_query_set: Default::default(),
                    timestamp_writes: Default::default(),
                };

                let mut command_encoder = state
                    .renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                command_encoder.push_debug_group("compute ray trace");
                {
                    // compute pass
                    let mut cpass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: Default::default(),
                        });
                    cpass.set_pipeline(state.renderer.compute_pipeline.as_ref().unwrap());
                    cpass.set_bind_group(
                        0,
                        state.renderer.compute_bind_group.as_ref().unwrap(),
                        &[],
                    );

                    for _ in 0..state.renderer.ray_bounces {
                        cpass.dispatch_workgroups(
                            state.screen_data.size.width / WORKGROUP_SIZE[0],
                            // + (self.screen_data.size.width % WORKGROUP_SIZE[0]),
                            state.screen_data.size.height / WORKGROUP_SIZE[1],
                            // + (self.screen_data.size.height % WORKGROUP_SIZE[1]),
                            WORKGROUP_SIZE[2],
                        );
                    }
                }
                command_encoder.pop_debug_group();

                command_encoder.push_debug_group("render texture");
                {
                    // render pass
                    let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
                    rpass.set_pipeline(state.renderer.render_pipeline.as_ref().unwrap());
                    rpass.set_bind_group(0, state.renderer.render_bind_group.as_ref().unwrap(), &[]);
                    // rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
                    // rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));
                    rpass.draw(0..3, 0..1);
                }
                command_encoder.pop_debug_group();

                state.renderer
                    .queue
                    .submit(std::iter::once(command_encoder.finish()));

                let _ = state.renderer.device.poll(wgpu::PollType::Wait);
                frame.present();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + target_frametime - time_since_last_frame,
            ))
        }

        state.renderer.window.request_redraw();
    }
}

/// A wgpu ray tracer
#[derive(Parser, Debug, Default)]
#[clap(about, author)]
pub struct Args {
    /// Path to scene definition YAML
    #[clap(short, long)]
    scene: String,

    /// Only redraw on mouse up
    #[clap(short, long)]
    draw_on_mouseup: bool,

    /// Use Whitted ray tracer renderer (deprecated - no longer works)
    #[clap(short, long)]
    whitted: bool,
    /// Number of rays per pixel
    #[clap(short, long, default_value_t = 8)]
    rays_per_pixel: u32,
    /// Number of bounces per ray
    #[clap(short, long, default_value_t = 8)]
    bounces: u32,
}

