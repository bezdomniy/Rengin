mod engine;
mod renderer;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
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
        profile: bool,
    ) -> Self {
        let physical_size = window.inner_size();

        let mut renderer = RenginWgpu::new(
            window,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
            profile,
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
            args.profile,
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
            state.last_update_inst = Instant::now();
            if state.screen_data.subpixel_idx < state.renderer.rays_per_pixel {
                state.update();

                let frame = match state.renderer.surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(_) => {
                        state
                            .renderer
                            .surface
                            .configure(&state.renderer.device, &state.renderer.config);
                        state
                            .renderer
                            .surface
                            .get_current_texture()
                            .expect("Failed to acquire next surface texture!")
                    }
                };
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // create render pass descriptor and its color attachments
                let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })];
                let mut command_encoder = state
                    .renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                if state.renderer.profile {
                    // === PROFILED MODE: separate passes with GPU timestamps ===
                    let mut query_idx: u32 = 0;
                    let qs = state.renderer.timestamp_query_set.as_ref().unwrap();

                    // 1. Raygen pass
                    {
                        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("raygen"),
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: qs,
                                beginning_of_pass_write_index: Some(query_idx),
                                end_of_pass_write_index: Some(query_idx + 1),
                            }),
                        });
                        cpass.set_pipeline(state.renderer.raygen_pipeline.as_ref().unwrap());
                        cpass.set_bind_group(0, state.renderer.raygen_bind_group.as_ref().unwrap(), &[]);
                        cpass.dispatch_workgroups(
                            state.screen_data.size.width / WORKGROUP_SIZE[0],
                            state.screen_data.size.height / WORKGROUP_SIZE[1],
                            WORKGROUP_SIZE[2],
                        );
                    }
                    query_idx += 2;

                    // 2. Each bounce in its own pass for per-bounce timing
                    for _bounce in 0..state.renderer.ray_bounces {
                        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pathtracer bounce"),
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: qs,
                                beginning_of_pass_write_index: Some(query_idx),
                                end_of_pass_write_index: Some(query_idx + 1),
                            }),
                        });
                        cpass.set_pipeline(state.renderer.compute_pipeline.as_ref().unwrap());
                        cpass.set_bind_group(0, state.renderer.compute_bind_group.as_ref().unwrap(), &[]);
                        cpass.dispatch_workgroups(
                            state.screen_data.size.width / WORKGROUP_SIZE[0],
                            state.screen_data.size.height / WORKGROUP_SIZE[1],
                            WORKGROUP_SIZE[2],
                        );
                        query_idx += 2;
                    }

                    // 3. Render pass
                    {
                        let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            multiview_mask: None,
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: None,
                            occlusion_query_set: Default::default(),
                            timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                                query_set: qs,
                                beginning_of_pass_write_index: Some(query_idx),
                                end_of_pass_write_index: Some(query_idx + 1),
                            }),
                        });
                        rpass.set_pipeline(state.renderer.render_pipeline.as_ref().unwrap());
                        rpass.set_bind_group(0, state.renderer.render_bind_group.as_ref().unwrap(), &[]);
                        rpass.draw(0..3, 0..1);
                    }
                    query_idx += 2;

                    // Resolve + copy timestamps
                    command_encoder.resolve_query_set(
                        qs, 0..query_idx,
                        state.renderer.timestamp_resolve_buffer.as_ref().unwrap(), 0,
                    );
                    command_encoder.copy_buffer_to_buffer(
                        state.renderer.timestamp_resolve_buffer.as_ref().unwrap(), 0,
                        state.renderer.timestamp_read_buffer.as_ref().unwrap(), 0,
                        (query_idx as u64) * 8,
                    );

                    state.renderer.queue.submit(std::iter::once(command_encoder.finish()));
                    let _ = state.renderer.device.poll(wgpu::PollType::Wait {
                        submission_index: None, timeout: None,
                    });

                    // Read back GPU timestamps
                    let read_buf = state.renderer.timestamp_read_buffer.as_ref().unwrap();
                    let buf_slice = read_buf.slice(..((query_idx as u64) * 8));
                    let (tx, rx) = std::sync::mpsc::channel();
                    buf_slice.map_async(wgpu::MapMode::Read, move |result| {
                        tx.send(result).unwrap();
                    });
                    let _ = state.renderer.device.poll(wgpu::PollType::Wait {
                        submission_index: None, timeout: None,
                    });
                    if let Ok(Ok(())) = rx.recv() {
                        let data = buf_slice.get_mapped_range();
                        let timestamps: &[u64] = bytemuck::cast_slice(&data);
                        let period = state.renderer.timestamp_period as f64;

                        let raygen_ns = (timestamps[1] - timestamps[0]) as f64 * period;
                        let mut bounce_strs = Vec::new();
                        let mut total_bounce_ns = 0.0;
                        for b in 0..state.renderer.ray_bounces {
                            let idx = (2 + b * 2) as usize;
                            let bounce_ns = (timestamps[idx + 1] - timestamps[idx]) as f64 * period;
                            total_bounce_ns += bounce_ns;
                            bounce_strs.push(format!("b{}={:.2}ms", b, bounce_ns / 1_000_000.0));
                        }
                        let render_base = (2 + state.renderer.ray_bounces * 2) as usize;
                        let render_ns = timestamps[render_base + 1].wrapping_sub(timestamps[render_base]) as f64 * period;
                        let render_ns = if render_ns > 1e12 { 0.0 } else { render_ns };

                        log::info!(
                            "GPU Sample {}/{} | raygen={:.2}ms | pathtracer={:.2}ms [{}] | render={:.2}ms | total={:.2}ms",
                            state.screen_data.subpixel_idx,
                            state.renderer.rays_per_pixel,
                            raygen_ns / 1_000_000.0,
                            total_bounce_ns / 1_000_000.0,
                            bounce_strs.join(", "),
                            render_ns / 1_000_000.0,
                            (raygen_ns + total_bounce_ns + render_ns) / 1_000_000.0,
                        );

                        drop(data);
                        read_buf.unmap();
                    }
                } else {
                    // === DEFAULT MODE: batched passes, simple fps logging ===

                    // 1. Raygen
                    {
                        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("raygen"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(state.renderer.raygen_pipeline.as_ref().unwrap());
                        cpass.set_bind_group(0, state.renderer.raygen_bind_group.as_ref().unwrap(), &[]);
                        cpass.dispatch_workgroups(
                            state.screen_data.size.width / WORKGROUP_SIZE[0],
                            state.screen_data.size.height / WORKGROUP_SIZE[1],
                            WORKGROUP_SIZE[2],
                        );
                    }

                    // 2. All bounces in a single compute pass
                    {
                        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pathtracer"),
                            timestamp_writes: None,
                        });
                        cpass.set_pipeline(state.renderer.compute_pipeline.as_ref().unwrap());
                        cpass.set_bind_group(0, state.renderer.compute_bind_group.as_ref().unwrap(), &[]);
                        for _ in 0..state.renderer.ray_bounces {
                            cpass.dispatch_workgroups(
                                state.screen_data.size.width / WORKGROUP_SIZE[0],
                                state.screen_data.size.height / WORKGROUP_SIZE[1],
                                WORKGROUP_SIZE[2],
                            );
                        }
                    }

                    // 3. Render pass
                    {
                        let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            multiview_mask: None,
                            color_attachments: &color_attachments,
                            depth_stencil_attachment: None,
                            occlusion_query_set: Default::default(),
                            timestamp_writes: None,
                        });
                        rpass.set_pipeline(state.renderer.render_pipeline.as_ref().unwrap());
                        rpass.set_bind_group(0, state.renderer.render_bind_group.as_ref().unwrap(), &[]);
                        rpass.draw(0..3, 0..1);
                    }

                    state.renderer.queue.submit(std::iter::once(command_encoder.finish()));
                    let _ = state.renderer.device.poll(wgpu::PollType::Wait {
                        submission_index: None, timeout: None,
                    });

                    log::info!(
                        "Sample {}/{}, framerate: {} fps",
                        state.screen_data.subpixel_idx,
                        state.renderer.rays_per_pixel,
                        1000u128 / time_since_last_frame.as_millis().max(1),
                    );
                }

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
    /// Enable detailed GPU profiling (per-bounce timestamps)
    #[clap(short = 'p', long)]
    profile: bool,
    /// Number of bounces per ray
    #[clap(short, long, default_value_t = 8)]
    bounces: u32,
}
