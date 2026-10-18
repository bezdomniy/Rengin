pub mod engine;
pub mod platform;
mod renderer;

#[cfg(target_arch = "wasm32")]
mod web;

use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent,
};
#[cfg(not(target_arch = "wasm32"))]
use winit::event_loop::ControlFlow;
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

use std::sync::Arc;
use std::time::Duration;

use platform::Instant;

#[cfg(not(target_arch = "wasm32"))]
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
    mouselook: bool,
    keys_held: KeysHeld,
    last_update_inst: Instant,
    /// The surface was resized, so the image must be presented again even if
    /// all samples are done (resizing a canvas clears it on the web).
    needs_present: bool,
}

#[derive(Default)]
struct KeysHeld {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    space: bool,
    shift: bool,
    arrow_left: bool,
    arrow_right: bool,
    arrow_up: bool,
    arrow_down: bool,
}

use std::cell::RefCell;
use std::rc::Rc;

pub struct RenderApp {
    state: Rc<RefCell<Option<State>>>,
}

impl Default for RenderApp {
    fn default() -> Self {
        RenderApp {
            state: Rc::new(RefCell::new(None)),
        }
    }
}

impl State {
    pub async fn new(
        window: Arc<Window>,
        render_size: PhysicalSize<u32>,
        scene: Scene,
        continous_motion: bool,
        rays_per_pixel: u32,
        ray_bounces: u32,
        renderer_type: RendererType,
        profile: bool,
        profile_stats: bool,
    ) -> Self {
        let physical_size = window.inner_size();
        // On web, inner_size may be 0 before layout; use render size as fallback.
        let surface_size = if physical_size.width > 0 {
            physical_size
        } else {
            render_size
        };

        let mut renderer = RenginWgpu::new(
            window,
            surface_size,
            continous_motion,
            rays_per_pixel,
            ray_bounces,
            profile,
            profile_stats,
        )
        .await;

        let game_state = GameState {
            camera: Camera::new(
                scene.camera.from,
                scene.camera.to,
                scene.camera.up,
            ),
        };

        let screen_data = ScreenData::new(
            game_state.camera.get_inverse_transform(),
            scene.object_params.len() as u32,
            scene.specular_offset as u32,
            scene.lights_offset as u32,
            ray_bounces,
            render_size,
            scene.camera.field_of_view,
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
            &scene.bvh,
            &screen_data,
            &scene.object_params,
        );
        renderer.create_pipelines(
            &scene.bvh,
            &screen_data,
            &scene.object_params,
        );

        // TODO: remove buffers as arg and move into RenginWgpu state
        renderer.create_bind_groups(&render_size);

        Self {
            renderer,
            screen_data,
            game_state,
            left_mouse_down: false,
            mouselook: false,
            keys_held: KeysHeld::default(),
            last_update_inst: Instant::now(),
            needs_present: false,
        }
    }

    fn resize_surface(&mut self, size: PhysicalSize<u32>) {
        let config = &self.renderer.config;
        if size.width > 0 && size.height > 0 && (size.width, size.height) != (config.width, config.height) {
            self.renderer.update_window_size(&size);
            self.needs_present = true;
        }
    }

    fn update_device_event(&mut self, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.mouselook {
                    self.game_state
                        .camera
                        .rotate(-delta.0 as f32, delta.1 as f32);

                    self.screen_data.inverse_camera_transform =
                        self.game_state.camera.get_inverse_transform();
                    self.screen_data.subpixel_idx = 0;
                }
            }
            _ => {}
        }
    }

    fn update_window_event(&mut self, event: WindowEvent) {
        match event {
            // Right-click toggles mouselook
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                self.mouselook = !self.mouselook;
            }
            // Track WASD + Space/Shift
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref c),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                match c.as_ref() {
                    "w" => self.keys_held.w = pressed,
                    "a" => self.keys_held.a = pressed,
                    "s" => self.keys_held.s = pressed,
                    "d" => self.keys_held.d = pressed,
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(named),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                match named {
                    NamedKey::Space => self.keys_held.space = pressed,
                    NamedKey::Shift => self.keys_held.shift = pressed,
                    NamedKey::ArrowLeft => self.keys_held.arrow_left = pressed,
                    NamedKey::ArrowRight => self.keys_held.arrow_right = pressed,
                    NamedKey::ArrowUp => self.keys_held.arrow_up = pressed,
                    NamedKey::ArrowDown => self.keys_held.arrow_down = pressed,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn apply_movement(&mut self) -> bool {
        let k = &self.keys_held;
        if !k.w && !k.a && !k.s && !k.d && !k.space && !k.shift && !k.arrow_left && !k.arrow_right && !k.arrow_up && !k.arrow_down {
            return false;
        }

        if k.w { self.game_state.camera.move_forward(1.0); }
        if k.s { self.game_state.camera.move_forward(-1.0); }
        if k.d { self.game_state.camera.move_right(1.0); }
        if k.a { self.game_state.camera.move_right(-1.0); }
        if k.space { self.game_state.camera.move_up(1.0); }
        if k.shift { self.game_state.camera.move_up(-1.0); }
        if k.arrow_left { self.game_state.camera.rotate(3.0, 0.0); }
        if k.arrow_right { self.game_state.camera.rotate(-3.0, 0.0); }
        if k.arrow_up { self.game_state.camera.rotate(0.0, 3.0); }
        if k.arrow_down { self.game_state.camera.rotate(0.0, -3.0); }

        self.screen_data.inverse_camera_transform =
            self.game_state.camera.get_inverse_transform();
        self.screen_data.subpixel_idx = 0;
        true
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

impl RenderApp {
    fn set_aspect_preserving_viewport(
        rpass: &mut wgpu::RenderPass<'_>,
        surface_size: PhysicalSize<u32>,
        render_size: PhysicalSize<u32>,
    ) {
        let surface_aspect = surface_size.width as f32 / surface_size.height as f32;
        let render_aspect = render_size.width as f32 / render_size.height as f32;
        let (width, height) = if surface_aspect > render_aspect {
            (
                surface_size.height as f32 * render_aspect,
                surface_size.height as f32,
            )
        } else {
            (
                surface_size.width as f32,
                surface_size.width as f32 / render_aspect,
            )
        };
        let x = (surface_size.width as f32 - width) * 0.5;
        let y = (surface_size.height as f32 - height) * 0.5;
        rpass.set_viewport(x, y, width, height, 0.0, 1.0);
    }

    /// Draws the accumulated image to the surface, letterboxed to its aspect ratio.
    fn encode_present_pass(
        state: &State,
        command_encoder: &mut wgpu::CommandEncoder,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
    ) {
        let mut rpass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            multiview_mask: None,
            color_attachments,
            depth_stencil_attachment: None,
            occlusion_query_set: Default::default(),
            timestamp_writes: None,
        });
        rpass.set_pipeline(state.renderer.render_pipeline.as_ref().unwrap());
        rpass.set_bind_group(0, state.renderer.render_bind_group.as_ref().unwrap(), &[]);
        Self::set_aspect_preserving_viewport(
            &mut rpass,
            PhysicalSize::new(state.renderer.config.width, state.renderer.config.height),
            state.screen_data.size,
        );
        rpass.draw(0..3, 0..1);
    }

    fn acquire_frame(state: &State) -> (wgpu::SurfaceTexture, wgpu::TextureView) {
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
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(state.renderer.config.format.add_srgb_suffix()),
            ..Default::default()
        });
        (frame, view)
    }

    fn color_attachments(view: &wgpu::TextureView) -> [Option<wgpu::RenderPassColorAttachment<'_>>; 1] {
        [Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })]
    }

    /// Presents the accumulated image again without tracing new samples.
    fn present_only(state: &State) {
        let (frame, view) = Self::acquire_frame(state);
        let mut command_encoder = state
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Self::encode_present_pass(state, &mut command_encoder, &Self::color_attachments(&view));
        state.renderer.queue.submit(std::iter::once(command_encoder.finish()));
        frame.present();
    }

    fn render_default_mode(
        state: &mut State,
        command_encoder: wgpu::CommandEncoder,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
        time_since_last_frame: Duration,
    ) {
        let mut command_encoder = command_encoder;
        // === DEFAULT MODE: wavefront path tracing with dead ray elimination ===

        // Initialize ray_counts: [0, 0, 0] (will be populated by first bounce)
        state.renderer.queue.write_buffer(
            state.renderer.buffers.as_ref().unwrap().get("ray_counts").unwrap(),
            0,
            bytemuck::cast_slice(&[0u32, 0u32, 0u32]),
        );

        // 1. Raygen
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raygen"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(state.renderer.raygen_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, state.renderer.raygen_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups(
                state.screen_data.size.width.div_ceil(WORKGROUP_SIZE[0]),
                state.screen_data.size.height.div_ceil(WORKGROUP_SIZE[1]),
                WORKGROUP_SIZE[2],
            );
        }

        // 2. Bounce 0: 2D dispatch (all pixels, spatially coherent)
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pathtracer bounce 0"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(state.renderer.compute_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, state.renderer.compute_bind_group.as_ref().unwrap(), &[]);
            cpass.dispatch_workgroups(
                state.screen_data.size.width.div_ceil(WORKGROUP_SIZE[0]),
                state.screen_data.size.height.div_ceil(WORKGROUP_SIZE[1]),
                WORKGROUP_SIZE[2],
            );
        }

        // 3. Subsequent bounces: compact 1D indirect dispatch
        for bounce in 1..state.renderer.ray_bounces {
            // Prepare: swap counts, compute indirect dispatch args
            {
                let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prepare dispatch"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(state.renderer.prepare_dispatch_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, state.renderer.prepare_bind_group.as_ref().unwrap(), &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Compact bounce: 1D indirect dispatch, only alive rays
            let bg = if bounce % 2 == 1 {
                state.renderer.compute_bind_group_b.as_ref().unwrap()
            } else {
                state.renderer.compute_bind_group.as_ref().unwrap()
            };
            {
                let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pathtracer compact"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(state.renderer.compute_compact_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, bg, &[]);
                cpass.dispatch_workgroups_indirect(
                    state.renderer.buffers.as_ref().unwrap().get("indirect_dispatch").unwrap(),
                    0,
                );
            }
        }

        // 4. Render pass
        Self::encode_present_pass(state, &mut command_encoder, color_attachments);

        state.renderer.queue.submit(std::iter::once(command_encoder.finish()));

        log::info!(
            "Sample {}/{}, framerate: {} fps",
            state.screen_data.subpixel_idx,
            state.renderer.rays_per_pixel,
            1000u128 / time_since_last_frame.as_millis().max(1),
        );
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let config = platform::AppConfig::from_url_params();
                let monitor_scale_factor = web_sys::window().unwrap().device_pixel_ratio();
            } else {
                let monitor = event_loop.available_monitors().next().unwrap();
                let monitor_scale_factor = monitor.scale_factor();
                let args = Args::parse();
                let config = platform::AppConfig::from(&args);
            }
        }

        let renderer_type = RendererType::PathTracer;

        let now = Instant::now();
        log::info!("Loading scene: {}", config.scene);

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                let scene = engine::embedded_assets::load_scene(&config.scene, &renderer_type);
            } else {
                let scene = Scene::from_file(&config.scene, &renderer_type);
            }
        }
        let scene = match scene {
            Ok(scene) => scene,
            Err(error) => {
                log::error!("Failed to load scene {}: {error}", config.scene);
                event_loop.exit();
                return;
            }
        };
        log::info!(
            "Finished loading scene in {} millis.",
            now.elapsed().as_millis()
        );

        let logical_size: LogicalSize<u32> = winit::dpi::LogicalSize::new(
            scene.camera.width,
            scene.camera.height,
        );
        let physical_size: PhysicalSize<u32> = logical_size.to_physical(monitor_scale_factor);
        let render_size = PhysicalSize::new(
            ((logical_size.width as f32 * config.render_scale).round() as u32).max(1),
            ((logical_size.height as f32 * config.render_scale).round() as u32).max(1),
        );

        let window_attributes = WindowAttributes::default()
            .with_title("Rengin Path Tracer")
            .with_resizable(true)
            .with_inner_size(physical_size);

        // Create window object
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        // On web, attach the canvas to the DOM. The page's CSS sizes it to fit the
        // viewport; winit reports that size as Resized and the render is letterboxed.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("Failed to get canvas from winit window");
            let web_window = web_sys::window().unwrap();
            let document = web_window.document().unwrap();
            let container = document
                .get_element_by_id("rengin-canvas")
                .expect("No #rengin-canvas element found in DOM");
            container
                .append_child(&web_sys::Element::from(canvas))
                .expect("Failed to append canvas to DOM");
        }

        let init_future = State::new(
            window.clone(),
            render_size,
            scene,
            config.continuous_motion,
            config.rays_per_pixel,
            config.bounces,
            renderer_type,
            config.profile,
            config.profile_stats,
        );

        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                // On web, we cannot block. Use spawn_local with shared state.
                let state_slot = self.state.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let state = init_future.await;
                    *state_slot.borrow_mut() = Some(state);
                    window.request_redraw();
                });
            } else {
                let state = pollster::block_on(init_future);
                *self.state.borrow_mut() = Some(state);
                window.request_redraw();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut guard = self.state.borrow_mut();
        let Some(state) = guard.as_mut() else { return };

        match event {
            WindowEvent::Resized(size) => state.resize_surface(size),
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
                #[cfg(not(target_arch = "wasm32"))]
                println!("{:#?}", state.renderer.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {}
            _ => {
                state.update_window_event(event);
            }
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let mut guard = self.state.borrow_mut();
        let Some(state) = guard.as_mut() else { return };
        state.update_device_event(event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let mut guard = self.state.borrow_mut();
        let Some(state) = guard.as_mut() else { return };
        #[cfg(not(target_arch = "wasm32"))]
        state.renderer.poll_profile_readbacks();

        // Profiling readback is double-buffered. If both slots are still owned
        // by the GPU, briefly yield instead of dropping the next sample's data
        // or blocking the CPU on device.poll(Wait).
        #[cfg(not(target_arch = "wasm32"))]
        if state.renderer.profile
            && state.screen_data.subpixel_idx < state.renderer.rays_per_pixel
            && !state.renderer.has_available_profile_readback_slot()
        {
            _event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(1),
            ));
            state.renderer.window.request_redraw();
            return;
        }
        let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
        let time_since_last_frame = state.last_update_inst.elapsed();

        // On the web the first Resized can arrive before the renderer exists
        // (initialisation is async), so also pick up the window size here.
        let window_size = state.renderer.window.inner_size();
        state.resize_surface(window_size);

        // Apply WASD movement every frame
        state.apply_movement();

        if state.needs_present && state.screen_data.subpixel_idx >= state.renderer.rays_per_pixel {
            Self::present_only(state);
        }
        state.needs_present = false;

        if (!state.left_mouse_down || state.renderer.continous_motion)
            && ((state.screen_data.subpixel_idx < state.renderer.rays_per_pixel)
                || (state.screen_data.subpixel_idx == 0
                    && time_since_last_frame >= target_frametime))
        {
            state.last_update_inst = Instant::now();
            if state.screen_data.subpixel_idx < state.renderer.rays_per_pixel {
                state.update();

                let (frame, view) = Self::acquire_frame(state);
                let color_attachments = Self::color_attachments(&view);
                #[allow(unused_mut)]
                let mut command_encoder = state
                    .renderer
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // Profiling mode: native-only (timestamp queries + asynchronous readback)
                #[cfg(not(target_arch = "wasm32"))]
                if state.renderer.profile {
                    // === PROFILED MODE: separate passes with GPU timestamps ===
                    state.renderer.queue.write_buffer(
                        state
                            .renderer
                            .buffers
                            .as_ref()
                            .unwrap()
                            .get("ray_counts")
                            .unwrap(),
                        0,
                        bytemuck::cast_slice(&[0u32, 0u32, 0u32]),
                    );
                    let mut query_idx: u32 = 0;
                    let readback_slot = state.renderer.acquire_profile_readback_slot();
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
                            state.screen_data.size.width.div_ceil(WORKGROUP_SIZE[0]),
                            state.screen_data.size.height.div_ceil(WORKGROUP_SIZE[1]),
                            WORKGROUP_SIZE[2],
                        );
                    }
                    query_idx += 2;

                    // 2. Bounce 0 covers the full image. Later bounces use the
                    // same compacted indirect dispatch path as normal rendering,
                    // including bounce-depth updates for Russian roulette.
                    for bounce in 0..state.renderer.ray_bounces {
                        if bounce > 0 {
                            let mut cpass = command_encoder.begin_compute_pass(
                                &wgpu::ComputePassDescriptor {
                                    label: Some("prepare profiled dispatch"),
                                    timestamp_writes: None,
                                },
                            );
                            cpass.set_pipeline(
                                state.renderer.prepare_dispatch_pipeline.as_ref().unwrap(),
                            );
                            cpass.set_bind_group(
                                0,
                                state.renderer.prepare_bind_group.as_ref().unwrap(),
                                &[],
                            );
                            cpass.dispatch_workgroups(1, 1, 1);
                        }

                        let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pathtracer bounce"),
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: qs,
                                beginning_of_pass_write_index: Some(query_idx),
                                end_of_pass_write_index: Some(query_idx + 1),
                            }),
                        });
                        if bounce == 0 {
                            cpass.set_pipeline(state.renderer.compute_pipeline.as_ref().unwrap());
                            cpass.set_bind_group(0, state.renderer.compute_bind_group.as_ref().unwrap(), &[]);
                            cpass.dispatch_workgroups(
                                state.screen_data.size.width.div_ceil(WORKGROUP_SIZE[0]),
                                state.screen_data.size.height.div_ceil(WORKGROUP_SIZE[1]),
                                WORKGROUP_SIZE[2],
                            );
                        } else {
                            let bg = if bounce % 2 == 1 {
                                state.renderer.compute_bind_group_b.as_ref().unwrap()
                            } else {
                                state.renderer.compute_bind_group.as_ref().unwrap()
                            };
                            cpass.set_pipeline(
                                state.renderer.compute_compact_pipeline.as_ref().unwrap(),
                            );
                            cpass.set_bind_group(0, bg, &[]);
                            cpass.dispatch_workgroups_indirect(
                                state
                                    .renderer
                                    .buffers
                                    .as_ref()
                                    .unwrap()
                                    .get("indirect_dispatch")
                                    .unwrap(),
                                0,
                            );
                        }
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
                        Self::set_aspect_preserving_viewport(
                            &mut rpass,
                            PhysicalSize::new(
                                state.renderer.config.width,
                                state.renderer.config.height,
                            ),
                            state.screen_data.size,
                        );
                        rpass.draw(0..3, 0..1);
                    }
                    query_idx += 2;

                    // Copy profiling data into an available readback slot. The
                    // buffers are mapped asynchronously after submission and
                    // consumed on a later event-loop iteration.
                    let stats_buf = state.renderer.buffers.as_ref().unwrap().get("stats").unwrap();
                    if let Some(slot_index) = readback_slot {
                        let slot = &state.renderer.profile_readbacks[slot_index];
                        command_encoder.copy_buffer_to_buffer(
                            stats_buf,
                            0,
                            &slot.stats_buffer,
                            0,
                            8 * 4,
                        );
                        command_encoder.resolve_query_set(
                            qs,
                            0..query_idx,
                            state.renderer.timestamp_resolve_buffer.as_ref().unwrap(),
                            0,
                        );
                        command_encoder.copy_buffer_to_buffer(
                            state.renderer.timestamp_resolve_buffer.as_ref().unwrap(),
                            0,
                            &slot.timestamp_buffer,
                            0,
                            (query_idx as u64) * 8,
                        );
                    }
                    command_encoder.clear_buffer(stats_buf, 0, None);

                    state.renderer.queue.submit(std::iter::once(command_encoder.finish()));
                    if let Some(slot_index) = readback_slot {
                        let total_pixels = u64::from(state.screen_data.size.width)
                            * u64::from(state.screen_data.size.height);
                        state.renderer.begin_profile_readback(
                            slot_index,
                            state.screen_data.subpixel_idx,
                            total_pixels,
                        );
                    }

                    frame.present();
                } else {
                    Self::render_default_mode(state, command_encoder, &color_attachments, time_since_last_frame);
                    frame.present();
                }

                #[cfg(target_arch = "wasm32")]
                {
                    Self::render_default_mode(state, command_encoder, &color_attachments, time_since_last_frame);
                    frame.present();
                }
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            {
                let _event_loop = _event_loop;
                _event_loop.set_control_flow(ControlFlow::WaitUntil(
                    Instant::now() + target_frametime - time_since_last_frame,
                ));
            }
        }

        state.renderer.window.request_redraw();
    }
}

/// A wgpu ray tracer
#[cfg(not(target_arch = "wasm32"))]
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
    /// Internal render-resolution multiplier (independent of window DPI)
    #[clap(long, default_value_t = 1.0)]
    render_scale: f32,
    /// Enable detailed GPU profiling (per-bounce timestamps)
    #[clap(short = 'p', long)]
    profile: bool,
    /// Enable instrumented shader counters (implies --profile and adds overhead)
    #[clap(long)]
    profile_stats: bool,
    /// Number of bounces per ray
    #[clap(short, long, default_value_t = 8)]
    bounces: u32,
}
