pub mod engine;
pub mod platform;
mod renderer;
mod ui;

#[cfg(target_arch = "wasm32")]
mod web;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
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

/// Camera movement is per second, but a frame never moves it further than this
/// many seconds' worth: when frames are slow, movement slows instead of jumping.
static MAX_MOVEMENT_STEP: f32 = 0.05;

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
    /// When the current image's first sample was rendered.
    render_started: Instant,
    /// The surface was resized, so the image must be presented again even if
    /// all samples are done (resizing a canvas clears it on the web).
    needs_present: bool,
    /// When the camera last moved from held keys.
    last_movement: Instant,
    /// Samples submitted to the GPU that it hasn't finished yet.
    #[cfg(target_arch = "wasm32")]
    samples_in_flight: Arc<std::sync::atomic::AtomicU32>,
    /// Kept to rebuild the GPU buffers when the render scale changes.
    scene: Scene,
    scene_name: String,
    /// The scenes the settings panel offers.
    scenes: Vec<String>,
    /// Where the native build loads `scenes` from.
    #[cfg(not(target_arch = "wasm32"))]
    scene_dir: std::path::PathBuf,
    /// A scene being downloaded and built in the background (web).
    loading_scene: Option<String>,
    render_scale: f32,
    /// How long the image took, once all its samples are done.
    render_time: Option<Duration>,
    ui: ui::Ui,
}

/// The scenes the settings panel offers, and the directory they are in.
#[cfg(not(target_arch = "wasm32"))]
fn scenes_next_to(scene_file: &str) -> (std::path::PathBuf, Vec<String>) {
    let dir = std::path::Path::new(scene_file)
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default();
    let mut scenes: Vec<String> = std::fs::read_dir(if dir.as_os_str().is_empty() { ".".as_ref() } else { dir.as_path() })
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            (path.extension()? == "yaml").then(|| path.file_stem().map(|stem| stem.to_string_lossy().into_owned()))?
        })
        .collect();
    scenes.sort();
    (dir, scenes)
}

/// The scene is rendered at its camera's size times `render_scale`,
/// independent of the window size.
fn render_size(scene: &Scene, render_scale: f32) -> PhysicalSize<u32> {
    PhysicalSize::new(
        ((scene.camera.width as f32 * render_scale).round() as u32).max(1),
        ((scene.camera.height as f32 * render_scale).round() as u32).max(1),
    )
}

fn scene_screen_data(
    scene: &Scene,
    camera: &Camera,
    render_size: PhysicalSize<u32>,
    ray_bounces: u32,
    rays_per_pixel: u32,
) -> ScreenData {
    ScreenData::new(
        camera.get_inverse_transform(),
        scene.object_params.len() as u32,
        scene.specular_offset as u32,
        scene.lights_offset as u32,
        scene.bvh.leaf_nodes.len() as u32,
        ray_bounces,
        render_size,
        scene.camera.field_of_view,
        (rays_per_pixel as f32).sqrt() as u32,
    )
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

impl KeysHeld {
    fn any(&self) -> bool {
        self.w || self.a || self.s || self.d || self.space || self.shift
            || self.arrow_left || self.arrow_right || self.arrow_up || self.arrow_down
    }
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
    /// Sets up the renderer for `scene`. `scenes` are offered in the settings
    /// panel; `scene_name` is the one being shown.
    pub async fn new(
        window: Arc<Window>,
        scene: Scene,
        scene_name: String,
        scenes: Vec<String>,
        config: &platform::AppConfig,
    ) -> Self {
        let render_size = render_size(&scene, config.render_scale);
        let physical_size = window.inner_size();
        // On web, inner_size may be 0 before layout; use render size as fallback.
        let surface_size = if physical_size.width > 0 {
            physical_size
        } else {
            render_size
        };

        let mut renderer = RenginWgpu::new(
            window.clone(),
            surface_size,
            config.continuous_motion,
            config.rays_per_pixel,
            config.bounces,
            config.profile,
            config.profile_stats,
        )
        .await;

        let game_state = GameState {
            camera: Camera::new(
                scene.camera.from,
                scene.camera.to,
                scene.camera.up,
            ),
        };

        let screen_data = scene_screen_data(
            &scene,
            &game_state.camera,
            render_size,
            config.bounces,
            renderer.rays_per_pixel,
        );

        log::debug!("screen_data: {:?}", screen_data);

        let now = Instant::now();
        log::info!("Building shaders...");
        renderer.create_shaders(RendererType::PathTracer);
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

        let ui = ui::Ui::new(&window, &renderer.device, renderer.config.format.remove_srgb_suffix());

        Self {
            renderer,
            screen_data,
            game_state,
            left_mouse_down: false,
            mouselook: false,
            keys_held: KeysHeld::default(),
            last_update_inst: Instant::now(),
            render_started: Instant::now(),
            needs_present: false,
            last_movement: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            samples_in_flight: Default::default(),
            scene,
            scene_name,
            scenes,
            #[cfg(not(target_arch = "wasm32"))]
            scene_dir: Default::default(),
            loading_scene: None,
            render_scale: config.render_scale,
            render_time: None,
            ui,
        }
    }

    /// Restarts the image, e.g. after a setting changed.
    fn restart(&mut self) {
        self.screen_data.subpixel_idx = 0;
        self.needs_present = true;
    }

    fn set_rays_per_pixel(&mut self, rays_per_pixel: u32) {
        self.renderer.rays_per_pixel = rays_per_pixel;
        // The samples are stratified on a grid of this size.
        self.screen_data.sqrt_rays_per_pixel = (rays_per_pixel as f32).sqrt() as u32;
        self.restart();
    }

    fn set_bounces(&mut self, bounces: u32) {
        self.renderer.ray_bounces = bounces;
        self.screen_data.ray_bounces = bounces;
        self.restart();
    }

    fn set_render_scale(&mut self, render_scale: f32) {
        self.render_scale = render_scale;
        self.rebuild_scene_buffers();
    }

    /// Shows `scene` from its own camera.
    fn set_scene(&mut self, name: String, scene: Scene) {
        self.game_state.camera = Camera::new(scene.camera.from, scene.camera.to, scene.camera.up);
        self.scene = scene;
        self.scene_name = name;
        self.rebuild_scene_buffers();
    }

    /// Recreates the buffers that depend on the scene and the render size.
    /// The pipelines don't, so they are kept.
    fn rebuild_scene_buffers(&mut self) {
        let render_size = render_size(&self.scene, self.render_scale);
        self.screen_data = scene_screen_data(
            &self.scene,
            &self.game_state.camera,
            render_size,
            self.renderer.ray_bounces,
            self.renderer.rays_per_pixel,
        );
        self.renderer.create_buffers(&self.scene.bvh, &self.screen_data, &self.scene.object_params);
        self.renderer.create_bind_groups(&render_size);
        self.restart();
    }

    /// Lays out the settings panel and returns the settings changed in it.
    fn run_ui(&mut self) -> Vec<ui::Change> {
        let samples_done = self.screen_data.subpixel_idx;
        let panel = ui::Panel {
            scenes: &self.scenes,
            scene: &self.scene_name,
            loading_scene: self.loading_scene.as_deref(),
            rays_per_pixel: self.renderer.rays_per_pixel,
            bounces: self.renderer.ray_bounces,
            bounces_editable: !self.renderer.profile,
            render_scale: self.render_scale,
            render_size: self.screen_data.size,
            samples_done,
            render_time: match self.render_time {
                Some(time) => time,
                None if samples_done == 0 => Duration::ZERO,
                None => self.render_started.elapsed(),
            },
        };
        self.ui.run(&self.renderer.window, &panel)
    }

    /// Puts the current settings in the page's URL, so reloading or sharing
    /// it shows the same image.
    #[cfg(target_arch = "wasm32")]
    fn update_url(&self) {
        web::set_url_params(&[
            (platform::SCENE_PARAMS[0], self.scene_name.clone()),
            (platform::RAYS_PER_PIXEL_PARAMS[0], self.renderer.rays_per_pixel.to_string()),
            (platform::BOUNCES_PARAMS[0], self.renderer.ray_bounces.to_string()),
            (platform::RENDER_SCALE_PARAMS[0], self.render_scale.to_string()),
        ]);
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
        // The step sizes below are per frame at FRAMERATE.
        let step = self.last_movement.elapsed().as_secs_f32().min(MAX_MOVEMENT_STEP) * FRAMERATE as f32;
        self.last_movement = Instant::now();
        let k = &self.keys_held;
        if !k.any() {
            return false;
        }

        if k.w { self.game_state.camera.move_forward(step); }
        if k.s { self.game_state.camera.move_forward(-step); }
        if k.d { self.game_state.camera.move_right(step); }
        if k.a { self.game_state.camera.move_right(-step); }
        if k.space { self.game_state.camera.move_up(step); }
        if k.shift { self.game_state.camera.move_up(-step); }
        if k.arrow_left { self.game_state.camera.rotate(3.0 * step, 0.0); }
        if k.arrow_right { self.game_state.camera.rotate(-3.0 * step, 0.0); }
        if k.arrow_up { self.game_state.camera.rotate(0.0, 3.0 * step); }
        if k.arrow_down { self.game_state.camera.rotate(0.0, -3.0 * step); }

        self.screen_data.inverse_camera_transform =
            self.game_state.camera.get_inverse_transform();
        self.screen_data.subpixel_idx = 0;
        true
    }

    /// Whether the GPU has caught up enough to take another sample. Nothing
    /// paces submissions on the web, so without this the GPU falls behind by
    /// many samples: the image lags input and then catches up in jumps. While
    /// the camera moves, wait for all samples so the next one shows the latest
    /// view; while it is still, keep one queued so the GPU stays busy.
    #[cfg(target_arch = "wasm32")]
    fn gpu_ready(&self) -> bool {
        let in_flight = self.samples_in_flight.load(std::sync::atomic::Ordering::Acquire);
        let camera_moving = self.keys_held.any() || self.screen_data.subpixel_idx == 0;
        in_flight == 0 || (in_flight == 1 && !camera_moving)
    }

    pub fn update(&mut self) {
        self.renderer.queue.write_buffer(
            self.renderer.buffers.as_ref().unwrap().get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.screen_data.generate_ubo()),
        );

        if self.screen_data.subpixel_idx == 0 {
            self.render_started = Instant::now();
            self.render_time = None;
        }
        self.screen_data.subpixel_idx += 1;
        if self.screen_data.subpixel_idx == self.renderer.rays_per_pixel {
            self.render_time = Some(self.render_started.elapsed());
            log::info!(
                "Rendered {} samples in {} millis.",
                self.renderer.rays_per_pixel,
                self.render_started.elapsed().as_millis()
            );
        }
    }
}

impl RenderApp {
    #[cfg(not(target_arch = "wasm32"))]
    fn resume_native(&mut self, event_loop: &ActiveEventLoop) {
        let monitor = event_loop.available_monitors().next().unwrap();
        let monitor_scale_factor = monitor.scale_factor();
        let args = Args::parse();
        let config = platform::AppConfig::from(&args);

        let now = Instant::now();
        log::info!("Loading scene: {}", config.scene);
        let scene = match Scene::from_file(&config.scene, &RendererType::PathTracer) {
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

        let logical_size = winit::dpi::LogicalSize::new(
            scene.camera.width,
            scene.camera.height,
        );
        let window_attributes = WindowAttributes::default()
            .with_title("Rengin Path Tracer")
            .with_resizable(true)
            .with_inner_size(logical_size.to_physical::<u32>(monitor_scale_factor));
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        let (scene_dir, scenes) = scenes_next_to(&config.scene);
        let scene_name = std::path::Path::new(&config.scene)
            .file_stem()
            .map_or_else(|| config.scene.clone(), |stem| stem.to_string_lossy().into_owned());
        let mut state = pollster::block_on(State::new(window.clone(), scene, scene_name, scenes, &config));
        state.scene_dir = scene_dir;
        *self.state.borrow_mut() = Some(state);
        window.request_redraw();
    }

    /// On the web the scene's models are downloaded, so the scene and renderer
    /// are set up asynchronously; events are ignored until the state exists.
    #[cfg(target_arch = "wasm32")]
    fn resume_web(&mut self, event_loop: &ActiveEventLoop) {
        let config = platform::AppConfig::from_url_params();
        let window_attributes = WindowAttributes::default().with_title("Rengin Path Tracer");
        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        web::attach_canvas(&window);

        let state_slot = self.state.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let scene = match web::load_scene(&config.scene).await {
                Ok(scene) => scene,
                Err(error) => {
                    log::error!("Failed to load scene {}: {error}", config.scene);
                    web::set_status(Some(&format!("Couldn't load scene {}: {error}", config.scene)));
                    return;
                }
            };
            let scenes = engine::embedded_assets::SCENES.iter().map(|(name, _)| name.to_string()).collect();
            let state = State::new(window.clone(), scene, config.scene.clone(), scenes, &config).await;
            web::set_status(None);
            *state_slot.borrow_mut() = Some(state);
            window.request_redraw();
        });
    }

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

    /// The next surface texture with an sRGB view of it for the image and a
    /// non-sRGB one for the settings panel, or None if there's no texture to
    /// draw to this frame (e.g. the window is hidden).
    fn acquire_frame(state: &State) -> Option<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::TextureView)> {
        use wgpu::CurrentSurfaceTexture as Current;
        let surface = &state.renderer.surface;
        let frame = match surface.get_current_texture() {
            Current::Success(frame) | Current::Suboptimal(frame) => frame,
            Current::Outdated | Current::Lost => {
                surface.configure(&state.renderer.device, &state.renderer.config);
                match surface.get_current_texture() {
                    Current::Success(frame) | Current::Suboptimal(frame) => frame,
                    _ => return None,
                }
            }
            Current::Timeout | Current::Occluded | Current::Validation => return None,
        };
        let view = |format| {
            frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(format),
                ..Default::default()
            })
        };
        let format = state.renderer.config.format;
        let (image_view, ui_view) = (view(format.add_srgb_suffix()), view(format.remove_srgb_suffix()));
        Some((frame, image_view, ui_view))
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

    /// Encodes one sample of the image, and drawing the image to the surface.
    fn encode_sample(
        state: &State,
        command_encoder: &mut wgpu::CommandEncoder,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
    ) {
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
        Self::encode_present_pass(state, command_encoder, color_attachments);
    }
}

impl RenderApp {
    /// Applies a setting changed in the settings panel.
    fn apply_change(slot: &Rc<RefCell<Option<State>>>, state: &mut State, change: ui::Change) {
        match change {
            ui::Change::RaysPerPixel(rays_per_pixel) => state.set_rays_per_pixel(rays_per_pixel),
            ui::Change::Bounces(bounces) => state.set_bounces(bounces),
            ui::Change::RenderScale(render_scale) => state.set_render_scale(render_scale),
            ui::Change::Scene(name) => Self::load_scene(slot, state, name),
        }
        #[cfg(target_arch = "wasm32")]
        state.update_url();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn load_scene(_slot: &Rc<RefCell<Option<State>>>, state: &mut State, name: String) {
        let path = state.scene_dir.join(format!("{name}.yaml"));
        let path = path.to_string_lossy();
        log::info!("Loading scene: {path}");
        match Scene::from_file(&path, &RendererType::PathTracer) {
            Ok(scene) => state.set_scene(name, scene),
            Err(error) => log::error!("Failed to load scene {path}: {error}"),
        }
    }

    /// Downloads and builds the scene in the background; the current scene
    /// keeps rendering until it is ready.
    #[cfg(target_arch = "wasm32")]
    fn load_scene(slot: &Rc<RefCell<Option<State>>>, state: &mut State, name: String) {
        state.loading_scene = Some(name.clone());
        let slot = slot.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let scene = web::load_scene(&name).await;
            let mut guard = slot.borrow_mut();
            let Some(state) = guard.as_mut() else { return };
            state.loading_scene = None;
            match scene {
                Ok(scene) => {
                    web::set_status(None);
                    state.set_scene(name, scene);
                    state.update_url();
                }
                Err(error) => {
                    log::error!("Failed to load scene {name}: {error}");
                    web::set_status(Some(&format!("Couldn't load scene {name}: {error}")));
                    // Show the panel's scene again.
                    state.needs_present = true;
                }
            }
        });
    }
    /// Encodes one sample with GPU timestamps around each pass, copying them
    /// and the shader counters to a readback slot. Returns the slot, if one was
    /// free.
    #[cfg(not(target_arch = "wasm32"))]
    fn encode_profiled_sample(
        state: &mut State,
        command_encoder: &mut wgpu::CommandEncoder,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
    ) -> Option<usize> {
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
        readback_slot
    }
}

impl ApplicationHandler for RenderApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(target_arch = "wasm32")]
        self.resume_web(event_loop);
        #[cfg(not(target_arch = "wasm32"))]
        self.resume_native(event_loop);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Only used to quit, which the web build doesn't do.
        #[cfg(target_arch = "wasm32")]
        let _ = event_loop;
        let mut guard = self.state.borrow_mut();
        let Some(state) = guard.as_mut() else { return };
        let used_by_panel = state.ui.on_window_event(&state.renderer.window, &event);

        match event {
            WindowEvent::Resized(size) => state.resize_surface(size),
            // On the web there is nothing to quit to, and Escape is how the
            // browser leaves fullscreen.
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            }
            | WindowEvent::CloseRequested if !used_by_panel => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(s),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if s == "R" && !used_by_panel => {
                #[cfg(not(target_arch = "wasm32"))]
                println!("{:#?}", state.renderer.instance.generate_report());
            }
            WindowEvent::RedrawRequested => {}
            _ if used_by_panel => {}
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
        #[cfg(not(target_arch = "wasm32"))]
        let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
        let time_since_last_frame = state.last_update_inst.elapsed();

        // On the web the first Resized can arrive before the renderer exists
        // (initialisation is async), so also pick up the window size here.
        let window_size = state.renderer.window.inner_size();
        state.resize_surface(window_size);

        #[cfg(target_arch = "wasm32")]
        if !state.gpu_ready() {
            state.renderer.window.request_redraw();
            return;
        }

        // Apply WASD movement every frame
        state.apply_movement();

        let sampling = |state: &State| {
            (!state.left_mouse_down || state.renderer.continous_motion)
                && state.screen_data.subpixel_idx < state.renderer.rays_per_pixel
        };
        if !sampling(state) && !state.needs_present && !state.ui.needs_repaint() {
            #[cfg(not(target_arch = "wasm32"))]
            _event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + target_frametime.saturating_sub(time_since_last_frame),
            ));
            state.renderer.window.request_redraw();
            return;
        }

        let Some((frame, view, ui_view)) = Self::acquire_frame(state) else {
            // Try again a frame later rather than spinning.
            #[cfg(not(target_arch = "wasm32"))]
            _event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + target_frametime));
            #[cfg(target_arch = "wasm32")]
            state.renderer.window.request_redraw();
            return;
        };

        // Settings changed in the panel apply to this frame.
        for change in state.run_ui() {
            Self::apply_change(&self.state, state, change);
        }

        let color_attachments = Self::color_attachments(&view);
        let mut command_encoder = state
            .renderer
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let sampling = sampling(state);
        #[cfg(not(target_arch = "wasm32"))]
        let mut readback_slot = None;
        if sampling {
            state.last_update_inst = Instant::now();
            state.update();
            // Profiling mode: native-only (timestamp queries + asynchronous readback)
            #[cfg(not(target_arch = "wasm32"))]
            if state.renderer.profile {
                readback_slot = Self::encode_profiled_sample(state, &mut command_encoder, &color_attachments);
            } else {
                Self::encode_sample(state, &mut command_encoder, &color_attachments);
            }
            #[cfg(target_arch = "wasm32")]
            Self::encode_sample(state, &mut command_encoder, &color_attachments);
        } else {
            Self::encode_present_pass(state, &mut command_encoder, &color_attachments);
        }
        state.needs_present = false;

        let surface_size = PhysicalSize::new(state.renderer.config.width, state.renderer.config.height);
        let ui_commands = state.ui.paint(
            &state.renderer.device,
            &state.renderer.queue,
            &mut command_encoder,
            &ui_view,
            surface_size,
        );
        state
            .renderer
            .queue
            .submit(ui_commands.into_iter().chain(std::iter::once(command_encoder.finish())));

        if sampling {
            #[cfg(not(target_arch = "wasm32"))]
            if let Some(slot_index) = readback_slot {
                let total_pixels = u64::from(state.screen_data.size.width)
                    * u64::from(state.screen_data.size.height);
                state.renderer.begin_profile_readback(
                    slot_index,
                    state.screen_data.subpixel_idx,
                    total_pixels,
                );
            }
            #[cfg(target_arch = "wasm32")]
            {
                use std::sync::atomic::Ordering;
                let in_flight = state.samples_in_flight.clone();
                in_flight.fetch_add(1, Ordering::AcqRel);
                state
                    .renderer
                    .queue
                    .on_submitted_work_done(move || _ = in_flight.fetch_sub(1, Ordering::AcqRel));
            }
            log::debug!(
                "Sample {}/{}, framerate: {} fps",
                state.screen_data.subpixel_idx,
                state.renderer.rays_per_pixel,
                1000u128 / time_since_last_frame.as_millis().max(1),
            );
        }

        state.renderer.queue.present(frame);
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
