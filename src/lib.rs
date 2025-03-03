#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod engine;
mod renderer;

use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowBuilder};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use crate::renderer::{wgpu_utils::RenginWgpu, RenginRenderer};
use clap::Parser;
use engine::rt_primitives::{Camera, ScreenData};
use engine::scene_importer::Scene;

static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;

pub enum RendererType {
    PathTracer,
    RayTracer,
}

struct GameState {
    pub camera: Camera,
}

pub struct RenderApp<'a> {
    renderer: RenginWgpu<'a>,
    screen_data: ScreenData,
    game_state: GameState,
}

impl<'a> RenderApp<'a> {
    pub async fn new(
        window: &'a Window,
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

        // TODO: remove buffers as arg and move into RenginWgpu state
        renderer.create_target_textures(&physical_size);
        
        renderer.create_pipelines(
            scene.bvh.as_ref().unwrap(),
            &screen_data,
            scene.object_params.as_ref().unwrap(),
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
        self.renderer.queue.write_buffer(
            self.renderer.buffers.get("ubo").unwrap(),
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
            cpass.set_bind_group(0, self.renderer.bind_groups.get("raygen").unwrap(), &[]);

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

    pub async fn render(mut self, event_loop: EventLoop<()>) {
        let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;

        let _ = event_loop.run(
            move |event: Event<()>, target: &EventLoopWindowTarget<()>| {
                match event {
                    winit::event::Event::WindowEvent { event, .. } => match event {
                        WindowEvent::Resized(size) => {
                            self.screen_data.update_dims(&size);
                            self.screen_data.subpixel_idx = 0;
                            self.renderer.update_window_size(&size);
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
                        | WindowEvent::CloseRequested => target.exit(),

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
                            println!("{:#?}", self.renderer.instance.generate_report());
                        }
                        WindowEvent::RedrawRequested => {}
                        _ => {
                            self.update_window_event(event, &mut left_mouse_down);
                        }
                    },
                    Event::DeviceEvent { event, .. } => {
                        self.update_device_event(event, &mut left_mouse_down);
                    }
                    Event::AboutToWait => {
                        let time_since_last_frame = last_update_inst.elapsed();

                        if (!left_mouse_down || self.renderer.continous_motion)
                            && ((self.screen_data.subpixel_idx < self.renderer.rays_per_pixel)
                                || (self.screen_data.subpixel_idx == 0
                                    && time_since_last_frame >= target_frametime))
                        {
                            log::info!(
                                "Drawing ray index: {}, framerate: {}",
                                self.screen_data.subpixel_idx,
                                1000u128 / time_since_last_frame.as_millis()
                            );

                            last_update_inst = Instant::now();
                            if self.screen_data.subpixel_idx < self.renderer.rays_per_pixel {
                                self.update();

                                let frame = match self.renderer.surface.get_current_texture() {
                                    Ok(frame) => frame,
                                    Err(_) => {
                                        self.renderer.surface.configure(
                                            &self.renderer.device,
                                            &self.renderer.config,
                                        );
                                        self.renderer
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

                                let mut command_encoder =
                                    self.renderer.device.create_command_encoder(
                                        &wgpu::CommandEncoderDescriptor { label: None },
                                    );

                                command_encoder.push_debug_group("compute ray trace");
                                {
                                    // compute pass
                                    let mut cpass = command_encoder.begin_compute_pass(
                                        &wgpu::ComputePassDescriptor {
                                            label: None,
                                            timestamp_writes: Default::default(),
                                        },
                                    );

                                    for _ in 0..self.renderer.ray_bounces {
                                        cpass.set_pipeline(
                                            self.renderer.compute_pipeline.as_ref().unwrap(),
                                        );
                                        cpass.set_bind_group(
                                            0,
                                            self.renderer.bind_groups.get("compute").unwrap(),
                                            &[],
                                        );
                                        
                                        cpass.dispatch_workgroups(
                                            // self.screen_data.size.width / WORKGROUP_SIZE[0],
                                            // // + (self.screen_data.size.width % WORKGROUP_SIZE[0]),
                                            // self.screen_data.size.height / WORKGROUP_SIZE[1],
                                            // // + (self.screen_data.size.height % WORKGROUP_SIZE[1]),
                                            // WORKGROUP_SIZE[2],
                                            (self.screen_data.size.width * self.screen_data.size.height) / (WORKGROUP_SIZE[0] * WORKGROUP_SIZE[1]), 
                                            1, 
                                            1
                                        );


                                        cpass.set_pipeline(self.renderer.raysort_pipeline.as_ref().unwrap());
                                        cpass.set_bind_group(0, self.renderer.bind_groups.get("raysort").unwrap(), &[]);

                                        cpass.dispatch_workgroups(
                                            1,1,1,
                                        );
                                    }
                                }
                                command_encoder.pop_debug_group();

                                command_encoder.push_debug_group("render texture");
                                {
                                    // render pass
                                    let mut rpass =
                                        command_encoder.begin_render_pass(&render_pass_descriptor);
                                    rpass.set_pipeline(
                                        self.renderer.render_pipeline.as_ref().unwrap(),
                                    );
                                    rpass.set_bind_group(
                                        0,
                                        self.renderer.bind_groups.get("render").unwrap(),
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
                        } else {
                            target.set_control_flow(ControlFlow::WaitUntil(
                                Instant::now() + target_frametime - time_since_last_frame,
                            ))
                        }

                        self.renderer.window.request_redraw();
                    }
                    _ => {}
                }
            },
        );
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

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Info).expect("Could't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let args = if cfg!(target_arch = "wasm32") {
        Args {
            ..Default::default()
        }
    } else {
        Args::parse()
    };
    let renderer_type = if args.whitted {
        panic!("Whitted Ray Tracer has been deprecated and will no longer work.");
        // RendererType::RayTracer
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

    let event_loop = EventLoop::new().unwrap();

    let monitor = event_loop.available_monitors().next().unwrap();
    let monitor_scale_factor = monitor.scale_factor();
    let resolution = monitor.size();

    let logical_size: LogicalSize<u32> = winit::dpi::LogicalSize::new(
        scene.camera.as_ref().unwrap().width,
        scene.camera.as_ref().unwrap().height,
    );
    let physical_size: PhysicalSize<u32> = logical_size.to_physical(monitor_scale_factor);

    let window = WindowBuilder::new()
        .with_title("Rengin")
        .with_resizable(true)
        .with_inner_size(physical_size)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        let _ = window.request_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let app = RenderApp::new(
        &window,
        &resolution,
        &scene,
        !args.draw_on_mouseup,
        args.rays_per_pixel,
        args.bounces,
        renderer_type,
    )
    .await;

    app.render(event_loop).await;
}
