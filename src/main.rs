#![feature(iter_partition_in_place)]

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

use std::collections::HashMap;
use std::env;

// use core::num;

// use wgpu::BufferUsage;
use glam::{const_vec3, Mat4};

use engine::scene_importer::Scene;

use crate::renderer::wgpu_utils::RenginWgpu;
use engine::rt_primitives::{Camera, BVH, UBO};

static WIDTH: u32 = 800;
static HEIGHT: u32 = 600;
static WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

static FRAMERATE: f64 = 60.0;
static RAYS_PER_PIXEL: u32 = 16;

//TODO: try doing passes over parts of the image instead of whole at a time
//      that way you can maintain framerate

enum RendererType {
    PathTracer,
    RayTracer,
}

// static RENDERER_TYPE: RendererType = RendererType::PathTracer;
static RENDERER_TYPE: RendererType = RendererType::RayTracer;

struct GameState {
    pub camera_angle_y: f32,
    pub camera_angle_xz: f32,
    pub camera_dist: f32,
    pub camera_centre: [f32; 3],
    pub camera_up: [f32; 3],
}

struct RenderApp {
    renderer: RenginWgpu,
    shaders: HashMap<&'static str, ShaderModule>,
    // compute_pipeline: Option<ComputePipeline>,
    // // compute_bind_group_layout: Option<BindGroupLayout>,
    // compute_bind_group: Option<BindGroup>,
    // render_pipeline: Option<RenderPipeline>,
    // render_bind_group_layout: Option<BindGroupLayout>,
    // render_bind_group: Option<BindGroup>,
    // sampler: Option<Sampler>,
    scene: Scene,
    buffers: HashMap<&'static str, Buffer>,
    texture: Texture,
    // object_params: Option<Vec<ObjectParams>>,
    ubo: UBO,
    game_state: GameState,
}

impl RenderApp {
    pub fn new(scene_path: &str, event_loop: &EventLoop<()>, continous_motion: bool) -> Self {
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
        ));

        // let object_params = scene.object_params.as_ref().unwrap().clone();

        // println!("{:#?}", self.object_params);

        // log::info!("tlas:{:?}, blas{:?}", dragon_tlas.len(), dragon_blas.len());
        // log::info!(
        //     "tlas:{:?}, blas{:?}",
        //     mem::size_of::<NodeTLAS>(),
        //     mem::size_of::<NodeBLAS>()
        // );

        // let camera_position = [-4f32, 2f32, -3f32];
        // let camera_centre = [0f32, 1f32, 0f32];

        // let camera_angle_y = 0.0;
        // let camera_angle_xz = 0.0;
        // let camera_dist = 9.0;
        // TODO: find out why models are appearing upside-down
        let camera_centre = scene.camera.as_ref().unwrap().to;
        let camera_up = scene.camera.as_ref().unwrap().up;
        let camera_position = scene.camera.as_ref().unwrap().from;

        let transform = Mat4::look_at_rh(
            const_vec3!(camera_position),
            const_vec3!(camera_centre),
            const_vec3!(camera_up),
        );

        // TODO: fix these
        let camera_angle_y = transform.row(2)[1].atan2(transform.row(1)[1]);
        let camera_angle_xz = (-transform.row(3)[1])
            .atan2((transform.row(3)[2].powf(2.0) + transform.row(3)[3].powf(2.0)).sqrt());
        let camera_dist = const_vec3!(camera_position).distance(const_vec3!(camera_centre));

        let game_state = GameState {
            camera_angle_xz,
            camera_angle_y,
            camera_dist,
            camera_centre,
            camera_up,
        };

        println!("{} {}", camera_angle_y, camera_angle_xz);

        let camera = Camera::new(
            camera_position,
            camera_centre,
            camera_up,
            scene.camera.as_ref().unwrap().width,
            scene.camera.as_ref().unwrap().height,
            std::f32::consts::FRAC_PI_3,
            // 1.0472f32,
        );

        let light_value = scene.lights.as_ref().unwrap()[0].at;
        let ubo = UBO::new(
            // [-4f32, 2f32, 3f32, 1f32],
            [light_value[0], light_value[1], light_value[2], 1.0],
            scene.object_params.as_ref().unwrap().len() as u32,
            (RAYS_PER_PIXEL as f32).sqrt() as u32,
            camera,
        );

        println!("ubo: {:?}", ubo);

        // log::info!("ubo:{:?},", ubo);
        now = Instant::now();
        log::info!("Building shaders...");
        let shaders = renderer.create_shaders();
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
            scene.object_params.as_ref().unwrap(),
        );
        renderer.create_pipelines(&buffers, &shaders, &texture, scene.bvh.as_ref().unwrap());

        Self {
            renderer,
            shaders,
            // compute_pipeline: None,
            // compute_bind_group_layout: None,
            // compute_bind_group: None,
            // render_pipeline: None,
            // render_bind_group_layout: None,
            // render_bind_group: None,
            // sampler: None,
            scene,
            buffers,
            texture,
            // object_params: None,
            ubo,
            game_state,
        }
    }

    fn update_device_event(
        &mut self,
        event: DeviceEvent,
        left_mouse_down: &mut bool,
        something_changed: &mut bool,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                // println!("x:{}, y:{}", position.x, position.y);
                if *left_mouse_down {
                    // let game_state: &mut GameState = self.game_state.as_mut().unwrap();
                    // println!(
                    //     "{} {}",
                    //     game_state.camera_angle_y, game_state.camera_angle_xz
                    // );

                    self.game_state.camera_angle_y += delta.0 as f32;
                    self.game_state.camera_angle_xz += delta.1 as f32;

                    let norm_x = self.game_state.camera_angle_y / self.renderer.config.width as f32;
                    let norm_y =
                        self.game_state.camera_angle_xz / self.renderer.config.height as f32;
                    let angle_y = norm_x * 5.0;
                    let angle_xz = -norm_y * 2.0;

                    let new_position = [
                        angle_xz.cos() * angle_y.sin() * self.game_state.camera_dist,
                        angle_xz.sin() * self.game_state.camera_dist
                            + self.game_state.camera_centre[1],
                        angle_xz.cos() * angle_y.cos() * self.game_state.camera_dist,
                    ];

                    self.ubo.camera.update_position(
                        new_position,
                        self.game_state.camera_centre,
                        self.game_state.camera_up,
                    );
                    self.ubo.subpixel_idx = 0;
                    self.ubo.update_random_seed();
                    *something_changed = true;

                    // if let Some(ref mut ubo) = self.ubo {
                    //     // no reference before Some
                    //     ubo.camera.update_position(
                    //         new_position,
                    //         game_state.camera_centre,
                    //         game_state.camera_up,
                    //     );
                    //     ubo.subpixel_idx = 0;
                    //     ubo.update_random_seed();
                    //     *something_changed = true;
                    // }
                }
            }
            DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(_, y) => {
                    // println!("{} {}", x, y);
                    // let game_state: &mut GameState = self.game_state.as_mut().unwrap();
                    self.game_state.camera_dist -= (y as f32) / 3.;

                    let norm_x = self.game_state.camera_angle_y / self.renderer.config.width as f32;
                    let norm_y =
                        self.game_state.camera_angle_xz / self.renderer.config.height as f32;
                    let angle_y = norm_x * 5.0;
                    let angle_xz = -norm_y * 2.0;

                    let new_position = [
                        angle_xz.cos() * angle_y.sin() * self.game_state.camera_dist,
                        angle_xz.sin() * self.game_state.camera_dist
                            + self.game_state.camera_centre[1],
                        angle_xz.cos() * angle_y.cos() * self.game_state.camera_dist,
                    ];

                    self.ubo.camera.update_position(
                        new_position,
                        self.game_state.camera_centre,
                        self.game_state.camera_up,
                    );
                    self.ubo.subpixel_idx = 0;
                    self.ubo.update_random_seed();
                    *something_changed = true;

                    // if let Some(ref mut ubo) = self.ubo {
                    //     // no reference before Some
                    //     ubo.camera.update_position(
                    //         new_position,
                    //         game_state.camera_centre,
                    //         game_state.camera_up,
                    //     );
                    //     ubo.subpixel_idx = 0;
                    //     ubo.update_random_seed();
                    //     *something_changed = true;
                    // }
                }
                _ => {}
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
            self.buffers.get("ubo").unwrap(),
            0,
            bytemuck::bytes_of(&self.ubo),
        );
    }

    pub fn render(mut self, event_loop: EventLoop<()>) {
        let mut last_update_inst = Instant::now();
        let mut left_mouse_down = false;
        let mut something_changed = false;

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            match event {
                Event::MainEventsCleared => {
                    // // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                    //     let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    //     let time_since_last_frame = last_update_inst.elapsed();

                    //     if (something_changed
                    //         || self.ubo.as_ref().unwrap().subpixel_idx < RAYS_PER_PIXEL)
                    //         && time_since_last_frame >= target_frametime
                    //         && (!left_mouse_down || self.renderer.continous_motion)
                    //     {
                    //         println!("Drawing ray index: {}", self.ubo.unwrap().subpixel_idx);

                    //         self.renderer.window.request_redraw();

                    //         // if let Some(ref mut x) = self.ubo {
                    //         //     x.subpixel_idx += 1;
                    //         // }

                    //         println!("render time: {:?}", time_since_last_frame);
                    //         something_changed = false;

                    //         last_update_inst = Instant::now();
                    //     } else {
                    //         *control_flow = ControlFlow::WaitUntil(
                    //             Instant::now() + target_frametime - time_since_last_frame,
                    //         );
                    //     }
                }
                Event::RedrawEventsCleared => {
                    let target_frametime = Duration::from_secs_f64(1.0 / FRAMERATE);
                    let time_since_last_frame = last_update_inst.elapsed();

                    if (something_changed || self.ubo.subpixel_idx < RAYS_PER_PIXEL)
                        && time_since_last_frame >= target_frametime
                        && (!left_mouse_down || self.renderer.continous_motion)
                    {
                        println!("Drawing ray index: {}", self.ubo.subpixel_idx);

                        // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                        // self.renderer.instance.poll_all(true);
                        self.renderer.device.poll(wgpu::Maintain::Wait);

                        self.renderer.window.request_redraw();

                        // if let Some(ref mut x) = self.ubo {
                        //     x.subpixel_idx += 1;
                        // }

                        println!("render time: {:?}", time_since_last_frame);
                        something_changed = false;

                        last_update_inst = Instant::now();
                    } else {
                        *control_flow = ControlFlow::WaitUntil(
                            Instant::now() + target_frametime - time_since_last_frame,
                        );
                    }
                }
                Event::WindowEvent {
                    event:
                    // TODO: resize makes picture darker - fix it
                        WindowEvent::Resized(size)
                        | WindowEvent::ScaleFactorChanged {
                            new_inner_size: &mut size,
                            ..
                        },
                    ..
                } => {
                    // println!("p: {} {}", size.width, size.height);
                    // Reconfigure the surface with the new size
                    self.renderer.config.width = size.width.max(1);
                    self.renderer.config.height = size.height.max(1);

                    let logical_size: LogicalSize<u32> =
                        winit::dpi::PhysicalSize::new(size.width, size.height)
                            .to_logical(self.renderer.scale_factor);

                    // println!("l: {} {}", logical_size.width, logical_size.height);

                    let texture_extent = wgpu::Extent3d {
                        width: logical_size.width,
                        height: logical_size.height,
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
                    something_changed = true;
                }
                Event::DeviceEvent { event, .. } => match event {
                    _ => {
                        self.update_device_event(
                            event,
                            &mut left_mouse_down,
                            &mut something_changed,
                        );
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
                Event::RedrawRequested(_) => {
                    // self.renderer.queue.submit(None);
                    // println!("blocking");
                    // futures::executor::block_on(self.renderer.queue.on_submitted_work_done());
                    // println!("done");
                    // println!("redrawing");
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

                    let mut command_encoder = self
                        .renderer
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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
                            self.renderer.config.width / WORKGROUP_SIZE[0],
                            self.renderer.config.height / WORKGROUP_SIZE[1],
                            WORKGROUP_SIZE[2],
                        );
                    }
                    command_encoder.pop_debug_group();

                    command_encoder.push_debug_group("render texture");
                    {
                        // render pass
                        let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
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

                    frame.present();

                    self.ubo.subpixel_idx += 1;

                    // if let Some(ref mut x) = self.ubo {
                    //     x.subpixel_idx += 1;
                    // }
                }
                _ => {}
            }
        });
    }
}

fn main() {
    env_logger::init();
    let args: Vec<String> = env::args().collect();

    let scene_path = &args[1];
    // let objects = import_obj(model_path);
    // let (tlas, blas) = objects.as_ref().unwrap().get(0).unwrap();

    // let inverseTransform = Mat4::IDENTITY;

    // let camera_angle_y = 0.0;
    // let camera_angle_xz = 0.0;
    // let camera_dist = 9.0;
    // // TODO: find out why models are appearing upside-down
    // let camera_centre = [0.0, 1.0, 0.0];
    // let camera_up = [0.0, -1.0, 0.0];

    // let game_state = Some(GameState {
    //     camera_angle_xz,
    //     camera_angle_y,
    //     camera_dist,
    //     camera_centre,
    //     camera_up,
    // });

    // let camera_position = [
    //     camera_angle_xz.cos() * camera_angle_y.sin() * camera_dist,
    //     camera_angle_xz.sin() * camera_dist + camera_centre[1],
    //     -camera_angle_xz.cos() * camera_angle_y.cos() * camera_dist,
    // ];

    // // println!("{} {}", camera_angle_y, camera_angle_xz);

    // let camera = Camera::new(
    //     camera_position,
    //     camera_centre,
    //     camera_up,
    //     WIDTH as u32,
    //     HEIGHT as u32,
    //     std::f32::consts::FRAC_PI_3,
    //     // 1.0472f32,
    // );

    // for x in 0..WIDTH {
    //     for y in 0..HEIGHT {
    //         let ray: Ray = rayForPixel(x, y, &camera);
    //         intersect(ray, inverseTransform, tlas);
    //     }
    // }

    let event_loop = EventLoop::new();
    let mut app = RenderApp::new(scene_path, &event_loop, args[2].parse::<bool>().unwrap());
    // app.init(scene_path);

    // let mut renderdoc_api: RenderDoc<renderdoc::V100> = RenderDoc::new().unwrap();
    // renderdoc_api.start_frame_capture(std::ptr::null(), std::ptr::null());
    app.render(event_loop);

    // drop(app);

    // log::info!("sleeping...");
    // thread::sleep(Duration::from_millis(4000));
    // log::info!("waking.");
    // renderdoc_api.end_frame_capture(std::ptr::null(), std::ptr::null());
}
