//! The settings panel, drawn with egui over the rendered image.

use std::time::Duration;

use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::window::Window;

/// What the panel shows: the current settings and the render's progress.
pub struct Panel<'a> {
    pub scenes: &'a [String],
    pub scene: &'a str,
    /// A scene that is downloading or being built, if any.
    pub loading_scene: Option<&'a str>,
    pub rays_per_pixel: u32,
    pub bounces: u32,
    /// Profiling sizes its timestamp queries for the starting bounce count.
    pub bounces_editable: bool,
    pub render_scale: f32,
    pub render_size: PhysicalSize<u32>,
    pub samples_done: u32,
    /// How long the image has taken so far, or took if it is done.
    pub render_time: Duration,
}

/// A setting changed in the panel.
pub enum Change {
    Scene(String),
    RaysPerPixel(u32),
    Bounces(u32),
    RenderScale(f32),
}

const MAX_RAYS_PER_PIXEL: u32 = 1 << 20;
const MAX_BOUNCES: u32 = 64;
const RENDER_SCALES: std::ops::RangeInclusive<f32> = 0.1..=2.0;

pub struct Ui {
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    /// Laid out but not painted yet.
    pending: Option<egui::FullOutput>,
    /// The render scale while its slider is dragged. Changing the scale rebuilds
    /// the GPU buffers, so it is applied when the slider is let go.
    dragged_render_scale: Option<f32>,
    repaint: bool,
}

impl Ui {
    pub fn new(window: &Window, device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        ctx.set_theme(egui::Theme::Dark);
        let state = egui_winit::State::new(
            ctx,
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer =
            egui_wgpu::Renderer::new(device, target_format, egui_wgpu::RendererOptions::default());
        Self {
            state,
            renderer,
            pending: None,
            dragged_render_scale: None,
            repaint: true,
        }
    }

    /// Passes a window event to the panel. Returns true if the panel used it,
    /// in which case the camera should ignore it.
    pub fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.state.on_window_event(window, event);
        self.repaint |= response.repaint;
        match event {
            // egui claims the keyboard while any widget has focus, which a
            // clicked dropdown keeps; only typing a value should stop the
            // camera keys. Releases always reach the camera so no key sticks.
            WindowEvent::KeyboardInput { event, .. } => {
                event.state.is_pressed() && self.state.egui_ctx().text_edit_focused()
            }
            _ => response.consumed,
        }
    }

    /// Whether the panel has changed since it was last drawn.
    pub fn needs_repaint(&self) -> bool {
        self.repaint
    }

    /// Lays out the panel and returns the settings changed in it.
    pub fn run(&mut self, window: &Window, panel: &Panel) -> Vec<Change> {
        let input = self.state.take_egui_input(window);
        let ctx = self.state.egui_ctx().clone();
        let mut changes = Vec::new();
        let mut output = ctx.run_pass(input, |ctx| {
            self.dragged_render_scale =
                show_panel(ctx, panel, self.dragged_render_scale, &mut changes);
        });
        self.state
            .handle_platform_output(window, std::mem::take(&mut output.platform_output));
        self.repaint = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .is_some_and(|viewport| viewport.repaint_delay.is_zero());

        // Texture updates (such as the font atlas) must not be lost if the
        // previous output was never painted.
        if let Some(previous) = self.pending.take() {
            let mut textures_delta = previous.textures_delta;
            textures_delta.append(std::mem::take(&mut output.textures_delta));
            output.textures_delta = textures_delta;
        }
        self.pending = Some(output);
        changes
    }

    /// Encodes a pass drawing the panel over `view`. The returned command
    /// buffers must be submitted before `encoder`.
    pub fn paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: PhysicalSize<u32>,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(output) = self.pending.take() else {
            return Vec::new();
        };
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: output.pixels_per_point,
        };
        let jobs = self
            .state
            .egui_ctx()
            .tessellate(output.shapes, output.pixels_per_point);
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }
        let command_buffers = self
            .renderer
            .update_buffers(device, queue, encoder, &jobs, &screen);
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("settings panel"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            self.renderer.render(&mut pass, &jobs, &screen);
        }
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        command_buffers
    }
}

/// Shows the panel, adding any changes to `changes`. Returns the render scale
/// being dragged, if any.
fn show_panel(
    ctx: &egui::Context,
    panel: &Panel,
    dragged_render_scale: Option<f32>,
    changes: &mut Vec<Change>,
) -> Option<f32> {
    let mut still_dragged = None;
    // A collapsing header rather than the window's title bar, which only
    // collapses from its arrow; the whole header row is easier to tap.
    egui::Window::new("Settings")
        .title_bar(false)
        .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
        .resizable(false)
        .show(ctx, |ui| {
            egui::CollapsingHeader::new("Settings")
                // On a phone the open panel would hide much of the image.
                .default_open(ctx.content_rect().width() >= 600.0)
                .show(ui, |ui| {
                    egui::Grid::new("settings").num_columns(2).show(ui, |ui| {
                        ui.label("Scene");
                        ui.add_enabled_ui(panel.loading_scene.is_none(), |ui| {
                            egui::ComboBox::from_id_salt("scene")
                                .selected_text(panel.loading_scene.unwrap_or(panel.scene))
                                .show_ui(ui, |ui| {
                                    for scene in panel.scenes {
                                        let current = scene == panel.scene;
                                        if ui.selectable_label(current, scene).clicked() && !current
                                        {
                                            changes.push(Change::Scene(scene.clone()));
                                        }
                                    }
                                });
                        });
                        ui.end_row();

                        ui.label("Rays per pixel");
                        let mut rays_per_pixel = panel.rays_per_pixel;
                        let drag =
                            egui::DragValue::new(&mut rays_per_pixel).range(1..=MAX_RAYS_PER_PIXEL);
                        if ui.add(drag).changed() && rays_per_pixel != panel.rays_per_pixel {
                            changes.push(Change::RaysPerPixel(rays_per_pixel));
                        }
                        ui.end_row();

                        ui.label("Bounces");
                        let mut bounces = panel.bounces;
                        let drag = egui::DragValue::new(&mut bounces).range(1..=MAX_BOUNCES);
                        let response = ui
                            .add_enabled(panel.bounces_editable, drag)
                            .on_disabled_hover_text("Fixed while profiling");
                        if response.changed() && bounces != panel.bounces {
                            changes.push(Change::Bounces(bounces));
                        }
                        ui.end_row();

                        ui.label("Render scale");
                        let mut render_scale = dragged_render_scale.unwrap_or(panel.render_scale);
                        let response = ui
                            .add(egui::Slider::new(&mut render_scale, RENDER_SCALES).step_by(0.05));
                        if response.dragged() {
                            still_dragged = Some(render_scale);
                        } else if render_scale != panel.render_scale {
                            changes.push(Change::RenderScale(render_scale));
                        }
                        ui.end_row();
                    });
                    ui.label(format!(
                        "{} × {} pixels",
                        panel.render_size.width, panel.render_size.height
                    ));

                    ui.separator();
                    let done = panel.samples_done.min(panel.rays_per_pixel);
                    ui.add(
                        egui::ProgressBar::new(done as f32 / panel.rays_per_pixel as f32)
                            .text(format!("{done} / {} samples", panel.rays_per_pixel)),
                    );
                    ui.label(format!("{:.1} s", panel.render_time.as_secs_f32()));

                    ui.collapsing("Controls", |ui| {
                        ui.label("W A S D, Space, Shift: move");
                        ui.label("Arrow keys: turn");
                        ui.label("Right-click: toggle mouse look");
                    });
                });
        });
    still_dragged
}
