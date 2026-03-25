#[cfg(not(target_arch = "wasm32"))]
use rengin::RenderApp;
#[cfg(not(target_arch = "wasm32"))]
use winit::event_loop::{ControlFlow, EventLoop};

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = RenderApp::default();
    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Web entry point is in src/web.rs via wasm_bindgen(start).
    // This binary is not used on wasm32.
}
