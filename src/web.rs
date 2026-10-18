use wasm_bindgen::prelude::*;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::RenderApp;

#[wasm_bindgen(start)]
pub fn web_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(log::Level::Info).expect("Failed to init console_log");

    let event_loop = EventLoop::new().unwrap();
    // Poll would schedule the next iteration immediately and starve the browser
    // (Firefox kills the script). With Wait, the request_redraw() at the end of
    // about_to_wait paces the loop to one iteration per animation frame.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = RenderApp::default();
    event_loop.run_app(&mut app).unwrap();
}
