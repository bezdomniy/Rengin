use rengin::run;

#[cfg(not(target_arch = "wasm32"))]
use futures::executor;

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        executor::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
