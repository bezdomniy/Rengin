use std::collections::HashMap;
use std::path::Path;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{CompressionFormat, DecompressionStream, ReadableWritablePair, Response};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowExtWebSys};
use winit::window::Window;

use crate::engine::embedded_assets;
use crate::engine::obj_loader::parse_obj_triangles;
use crate::engine::scene_importer::Scene;
use crate::platform::Instant;
use crate::{RenderApp, RendererType};

#[wasm_bindgen(start)]
pub fn web_main() {
    std::panic::set_hook(Box::new(|info| {
        console_error_panic_hook::hook(info);
        set_status(Some("Something went wrong. The browser console has the details."));
    }));
    console_log::init_with_level(log::Level::Info).expect("Failed to init console_log");

    // index.html explains that WebGPU is missing.
    if !webgpu_available() {
        return;
    }

    let event_loop = EventLoop::new().unwrap();
    // Poll would schedule the next iteration immediately and starve the browser
    // (Firefox kills the script). With Wait, the request_redraw() at the end of
    // about_to_wait paces the loop to one iteration per animation frame.
    event_loop.set_control_flow(ControlFlow::Wait);

    // spawn_app hands the app to the browser's event loop and returns, instead of
    // run_app's throwing a JS exception to unwind (logged as an uncaught error).
    event_loop.spawn_app(RenderApp::default());
}

fn webgpu_available() -> bool {
    let navigator = js_sys::Reflect::get(&js_sys::global(), &"navigator".into()).unwrap_or_default();
    js_sys::Reflect::get(&navigator, &"gpu".into()).is_ok_and(|gpu| gpu.is_truthy())
}

/// Attaches the window's canvas to the page and focuses it, so the keyboard
/// moves the camera without clicking first. The page's CSS sizes it to fit the
/// viewport; winit reports that size as Resized and the render is letterboxed.
pub(crate) fn attach_canvas(window: &Window) {
    let canvas = window.canvas().expect("Failed to get canvas from winit window");
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("rengin-canvas"))
        .expect("No #rengin-canvas element found in DOM")
        .append_child(&canvas)
        .expect("Failed to append canvas to DOM");
    window.focus_window();
}

/// Shows `message` in the page's status line, or hides it.
pub(crate) fn set_status(message: Option<&str>) {
    let Some(status) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id("status"))
    else {
        return;
    };
    status.set_text_content(message);
    let _ = match message {
        Some(_) => status.remove_attribute("hidden"),
        None => status.set_attribute("hidden", ""),
    };
}

/// Loads an embedded scene, downloading the models it uses.
pub(crate) async fn load_scene(name: &str) -> Result<Scene, String> {
    let now = Instant::now();
    log::info!("Loading scene: {name}");
    let yaml = embedded_assets::scene_yaml(name).map_err(|error| error.to_string())?;
    let scene = Scene::parse(yaml, None, &RendererType::PathTracer).map_err(|error| error.to_string())?;

    let files = scene.model_files().to_vec();
    if !files.is_empty() {
        set_status(Some("Downloading models…"));
    }
    let downloads = files
        .iter()
        .map(|file| fetch_gzipped(embedded_assets::model_url(file)));
    let data = futures::future::try_join_all(downloads).await?;
    let models: HashMap<&str, Vec<u8>> = files.iter().map(String::as_str).zip(data).collect();
    if !models.is_empty() {
        // Building the BVH blocks the page, so let the message show first.
        set_status(Some("Building the scene…"));
        after_next_paint().await;
    }

    let scene = scene
        .build(|file| parse_obj_triangles(&models[file], Path::new(file)))
        .map_err(|error| error.to_string())?;
    log::info!("Finished loading scene in {} millis.", now.elapsed().as_millis());
    Ok(scene)
}

/// Downloads a gzipped file and decompresses it.
async fn fetch_gzipped(url: String) -> Result<Vec<u8>, String> {
    let error = |error: JsValue| format!("couldn't download {url}: {}", js_error_message(&error));
    let window = web_sys::window().expect("no global window");
    let response: Response = JsFuture::from(window.fetch_with_str(&url))
        .await
        .map_err(error)?
        .unchecked_into();
    if !response.ok() {
        return Err(format!("couldn't download {url}: HTTP {}", response.status()));
    }
    let body = response
        .body()
        .ok_or_else(|| format!("couldn't download {url}: the response is empty"))?;
    let gunzip = DecompressionStream::new(CompressionFormat::Gzip).map_err(error)?;
    let decompressed = body.pipe_through(&ReadableWritablePair::new(
        &gunzip.readable(),
        &gunzip.writable(),
    ));
    let buffer = Response::new_with_opt_readable_stream(Some(&decompressed))
        .and_then(|response| response.array_buffer())
        .map_err(error)?;
    let buffer = JsFuture::from(buffer).await.map_err(error)?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

fn js_error_message(error: &JsValue) -> String {
    error
        .dyn_ref::<js_sys::Error>()
        .map(|error| String::from(error.message()))
        .or_else(|| error.as_string())
        .unwrap_or_else(|| format!("{error:?}"))
}

/// Resolves once the browser has painted, so DOM changes show before
/// long-running work blocks the page.
async fn after_next_paint() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let after_frame = Closure::once_into_js(move || {
            let _ = web_sys::window()
                .expect("no global window")
                .set_timeout_with_callback(&resolve);
        });
        let _ = web_sys::window()
            .expect("no global window")
            .request_animation_frame(after_frame.unchecked_ref());
    });
    let _ = JsFuture::from(promise).await;
}
