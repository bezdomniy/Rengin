// Platform abstraction layer for native vs web builds.

cfg_if::cfg_if! {
    if #[cfg(target_arch = "wasm32")] {
        pub use web_time::Instant;
    } else {
        pub use std::time::Instant;
    }
}

/// Application configuration, parsed from CLI args (native) or URL params (web).
pub struct AppConfig {
    pub scene: String,
    pub rays_per_pixel: u32,
    pub bounces: u32,
    pub profile: bool,
    pub continuous_motion: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<&super::Args> for AppConfig {
    fn from(args: &super::Args) -> Self {
        AppConfig {
            scene: args.scene.clone(),
            rays_per_pixel: args.rays_per_pixel,
            bounces: args.bounces,
            profile: args.profile,
            continuous_motion: !args.draw_on_mouseup,
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl AppConfig {
    pub fn from_url_params() -> Self {
        let window = web_sys::window().expect("no global window");
        let search = window.location().search().unwrap_or_default();
        let params = web_sys::UrlSearchParams::new_with_str(&search).unwrap();

        AppConfig {
            scene: params.get("scene").unwrap_or_else(|| "cornellbox".into()),
            rays_per_pixel: params
                .get("rpp")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8),
            bounces: params
                .get("bounces")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8),
            profile: false,
            continuous_motion: true,
        }
    }
}
