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
    pub render_scale: f32,
    pub rays_per_pixel: u32,
    pub bounces: u32,
    pub profile: bool,
    pub profile_stats: bool,
    pub continuous_motion: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl From<&super::Args> for AppConfig {
    fn from(args: &super::Args) -> Self {
        let render_scale = if args.render_scale.is_finite() && args.render_scale > 0.0 {
            args.render_scale
        } else {
            1.0
        };
        AppConfig {
            scene: args.scene.clone(),
            render_scale,
            rays_per_pixel: args.rays_per_pixel,
            bounces: args.bounces,
            profile: args.profile || args.profile_stats,
            profile_stats: args.profile_stats,
            continuous_motion: !args.draw_on_mouseup,
        }
    }
}

/// URL parameter names for each option: the CLI long and short flag names,
/// plus older web-only aliases. Profiling options are native-only.
#[cfg(target_arch = "wasm32")]
pub const SCENE_PARAMS: &[&str] = &["scene", "s"];
#[cfg(target_arch = "wasm32")]
pub const RAYS_PER_PIXEL_PARAMS: &[&str] = &["rays-per-pixel", "r", "rpp"];
#[cfg(target_arch = "wasm32")]
pub const BOUNCES_PARAMS: &[&str] = &["bounces", "b"];
#[cfg(target_arch = "wasm32")]
pub const RENDER_SCALE_PARAMS: &[&str] = &["render-scale", "render_scale"];

#[cfg(target_arch = "wasm32")]
impl AppConfig {
    pub fn from_url_params() -> Self {
        let window = web_sys::window().expect("no global window");
        let search = window.location().search().unwrap_or_default();
        let params = web_sys::UrlSearchParams::new_with_str(&search).unwrap();

        let known = [SCENE_PARAMS, RAYS_PER_PIXEL_PARAMS, BOUNCES_PARAMS, RENDER_SCALE_PARAMS];
        for name in params.keys() {
            let name = name.ok().and_then(|name| name.as_string()).unwrap_or_default();
            if !known.iter().any(|names| names.contains(&name.as_str())) {
                log::warn!("Ignoring unknown URL parameter `{name}`");
            }
        }

        AppConfig {
            scene: url_param(&params, SCENE_PARAMS, "cornellbox".to_string(), |_| true),
            render_scale: url_param(&params, RENDER_SCALE_PARAMS, 1.0, |scale: &f32| {
                scale.is_finite() && *scale > 0.0
            }),
            rays_per_pixel: url_param(&params, RAYS_PER_PIXEL_PARAMS, 8, |rays| *rays > 0),
            bounces: url_param(&params, BOUNCES_PARAMS, 8, |bounces| *bounces > 0),
            profile: false,
            profile_stats: false,
            continuous_motion: true,
        }
    }
}

/// Reads the first of `names` present in the URL, falling back to `default`
/// (with a warning) when its value doesn't parse or isn't `valid`.
#[cfg(target_arch = "wasm32")]
fn url_param<T: std::str::FromStr>(
    params: &web_sys::UrlSearchParams,
    names: &[&str],
    default: T,
    valid: impl Fn(&T) -> bool,
) -> T {
    let Some((name, value)) = names
        .iter()
        .find_map(|name| params.get(name).map(|value| (name, value)))
    else {
        return default;
    };
    match value.parse() {
        Ok(parsed) if valid(&parsed) => parsed,
        _ => {
            log::warn!("Ignoring invalid URL parameter `{name}={value}`, using the default");
            default
        }
    }
}
