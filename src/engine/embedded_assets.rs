// Compile-time embedded assets for web builds.
// On native, scenes are loaded from the filesystem.
// On web, a fixed set of scenes are embedded in the binary.

use super::scene_importer::{Scene, SceneError};
use crate::RendererType;
use std::collections::HashMap;

struct EmbeddedScenes {
    scenes: HashMap<&'static str, &'static str>,
}

impl EmbeddedScenes {
    fn new() -> Self {
        let mut scenes = HashMap::new();
        scenes.insert("cornellbox", include_str!("../../assets/scenes/cornellbox.yaml"));
        scenes.insert("cornellbox2", include_str!("../../assets/scenes/cornellbox2.yaml"));
        scenes.insert("fresnel", include_str!("../../assets/scenes/fresnel.yaml"));
        scenes.insert("glass_test", include_str!("../../assets/scenes/glass_test.yaml"));
        Self { scenes }
    }
}

/// Load an embedded scene by name.
pub fn load_scene(name: &str, renderer_type: &RendererType) -> Result<Scene, SceneError> {
    let embedded = EmbeddedScenes::new();
    let Some(yaml) = embedded.scenes.get(name) else {
        let mut available: Vec<_> = embedded.scenes.keys().collect();
        available.sort();
        return Err(SceneError::Invalid(format!(
            "scene '{name}' not found in embedded assets. Available: {available:?}"
        )));
    };
    Scene::from_yaml_str(yaml, renderer_type)
}
