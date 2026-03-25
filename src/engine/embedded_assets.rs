// Compile-time embedded assets for web builds.
// On native, scenes are loaded from the filesystem.
// On web, a fixed set of scenes are embedded in the binary.

use super::scene_importer::Scene;
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

/// Load an embedded scene by name. Panics if the scene is not found.
pub fn load_scene(name: &str, renderer_type: &RendererType) -> Scene {
    let embedded = EmbeddedScenes::new();
    let yaml = embedded
        .scenes
        .get(name)
        .unwrap_or_else(|| {
            let available: Vec<_> = embedded.scenes.keys().collect();
            panic!(
                "Scene '{}' not found in embedded assets. Available: {:?}",
                name, available
            );
        });
    Scene::from_yaml_str(yaml, renderer_type)
}
