// Scenes compiled into the web build. Their OBJ models are not embedded: the
// web build fetches each one from `model_url`, and scripts/gzip-models.sh adds
// them to the trunk build output.

use super::scene_importer::SceneError;
use std::path::Path;

/// The directory the embedded scenes' model paths are relative to.
pub const SCENE_DIR: &str = "assets/scenes";

pub const SCENES: &[(&str, &str)] = &[
    ("cornellbox", include_str!("../../assets/scenes/cornellbox.yaml")),
    ("cornellbox2", include_str!("../../assets/scenes/cornellbox2.yaml")),
    ("coverScene", include_str!("../../assets/scenes/coverScene.yaml")),
    ("doom", include_str!("../../assets/scenes/doom.yaml")),
    ("fresnel", include_str!("../../assets/scenes/fresnel.yaml")),
    ("glass_test", include_str!("../../assets/scenes/glass_test.yaml")),
    ("models", include_str!("../../assets/scenes/models.yaml")),
    ("reflectionScene", include_str!("../../assets/scenes/reflectionScene.yaml")),
    ("sponza", include_str!("../../assets/scenes/sponza.yaml")),
    ("table", include_str!("../../assets/scenes/table.yaml")),
];

/// The YAML of an embedded scene.
pub fn scene_yaml(name: &str) -> Result<&'static str, SceneError> {
    SCENES
        .iter()
        .find(|(scene, _)| *scene == name)
        .map(|(_, yaml)| *yaml)
        .ok_or_else(|| {
            let available: Vec<_> = SCENES.iter().map(|(scene, _)| *scene).collect();
            SceneError::Invalid(format!(
                "scene '{name}' not found in embedded assets. Available: {available:?}"
            ))
        })
}

/// Where the web build serves a scene's model file, relative to the page:
/// the gzipped file under `models/`, by file name.
pub fn model_url(file: &str) -> String {
    let name = Path::new(file)
        .file_name()
        .map_or_else(|| file.into(), |name| name.to_string_lossy());
    format!("models/{name}.gz")
}
