use super::super::BVH;
use image::{ImageBuffer, Rgba};
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Command {
    Add(Add),
    Define(Define),
    Fail(serde_yaml::Value),
}

#[derive(Debug)]
struct Scene {
    commands: Vec<Command>,
    bvh: Option<BVH>,
    textures: Option<Vec<Texture>>,
}

#[derive(Debug)]
struct Texture {
    path: String,
    data: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

impl Texture {
    pub fn new(file_path: &str) -> Self {
        let img = image::open(file_path).unwrap();
        let buf = img.to_rgba8();

        Texture {
            path: file_path.to_string(),
            data: buf,
        }
    }
}

impl Scene {
    pub fn new(file_path: &str) -> Self {
        let path = Path::new(file_path);
        let display = path.display();

        // Open the path in read-only mode, returns `io::Result<File>`
        let f = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}", display, why),
            Ok(file) => file,
        };
        // let scene: Vec<serde_yaml::Value> = serde_yaml::from_reader(f).unwrap();

        Scene {
            commands: serde_yaml::from_reader(f).expect("Failed to load scene description."),
            bvh: None,
            textures: None,
        }
    }

    fn validate(&self) {
        todo!();
    }

    fn load_assets(&self) {
        todo!();
    }
}

#[derive(Debug, Deserialize)]
struct Define {
    define: String,
    extend: Option<String>,
    value: DefineValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DefineValue {
    Shape(Shape),
    MaterialDefinition(MaterialDefinition),
    TransformDefinition(Vec<Transform>),
    Fail(serde_yaml::Value),
}

#[derive(Debug, Deserialize)]
// #[serde(tag = "add")]
#[serde(untagged)]
// #[serde(rename_all(deserialize = "camelCase"))]
enum Add {
    Camera(Camera),
    Light(Light),
    Shape(Shape),
}

#[derive(Debug, Deserialize)]
#[serde(rename_all(deserialize = "kebab-case"))]
struct Camera {
    width: u32,
    height: u32,
    // #[serde(rename(deserialize = "field-of-view"))]
    field_of_view: f32,
    from: [f32; 3],
    to: [f32; 3],
    up: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct Light {
    at: [f32; 3],
    intensity: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct Shape {
    add: String,
    args: Option<[f32; 3]>,
    material: Option<Material>,
    transform: Option<Vec<Transform>>,
    children: Option<Vec<Shape>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Material {
    MaterialDefinition(MaterialDefinition),
    MaterialReference(String),
}

// #[derive(Debug, Deserialize)]
// #[serde(untagged)]
// enum Transform {
//     TransformDefinition(Vec<TransformElement>),
//     TransformReference(String),
// }

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Transform {
    Vector(VectorTransform),
    Scalar(ScalarTransform),
    Reference(String),
}

#[derive(Debug, Deserialize)]
struct ScalarTransform {
    name: String,
    value: f32,
}

#[derive(Debug, Deserialize)]
struct VectorTransform {
    name: String,
    value1: f32,
    value2: f32,
    value3: f32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all(deserialize = "kebab-case"))]
struct MaterialDefinition {
    color: Option<[f32; 3]>,
    ambient: Option<f32>,
    diffuse: Option<f32>,
    specular: Option<f32>,
    shininess: Option<f32>,
    emissiveness: Option<[f32; 3]>,
    reflective: Option<f32>,
    transparency: Option<f32>,
    refractive_index: Option<f32>,
    pattern: Option<Pattern>,
}

// TODO
#[derive(Debug, Deserialize)]
struct Pattern {}

#[cfg(test)]
mod tests {
    use crate::engine::scene_importer::Scene;

    // use super::Scene;
    #[test]
    fn load_scene() {
        let scene = Scene::new("./assets/scenes/skybox.yaml");

        // let x = scene[0].is_sequence()
        println!("{:#?}", scene);

        assert_eq!(2 + 2, 4);
    }
}
