use super::asset_importer::import_objs;
use super::{super::BVH, rt_primitives::ObjectParams};
use image::{ImageBuffer, Rgba};
use serde::Deserialize;
use std::{fs::File, path::Path};

static BUILTIN_SHAPES: [&'static str; 5] = ["camera", "light", "plane", "sphere", "sphere"];

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Command {
    Add(Add),
    Define(Define),
    // Fail(serde_yaml::Value),
}

#[derive(Debug)]
pub struct Scene {
    commands: Vec<Command>,
    pub bvh: Option<BVH>,
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

        let commands: Vec<Command> =
            serde_yaml::from_reader(f).expect("Failed to load scene description.");
        let bvh = Scene::load_assets(&commands);

        let scene = Scene {
            commands,
            bvh,
            textures: None,
        };

        // scene.load_assets();

        scene
    }

    fn validate(&self) {
        todo!();
    }

    fn load_textures(commands: &Vec<Command>) -> Option<Vec<Texture>> {
        todo!()
    }

    // I think all bvh object params have to be at the start of the vec
    fn get_object_params(&self) -> Vec<ObjectParams> {
        todo!()
    }

    // TODO: we have to be able to get object params in order of adding to work with bvh in shader
    fn load_assets(commands: &Vec<Command>) -> Option<BVH> {
        let mut add_obj_paths: Vec<String> = vec![];
        for command in commands {
            match command {
                Command::Add(add_command) => match add_command {
                    Add::Shape(shape) => {
                        Scene::_find_asset_paths(shape, &mut add_obj_paths, commands)
                    }
                    _ => {}
                },
                // Command::Define(define_command) => {
                //     match &define_command.value {
                //         DefineValue::Shape(shape) => {
                //             Scene::_find_asset_paths(&shape, &mut define_obj_paths)
                //         }
                //         _ => {}
                //     };
                // }
                _ => {}
            };
        }
        println!("{:#?}", add_obj_paths);

        import_objs(add_obj_paths)
    }

    fn _find_asset_paths(curr_shape: &Shape, accum: &mut Vec<String>, commands: &Vec<Command>) {
        if curr_shape.file.is_some() {
            accum.push(curr_shape.file.as_ref().unwrap().clone());
        } else if !BUILTIN_SHAPES.contains(&curr_shape.add.as_str()) {
            for command in commands {
                match command {
                    Command::Define(define_command) => {
                        // println!("{:?}", define_command);
                        if define_command.define == curr_shape.add {
                            match &define_command.value {
                                DefineValue::Shape(defined_shape) => {
                                    if defined_shape.file.is_some() {
                                        accum.push(defined_shape.file.as_ref().unwrap().clone());
                                    } else {
                                        // break;
                                        panic!("Could not find model for added shape.");
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                };
            }
        }
        if curr_shape.children.is_some() {
            for child in curr_shape.children.as_ref().unwrap() {
                Scene::_find_asset_paths(child, accum, commands);
            }
        }
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
    // Fail(serde_yaml::Value),
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
    file: Option<String>,
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
        let scene = Scene::new("./assets/scenes/model3.yaml");

        // let x = scene[0].is_sequence()
        // println!("{:#?}", scene);

        assert_eq!(2 + 2, 4);
    }
}
