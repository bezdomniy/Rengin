extern crate yaml_rust;
use std::{collections::HashMap, default, ops::Deref};
use yaml_rust::{Yaml, YamlLoader};

use crate::engine::rt_primitives::Camera;

#[derive(Debug)]
enum ShapeType {
    Sphere,
    Plane,
    Model(&'static str),
    Group,
    Camera,
    Light,
}

#[derive(Debug)]
struct ShapeDefinition<'a> {
    shape_type: ShapeType,
    parameters: &'a ParameterDefinition<'a>,
    children: Option<Vec<&'a ShapeDefinition<'a>>>,
}

// impl<'a> Default for ShapeDefinition<'a> {
//     fn default() -> Self {
//         ShapeDefinition {
//             shape_type: ShapeType::Sphere,
//             parameters: &ParameterDefinition::default(),
//             children: None,
//         }
//     }
// }

#[derive(Default, Debug, Clone)]
struct ParameterDefinition<'a> {
    material: Option<MaterialDefinition>,
    transform: Option<TransformDefinition>,
    args: Option<Vec<f32>>,
    file_path: Option<&'static str>,
    light_definition: Option<LightDefinition>,
    camera_defintion: Option<CameraDefinition>,
    extends: Option<&'a ParameterDefinition<'a>>,
}

#[derive(Default, Debug, Clone)]
struct MaterialDefinition {}

#[derive(Default, Debug, Clone)]
struct TransformDefinition {}

#[derive(Default, Debug, Clone)]
struct CameraDefinition {
    width: u32,
    height: u32,
    field_of_view: f32,
    from: [f32; 3],
    to: [f32; 3],
    up: [f32; 3],
}

#[derive(Default, Debug, Clone)]
struct LightDefinition {
    at: [f32; 3],
    intensity: [f32; 3],
}

#[derive(Default, Debug)]
struct Scene<'a> {
    shape_definitions: HashMap<&'static str, ShapeDefinition<'a>>,
    parameter_definitions: HashMap<&'static str, ParameterDefinition<'a>>,
    shapes: Vec<&'a ShapeDefinition<'a>>,
}

impl<'a> Scene<'a> {
    pub fn new(path: &str) -> Self {
        // append hash to key if using in an add statement to deferentiate without explictly naming
        let mut shape_definitions: HashMap<&str, ShapeDefinition> = HashMap::default();
        let mut parameter_definitions: HashMap<&str, ParameterDefinition> = HashMap::default();
        let mut shapes: Vec<&ShapeDefinition> = vec![];

        let scene_str = std::fs::read_to_string(path).expect("Unable to read file");
        let scene_yaml = &YamlLoader::load_from_str(&scene_str).unwrap()[0];

        for item_yaml in scene_yaml.as_vec().unwrap() {
            let item = item_yaml.as_hash().unwrap();
            let (item_key, item_value) = item.front().unwrap();
            match item_key.as_str() {
                Some("add") => {
                    // let mut shape_def = ShapeDefinition {
                    //     ..Default::default()
                    // };

                    // let mut param_def = ParameterDefinition::default();

                    match item_value.as_str() {
                        Some("camera") => {
                            let mut definition = CameraDefinition {
                                ..Default::default()
                            };

                            for (k, v) in item.iter().skip(1) {
                                println!("{:?} {:?}", k, v);
                                match k.as_str() {
                                    Some("width") => definition.width = v.as_i64().unwrap() as u32,
                                    Some("height") => {
                                        definition.height = v.as_i64().unwrap() as u32
                                    }
                                    Some("field-of-view") => {
                                        definition.field_of_view = v.as_f64().unwrap() as f32
                                    }
                                    Some("from") => definition.from = Scene::yaml_to_arr3(v),
                                    Some("to") => definition.to = Scene::yaml_to_arr3(v),
                                    Some("up") => definition.up = Scene::yaml_to_arr3(v),
                                    _ => {}
                                }
                            }

                            parameter_definitions.insert(
                                "camera",
                                ParameterDefinition {
                                    camera_defintion: Some(definition),
                                    ..Default::default()
                                },
                            );

                            // let params = ParameterDefinition {camera_defintion}
                            println!("camera");
                        }
                        Some("light") => {
                            println!("light");
                        }
                        Some(_) => {
                            println!("Other str");
                        }
                        _ => {
                            println!("Other ??");
                        }
                    }
                }
                Some("define") => {}
                _ => {}
            }
        }

        Scene {
            shape_definitions,
            parameter_definitions,
            shapes,
        }
    }

    fn yaml_to_arr3(yaml: &Yaml) -> [f32; 3] {
        yaml.as_vec()
            .unwrap()
            .iter()
            .enumerate()
            .fold([0.0, 0.0, 0.0], |acc, (i, v)| {
                let mut new = acc;
                new[i] = v.as_f64().unwrap_or_else(|| v.as_i64().unwrap() as f64) as f32;
                new
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{Scene, ShapeDefinition};
    #[test]
    fn load_scene() {
        let scene = Scene::new("./assets/scenes/test.yaml");

        println!("{:?}", scene);

        // let mut shape_def = ShapeDefinition {
        //     ..Default::default()
        // };
        assert_eq!(2 + 2, 4);
    }
}
