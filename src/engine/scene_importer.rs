use super::asset_importer::import_objs;
use super::{super::BVH, rt_primitives::Material, rt_primitives::ObjectParams};
use glam::{Mat4, Vec3, Vec4};
use image::{ImageBuffer, Rgba};
use itertools::izip;
use serde::Deserialize;
use std::{fs::File, path::Path};

static BUILTIN_SHAPES: [&'static str; 5] = ["camera", "light", "plane", "sphere", "group"];

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
    pub camera: Option<CameraValue>,
    pub lights: Option<Vec<LightValue>>,
    pub object_params: Option<Vec<ObjectParams>>,
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

#[derive(Debug, Deserialize)]
struct Define {
    define: String,
    extend: Option<String>, //TODO
    value: DefineValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DefineValue {
    ShapeDefinition(ShapeValue),
    MaterialDefinition(MaterialValue),
    TransformDefinition(Vec<TransformValue>),
    // Fail(serde_yaml::Value),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Add {
    Camera(CameraValue),
    Light(LightValue),
    Shape(ShapeValue),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all(deserialize = "kebab-case"))]
pub struct CameraValue {
    width: u32,
    height: u32,
    // #[serde(rename(deserialize = "field-of-view"))]
    field_of_view: f32,
    from: [f32; 3],
    to: [f32; 3],
    up: [f32; 3],
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct LightValue {
    pub at: [f32; 3],
    pub intensity: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct ShapeValue {
    add: String,
    args: Option<[f32; 3]>,
    file: Option<String>,
    material: Option<MaterialValue>,
    transform: Option<Vec<TransformValue>>,
    children: Option<Vec<ShapeValue>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MaterialValue {
    Definition(MaterialDefinition),
    Reference(String),
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TransformValue {
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
    pattern: Option<PatternValue>,
}

impl MaterialValue {
    pub fn set_material(&self, material: &mut Material, commands: &Vec<Command>) {
        match self {
            MaterialValue::Definition(material_def) => {
                MaterialValue::_set_material(material_def, material);
            }
            MaterialValue::Reference(material_ref) => {
                MaterialValue::_find_definition_and_set(material_ref, material, commands);
            }
        }
    }

    fn _find_definition_and_set(
        material_name: &String,
        material: &mut Material,
        commands: &Vec<Command>,
    ) {
        for command in commands {
            match command {
                Command::Define(define_command) => {
                    if define_command.define == *material_name {
                        if define_command.extend.is_some() {
                            MaterialValue::_find_definition_and_set(
                                define_command.extend.as_ref().unwrap(),
                                material,
                                commands,
                            );
                        }
                        match &define_command.value {
                            DefineValue::MaterialDefinition(material_def) => {
                                material_def.set_material(material, commands);
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn _set_material(material_def: &MaterialDefinition, material: &mut Material) {
        if material_def.color.is_some() {
            let color = material_def.color.unwrap();
            (*material).colour = Vec4::new(color[0], color[1], color[2], 1f32);
        }
        if material_def.emissiveness.is_some() {
            let emissiveness = material_def.emissiveness.unwrap();
            (*material).emissiveness =
                Vec4::new(emissiveness[0], emissiveness[1], emissiveness[2], 1f32);
        }
        if material_def.ambient.is_some() {
            (*material).ambient = material_def.ambient.unwrap();
        }
        if material_def.diffuse.is_some() {
            (*material).diffuse = material_def.diffuse.unwrap();
        }
        if material_def.specular.is_some() {
            (*material).specular = material_def.specular.unwrap();
        }
        if material_def.shininess.is_some() {
            (*material).shininess = material_def.shininess.unwrap();
        }
        if material_def.reflective.is_some() {
            (*material).reflective = material_def.reflective.unwrap();
        }
        if material_def.transparency.is_some() {
            (*material).transparency = material_def.transparency.unwrap();
        }
        if material_def.refractive_index.is_some() {
            (*material).refractive_index = material_def.refractive_index.unwrap();
        }
        // if material_def.pattern.is_some() {
        //     (*material).pattern = material_def.pattern.unwrap();
        // }},
    }
}

// TODO
#[derive(Debug, Deserialize)]
struct PatternValue {}

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
        let (camera, lights, object_params, bvh) = Scene::load_assets(&commands);

        let scene = Scene {
            commands,
            bvh,
            lights,
            object_params,
            camera,
            textures: None,
        };

        scene
    }

    fn validate(&self) {
        todo!();
    }

    fn load_textures(commands: &Vec<Command>) -> Option<Vec<Texture>> {
        todo!()
    }

    fn load_assets(
        commands: &Vec<Command>,
    ) -> (
        Option<CameraValue>,
        Option<Vec<LightValue>>,
        Option<Vec<ObjectParams>>,
        Option<BVH>,
    ) {
        let mut lights: Vec<LightValue> = vec![];
        let mut object_params: Vec<ObjectParams> = vec![];
        let mut model_paths: Vec<String> = vec![];
        let mut camera: Option<CameraValue> = None;

        for command in commands {
            match command {
                Command::Add(add_command) => match add_command {
                    Add::Camera(add_camera) => {
                        camera = Some(add_camera.clone());
                    }
                    Add::Shape(add_shape) => {
                        Scene::_get_model_params(
                            add_shape,
                            &mut object_params,
                            &mut model_paths,
                            commands,
                        );
                        // Scene::_get_object_params(add_shape, &mut object_params, commands);
                    }
                    Add::Light(add_light) => {
                        lights.push(add_light.clone());
                    }
                    _ => continue,
                },
                _ => continue,
            };
        }
        println!("{:#?}", model_paths);

        let bvh = import_objs(model_paths);

        for (object_param, len_inners, len_leafs) in izip!(
            &mut object_params,
            &bvh.as_ref().unwrap().len_inner_nodes,
            &bvh.as_ref().unwrap().len_leaf_nodes,
        ) {
            object_param.len_inner_nodes = *len_inners;
            object_param.len_leaf_nodes = *len_leafs;

            // println!("{:?} {} {}", object_param, len_leafs, len_inners);
        }

        return (camera, Some(lights), Some(object_params), bvh);
    }

    // I think all bvh object params have to be at the start of the vec
    fn _get_object_params(
        curr_shape: &ShapeValue,
        accum: &mut Vec<ObjectParams>,
        commands: &Vec<Command>,
    ) {
        todo!()
    }

    fn get_inverse_transform(vec_transforms: &Vec<TransformValue>) -> Mat4 {
        let mut transform = Mat4::IDENTITY;
        for t in vec_transforms {
            transform *= match t {
                TransformValue::Scalar(s) => match s.name.as_str() {
                    "rotate-x" => Mat4::from_rotation_x(s.value),
                    "rotate_y" => Mat4::from_rotation_y(s.value),
                    "rotate_z" => Mat4::from_rotation_z(s.value),
                    _ => Mat4::IDENTITY,
                },
                TransformValue::Vector(v) => match v.name.as_str() {
                    "scale" => Mat4::from_scale(Vec3::new(v.value1, v.value2, v.value3)),
                    "translate" => Mat4::from_translation(Vec3::new(v.value1, v.value2, v.value3)),
                    _ => Mat4::IDENTITY,
                },
                TransformValue::Reference(r) => Mat4::IDENTITY, //TODO
            };
        }
        transform.inverse()
    }

    // fn get_material(materialValue: &MaterialValue, material: &Material) -> Material {
    //     materialValue.set_material(material)
    //     // match materialValue {
    //     //     MaterialValue::Definition(mat_def) => Material::default(),
    //     //     MaterialValue::Reference(mat_ref) => Material::default(),
    //     // }
    // }

    fn _get_model_params(
        curr_shape: &ShapeValue,
        accum_object_params: &mut Vec<ObjectParams>,
        accum_model_paths: &mut Vec<String>,
        commands: &Vec<Command>,
    ) {
        let mut object_param: ObjectParams = ObjectParams::default();
        let mut model_path_found: bool = true;

        // if !BUILTIN_SHAPES.contains(&curr_shape.add.as_str()) {
        // model_path_found = false;
        for command in commands {
            match command {
                Command::Define(define_command) => {
                    // println!("{:?}", define_command);
                    if define_command.define == curr_shape.add {
                        match &define_command.value {
                            DefineValue::ShapeDefinition(defined_shape) => {
                                if !BUILTIN_SHAPES.contains(&curr_shape.add.as_str()) {
                                    model_path_found = false;
                                    if defined_shape.file.is_some() {
                                        accum_model_paths
                                            .push(defined_shape.file.as_ref().unwrap().clone());
                                        model_path_found = true;
                                    }
                                }
                                if defined_shape.transform.is_some() {
                                    object_param.inverse_transform = Scene::get_inverse_transform(
                                        defined_shape.transform.as_ref().unwrap().clone(),
                                    );
                                    //todo
                                }
                                if defined_shape.material.is_some() {
                                    defined_shape
                                        .material
                                        .as_ref()
                                        .unwrap()
                                        .set_material(&mut object_param.material, commands);

                                    // object_param.material.colour.x = 1.0;
                                    // object_param.material.colour.w = 1.0;
                                    // object_param.material.ambient = 0.2;
                                    //todo
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            };
        }
        // }

        if curr_shape.file.is_some() {
            if model_path_found {
                accum_model_paths.pop();
            }

            accum_model_paths.push(curr_shape.file.as_ref().unwrap().clone());
        } else if !model_path_found {
            panic!("Model definition does not contain file path.");
        }

        if curr_shape.transform.is_some() {
            object_param.inverse_transform =
                Scene::get_inverse_transform(curr_shape.transform.as_ref().unwrap().clone());
            //todo
        }
        if curr_shape.material.is_some() {
            curr_shape
                .material
                .as_ref()
                .unwrap()
                .set_material(&mut object_param.material, commands);
        }

        accum_object_params.push(object_param);

        if curr_shape.children.is_some() {
            for child in curr_shape.children.as_ref().unwrap() {
                Scene::_get_model_params(child, accum_object_params, accum_model_paths, commands);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::scene_importer::Scene;

    // use super::Scene;
    #[test]
    fn load_scene() {
        let scene = Scene::new("./assets/scenes/model3.yaml");

        // let x = scene[0].is_sequence()
        println!("{:#?}", scene);

        assert_eq!(2 + 2, 4);
    }
}
