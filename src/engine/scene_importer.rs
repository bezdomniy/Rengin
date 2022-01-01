use super::asset_importer::import_objs;
use super::{super::BVH, rt_primitives::Material, rt_primitives::ObjectParams};
use glam::{Mat4, Vec3, Vec4};
use image::{ImageBuffer, Rgba};
use itertools::{izip, Itertools};
use rand::{distributions::Alphanumeric, Rng};
use serde::Deserialize;
// use std::collections::HashMap;
use linked_hash_map::LinkedHashMap;
use std::{fs::File, path::Path};

static BUILTIN_SHAPES: [&'static str; 6] = ["camera", "light", "sphere", "plane", "cube", "group"];

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

type TransformDefinition = Vec<TransformValue>;

trait TransformDefinitionMethods {
    fn set_inverse_transform(&self, transform: &mut Mat4, commands: &Vec<Command>);
    fn get_transform(transform_name: &String, commands: &Vec<Command>) -> Mat4;
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DefineValue {
    ShapeDefinition(ShapeValue),
    MaterialDefinition(MaterialValue),
    TransformDefinition(TransformDefinition),
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
    pub width: u32,
    pub height: u32,
    // #[serde(rename(deserialize = "field-of-view"))]
    pub field_of_view: f32,
    pub from: [f32; 3],
    pub to: [f32; 3],
    pub up: [f32; 3],
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
    transform: Option<TransformDefinition>,
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

impl TransformDefinitionMethods for TransformDefinition {
    // pub fn set_transform(&self, transform: &mut , commands: &Vec<Command>) {
    //     match self {}
    // }

    fn set_inverse_transform(&self, transform: &mut Mat4, commands: &Vec<Command>) {
        let mut out_transform = Mat4::IDENTITY;
        for t in self.iter().rev() {
            out_transform *= match t {
                TransformValue::Scalar(s) => match s.name.as_str() {
                    "rotate-x" => Mat4::from_rotation_x(s.value),
                    "rotate-y" => Mat4::from_rotation_y(s.value),
                    "rotate-z" => Mat4::from_rotation_z(s.value),
                    _ => Mat4::IDENTITY,
                },
                TransformValue::Vector(v) => match v.name.as_str() {
                    "scale" => Mat4::from_scale(Vec3::new(v.value1, v.value2, v.value3)),
                    "translate" => Mat4::from_translation(Vec3::new(v.value1, v.value2, v.value3)),
                    _ => Mat4::IDENTITY,
                },
                TransformValue::Reference(r) => TransformDefinition::get_transform(r, commands), //TODO
            };
        }
        *transform = out_transform.inverse();
    }

    fn get_transform(transform_name: &String, commands: &Vec<Command>) -> Mat4 {
        let mut out_transform = Mat4::IDENTITY;
        for command in commands {
            match command {
                Command::Define(define_command) => {
                    if define_command.define == *transform_name {
                        if define_command.extend.is_some() {
                            out_transform *= TransformDefinition::get_transform(
                                define_command.extend.as_ref().unwrap(),
                                commands,
                            );
                        }
                        match &define_command.value {
                            DefineValue::TransformDefinition(transform_def) => {
                                let mut tranform_val = Mat4::IDENTITY;
                                transform_def.set_inverse_transform(&mut tranform_val, commands);
                                out_transform *= tranform_val.inverse();
                            }
                            _ => {
                                panic!("Transform definition found, but has no transform value.");
                            }
                        }
                    }
                }
                _ => {
                    continue;
                }
            }
        }
        return out_transform;
        // panic!("Transform definition: {} not found.", transform_name);
        // todo!()
    }
}

impl MaterialValue {
    pub fn set_material(&self, material: &mut Material, commands: &Vec<Command>) {
        match self {
            MaterialValue::Definition(material_def) => {
                MaterialValue::_set_material(material_def, material);
            }
            MaterialValue::Reference(material_ref) => {
                MaterialValue::_find_definition_and_set_material(material_ref, material, commands);
            }
        }
    }

    fn _find_definition_and_set_material(
        material_name: &String,
        material: &mut Material,
        commands: &Vec<Command>,
    ) {
        for command in commands {
            match command {
                Command::Define(define_command) => {
                    if define_command.define == *material_name {
                        if define_command.extend.is_some() {
                            MaterialValue::_find_definition_and_set_material(
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

// TODO: why are transforms in definition and add not combining properly
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
        let mut object_params: LinkedHashMap<(String, String), ObjectParams> = LinkedHashMap::new();
        let mut model_paths: Vec<String> = vec![];
        let mut camera: Option<CameraValue> = None;

        for command in commands {
            match command {
                Command::Add(add_command) => match add_command {
                    Add::Camera(add_camera) => {
                        camera = Some(add_camera.clone());
                    }
                    Add::Shape(add_shape) => {
                        Scene::_get_object_params(
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

        model_paths = model_paths.into_iter().unique().collect();
        println!("{:#?}", model_paths);

        // TODO: this approach doesn't fully work,
        //       it can't handle multiple instances of same model
        //       Make it only load each instance of model once, and
        //       handle drawing it multiple times
        let bvh = import_objs(&model_paths);

        // for (i, (object_param, offset_inners, len_inners, offset_leafs)) in izip!(
        //     // TODO: need to filter considering duplicate models too...
        //     object_params.iter_mut().filter(|x| { x.model_type >= 10 }),
        //     // &mut object_params,
        //     &bvh.as_ref().unwrap().offset_inner_nodes,
        //     &bvh.as_ref().unwrap().len_inner_nodes,
        //     &bvh.as_ref().unwrap().offset_leaf_nodes,
        // )
        // .enumerate()
        // {
        //     object_param.len_inner_nodes = *len_inners;
        //     object_param.offset_inner_nodes = *offset_inners;
        //     object_param.offset_leaf_nodes = *offset_leafs;
        //     object_param.model_type += i as u32;

        //     // println!("{:?} {} {}", object_param, len_leafs, len_inners);
        // }

        for (i, (obparam_key, obparam_value)) in object_params
            .iter_mut()
            .filter(|(_, v)| v.model_type >= 10)
            .enumerate()
        {
            let (inner_offset, inner_len, leaf_offset) =
                bvh.as_ref().unwrap().find_model_locations(&obparam_key.0);
            obparam_value.offset_inner_nodes = inner_offset;
            obparam_value.len_inner_nodes = inner_len;
            obparam_value.offset_leaf_nodes = leaf_offset;
            obparam_value.model_type += i as u32;
        }

        return (
            camera,
            Some(lights),
            Some(object_params.into_iter().map(|(k, v)| v).collect()),
            bvh,
        );
    }

    fn _set_primitive_type(type_name: &String, object_param: &mut ObjectParams) {
        match type_name.as_str() {
            "sphere" => {
                object_param.model_type = 0;
            }
            "plane" => {
                object_param.model_type = 1;
            }
            "cube" => {
                object_param.model_type = 2;
            }
            "group" => {
                object_param.model_type = 3;
            }
            _ => {}
        }
    }

    fn _get_object_params(
        curr_shape: &ShapeValue,
        accum_object_params: &mut LinkedHashMap<(String, String), ObjectParams>,
        accum_model_paths: &mut Vec<String>,
        commands: &Vec<Command>,
    ) {
        let mut object_param: ObjectParams = ObjectParams::default();
        let mut model_path_found: bool = true;

        let mut object_map_key = curr_shape.add.clone();

        let hash: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(8)
            .map(char::from)
            .collect();

        let is_model: bool = !BUILTIN_SHAPES.contains(&curr_shape.add.as_str());

        // if !BUILTIN_SHAPES.contains(&curr_shape.add.as_str()) {
        // model_path_found = false;
        for command in commands {
            match command {
                Command::Define(define_command) => {
                    // println!("{:?}", define_command);
                    if define_command.define == curr_shape.add {
                        match &define_command.value {
                            DefineValue::ShapeDefinition(defined_shape) => {
                                if is_model {
                                    model_path_found = false;
                                    if defined_shape.file.is_some() {
                                        object_map_key =
                                            defined_shape.file.as_ref().unwrap().clone();
                                        accum_model_paths.push(object_map_key.clone());
                                        model_path_found = true;
                                    }
                                    object_param.model_type = 10;
                                } else {
                                    Scene::_set_primitive_type(&curr_shape.add, &mut object_param);
                                }
                                if defined_shape.transform.is_some() {
                                    defined_shape
                                        .transform
                                        .as_ref()
                                        .unwrap()
                                        // .clone()
                                        .set_inverse_transform(
                                            &mut object_param.inverse_transform,
                                            commands,
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
            object_param.model_type = 10;
        } else if !model_path_found {
            panic!("Model definition does not contain file path.");
        }

        if !is_model {
            Scene::_set_primitive_type(&curr_shape.add, &mut object_param);
        }

        if curr_shape.transform.is_some() {
            curr_shape
                .transform
                .as_ref()
                .unwrap()
                // .clone()
                .set_inverse_transform(&mut object_param.inverse_transform, commands);
            //todo
        }
        if curr_shape.material.is_some() {
            curr_shape
                .material
                .as_ref()
                .unwrap()
                .set_material(&mut object_param.material, commands);
        }

        accum_object_params.insert((object_map_key.clone(), hash.clone()), object_param);

        if curr_shape.children.is_some() {
            for child in curr_shape.children.as_ref().unwrap() {
                Scene::_get_object_params(child, accum_object_params, accum_model_paths, commands);
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
