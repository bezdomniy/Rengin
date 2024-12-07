use super::{bvh::Bvh, rt_primitives::Material, rt_primitives::ObjectParam};
use crate::RendererType;
use glam::{Mat4, Vec3, Vec4};
use image::{ImageBuffer, Rgba};
use itertools::Itertools;
use rand::{distributions::Alphanumeric, Rng};
use serde::Deserialize;
// use std::collections::HashMap;
use linked_hash_map::LinkedHashMap;
use std::{fs::File, path::Path};

static BUILTIN_SHAPES: [&str; 6] = ["camera", "light", "sphere", "plane", "cube", "group"];

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Command {
    Add(Add),
    Define(Define),
    // Fail(serde_yaml::Value),
}

#[derive(Debug)]
pub struct Scene {
    pub bvh: Option<Bvh>,
    pub camera: Option<CameraValue>,
    pub object_params: Option<Vec<ObjectParam>>,
    pub specular_offset: usize,
    pub lights_offset: usize,
    _textures: Option<Vec<Texture>>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct Texture {
    path: String,
    data: ImageBuffer<Rgba<u8>, Vec<u8>>,
}

impl Texture {
    #[allow(dead_code)]
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
    fn set_transform(&self, transform: &mut Mat4, commands: &[Command]);
    fn get_transform(transform_name: &str, commands: &[Command]) -> Mat4;
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum DefineValue {
    Shape(ShapeValue),
    Material(MaterialValue),
    Transform(TransformDefinition),
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

#[derive(Debug, Default, Deserialize)]
struct ShapeValue {
    add: String,
    #[allow(dead_code)]
    args: Option<[f32; 3]>,
    file: Option<String>,
    material: Option<MaterialValue>,
    transform: Option<TransformDefinition>,
    children: Option<Vec<ShapeValue>>,
}

impl From<LightValue> for ShapeValue {
    fn from(light: LightValue) -> Self {
        let transform = Some(vec![TransformValue::Vector(VectorTransform {
            name: "translate".to_string(),
            value1: light.at[0],
            value2: light.at[1],
            value3: light.at[2],
        })]);

        let emissiveness = Some([light.intensity[0], light.intensity[1], light.intensity[2]]);

        let material = Some(MaterialValue::Definition(MaterialDefinition {
            emissiveness,
            ..Default::default()
        }));

        ShapeValue {
            add: "light".to_string(),
            transform,
            material,
            ..Default::default()
        }
    }
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

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all(deserialize = "kebab-case"))]
struct MaterialDefinition {
    color: Option<[f32; 3]>,
    emissiveness: Option<[f32; 3]>,
    ambient: Option<f32>,
    diffuse: Option<f32>,
    specular: Option<f32>,
    shininess: Option<f32>,
    reflective: Option<f32>,
    transparency: Option<f32>,
    refractive_index: Option<f32>,
    _pattern: Option<PatternValue>,
}

impl TransformDefinitionMethods for TransformDefinition {
    // pub fn set_transform(&self, transform: &mut , commands: &Vec<Command>) {
    //     match self {}
    // }

    fn set_transform(&self, transform: &mut Mat4, commands: &[Command]) {
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
                TransformValue::Reference(r) => TransformDefinition::get_transform(r, commands),
            };
        }
        *transform = out_transform;
    }

    fn get_transform(transform_name: &str, commands: &[Command]) -> Mat4 {
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
                            DefineValue::Transform(transform_def) => {
                                let mut tranform_val = Mat4::IDENTITY;
                                transform_def.set_transform(&mut tranform_val, commands);
                                out_transform *= tranform_val;
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
        out_transform
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
            if let Command::Define(define_command) = command {
                if define_command.define == *material_name {
                    if define_command.extend.is_some() {
                        MaterialValue::_find_definition_and_set_material(
                            define_command.extend.as_ref().unwrap(),
                            material,
                            commands,
                        );
                    }
                    if let DefineValue::Material(material_def) = &define_command.value {
                        material_def.set_material(material, commands);
                    }
                }
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
        // }
    }
}

// TODO
#[derive(Debug, Deserialize)]
struct PatternValue {}

// TODO: why are transforms in definition and add not combining properly
impl Scene {
    pub fn new(file_path: &str, renderer_type: &RendererType) -> Self {
        let path = Path::new(file_path);
        let display = path.display();

        // Open the path in read-only mode, returns `io::Result<File>`
        let f = match File::open(&path) {
            Err(why) => panic!("couldn't open {}: {}", display, why),
            Ok(file) => file,
        };
        // let scene: Vec<serde_yaml::Value> = serde_yaml::from_reader(f).unwrap();

        let commands: Vec<Command> =
            serde_yml::from_reader(f).expect("Failed to load scene description.");
        let (camera, object_params, bvh, specular_offset, lights_offset) =
            Scene::load_assets(&commands, renderer_type);

        // for item in object_params.as_ref().unwrap() {
        //     println!("{:?}", item.model_type);
        // }

        log::debug!(
            "op: {:?}",
            object_params
                .as_ref()
                .unwrap()
                .iter()
                .map(|x| x.material.colour)
                .collect::<Vec<Vec4>>()
        );
        Scene {
            bvh,
            object_params,
            specular_offset,
            lights_offset,
            camera,
            _textures: None,
        }
    }

    #[allow(dead_code)]
    fn validate(&self) {
        todo!();
    }

    #[allow(dead_code)]
    fn load_textures(_commands: &[Command]) -> Option<Vec<Texture>> {
        todo!()
    }

    fn load_assets(
        commands: &Vec<Command>,
        renderer_type: &RendererType,
    ) -> (
        Option<CameraValue>,
        Option<Vec<ObjectParam>>,
        Option<Bvh>,
        usize,
        usize,
    ) {
        let mut lambertian_params: LinkedHashMap<(String, String), ObjectParam> =
            LinkedHashMap::new();
        let mut specular_params: LinkedHashMap<(String, String), ObjectParam> =
            LinkedHashMap::new();
        let mut light_params: LinkedHashMap<(String, String), ObjectParam> = LinkedHashMap::new();
        let mut model_paths: Vec<String> = vec![];
        let mut camera: Option<CameraValue> = None;

        for command in commands {
            match command {
                Command::Add(add_command) => match add_command {
                    Add::Camera(add_camera) => {
                        camera = Some(add_camera.clone());
                    }
                    Add::Shape(add_shape) => {
                        let no_emissive_shapes = match renderer_type {
                            RendererType::RayTracer => true,
                            RendererType::PathTracer => false,
                        };

                        Scene::_get_object_params(
                            add_shape,
                            &mut lambertian_params,
                            &mut specular_params,
                            &mut light_params,
                            &mut model_paths,
                            commands,
                            no_emissive_shapes,
                        );
                    }
                    Add::Light(add_light) => match renderer_type {
                        RendererType::RayTracer => {
                            let light_shape = (*add_light).into();

                            Scene::_get_object_params(
                                &light_shape,
                                &mut lambertian_params,
                                &mut specular_params,
                                &mut light_params,
                                &mut model_paths,
                                commands,
                                false,
                            );
                        }
                        RendererType::PathTracer => {}
                    },
                },
                _ => continue,
            };
        }

        let specular_offset = lambertian_params.len();
        lambertian_params.extend(specular_params);
        let lights_offset = lambertian_params.len();
        lambertian_params.extend(light_params);

        // for item in &object_params {
        //     println!("{:?}", item.1.model_type);
        // }

        model_paths = model_paths.into_iter().unique().collect();
        // println!("{:#?}", model_paths);

        let bvh = Some(Bvh::new(&model_paths));

        for (i, (obparam_key, obparam_value)) in lambertian_params
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

        // TODO: change so n_objects is calculated here
        (
            camera,
            Some(lambertian_params.into_iter().map(|(_, v)| v).collect()),
            bvh,
            specular_offset,
            lights_offset,
        )
    }

    fn _set_primitive_type(type_name: &str, object_param: &mut ObjectParam) {
        match type_name {
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
            "light" => {
                object_param.model_type = 9;
            }
            _ => {}
        }
    }

    fn _get_object_params(
        curr_shape: &ShapeValue,
        accum_lambertian_params: &mut LinkedHashMap<(String, String), ObjectParam>,
        accum_specular_params: &mut LinkedHashMap<(String, String), ObjectParam>,
        accum_light_params: &mut LinkedHashMap<(String, String), ObjectParam>,
        accum_model_paths: &mut Vec<String>,
        commands: &Vec<Command>,
        no_emissive_shapes: bool,
    ) {
        let mut object_param: ObjectParam = ObjectParam::default();
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
            if let Command::Define(define_command) = command {
                // println!("{:?}", define_command);
                if define_command.define == curr_shape.add {
                    if let DefineValue::Shape(defined_shape) = &define_command.value {
                        if is_model {
                            model_path_found = false;
                            if defined_shape.file.is_some() {
                                object_map_key = defined_shape.file.as_ref().unwrap().clone();
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
                                .set_transform(&mut object_param.transform, commands);
                            object_param.inverse_transform = object_param.transform.inverse();
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
                }
            }
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
                .set_transform(&mut object_param.transform, commands);
            object_param.inverse_transform = object_param.transform.inverse();
            //todo
        }
        if curr_shape.material.is_some() {
            curr_shape
                .material
                .as_ref()
                .unwrap()
                .set_material(&mut object_param.material, commands);
            // println!("{:#?}", object_param.material);
        }

        if object_param.material.emissiveness == Vec4::from_array([0.0; 4]) {
            if object_param.material.reflective > 0f32 || object_param.material.transparency > 0f32
            {
                accum_specular_params.insert((object_map_key, hash), object_param);
            } else {
                accum_lambertian_params.insert((object_map_key, hash), object_param);
            }
        } else if !no_emissive_shapes || object_param.model_type == 9 {
            accum_light_params.insert((object_map_key, hash), object_param);
        }
        // // Add this if you want emissive objects to appear in whitted renderer type
        // else {
        //     object_param.material.emissiveness = const_vec4!([0.0; 4]);
        //     accum_object_params.insert((object_map_key, hash), object_param);
        // }

        if curr_shape.children.is_some() {
            for child in curr_shape.children.as_ref().unwrap() {
                Scene::_get_object_params(
                    child,
                    accum_lambertian_params,
                    accum_specular_params,
                    accum_light_params,
                    accum_model_paths,
                    commands,
                    no_emissive_shapes,
                );
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
        let scene = Scene::new(
            "./assets/scenes/test.yaml",
            &crate::RendererType::PathTracer,
        );

        // let x = scene[0].is_sequence()
        println!("{:#?}", scene);

        assert_eq!(2 + 2, 4);
    }
}
