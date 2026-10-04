//! Scene loading for the YAML format from the Ray Tracer Challenge book.
//!
//! Loading happens in two passes. Each top-level entry is first parsed
//! strictly into a typed command or definition, so unknown keys are reported
//! instead of silently ignored. The `add` commands are then resolved against
//! the named definitions into the object parameters uploaded to the GPU.

use super::{
    bvh::Bvh,
    obj_loader::ModelError,
    rt_primitives::{Material, ObjectParam},
};
use crate::RendererType;
use glam::{Mat4, Vec3, Vec4};
use serde::{Deserialize, Deserializer, de::Error as _};
use serde_yaml_ng::Value;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};

const MODEL_TYPE_SPHERE: u32 = 0;
const MODEL_TYPE_PLANE: u32 = 1;
const MODEL_TYPE_CUBE: u32 = 2;
const MODEL_TYPE_POINT_LIGHT: u32 = 9;
/// Meshes are numbered from here, one per mesh object.
const MODEL_TYPE_MESH: u32 = 10;

static BUILTIN_SHAPES: [&str; 5] = ["sphere", "plane", "cube", "group", "obj"];

#[derive(Debug)]
pub enum SceneError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Yaml(serde_yaml_ng::Error),
    /// A problem with one top-level entry. `index` is 1-based.
    Entry {
        index: usize,
        label: String,
        message: String,
    },
    Model(ModelError),
    Invalid(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SceneError::Io { path, source } => {
                write!(f, "couldn't read {}: {source}", path.display())
            }
            SceneError::Yaml(error) => write!(f, "invalid scene YAML: {error}"),
            SceneError::Entry {
                index,
                label,
                message,
            } => write!(f, "entry {index} ({label}): {message}"),
            SceneError::Model(error) => write!(f, "{error}"),
            SceneError::Invalid(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SceneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SceneError::Io { source, .. } => Some(source),
            SceneError::Yaml(error) => Some(error),
            SceneError::Model(error) => Some(error),
            SceneError::Entry { .. } | SceneError::Invalid(_) => None,
        }
    }
}

#[derive(Debug)]
pub struct Scene {
    pub bvh: Bvh,
    pub camera: CameraValue,
    pub object_params: Vec<ObjectParam>,
    pub specular_offset: usize,
    pub lights_offset: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct CameraValue {
    pub width: u32,
    pub height: u32,
    pub field_of_view: f32,
    pub from: [f32; 3],
    pub to: [f32; 3],
    pub up: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LightValue {
    at: [f32; 3],
    intensity: [f32; 3],
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShapeValue {
    add: String,
    file: Option<String>,
    material: Option<MaterialValue>,
    transform: Option<Vec<TransformOp>>,
    children: Option<Vec<ShapeValue>>,
    // Accepted so book scenes load, but not supported by this renderer.
    args: Option<Value>,
    shadow: Option<bool>,
}

#[derive(Debug)]
enum MaterialValue {
    Named(String),
    Inline(MaterialDefinition),
}

impl<'de> Deserialize<'de> for MaterialValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match Value::deserialize(deserializer)? {
            Value::String(name) => Ok(MaterialValue::Named(name)),
            value => serde_yaml_ng::from_value(value)
                .map(MaterialValue::Inline)
                .map_err(D::Error::custom),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
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
    // Accepted so book scenes load, but not rendered yet.
    pattern: Option<Value>,
}

#[derive(Debug)]
enum TransformOp {
    Translate(Vec3),
    Scale(Vec3),
    RotateX(f32),
    RotateY(f32),
    RotateZ(f32),
    Named(String),
}

impl<'de> Deserialize<'de> for TransformOp {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        TransformOp::from_value(Value::deserialize(deserializer)?).map_err(D::Error::custom)
    }
}

impl TransformOp {
    fn from_value(value: Value) -> Result<Self, String> {
        let items = match value {
            Value::String(name) => return Ok(TransformOp::Named(name)),
            Value::Sequence(items) => items,
            _ => {
                return Err(
                    "expected a transform such as [translate, x, y, z] or the name of a defined transform"
                        .to_string(),
                );
            }
        };
        let Some((name, args)) = items.split_first() else {
            return Err("empty transform".to_string());
        };
        let Some(name) = name.as_str() else {
            return Err("a transform must start with its name, e.g. [scale, x, y, z]".to_string());
        };
        let args = args
            .iter()
            .map(|arg| arg.as_f64().map(|arg| arg as f32))
            .collect::<Option<Vec<f32>>>()
            .ok_or_else(|| format!("`{name}` arguments must be numbers"))?;
        let expect_args = |count: usize| {
            if args.len() == count {
                Ok(())
            } else {
                Err(format!(
                    "`{name}` takes {count} value(s), got {}",
                    args.len()
                ))
            }
        };

        match name {
            "translate" => expect_args(3).map(|_| TransformOp::Translate(Vec3::from_slice(&args))),
            "scale" => expect_args(3).map(|_| TransformOp::Scale(Vec3::from_slice(&args))),
            "rotate-x" => expect_args(1).map(|_| TransformOp::RotateX(args[0])),
            "rotate-y" => expect_args(1).map(|_| TransformOp::RotateY(args[0])),
            "rotate-z" => expect_args(1).map(|_| TransformOp::RotateZ(args[0])),
            _ => Err(format!(
                "unknown transform `{name}`, expected one of translate, scale, rotate-x, rotate-y, rotate-z"
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DefineEntry {
    define: String,
    extend: Option<String>,
    value: Value,
}

#[derive(Debug)]
enum Definition {
    Shape(ShapeValue),
    Material(MaterialDefinition),
    Transform(Vec<TransformOp>),
}

impl Definition {
    fn from_value(value: Value) -> Result<Self, String> {
        let definition = match value {
            Value::Sequence(_) => Definition::Transform(
                serde_yaml_ng::from_value(value).map_err(|error| error.to_string())?,
            ),
            Value::Mapping(ref mapping) if mapping.contains_key("add") => Definition::Shape(
                serde_yaml_ng::from_value(value).map_err(|error| error.to_string())?,
            ),
            Value::Mapping(_) => Definition::Material(
                serde_yaml_ng::from_value(value).map_err(|error| error.to_string())?,
            ),
            _ => {
                return Err(
                    "`value` must be a material, a shape or a list of transforms".to_string(),
                );
            }
        };
        Ok(definition)
    }

    fn kind(&self) -> &'static str {
        match self {
            Definition::Shape(_) => "shape",
            Definition::Material(_) => "material",
            Definition::Transform(_) => "transform",
        }
    }
}

#[derive(Debug)]
struct Define {
    extend: Option<String>,
    value: Definition,
}

#[derive(Debug)]
enum Command {
    Camera(CameraValue),
    Light(LightValue),
    Shape(Box<ShapeValue>),
}

struct ParsedScene {
    /// Top-level `add` entries with their 1-based entry index and label.
    commands: Vec<(usize, String, Command)>,
    defines: HashMap<String, Define>,
}

fn parse_scene(document: Value) -> Result<ParsedScene, SceneError> {
    let Value::Sequence(entries) = document else {
        return Err(SceneError::Invalid(
            "a scene must be a YAML list of `add` and `define` entries".to_string(),
        ));
    };

    let mut commands = Vec::new();
    let mut defines = HashMap::new();
    for (position, entry) in entries.into_iter().enumerate() {
        let index = position + 1;
        let Value::Mapping(mut mapping) = entry else {
            return Err(SceneError::Entry {
                index,
                label: "entry".to_string(),
                message: "expected a mapping with an `add` or `define` key".to_string(),
            });
        };
        let add = mapping.get("add").map(|value| value.as_str());
        let define = mapping.get("define").map(|value| value.as_str());
        let label = match (add, define) {
            (Some(Some(name)), _) => format!("add: {name}"),
            (_, Some(Some(name))) => format!("define: {name}"),
            _ => "entry".to_string(),
        };
        let entry_error = |message: String| SceneError::Entry {
            index,
            label: label.clone(),
            message,
        };

        match (add, define) {
            (Some(_), Some(_)) => {
                return Err(entry_error(
                    "an entry can't have both `add` and `define`".to_string(),
                ));
            }
            (Some(None), _) | (_, Some(None)) => {
                return Err(entry_error(
                    "`add` and `define` must be followed by a name".to_string(),
                ));
            }
            (Some(Some(add)), None) => {
                let add = add.to_string();
                let command = match add.as_str() {
                    "camera" | "light" => {
                        mapping.remove("add");
                        let value = Value::Mapping(mapping);
                        if add == "camera" {
                            serde_yaml_ng::from_value(value).map(Command::Camera)
                        } else {
                            serde_yaml_ng::from_value(value).map(Command::Light)
                        }
                    }
                    _ => serde_yaml_ng::from_value(Value::Mapping(mapping))
                        .map(|shape| Command::Shape(Box::new(shape))),
                }
                .map_err(|error| entry_error(error.to_string()))?;
                commands.push((index, label.clone(), command));
            }
            (None, Some(Some(_))) => {
                let entry: DefineEntry = serde_yaml_ng::from_value(Value::Mapping(mapping))
                    .map_err(|error| entry_error(error.to_string()))?;
                let value = Definition::from_value(entry.value).map_err(&entry_error)?;
                if let Definition::Shape(_) = value {
                    if BUILTIN_SHAPES.contains(&entry.define.as_str())
                        || ["camera", "light"].contains(&entry.define.as_str())
                    {
                        return Err(entry_error(format!(
                            "`{}` is a built-in name and can't be redefined",
                            entry.define
                        )));
                    }
                    if entry.extend.is_some() {
                        return Err(entry_error(
                            "`extend` is only supported for materials and transforms".to_string(),
                        ));
                    }
                }
                if defines.contains_key(&entry.define) {
                    return Err(entry_error(format!(
                        "`{}` is already defined",
                        entry.define
                    )));
                }
                defines.insert(
                    entry.define,
                    Define {
                        extend: entry.extend,
                        value,
                    },
                );
            }
            (None, None) => {
                return Err(entry_error("expected an `add` or `define` key".to_string()));
            }
        }
    }

    Ok(ParsedScene { commands, defines })
}

enum ShapeKind {
    Primitive(u32),
    Group,
    Mesh(Option<String>),
}

struct ResolvedShape<'a> {
    kind: ShapeKind,
    /// Object space to parent space.
    transform: Mat4,
    material: Material,
    children: Vec<&'a ShapeValue>,
}

/// An object waiting to be uploaded, with the index of its model in
/// `SceneBuilder::models` if it is a mesh.
type PendingObject = (ObjectParam, Option<usize>);

struct SceneBuilder<'a> {
    defines: &'a HashMap<String, Define>,
    renderer_type: &'a RendererType,
    base_dir: Option<&'a Path>,
    lambertian: Vec<PendingObject>,
    specular: Vec<PendingObject>,
    lights: Vec<PendingObject>,
    models: Vec<String>,
    warnings: BTreeSet<String>,
}

impl<'a> SceneBuilder<'a> {
    fn lookup(&self, name: &str, expected: &str) -> Result<&'a Define, String> {
        let defines: &'a HashMap<String, Define> = self.defines;
        let define = defines
            .get(name)
            .ok_or_else(|| format!("{expected} `{name}` is not defined"))?;
        if define.value.kind() != expected {
            return Err(format!(
                "`{name}` is a {}, not a {expected}",
                define.value.kind()
            ));
        }
        Ok(define)
    }

    fn add_light(&mut self, light: &LightValue) -> Result<(), String> {
        let intensity = Vec3::from_array(light.intensity);
        if intensity == Vec3::ZERO {
            self.warnings.insert(format!(
                "light at {:?} has zero intensity and was skipped",
                light.at
            ));
            return Ok(());
        }
        let material = Material {
            emissiveness: intensity.extend(1.0),
            ..Default::default()
        };
        // The path tracer needs finite geometry that rays can hit, so point
        // lights become unit sphere emitters.
        let model_type = match self.renderer_type {
            RendererType::PathTracer => MODEL_TYPE_SPHERE,
            RendererType::RayTracer => MODEL_TYPE_POINT_LIGHT,
        };
        let transform = Mat4::from_translation(Vec3::from_array(light.at));
        self.push_object(model_type, transform, material, None)
    }

    /// Adds `shape` and its children. `ancestry` holds the shape definitions
    /// being expanded, so a definition that contains itself is reported.
    fn add_shape(
        &mut self,
        shape: &'a ShapeValue,
        parent_transform: Mat4,
        inherited_material: Material,
        ancestry: &mut Vec<&'a str>,
    ) -> Result<(), String> {
        let ancestry_len = ancestry.len();
        let ResolvedShape {
            kind,
            transform,
            material,
            children,
        } = self.resolve_shape(shape, inherited_material, ancestry)?;
        let transform = parent_transform * transform;

        match kind {
            ShapeKind::Group => {}
            ShapeKind::Primitive(model_type) => {
                self.push_object(model_type, transform, material, None)?;
            }
            ShapeKind::Mesh(None) => {
                return Err(format!("mesh `{}` has no `file`", shape.add));
            }
            ShapeKind::Mesh(Some(file)) => {
                let model = self.model_index(&file)?;
                self.push_object(MODEL_TYPE_MESH, transform, material, Some(model))?;
            }
        }

        for child in children {
            self.add_shape(child, transform, material, ancestry)?;
        }
        ancestry.truncate(ancestry_len);
        Ok(())
    }

    /// Expands `shape` through any shape definitions it refers to. Settings
    /// on `shape` itself are applied on top of the definition's: its
    /// transform after the definition's, its material fields over the
    /// definition's, and its `file` instead of the definition's.
    fn resolve_shape(
        &mut self,
        shape: &'a ShapeValue,
        inherited_material: Material,
        ancestry: &mut Vec<&'a str>,
    ) -> Result<ResolvedShape<'a>, String> {
        let builtin = |kind| ResolvedShape {
            kind,
            transform: Mat4::IDENTITY,
            material: inherited_material,
            children: Vec::new(),
        };
        let mut resolved = match shape.add.as_str() {
            "sphere" => builtin(ShapeKind::Primitive(MODEL_TYPE_SPHERE)),
            "plane" => builtin(ShapeKind::Primitive(MODEL_TYPE_PLANE)),
            "cube" => builtin(ShapeKind::Primitive(MODEL_TYPE_CUBE)),
            "group" => builtin(ShapeKind::Group),
            "obj" => builtin(ShapeKind::Mesh(None)),
            "camera" | "light" => {
                return Err(format!(
                    "`add: {}` is only allowed at the top level of a scene",
                    shape.add
                ));
            }
            name => {
                if ancestry.contains(&name) {
                    return Err(format!(
                        "shape definition cycle: {} -> {name}",
                        ancestry.join(" -> ")
                    ));
                }
                let define =
                    self.lookup(name, "shape")
                        .map_err(|_| match self.defines.get(name) {
                            Some(define) => {
                                format!("`{name}` is a {}, not a shape", define.value.kind())
                            }
                            None => format!(
                                "unknown shape `{name}`, expected one of {} or a defined shape",
                                BUILTIN_SHAPES.join(", ")
                            ),
                        })?;
                let Definition::Shape(defined_shape) = &define.value else {
                    unreachable!("lookup checked the definition kind");
                };
                ancestry.push(name);
                self.resolve_shape(defined_shape, inherited_material, ancestry)?
            }
        };

        if let Some(file) = &shape.file {
            match &mut resolved.kind {
                ShapeKind::Mesh(path) => *path = Some(file.clone()),
                _ => return Err("`file` is only valid for `obj` meshes".to_string()),
            }
        }
        if let Some(material) = &shape.material {
            self.apply_material(material, &mut resolved.material)?;
        }
        if let Some(ops) = &shape.transform {
            resolved.transform = self.transform_matrix(ops, &mut Vec::new())? * resolved.transform;
        }
        if shape.args.is_some() {
            self.warnings.insert(format!(
                "`args` on `{}` is not supported and was ignored",
                shape.add
            ));
        }
        if shape.shadow.is_some() {
            self.warnings.insert(format!(
                "`shadow` on `{}` is not supported and was ignored",
                shape.add
            ));
        }
        if let Some(children) = &shape.children {
            resolved.children.extend(children);
        }
        Ok(resolved)
    }

    fn apply_material(
        &mut self,
        value: &MaterialValue,
        material: &mut Material,
    ) -> Result<(), String> {
        match value {
            MaterialValue::Inline(definition) => {
                self.apply_material_definition(definition, material);
                Ok(())
            }
            MaterialValue::Named(name) => {
                self.apply_named_material(name, material, &mut Vec::new())
            }
        }
    }

    /// Applies a material's `extend` chain first, then its own fields.
    fn apply_named_material(
        &mut self,
        name: &str,
        material: &mut Material,
        stack: &mut Vec<String>,
    ) -> Result<(), String> {
        if stack.iter().any(|visited| visited == name) {
            return Err(format!(
                "material `extend` cycle: {} -> {name}",
                stack.join(" -> ")
            ));
        }
        let define = self.lookup(name, "material")?;
        let Definition::Material(definition) = &define.value else {
            unreachable!("lookup checked the definition kind");
        };
        stack.push(name.to_string());
        if let Some(parent) = &define.extend {
            self.apply_named_material(parent, material, stack)?;
        }
        stack.pop();
        self.apply_material_definition(definition, material);
        Ok(())
    }

    fn apply_material_definition(
        &mut self,
        definition: &MaterialDefinition,
        material: &mut Material,
    ) {
        if let Some(color) = definition.color {
            material.colour = Vec3::from_array(color).extend(1.0);
        }
        if let Some(emissiveness) = definition.emissiveness {
            material.emissiveness = Vec3::from_array(emissiveness).extend(1.0);
        }
        if let Some(ambient) = definition.ambient {
            material.ambient = ambient;
        }
        if let Some(diffuse) = definition.diffuse {
            material.diffuse = diffuse;
        }
        if let Some(specular) = definition.specular {
            material.specular = specular;
        }
        if let Some(shininess) = definition.shininess {
            material.shininess = shininess;
        }
        if let Some(reflective) = definition.reflective {
            material.reflective = reflective;
        }
        if let Some(transparency) = definition.transparency {
            material.transparency = transparency;
        }
        if let Some(refractive_index) = definition.refractive_index {
            material.refractive_index = refractive_index;
        }
        if definition.pattern.is_some() {
            self.warnings
                .insert("material patterns are not supported yet and were ignored".to_string());
        }
    }

    /// Combines `ops` in list order: the first transform is applied first.
    fn transform_matrix(
        &self,
        ops: &[TransformOp],
        stack: &mut Vec<String>,
    ) -> Result<Mat4, String> {
        let mut matrix = Mat4::IDENTITY;
        for op in ops {
            let op_matrix = match op {
                TransformOp::Translate(offset) => Mat4::from_translation(*offset),
                TransformOp::Scale(scale) => Mat4::from_scale(*scale),
                TransformOp::RotateX(angle) => Mat4::from_rotation_x(*angle),
                TransformOp::RotateY(angle) => Mat4::from_rotation_y(*angle),
                TransformOp::RotateZ(angle) => Mat4::from_rotation_z(*angle),
                TransformOp::Named(name) => self.named_transform(name, stack)?,
            };
            matrix = op_matrix * matrix;
        }
        Ok(matrix)
    }

    /// A transform that `extend`s another applies the extended transforms
    /// first, then its own, as if its list were appended to the parent's.
    fn named_transform(&self, name: &str, stack: &mut Vec<String>) -> Result<Mat4, String> {
        if stack.iter().any(|visited| visited == name) {
            return Err(format!(
                "transform definition cycle: {} -> {name}",
                stack.join(" -> ")
            ));
        }
        let define = self.lookup(name, "transform")?;
        let Definition::Transform(ops) = &define.value else {
            unreachable!("lookup checked the definition kind");
        };
        stack.push(name.to_string());
        let parent = match &define.extend {
            Some(parent) => self.named_transform(parent, stack)?,
            None => Mat4::IDENTITY,
        };
        let own = self.transform_matrix(ops, stack)?;
        stack.pop();
        Ok(own * parent)
    }

    fn push_object(
        &mut self,
        model_type: u32,
        transform: Mat4,
        material: Material,
        model: Option<usize>,
    ) -> Result<(), String> {
        let inverse_transform = transform.inverse();
        if !transform.is_finite() || !inverse_transform.is_finite() {
            return Err("transform is not invertible (is a scale component zero?)".to_string());
        }
        let object = ObjectParam {
            transform,
            inverse_transform,
            material,
            model_type,
            ..Default::default()
        };

        if material.emissiveness.truncate() != Vec3::ZERO {
            if matches!(self.renderer_type, RendererType::RayTracer)
                && model_type != MODEL_TYPE_POINT_LIGHT
            {
                self.warnings.insert(
                    "emissive shapes are only rendered by the path tracer and were skipped"
                        .to_string(),
                );
                return Ok(());
            }
            self.lights.push((object, model));
        } else if material.reflective > 0.0 || material.transparency > 0.0 {
            self.specular.push((object, model));
        } else {
            self.lambertian.push((object, model));
        }
        Ok(())
    }

    fn model_index(&mut self, file: &str) -> Result<usize, String> {
        let path = self.resolve_model_path(file)?;
        let key = std::fs::canonicalize(&path)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        if let Some(index) = self.models.iter().position(|model| *model == key) {
            return Ok(index);
        }
        self.models.push(key);
        Ok(self.models.len() - 1)
    }

    /// Model paths are relative to the scene file. Paths relative to the
    /// working directory are still accepted, with a warning.
    fn resolve_model_path(&mut self, file: &str) -> Result<PathBuf, String> {
        let path = Path::new(file);
        let Some(base_dir) = self.base_dir.filter(|_| path.is_relative()) else {
            return Ok(path.to_path_buf());
        };
        let scene_relative = base_dir.join(path);
        if scene_relative.exists() {
            return Ok(scene_relative);
        }
        if path.exists() {
            self.warnings.insert(format!(
                "model `{file}` was found relative to the working directory; \
                 model paths should be relative to the scene file"
            ));
            return Ok(path.to_path_buf());
        }
        Err(format!(
            "model file `{file}` not found (looked for {} and in the working directory)",
            scene_relative.display()
        ))
    }

    fn finish(self, camera: CameraValue) -> Result<Scene, SceneError> {
        let bvh = Bvh::new(&self.models).map_err(SceneError::Model)?;
        let specular_offset = self.lambertian.len();
        let lights_offset = specular_offset + self.specular.len();
        if lights_offset + self.lights.len() == 0 {
            return Err(SceneError::Invalid("scene contains no objects".to_string()));
        }
        if self.lights.is_empty() {
            log::warn!("scene contains no lights");
        }

        let mut mesh_count = 0;
        let object_params = self
            .lambertian
            .into_iter()
            .chain(self.specular)
            .chain(self.lights)
            .map(|(mut object, model)| {
                match model {
                    Some(model) => {
                        let (inner_offset, inner_len, leaf_offset) = bvh.model_location(model);
                        object.offset_inner_nodes = inner_offset;
                        object.len_inner_nodes = inner_len;
                        object.offset_leaf_nodes = leaf_offset;
                        object.model_type = MODEL_TYPE_MESH + mesh_count;
                        mesh_count += 1;
                        let root_node = &bvh.inner_nodes[inner_offset as usize];
                        object.compute_world_aabb(root_node.first, root_node.second);
                    }
                    // Planes are infinite; the shader skips their AABB test.
                    None if object.model_type == MODEL_TYPE_PLANE => {}
                    // Spheres, cubes and point lights fill [-1, 1]^3.
                    None => object.compute_world_aabb(Vec3::NEG_ONE, Vec3::ONE),
                }
                object
            })
            .collect::<Vec<_>>();

        for warning in &self.warnings {
            log::warn!("{warning}");
        }
        log::debug!(
            "object colours: {:?}",
            object_params
                .iter()
                .map(|object| object.material.colour)
                .collect::<Vec<Vec4>>()
        );

        Ok(Scene {
            bvh,
            camera,
            object_params,
            specular_offset,
            lights_offset,
        })
    }
}

fn validate_camera(camera: &CameraValue) -> Result<(), String> {
    if camera.width == 0 || camera.height == 0 {
        return Err("`width` and `height` must be greater than zero".to_string());
    }
    if !(camera.field_of_view > 0.0 && camera.field_of_view < std::f32::consts::PI) {
        return Err("`field-of-view` must be in radians, between 0 and pi".to_string());
    }
    if camera.from == camera.to {
        return Err("`from` and `to` must be different points".to_string());
    }
    Ok(())
}

impl Scene {
    /// Loads a scene file. Model paths in it are resolved relative to the
    /// scene file.
    pub fn from_file(file_path: &str, renderer_type: &RendererType) -> Result<Self, SceneError> {
        let path = Path::new(file_path);
        let yaml = std::fs::read_to_string(path).map_err(|source| SceneError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Scene::load(&yaml, path.parent(), renderer_type)
    }

    /// Loads a scene from a YAML string (used for embedded assets on web).
    /// Model paths are used as written.
    pub fn from_yaml_str(yaml: &str, renderer_type: &RendererType) -> Result<Self, SceneError> {
        Scene::load(yaml, None, renderer_type)
    }

    fn load(
        yaml: &str,
        base_dir: Option<&Path>,
        renderer_type: &RendererType,
    ) -> Result<Self, SceneError> {
        let document: Value = serde_yaml_ng::from_str(yaml).map_err(SceneError::Yaml)?;
        let ParsedScene { commands, defines } = parse_scene(document)?;

        let mut builder = SceneBuilder {
            defines: &defines,
            renderer_type,
            base_dir,
            lambertian: Vec::new(),
            specular: Vec::new(),
            lights: Vec::new(),
            models: Vec::new(),
            warnings: BTreeSet::new(),
        };
        let mut camera: Option<(usize, CameraValue)> = None;

        for (index, label, command) in &commands {
            let result = match command {
                Command::Camera(value) => match &camera {
                    Some((first_index, _)) => Err(format!(
                        "the scene already has a camera (entry {first_index})"
                    )),
                    None => validate_camera(value).map(|()| camera = Some((*index, value.clone()))),
                },
                Command::Light(light) => builder.add_light(light),
                Command::Shape(shape) => {
                    builder.add_shape(shape, Mat4::IDENTITY, Material::default(), &mut Vec::new())
                }
            };
            result.map_err(|message| SceneError::Entry {
                index: *index,
                label: label.clone(),
                message,
            })?;
        }

        let Some((_, camera)) = camera else {
            return Err(SceneError::Invalid(
                "scene has no camera (`add: camera`)".to_string(),
            ));
        };
        builder.finish(camera)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAMERA: &str = "
- add: camera
  width: 10
  height: 10
  field-of-view: 1
  from: [0, 0, -5]
  to: [0, 0, 0]
  up: [0, 1, 0]
";

    fn load(body: &str) -> Result<Scene, SceneError> {
        Scene::from_yaml_str(&format!("{CAMERA}{body}"), &RendererType::PathTracer)
    }

    fn load_error(body: &str) -> String {
        load(body).unwrap_err().to_string()
    }

    fn assert_mat_eq(actual: Mat4, expected: Mat4) {
        assert!(
            actual.abs_diff_eq(expected, 1e-5),
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn shipped_scenes_load() {
        for entry in std::fs::read_dir("./assets/scenes").unwrap() {
            let path = entry.unwrap().path();
            if let Err(error) = Scene::from_file(path.to_str().unwrap(), &RendererType::PathTracer)
            {
                panic!("{}: {error}", path.display());
            }
        }
    }

    #[test]
    fn embedded_scenes_load() {
        for name in ["cornellbox", "cornellbox2", "fresnel", "glass_test"] {
            super::super::embedded_assets::load_scene(name, &RendererType::PathTracer).unwrap();
        }
    }

    #[test]
    fn pathtracer_keeps_legacy_point_lights_as_finite_emitters() {
        let scene =
            Scene::from_file("./assets/scenes/models.yaml", &RendererType::PathTracer).unwrap();
        let lights = &scene.object_params[scene.lights_offset..];

        assert!(lights.iter().any(|light| {
            light.model_type == MODEL_TYPE_SPHERE
                && light
                    .transform
                    .w_axis
                    .truncate()
                    .distance(Vec3::new(-4.0, 2.0, 3.0))
                    < 1e-5
                && light.material.emissiveness.truncate() == Vec3::ONE
        }));
    }

    #[test]
    fn transforms_apply_in_list_order() {
        let scene = load(
            "- add: sphere\n  transform:\n    - [scale, 2, 2, 2]\n    - [translate, 1, 0, 0]\n",
        )
        .unwrap();

        assert_mat_eq(
            scene.object_params[0].transform,
            Mat4::from_translation(Vec3::X) * Mat4::from_scale(Vec3::splat(2.0)),
        );
    }

    #[test]
    fn instance_transform_composes_with_definition_transform() {
        let scene = load(
            "- define: big-ball\n  value:\n    add: sphere\n    transform:\n      - [scale, 2, 2, 2]\n- add: big-ball\n  transform:\n    - [translate, 5, 0, 0]\n",
        )
        .unwrap();
        let object = &scene.object_params[0];

        assert_eq!(object.model_type, MODEL_TYPE_SPHERE);
        assert_mat_eq(
            object.transform,
            Mat4::from_translation(Vec3::new(5.0, 0.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.0)),
        );
    }

    #[test]
    fn extended_transform_applies_parent_first() {
        let scene = load(
            "- define: base\n  value:\n    - [translate, 1, 0, 0]\n- define: bigger\n  extend: base\n  value:\n    - [scale, 2, 2, 2]\n- add: sphere\n  transform:\n    - bigger\n",
        )
        .unwrap();

        assert_mat_eq(
            scene.object_params[0].transform,
            Mat4::from_scale(Vec3::splat(2.0)) * Mat4::from_translation(Vec3::X),
        );
    }

    #[test]
    fn material_extend_and_inline_fields_merge() {
        let scene = load(
            "- define: base\n  value:\n    color: [1, 0, 0]\n    diffuse: 0.5\n- define: shiny\n  extend: base\n  value:\n    shininess: 10\n- add: sphere\n  material: shiny\n",
        )
        .unwrap();
        let material = scene.object_params[0].material;

        assert_eq!(material.colour, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(material.diffuse, 0.5);
        assert_eq!(material.shininess, 10.0);
    }

    #[test]
    fn group_children_inherit_transform_and_material() {
        let scene = load(
            "- add: group\n  material:\n    color: [0, 1, 0]\n  transform:\n    - [translate, 10, 0, 0]\n  children:\n    - add: sphere\n    - add: cube\n      transform:\n        - [translate, 0, 1, 0]\n",
        )
        .unwrap();
        let objects = &scene.object_params;

        assert_eq!(objects.len(), 2, "the group itself is not an object");
        assert_eq!(objects[0].transform.w_axis, Vec4::new(10.0, 0.0, 0.0, 1.0));
        assert_eq!(objects[1].transform.w_axis, Vec4::new(10.0, 1.0, 0.0, 1.0));
        assert!(
            objects
                .iter()
                .all(|object| object.material.colour == Vec4::new(0.0, 1.0, 0.0, 1.0))
        );
    }

    #[test]
    fn explicit_zero_emissiveness_is_not_a_light() {
        let scene = load("- add: sphere\n  material:\n    emissiveness: [0, 0, 0]\n").unwrap();

        assert_eq!(scene.lights_offset, 1);
    }

    #[test]
    fn mesh_file_on_instance_overrides_definition() {
        let dir = std::env::temp_dir().join(format!("rengin-scene-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let triangle = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        std::fs::write(dir.join("one.obj"), triangle).unwrap();
        std::fs::write(dir.join("two.obj"), format!("{triangle}f 1 3 2\n")).unwrap();
        std::fs::write(
            dir.join("scene.yaml"),
            format!(
                "{CAMERA}- define: model\n  value:\n    add: obj\n    file: one.obj\n- add: model\n  file: two.obj\n- add: obj\n  file: one.obj\n- add: obj\n  file: ./one.obj\n"
            ),
        )
        .unwrap();

        let scene = Scene::from_file(
            dir.join("scene.yaml").to_str().unwrap(),
            &RendererType::PathTracer,
        )
        .unwrap();

        // two.obj, then one.obj shared by both instances that name it.
        assert_eq!(scene.bvh.model_tags.len(), 2);
        assert_eq!(scene.bvh.leaf_nodes.len(), 3);
        let meshes = &scene.object_params;
        assert_eq!(
            meshes
                .iter()
                .map(|mesh| mesh.model_type)
                .collect::<Vec<_>>(),
            [10, 11, 12]
        );
        assert_eq!(meshes[0].offset_leaf_nodes, 0);
        assert_eq!(meshes[1].offset_leaf_nodes, 2);
        assert_eq!(meshes[2].offset_leaf_nodes, 2);
    }

    #[test]
    fn missing_model_file_is_reported() {
        let error = load("- add: obj\n  file: does-not-exist.obj\n").unwrap_err();

        assert!(matches!(error, SceneError::Model(_)), "{error}");
        assert!(error.to_string().contains("does-not-exist.obj"));
    }

    #[test]
    fn typos_are_reported() {
        let cases = [
            ("- add: shpere\n", "unknown shape `shpere`"),
            (
                "- add: sphere\n  material:\n    colour: [1, 0, 0]\n",
                "unknown field `colour`",
            ),
            (
                "- add: sphere\n  materail: red\n",
                "unknown field `materail`",
            ),
            (
                "- add: sphere\n  transform:\n    - [rotate_y, 1]\n",
                "unknown transform `rotate_y`",
            ),
            (
                "- add: sphere\n  transform:\n    - [translate, 1, 2]\n",
                "`translate` takes 3 value(s), got 2",
            ),
            (
                "- add: sphere\n  material: missing\n",
                "material `missing` is not defined",
            ),
            (
                "- add: sphere\n  transform:\n    - missing\n",
                "transform `missing` is not defined",
            ),
            (
                "- add: light\n  at: [0, 0, 0]\n  intensty: [1, 1, 1]\n",
                "unknown field `intensty`",
            ),
        ];
        for (body, expected) in cases {
            let error = load_error(body);
            assert!(
                error.contains(expected),
                "expected `{expected}` in `{error}`"
            );
        }
    }

    #[test]
    fn camera_typo_is_reported_instead_of_becoming_a_shape() {
        let error = Scene::from_yaml_str(
            "- add: camera\n  width: 10\n  height: 10\n  field_of_view: 1\n  from: [0, 0, -5]\n  to: [0, 0, 0]\n  up: [0, 1, 0]\n",
            &RendererType::PathTracer,
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.starts_with("entry 1 (add: camera): unknown field `field_of_view`"),
            "{error}"
        );
    }

    #[test]
    fn structural_errors_are_reported() {
        let cases = [
            (
                "- define: a\n  extend: a\n  value:\n    color: [1, 0, 0]\n- add: sphere\n  material: a\n",
                "material `extend` cycle: a -> a",
            ),
            (
                "- define: a\n  extend: b\n  value: [[scale, 1, 1, 1]]\n- define: b\n  extend: a\n  value: [[scale, 1, 1, 1]]\n- add: sphere\n  transform: [a]\n",
                "transform definition cycle: a -> b -> a",
            ),
            (
                "- define: loop\n  value:\n    add: group\n    children:\n      - add: loop\n- add: loop\n",
                "shape definition cycle: loop -> loop",
            ),
            (
                "- define: red\n  value:\n    color: [1, 0, 0]\n- add: red\n",
                "`red` is a material, not a shape",
            ),
            (
                "- define: sphere\n  value:\n    add: cube\n",
                "`sphere` is a built-in name",
            ),
            (
                "- define: x\n  value: [[scale, 1, 1, 1]]\n- define: x\n  value: [[scale, 2, 2, 2]]\n",
                "`x` is already defined",
            ),
            (
                "- add: sphere\n  file: model.obj\n",
                "`file` is only valid for `obj` meshes",
            ),
            ("- add: obj\n", "mesh `obj` has no `file`"),
            (
                "- add: sphere\n  transform:\n    - [scale, 0, 1, 1]\n",
                "transform is not invertible",
            ),
            (
                "- add: group\n  children:\n    - add: camera\n",
                "only allowed at the top level",
            ),
            (
                "- add: camera\n  width: 1\n  height: 1\n  field-of-view: 1\n  from: [0, 0, -5]\n  to: [0, 0, 0]\n  up: [0, 1, 0]\n",
                "already has a camera (entry 1)",
            ),
        ];
        for (body, expected) in cases {
            let error = load_error(body);
            assert!(
                error.contains(expected),
                "expected `{expected}` in `{error}`"
            );
        }
    }

    #[test]
    fn scene_without_camera_or_objects_is_rejected() {
        let no_camera = Scene::from_yaml_str("- add: sphere\n", &RendererType::PathTracer);
        assert!(no_camera.unwrap_err().to_string().contains("no camera"));
        assert!(load("").unwrap_err().to_string().contains("no objects"));
    }

    #[test]
    fn unsupported_book_fields_are_accepted() {
        load(
            "- add: sphere\n  args: [1, 2, 3]\n  shadow: false\n  material:\n    pattern:\n      type: stripes\n      colors: [[1, 1, 1], [0, 0, 0]]\n",
        )
        .unwrap();
    }
}
