//! Wavefront OBJ loading for mesh primitives.

use std::fmt;
use std::path::{Path, PathBuf};

use glam::{Mat3, Vec3};

use super::bvh::TrianglePrimitive;

#[derive(Debug)]
pub enum ModelError {
    Load {
        path: PathBuf,
        source: tobj::LoadError,
    },
    Empty {
        path: PathBuf,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Load { path, source } => {
                write!(f, "failed to load model {}: {source}", path.display())
            }
            ModelError::Empty { path } => {
                write!(f, "model {} contains no usable triangles", path.display())
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Load { source, .. } => Some(source),
            ModelError::Empty { .. } => None,
        }
    }
}

fn load_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    }
}

/// Loads every triangle of an OBJ file. Polygons are triangulated, vertex
/// normals that are missing or invalid fall back to the face normal, and
/// zero-area triangles are skipped.
pub fn load_obj_triangles(path: &Path) -> Result<Vec<TrianglePrimitive>, ModelError> {
    log::info!("Loading {}", path.display());
    triangles_from_obj(tobj::load_obj(path, &load_options()), path)
}

/// Like [`load_obj_triangles`], for OBJ data already in memory. `path` is
/// only used in messages. Material libraries are not loaded.
pub fn parse_obj_triangles(
    mut bytes: &[u8],
    path: &Path,
) -> Result<Vec<TrianglePrimitive>, ModelError> {
    log::info!("Parsing {}", path.display());
    let result = tobj::load_obj_buf(&mut bytes, &load_options(), |_| {
        Err(tobj::LoadError::OpenFileFailed)
    });
    triangles_from_obj(result, path)
}

fn triangles_from_obj(
    result: tobj::LoadResult,
    path: &Path,
) -> Result<Vec<TrianglePrimitive>, ModelError> {
    let (models, materials) = result.map_err(|source| ModelError::Load {
        path: path.to_path_buf(),
        source,
    })?;

    if matches!(&materials, Ok(materials) if !materials.is_empty()) {
        log::warn!(
            "{}: OBJ materials are not supported; the scene material is used instead",
            path.display()
        );
    }

    let mut triangles = Vec::new();
    let mut degenerate_count = 0usize;
    let mut fallback_normal_count = 0usize;

    for model in &models {
        let mesh = &model.mesh;
        // With `single_index`, normals are either absent or one per position.
        // Anything else means only some faces had normals, so ignore them.
        let has_vertex_normals = mesh.normals.len() == mesh.positions.len();
        if !has_vertex_normals && !mesh.normals.is_empty() {
            log::warn!(
                "{}: only some faces of `{}` have normals; using face normals instead",
                path.display(),
                model.name
            );
        }

        let (faces, _) = mesh.indices.as_chunks::<3>();
        for &indices in faces {
            let points = indices.map(|index| vec3_at(&mesh.positions, index));
            let Some(face_normal) = (points[1] - points[0])
                .cross(points[2] - points[0])
                .try_normalize()
            else {
                degenerate_count += 1;
                continue;
            };

            let normals = indices.map(|index| {
                if has_vertex_normals
                    && let Some(normal) = vec3_at(&mesh.normals, index).try_normalize()
                {
                    return normal;
                }
                fallback_normal_count += 1;
                face_normal
            });

            triangles.push(TrianglePrimitive {
                points: Mat3::from_cols(points[0], points[1], points[2]),
                normals: Mat3::from_cols(normals[0], normals[1], normals[2]),
            });
        }
    }

    if degenerate_count > 0 {
        log::warn!(
            "{}: skipped {degenerate_count} zero-area triangles",
            path.display()
        );
    }
    if fallback_normal_count > 0 {
        log::info!(
            "{}: used face normals for {fallback_normal_count} vertices without a valid normal",
            path.display()
        );
    }
    if triangles.is_empty() {
        return Err(ModelError::Empty {
            path: path.to_path_buf(),
        });
    }

    Ok(triangles)
}

fn vec3_at(values: &[f32], index: u32) -> Vec3 {
    let start = 3 * index as usize;
    Vec3::from_slice(&values[start..start + 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_obj(name: &str, contents: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rengin-obj-tests-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn missing_normals_fall_back_to_face_normals() {
        let path = write_obj("no_normals.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        let triangles = load_obj_triangles(&path).unwrap();

        assert_eq!(triangles.len(), 1);
        assert_eq!(
            triangles[0].normals,
            Mat3::from_cols(Vec3::Z, Vec3::Z, Vec3::Z)
        );
    }

    #[test]
    fn partial_normals_are_ignored() {
        let path = write_obj(
            "partial_normals.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\nf 2 4 3\n",
        );
        let triangles = load_obj_triangles(&path).unwrap();

        assert_eq!(triangles.len(), 2);
        assert!(
            triangles
                .iter()
                .all(|triangle| triangle.normals == Mat3::from_cols(Vec3::Z, Vec3::Z, Vec3::Z))
        );
    }

    #[test]
    fn vertex_normals_are_normalized_and_zero_normals_replaced() {
        let path = write_obj(
            "bad_normals.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 5\nvn 0 0 0\nf 1//1 2//1 3//2\n",
        );
        let triangles = load_obj_triangles(&path).unwrap();

        assert_eq!(
            triangles[0].normals,
            Mat3::from_cols(Vec3::Z, Vec3::Z, Vec3::Z)
        );
    }

    #[test]
    fn degenerate_triangles_are_skipped() {
        let path = write_obj(
            "degenerate.obj",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 2 0 0\nf 1 2 3\nf 1 2 4\n",
        );

        assert_eq!(load_obj_triangles(&path).unwrap().len(), 1);
    }

    #[test]
    fn empty_model_is_an_error() {
        let path = write_obj("empty.obj", "# nothing here\n");

        assert!(matches!(
            load_obj_triangles(&path),
            Err(ModelError::Empty { .. })
        ));
    }

    #[test]
    fn missing_file_error_names_the_path() {
        let path = std::env::temp_dir().join("rengin-definitely-missing.obj");
        let error = load_obj_triangles(&path).unwrap_err();

        assert!(error.to_string().contains("rengin-definitely-missing.obj"));
    }
}
