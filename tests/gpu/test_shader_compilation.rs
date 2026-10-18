use super::ShaderCompiler;
use naga_oil::compose::ShaderDefValue;
use std::collections::HashMap;

#[test]
fn production_pathtracer_shader_composes() {
    let mut compiler = ShaderCompiler::new();
    compiler.compile(
        include_str!("../../src/shaders/pathtracer.wgsl"),
        "shaders/pathtracer.wgsl",
    );
}

#[test]
fn production_profiled_pathtracer_shader_composes() {
    let mut compiler = ShaderCompiler::new();
    let mut definitions = HashMap::new();
    definitions.insert("PROFILE_STATS".to_owned(), ShaderDefValue::Bool(true));
    compiler.compile_with_defs(
        include_str!("../../src/shaders/pathtracer.wgsl"),
        "shaders/pathtracer.wgsl",
        definitions,
    );
}

#[test]
fn production_raygen_shader_composes() {
    let mut compiler = ShaderCompiler::new();
    compiler.compile(
        include_str!("../../src/shaders/raygen.wgsl"),
        "shaders/raygen.wgsl",
    );
}
