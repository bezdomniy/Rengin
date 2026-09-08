use super::ShaderCompiler;

#[test]
fn production_pathtracer_shader_composes() {
    let mut compiler = ShaderCompiler::new();
    compiler.compile(
        include_str!("../../src/shaders/pathtracer.wgsl"),
        "shaders/pathtracer.wgsl",
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
