struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let x: f32 = f32(i32(vertex_index & 1u) << 2u) - 1.0;
    let y: f32 = f32(i32(vertex_index & 2u) << 1u) - 1.0;
    var output: VertexOutput;
    output.position = vec4<f32>(-x, y, 0.0, 1.0);
    output.tex_coords = vec2<f32>(x + 1.0, y + 1.0) * 0.5;
    return output;
}