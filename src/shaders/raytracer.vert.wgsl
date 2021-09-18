[[block]]
struct gl_PerVertex {
    [[builtin(position)]] gl_Position: vec4<f32>;
};

struct VertexOutput {
    [[location(0)]] member: vec2<f32>;
    [[builtin(position)]] gl_Position: vec4<f32>;
};

var<private> outUV: vec2<f32>;
var<private> gl_VertexIndex1: i32;
var<private> perVertexStruct: gl_PerVertex = gl_PerVertex(vec4<f32>(0.0, 0.0, 0.0, 1.0));

fn main1() {
    let _e14: i32 = gl_VertexIndex1;
    let _e19: i32 = gl_VertexIndex1;
    outUV = vec2<f32>(f32(((_e14 << u32(1)) & 2)), f32((_e19 & 2)));
    let _e23: vec2<f32> = outUV;
    let _e26: vec2<f32> = ((_e23 * 2.0) + vec2<f32>(-1.0));
    perVertexStruct.gl_Position = vec4<f32>(_e26.x, _e26.y, 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] gl_VertexIndex: u32) -> VertexOutput {
    gl_VertexIndex1 = i32(gl_VertexIndex);
    main1();
    let _e6: vec2<f32> = outUV;
    let _e7: vec4<f32> = perVertexStruct.gl_Position;
    return VertexOutput(_e6, _e7);
}

// [[stage(vertex)]]
// fn vs_main([[builtin(vertex_index)]] in_vertex_index: u32) -> [[builtin(position)]] vec4<f32> {
//     let x = f32(i32(in_vertex_index) - 1);
//     let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
//     return vec4<f32>(x, y, 0.0, 1.0);
// }
