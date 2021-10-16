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
    let e14: i32 = gl_VertexIndex1;
    let e19: i32 = gl_VertexIndex1;
    outUV = vec2<f32>(f32(((e14 << bitcast<u32>(1)) & 2)), f32((e19 & 2)));
    let e23: vec2<f32> = outUV;
    let e26: vec2<f32> = ((e23 * 2.0) + vec2<f32>(-1.0));
    perVertexStruct.gl_Position = vec4<f32>(e26.x, e26.y, 0.0, 1.0);
    return;
}

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] gl_VertexIndex: u32) -> VertexOutput {
    gl_VertexIndex1 = i32(gl_VertexIndex);
    main1();
    let e6: vec2<f32> = outUV;
    let e7: vec4<f32> = perVertexStruct.gl_Position;
    return VertexOutput(e6, e7);
}
