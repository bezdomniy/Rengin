// #version 450

// // layout(location = 0) in vec3 fragColor;
// layout(location = 0) out vec4 outColor;

// void main() {
//     outColor = vec4(1.0,0.0,0.0,0.0);
// }

#version 450

layout (location = 0) out vec2 outUV;

out gl_PerVertex 
{
	vec4 gl_Position;
};

void main() 
{
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
}