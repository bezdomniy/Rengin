// #version 450

// layout(location = 0) in vec3 fragColor;
// layout(location = 0) out vec4 outColor;

// void main() {
//     outColor = vec4(fragColor, 1.0);
// }


// #version 450

// layout (binding = 0) uniform sampler2D samplerColor;

// layout (location = 0) in vec2 inUV;

// layout (location = 0) out vec4 outFragColor;

// void main() 
// {
//   outFragColor = texture(samplerColor, vec2(inUV.s, 1.0 - inUV.t));
// //   outFragColor = vec4(1.0,0.0,0.0,1.0);
// }

#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform texture2D u_Textures;
layout(set = 0, binding = 1) uniform sampler u_Sampler;

void main() {
    outFragColor = vec4(texture(sampler2D(u_Textures, u_Sampler), inUV).rgb, 1.0);
}
