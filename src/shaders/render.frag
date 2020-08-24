#version 450

#extension GL_ARB_separate_shader_objects : enable

//layout(early_fragment_tests) in;

layout(location = 0) in vec4 p_color;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 color;

layout (set = 0, binding = 0) uniform ProjView {
    mat4 proj;
    mat4 view;
    vec4 camera;
};

layout(set = 1, binding = 0) uniform texture2D color_map;
layout(set = 1, binding = 1) uniform sampler color_sampler;

void main() {
    proj;
    color = texture(sampler2D(color_map, color_sampler), uv) * p_color;
}
