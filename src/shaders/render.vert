#version 450
//#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

layout (location = 2) in vec4 p_color;
layout (location = 3) in vec4 p_position;
// layout (location = 5) in vec3 velocity;
// layout (location = 7) in float lifetime;
// layout (location = 6) in uint emitter;
// layout (location = 7) in uint gen;

layout (set = 0, binding = 0) uniform ProjView {
    mat4 proj_view;
};

layout (location = 0) out vec4 color;

void main() {
    mat4 identity = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );

    color = p_color;
    gl_Position = proj_view * vec4(position + p_position.xyz, 1.0);
}
