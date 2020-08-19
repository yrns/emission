#version 450
//#extension GL_ARB_separate_shader_objects : enable

// layout (set = 0, binding = 0) buffer Particles {
//     Particle particles[];
// };

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

layout (location = 2) in vec4 p_color;
layout (location = 3) in vec3 pos;
// layout (location = 5) in vec3 velocity;
// layout (location = 7) in float lifetime;
// layout (location = 6) in uint emitter;
// layout (location = 7) in uint gen;

layout (location = 0) out vec4 color;

void main() {
    //color = vec4(1.0, 0.0, 0.0, 1.0);
    //color = vec4(p_color.rgb, 1.0);
    color = p_color;
    gl_Position = vec4(position + pos, 1.0);
}
