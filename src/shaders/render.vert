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
    mat4 proj;
    mat4 view;
};

layout (location = 0) out vec4 color;

void main() {
    color = p_color;

    // Spherical billboard. t is a transform with just the particle
    // position. r is the inverse of the view rotation.
    vec4 p = vec4(p_position.xyz, 1.0);
    mat4 t = mat4(vec4(1.0, 0.0, 0.0, 0.0),
                  vec4(0.0, 1.0, 0.0, 0.0),
                  vec4(0.0, 0.0, 1.0, 0.0),
                  p);

    mat4 r = inverse(mat4(vec4(view[0].xyz, 0.0),
                          vec4(view[1].xyz, 0.0),
                          vec4(view[2].xyz, 0.0),
                          vec4(0.0, 0.0, 0.0, 1.0)));

    gl_Position = proj * view * t * r * vec4(position.xyz, 1.0);
}
