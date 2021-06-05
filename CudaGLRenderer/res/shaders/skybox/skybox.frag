#version 450 core

uniform samplerCube u_SkyboxTexture;
uniform bool u_EnableHDR;

layout(location = 0) in vec3 in_FragmentPosition;

layout(location = 0) out vec4 out_FinalColor;


void main() {

    vec4 color = texture(u_SkyboxTexture, in_FragmentPosition);

    if(u_EnableHDR) 
    {
        // HDR tonemap and gamma correct
        color /= (color + vec4(1.0));
        color = pow(color, vec4(1.0 / 2.2));
    }

    out_FinalColor = color;
}
