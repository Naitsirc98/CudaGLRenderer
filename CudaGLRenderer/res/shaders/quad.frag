#version 330 core

uniform sampler2D u_Texture;

in vec2 frag_TexCoords;

out vec4 out_Color;

void main()
{
	out_Color = texture(u_Texture, frag_TexCoords);
}