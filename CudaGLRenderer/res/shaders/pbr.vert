#version 450 core

uniform mat4 u_ViewMatrix;
uniform mat4 u_ProjMatrix;
uniform mat4 u_ModelMatrix;

layout(location = 0) in vec3 in_Position;
layout(location = 1) in vec3 in_Normal;
layout(location = 2) in vec2 in_TexCoords;

layout(location = 0) out Fragment
{
	vec3 position;
	vec3 normal;
	vec2 texCoords;
} fragment;

void main() 
{	
	vec4 worldPos = u_ModelMatrix * vec4(in_Position, 1.0);

	fragment.position = worldPos.xyz;
	fragment.normal = in_Normal;
	fragment.texCoords = in_TexCoords;//vec2(in_TexCoords.x, 1.0 - in_TexCoords.y);

	gl_Position = u_ProjMatrix * u_ViewMatrix * worldPos;
}