#version 330

uniform mat4    ciModelViewProjection;
uniform mat3	ciNormalMatrix;

in vec4			ciPosition;
in vec3			ciNormal;
in vec2			ciTexCoord0;

out vec2		vTexCoord;

void main()
{
	vTexCoord	= ciTexCoord0;
	gl_Position	= ciModelViewProjection * ciPosition;
}