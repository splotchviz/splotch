#version 120

uniform mat4 MVP;        // ModelViewProjectionMatrix

void main()
{	
	//pass through colour and position after multiplying pos by matrices
	gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = MVP * gl_Vertex;
}