#version 120

uniform mat4 MVP;        // ModelViewProjectionMatrix

void main()
{	
	//pass through colour and position after multiplying pos by matrices
	gl_FrontColor = vec4(gl_Color.x, gl_Color.y, gl_Color.z, 0.5);
    gl_Position = MVP * gl_Vertex;
}