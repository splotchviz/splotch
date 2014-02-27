#version 120
#extension GL_EXT_gpu_shader4 : enable

uniform sampler2D  Tex0;

void main()
{	
	//map texture to fragment, multiply by colour to get coloured blobs
    vec4 tex = texture2D(Tex0, gl_TexCoord[0].xy);
	gl_FragColor = tex * gl_Color;
}



