#version 430

in block {
	vec4 color;
	vec2 texCoord;
} In;

layout( location = 0 ) out vec4 fragColor;



void main()
{
	float r = length( In.texCoord * 2.0 - 1.0 ) * 3.0;
	float i = exp( -r * r );
	if( i < 0.01 )
	{
		discard;
	}

    fragColor = vec4( In.color.rgb, 1.0 );
}