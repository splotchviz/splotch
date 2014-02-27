#version 330 core

uniform sampler2D image;

out vec4 FragmentColor;

uniform float offset[3] = float[]( 0.0, 1.3846153846, 3.2307692308 );
uniform float weight[3] = float[]( 0.2270270270, 0.3162162162, 0.0702702703 );

void main(void)
{
	ivec2 size = textureSize(image,0);
	FragmentColor = texture2D( image, vec2(gl_FragCoord)/size.x ) * weight[0];
	for (int i=1; i<3; i++) {
		FragmentColor += texture2D( image, ( vec2(gl_FragCoord)+vec2(offset[i], 0.0) )/size.x ) * weight[i];
		FragmentColor += texture2D( image, ( vec2(gl_FragCoord)-vec2(offset[i], 0.0) )/size.x ) * weight[i];
	}
}
