#version 120

uniform sampler2D  Tex0;	

void main(void)
{
    vec4 tex = texture2D(Tex0, gl_TexCoord[0].xy);
    gl_FragColor = tex;
}