#version 120

uniform sampler2D  Tex0;	

void main(void)
{
    vec4 tex = texture2D(Tex0, gl_TexCoord[0].xy);

    // Basic Tonemapping

    float k = 1.5;
    float h = 9.0;
    float r = pow(tex.r, 1.0/k)/h;
    float g = pow(tex.g, 1.0/k)/h;
    float b = pow(tex.b, 1.0/k)/h;

    gl_FragColor = vec4(r,g,b,1.0);
}