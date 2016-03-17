#version 330

uniform sampler2D	tex0;
uniform vec2		sampleOffset;
uniform float		colorModifier;

in vec2		vTexCoord;

layout (location = 0) out vec4 oFragColor;

void main()
{ 
	vec3 sum = vec3( 0.0, 0.0, 0.0 );	
	sum += texture( tex0, vTexCoord + -10.0 * sampleOffset ).rgb * 0.009167927656011385;
	sum += texture( tex0, vTexCoord +  -9.0 * sampleOffset ).rgb * 0.014053461291849008;
	sum += texture( tex0, vTexCoord +  -8.0 * sampleOffset ).rgb * 0.020595286319257878;
	sum += texture( tex0, vTexCoord +  -7.0 * sampleOffset ).rgb * 0.028855245532226279;
	sum += texture( tex0, vTexCoord +  -6.0 * sampleOffset ).rgb * 0.038650411513543079;
	sum += texture( tex0, vTexCoord +  -5.0 * sampleOffset ).rgb * 0.049494378859311142;
	sum += texture( tex0, vTexCoord +  -4.0 * sampleOffset ).rgb * 0.060594058578763078;
	sum += texture( tex0, vTexCoord +  -3.0 * sampleOffset ).rgb * 0.070921288047096992;
	sum += texture( tex0, vTexCoord +  -2.0 * sampleOffset ).rgb * 0.079358891804948081;
	sum += texture( tex0, vTexCoord +  -1.0 * sampleOffset ).rgb * 0.084895951965930902;
	sum += texture( tex0, vTexCoord +   0.0 * sampleOffset ).rgb * 0.086826196862124602;
	sum += texture( tex0, vTexCoord +  +1.0 * sampleOffset ).rgb * 0.084895951965930902;
	sum += texture( tex0, vTexCoord +  +2.0 * sampleOffset ).rgb * 0.079358891804948081;
	sum += texture( tex0, vTexCoord +  +3.0 * sampleOffset ).rgb * 0.070921288047096992;
	sum += texture( tex0, vTexCoord +  +4.0 * sampleOffset ).rgb * 0.060594058578763078;
	sum += texture( tex0, vTexCoord +  +5.0 * sampleOffset ).rgb * 0.049494378859311142;
	sum += texture( tex0, vTexCoord +  +6.0 * sampleOffset ).rgb * 0.038650411513543079;
	sum += texture( tex0, vTexCoord +  +7.0 * sampleOffset ).rgb * 0.028855245532226279;
	sum += texture( tex0, vTexCoord +  +8.0 * sampleOffset ).rgb * 0.020595286319257878;
	sum += texture( tex0, vTexCoord +  +9.0 * sampleOffset ).rgb * 0.014053461291849008;
	sum += texture( tex0, vTexCoord + +10.0 * sampleOffset ).rgb * 0.009167927656011385;

	oFragColor.rgb = sum * colorModifier;
	oFragColor.a = 1.0;
}