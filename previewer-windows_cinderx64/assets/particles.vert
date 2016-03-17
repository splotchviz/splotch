#version 430
#extension GL_ARB_shader_storage_buffer_object : require

uniform mat4 ciProjectionMatrix;
uniform mat4 ciModelView;
uniform vec4 iColor;
uniform float brightnessMod;
uniform	float saturation;
uniform	float contrast;
uniform float iparticleSize;


layout( std140, binding=1 ) buffer Pos {
	vec4 pos[];
};

layout( std140, binding=2 ) buffer Color {
	vec4 color[];
};

out gl_PerVertex {
	vec4 gl_Position;
};

out block {
	vec4 color;
	vec2 texCoord;
} Out;

vec3 ContrastSaturationBrightness(vec3 color, float brt, float sat, float con)
{
	const float AvgLumR = 0.5;
	const float AvgLumG = 0.5;
	const float AvgLumB = 0.5;
	
	const vec3 LumCoeff = vec3(0.2125, 0.7154, 0.0721);
	
	vec3 AvgLumin = vec3(AvgLumR, AvgLumG, AvgLumB);
	vec3 brtColor = color * brt;
	vec3 intensity = vec3(dot(brtColor, LumCoeff));
	vec3 satColor = mix(intensity, brtColor, sat);
	vec3 conColor = mix(AvgLumin, satColor, con);
	return conColor;
}

void main()
{
	int particleID = gl_VertexID >> 2; // 4 vertices per particle

	vec4 particlePos = pos[particleID];
	vec4 colorV = color[particleID];

	vec3 colorV3 = vec3(colorV.x, colorV.y, colorV.z);	//Convert into a vec3
	colorV3 = ContrastSaturationBrightness(colorV3, brightnessMod, saturation, contrast);	//Read into function
	colorV = vec4(colorV3.x, colorV3.y, colorV3.z, colorV.w);	//Convert back into vec4
	Out.color = colorV;

	//map vertex ID to quad vertex
	vec2 quadPos = vec2( ( ( gl_VertexID - 1 ) & 2 ) >> 1, ( gl_VertexID & 2 ) >> 1 );

	vec4 particlePosEye = ciModelView * particlePos;
	vec4 vertexPosEye = particlePosEye + vec4( ( quadPos * 2.0 - 1.0 ) * iparticleSize, 0, 0 );

	Out.texCoord = quadPos;
	gl_Position = ciProjectionMatrix * vertexPosEye;
}