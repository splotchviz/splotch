#version 120

uniform mat4 MVP;
uniform float inBrightness[10];
uniform float inSmoothingLength[10];
uniform float inRadialMod;

attribute vec3 inPosition;
attribute vec3 inColor;
attribute float inRadius;
attribute float inTensity;
attribute float inType;

varying float radius;

void main()
{	
	
	// Multiply color by type-dependant brightness
	vec3 col = inColor;
	float thisbrightness = inBrightness[int(inType)];
	float thissmooth = inSmoothingLength[int(inType)];

	// Compress further due to lack of hdr 
	col = inColor * thisbrightness * 0.075;

	// Pass through radius to geom shader
	if(thissmooth==0)
	{
    	radius = inRadius * inRadialMod;
    }
    else
    {
    	radius = thissmooth * inRadialMod;
    }

    gl_FrontColor = vec4(col, 1.00);
    gl_Position = MVP * vec4(inPosition,1.0);
}


