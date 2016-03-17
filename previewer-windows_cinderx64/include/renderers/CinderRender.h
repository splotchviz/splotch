/*
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*
* 		Elliott Ayling
* 		University of Portsmouth
*
*/

#ifndef CINDER_RENDER
#define CINDER_RENDER

//Debug include
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder\gl\Fbo.h"
#include "cinder\gl\Vbo.h"
#include "cinder\gl\GlslProg.h"
#include "cinder/Camera.h"
#include "cinder/gl/gl.h"
#include "cinder/gl/Texture.h"
#include "cinder/gl/Ssbo.h"
#include "cinder/CameraUi.h"
#include "../previewer_n/application.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class CinderRender
{
public:
	void Load(previewer::ParticleData pData);
	void Draw();
	CameraPersp mCam;
	float brightnessMod;

	int numberOfParticles;
	
	bool blurOn;
	float blurStrength;
	float blurColorModifier;
	float particleSize;
	float saturation, contrast;

private:
	void RenderToFBO();
	previewer::ParticleList particles;
	gl::GlslProgRef mParticlesShader;
	gl::GlslProgRef mBlurShader;

	gl::SsboRef mPos;
	gl::SsboRef mColor;
	gl::VboRef mIndicesVbo;

	gl::FboRef mFbo;
	gl::FboRef mFboBlur1;
	gl::FboRef mFboBlur2;



};

#endif