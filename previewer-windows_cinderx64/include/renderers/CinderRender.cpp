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

#include "CinderRender.h"

using namespace ci;
using namespace ci::app;
using namespace std;

void CinderRender::Load(previewer::ParticleData pData)
{
	// Multiple render targets shader updates the positions/velocities
	mParticlesShader = gl::GlslProg::create(gl::GlslProg::Format().vertex(loadAsset("particles.vert"))
		.fragment(loadAsset("particles.frag")));

	mBlurShader = gl::GlslProg::create(gl::GlslProg::Format().vertex(loadAsset("blur_pass.vert"))
		.fragment(loadAsset("blur.frag")));

	mCam.lookAt(ci::vec3(1, -10000, 0), ci::vec3(0, 0, 0));
	mCam.setFarClip(100000.0);

	particles = pData.GetParticleList();
	numberOfParticles = particles.size();
	
	mPos = gl::Ssbo::create(sizeof(ci::vec4) * particles.size(), nullptr, GL_STATIC_DRAW);
	mColor = gl::Ssbo::create(sizeof(ci::vec4) * particles.size(), nullptr, GL_STATIC_DRAW);

	std::vector<uint32_t> indices(particles.size() * 6);
	for (size_t i = 0, j = 0; i < particles.size(); ++i) {
		size_t index = i << 2;
		indices[j++] = index;
		indices[j++] = index + 1;
		indices[j++] = index + 2;
		indices[j++] = index;
		indices[j++] = index + 2;
		indices[j++] = index + 3;
	}

	mIndicesVbo = gl::Vbo::create<uint32_t>(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

	ci::vec4 *pos = reinterpret_cast<ci::vec4*>(mPos->map(GL_WRITE_ONLY));
	ci::vec4 *color = reinterpret_cast<ci::vec4*>(mColor->map(GL_WRITE_ONLY));
	for (size_t i = 0; i < particles.size(); ++i) {
		pos[i] = ci::vec4(particles[i].x, particles[i].y, particles[i].z, 1.0f);
		color[i] = ci::vec4(particles[i].e.r, particles[i].e.g, particles[i].e.b, 1.0f);
	}
	mPos->unmap();
	mColor->unmap();

	
	gl::Fbo::Format fmt;
	mFbo = gl::Fbo::create(getWindowWidth(), getWindowHeight(), fmt);
	mFboBlur1 = gl::Fbo::create(getWindowWidth(), getWindowHeight());
	mFboBlur2 = gl::Fbo::create(getWindowWidth(), getWindowHeight());


	brightnessMod = 1.0;
	saturation = 1.0;
	contrast   = 1.0;
	particleSize = 0.6;

	blurStrength = 0.13f;
	blurColorModifier = 1.2f;

	blurOn = true;
}

void CinderRender::Draw()
{

	mParticlesShader->uniform("brightnessMod", brightnessMod);
	mParticlesShader->uniform("saturation", saturation);
	mParticlesShader->uniform("contrast", contrast);
	float nParticleSize = particleSize * 10;
	mParticlesShader->uniform("iparticleSize", nParticleSize);

	RenderToFBO();
	gl::setMatrices(mCam);
	gl::viewport(getWindowSize());
	gl::setMatricesWindow(getWindowSize());
	gl::draw(mFbo->getColorTexture(), getWindowBounds());

	if (blurOn)
	{
		gl::enableAdditiveBlending();
		gl::draw(mFboBlur2->getColorTexture(), getWindowBounds());
		gl::disableAlphaBlending();
	}

}

void CinderRender::RenderToFBO()
{
	{
		gl::ScopedViewport scpVp(mFbo->getSize());
		gl::ScopedFramebuffer fbScp(mFbo);
		gl::ScopedMatrices matScope;
		gl::clear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		gl::setMatrices(mCam);

		gl::ScopedGlslProg scopedRenderProg(mParticlesShader);

		gl::context()->setDefaultShaderVars();

		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);

		{
			gl::bindBufferBase(mPos->getTarget(), 1, mPos);
			gl::bindBufferBase(mColor->getTarget(), 2, mColor);
			gl::ScopedBuffer scopedIndicex(mIndicesVbo);
			gl::drawElements(GL_TRIANGLES, particles.size() * 6, GL_UNSIGNED_INT, 0);
		}

		gl::disableAlphaBlending();

	}

	if (blurOn)
	{

		{
			gl::ScopedViewport scpVp(0, 0, mFboBlur1->getWidth(), mFboBlur1->getHeight());
			gl::ScopedFramebuffer fbScp(mFboBlur1);
			gl::ScopedGlslProg scopedblur(mBlurShader);
			gl::ScopedTextureBind tex0(mFbo->getColorTexture(), (uint8_t)0);
			gl::setMatricesWindowPersp(mFboBlur1->getWidth(), mFboBlur1->getHeight());

			gl::clear(Color::black());

			mBlurShader->uniform("tex0", 0);

			mBlurShader->uniform("sampleOffset", ci::vec2(blurStrength / mFboBlur1->getWidth(), 0.0f));
			mBlurShader->uniform("colorModifier", blurColorModifier);

			gl::drawSolidRect(mFboBlur1->getBounds());
		}

		{
			gl::ScopedViewport scpVp(0, 0, mFboBlur2->getWidth(), mFboBlur2->getHeight());
			gl::ScopedFramebuffer fbScp(mFboBlur2);
			gl::ScopedGlslProg scopedblur(mBlurShader);
			gl::ScopedTextureBind tex0(mFboBlur1->getColorTexture(), (uint8_t)0);
			gl::setMatricesWindowPersp(mFboBlur2->getWidth(), mFboBlur2->getHeight());

			gl::clear(Color::black());

			mBlurShader->uniform("tex0", 0);

			mBlurShader->uniform("sampleOffset", ci::vec2(0.0f, blurStrength / mFboBlur2->getHeight()));
			mBlurShader->uniform("colorModifier", blurColorModifier);

			gl::drawSolidRect(mFboBlur2->getBounds());
		}
	}
}