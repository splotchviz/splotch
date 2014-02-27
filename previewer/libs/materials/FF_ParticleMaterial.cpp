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
 * 		Tim Dykes and Ian Cant
 * 		University of Portsmouth
 *
 */

#include "FF_ParticleMaterial.h"

namespace previewer
{

	void FF_ParticleMaterial::Load()
	{
		SetAlphaTest(false);
		//SetAlphaValue();
		SetBlend(false);
		//SetBlendSrc();
		//SetBlendDst();
		SetCull(false);
		SetDepthTest(false);
		//SetDepthValue();
		//SetDepthMask();
		SetTexture(false);
		//LoadTexture();

	}

	void FF_ParticleMaterial::Load(std::string, bool)
	{
		// Unused in this material
	}

	void FF_ParticleMaterial::Unload()
	{

	}


	void FF_ParticleMaterial::Bind()
	{
		// Set all necessary GL capabilities

		if(alphaTest)
		{
			glEnable(GL_ALPHA_TEST);
			// Alpha always drawn at the moment - alphaValue irrelevant
			// Possible alternatives GL_LESS/GL_GREATER/GL_NEVER etc
			glAlphaFunc(GL_ALWAYS, alphaValue);
		}
		else 
		{
			glDisable(GL_ALPHA_TEST);
		}

		if(blend)
		{
			glEnable(GL_BLEND);
			glBlendFunc(blendSrc, blendDst);
		}
		else
		{
			glDisable(GL_BLEND);
		}

	    if(cull)
	    {
	    	glEnable(GL_CULL_FACE);
	    }
	    else
	    {
	    	glDisable(GL_CULL_FACE);
	    }

	    if(depthTest)
	    {
	    	glEnable(GL_DEPTH_TEST);
	    	glDepthMask(depthMask);
	    	// Depth func similar to alphaFunc
	    	glDepthFunc(GL_LESS);
	    }
	    else
	    {
	    	glDisable(GL_DEPTH_TEST);
	    }

	    if(hasTexture)
	    {
	    	glEnable(GL_TEXTURE_2D);
	    	//Bind texture
	    	texture0.Bind();
	    }
	    else
	    {
	    	glDisable(GL_TEXTURE_2D);
	    }

	}

	void FF_ParticleMaterial::Bind(const Matrix4&)
	{
		//Do nothing, FF does not need mvp matrix here
	}

	void FF_ParticleMaterial::Unbind()
	{
		if(hasTexture)
			texture0.Unbind();
	}


}
