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

#ifndef SPLOTCH_PREVIEWER_LIBS_MATERIALS_IMATERIAL
#define SPLOTCH_PREVIEWER_LIBS_MATERIALS_IMATERIAL

//Debug include
#include "previewer/libs/core/Debug.h"

#include "../core/Texture.h"
#include "../core/MathLib.h"

namespace previewer
{

	// Provides the interface for access to materials. Note all materials
	// must inherit from the IMaterial interface for them to be accessible
	// to the renderers
	class IMaterial
	{
		public:

		virtual ~IMaterial() {};

		virtual void Load() = 0;
		virtual void Load(std::string, bool) = 0;
		virtual void Unload() = 0;

		virtual void Bind() = 0;
		virtual void Bind(const Matrix4&) = 0;
		virtual void Unbind() = 0;

		virtual GLuint GetShaderHandle() = 0;
		virtual void SetShaderAttribute(std::string) = 0;
		virtual GLint GetAttributeLocation(std::string) = 0;
		virtual void SetShaderUniformf(std::string, int, GLfloat*) = 0;
		virtual GLint GetUniformLocation(std::string) = 0;

		void SetAlphaTest(bool _alphaTest)
		{
			alphaTest = _alphaTest;
		}

		void SetAlphaValue(float _alphaValue)
		{
			alphaValue = _alphaValue;
		}
		void SetBlend(bool _blend)
		{
			blend = _blend;
		}

		void SetBlendSrc(GLenum _blendSrc)
		{
			blendSrc = _blendSrc;
		}

		void SetBlendDst(GLenum _blendDst)
		{
			blendDst = _blendDst;
		}
		void SetCull(bool _cull)
		{
			cull = _cull;
		}

		void SetDepthTest(bool _depthTest)
		{
			depthTest = _depthTest;
		}	

		void SetDepthValue(float _depthValue)
		{
			depthValue = _depthValue;
		}

		void SetDepthMask(bool _depthMask)
		{
			depthMask = _depthMask;
		}

		void SetTexture(bool _hasTexture)
		{
			hasTexture = _hasTexture;
		}

		// Load texture from file
		void LoadTexture(std::string filename, GLenum texType)
		{
			texture0.SetTexture(filename, texType);
		}

		// Use preloaded texture
		void LoadTexture(GLuint& newTex, GLenum texType)
		{
			texture0.SetTexture(newTex, texType);
		}

		int GetTexWidth()
		{
			return texture0.GetWidth();
		}

		int GetTexHeight()
		{
			return texture0.GetHeight();
		}

		protected:

		bool        blend;
	    GLenum      blendSrc;
	    GLenum      blendDst;

	    bool        alphaTest;
	    float       alphaValue;

	    bool        cull;
	    
	    bool        depthTest;
	    bool        depthMask;
	    float 		depthValue;

	    bool		hasTexture;
	    Texture		texture0;

	};

}

#endif