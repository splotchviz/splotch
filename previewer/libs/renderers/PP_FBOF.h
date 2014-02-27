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
 
#ifndef SPLOTCH_PREVIEWER_LIBS_RENDERERS_PP_GEOM
#define SPLOTCH_PREVIEWER_LIBS_RENDERERS_PP_GEOM

#define GL_GLEXT_PROTOTYPES

#include "IRenderer.h"
#include "previewer/libs/core/Camera.h"
#include "previewer/libs/materials/PP_ParticleMaterial.h"
#include "previewer/libs/core/ParticleSimulation.h"
#include "previewer/libs/core/Fbo.h"
#include "cxxsupport/vec3.h"
#include "previewer/libs/core/WindowManager.h"

namespace previewer
{
#define TEXTURE_WIDTH ParticleSimulation::GetRenderWidth()
#define TEXTURE_HEIGHT ParticleSimulation::GetRenderWidth()

	class PP_FBOF : public IRenderer
	{
	public:

		void Load(const ParticleData&);
		void Draw();
		void Unload();
		void Update();

		void OnKeyPress(Event);
		void OnMotion(Event);

	private:	
		void genVBO();
		void DrawGeom();

	public:

	private:
		// Vertex Buffer Object
		GLuint VBO;

		// Keep track of mouse motion
		float mouseMotionX;
		float mouseMotionY;

		// Offsets for drawing interleaved data
		int colorOffset;
		int byteInterval;

		// Fbos for hdr rendering
		Fbo Fbo_Passthrough;
		Fbo Fbo_ToneMap;

		//Fbos for further post-processing
		Fbo Fbo_horizontal;
		Fbo Fbo_vertical;

		IMaterial* fboHZMaterial;
		IMaterial* fboVTMaterial;

		IMaterial* fboPTMaterial;
		IMaterial* fboTMMaterial;
		Matrix4 ident;
	};

}

#endif