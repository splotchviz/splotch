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

#ifndef SPLOTCH_PREVIEWER_LIBS_RENDERERS_FF_VBO
#define SPLOTCH_PREVIEWER_LIBS_RENDERERS_FF_VBO

#define GL_GLEXT_PROTOTYPES

#include "IRenderer.h"
#include "../core/Camera.h"
#include "../materials/FF_ParticleMaterial.h"
#include "../core/ParticleSimulation.h"
#include "cxxsupport/vec3.h"

namespace previewer
{

	class FF_VBO : public IRenderer
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

	public:

	private:
		// Vertex Buffer Object
		GLuint VBO;

		// Keep track of mouse motion
		float mouseMotionX;
		float mouseMotionY;

		//Storage for vertices and colours, to allow editing before send to fixed function shaders
		std::vector<vec3f> vertices;
		std::vector<vec3f> colours;
	};

}

#endif