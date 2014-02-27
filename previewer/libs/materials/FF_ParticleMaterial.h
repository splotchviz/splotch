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

#ifndef SPLOTCH_PREVIEWER_LIBS_MATERIALS_IMATERIAL_FF_PARTICLE_MATERIAL
#define SPLOTCH_PREVIEWER_LIBS_MATERIALS_IMATERIAL_FF_PARTICLE_MATERIAL

#include "IMaterial.h"

namespace previewer
{
	// Material class used to draw a particle
	class FF_ParticleMaterial : public IMaterial
	{
	public:
		
		void Bind();
		void Bind(const Matrix4&);
		void Unbind();

		void Load();
		void Load(std::string, bool);
		void Unload();

		GLuint GetShaderHandle() { return (GLuint)-1;}
		void SetShaderAttribute(std::string) {}
		GLint GetAttributeLocation(std::string) {return (GLint)-1;}
		void SetShaderUniformf(std::string,int, GLfloat*) {};
		GLint GetUniformLocation(std::string) {return (GLint)-1;}
	private:

	public:

	private:
	};

}

#endif