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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_SHADER
#define SPLOTCH_PREVIEWER_LIBS_CORE_SHADER

//Debug include
#include "previewer/libs/core/Debug.h"

#define GL_GLEXT_PROTOTYPES
#define GLX_GLXEXT_PROTOTYPES

#include <GL/gl.h>	
#include <GL/glx.h>	

#include <iostream>
#include <stdio.h>
#include <string>
#include <map>
#include <utility>
#include "FileLib.h"
#include "previewer/libs/core/MathLib.h"



namespace previewer
{
	// Provides the functionality to load and use a shader from within
	// other sections of the application (abstraction). It will load an
	// OpenGL shader where possible
	template<typename T>
	struct uniformT{
		std::string name;
		GLint size;
		GLint loc;
		T* ptr;
	};

	typedef uniformT<GLint> uniformi;
	typedef uniformT<GLfloat> uniformf;

	class Shader
	{
	public:

		// Public Methods
		void Load(std::string, bool);
		void Unload();

		void Bind(const Matrix4&);
		void Unbind();

		void SetTextureSrc(GLuint* texSrc);

		GLuint GetProgHandle();

		void SetAttribute(std::string);
		GLint GetAttributeLocation(std::string);

		void SetUniformf(std::string, int, GLfloat*);
		GLint GetUniformLocation(std::string);

	private:
		// Private methods
		GLuint LoadShader(const std::string&, const int);

	public:
		// Public variables

	private:
		// Private variables

		// Shader programs
		GLuint 	program;
		GLuint 	v_shader;
		GLuint 	f_shader;
		GLuint	g_shader;

		// Uniform locations for ModelviewProjection Matrix and Texture/s
		GLint 	uniformMVP; 

		std::map<std::string, GLint> attributes;
		std::vector<uniformf> scalarUniformfs;

		typedef std::map<std::string, GLint>::iterator str_glint_it;


		PFNGLPROGRAMPARAMETERIEXTPROC glProgramParameteriEXT;
		PFNGLPROGRAMUNIFORM1FVEXTPROC glProgramUniform1fv;
		PFNGLPROGRAMUNIFORM1FEXTPROC glProgramUniform1f;

	};
}

#endif
