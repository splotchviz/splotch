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

#define GL_GLEXT_PROTOTYPES
#define GLX_GLXEXT_PROTOTYPES
#include "Shader.h"

namespace previewer
{
	
	void Shader::Load(std::string filename, bool withGeometryShader)
	{
		program = 0;
		glProgramParameteriEXT = (PFNGLPROGRAMPARAMETERIEXTPROC)glXGetProcAddress((const GLubyte*)"glProgramParameteriEXT");
		glProgramUniform1fv = (PFNGLPROGRAMUNIFORM1FVEXTPROC)glXGetProcAddress((const GLubyte*)"glProgramUniform1fvEXT");
		glProgramUniform1f = (PFNGLPROGRAMUNIFORM1FEXTPROC)glXGetProcAddress((const GLubyte*)"glProgramUniform1fEXT");

		// Set paths to find shader files
		std::string vsPath = filename + ".vert";
		std::string fsPath = filename + ".frag";

		// Load vertex/fragment shaders
		v_shader = LoadShader(vsPath, GL_VERTEX_SHADER);
		f_shader = LoadShader(fsPath, GL_FRAGMENT_SHADER);


		// Create shader program and attach vertex/fragment shaders
		program = glCreateProgram();
		glAttachShader(program, v_shader);
		glAttachShader(program, f_shader);

		// Check if we are using a geometry shader, if so, load and attach that too
		if(withGeometryShader)
		{
			std::string gsPath = filename + ".geom";

			g_shader = LoadShader(gsPath, GL_GEOMETRY_SHADER_ARB);
			glAttachShader(program, g_shader);

			// Set input and output parameters
			glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS);
			glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP);


			// Setting output vertices to maximum possible caused large fps drop on linux AMD drivers (catalyst 12.6-4)
			// So has been limited to 4 as that is all we need for now.  
			
			// GLint maxPossibleOutVerts;
			// glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&maxPossibleOutVerts);
        	glProgramParameteriEXT(program,GL_GEOMETRY_VERTICES_OUT_EXT, 4);

		}	

		// Link
		glLinkProgram(program);

		// Use program while setting parameters
		glUseProgram(program);

		// glGetUniformLocation will return -1 if uniform is nonexistant
		// Compiler can/will optimise out unused uniforms from shaders

		// Get model view projection matrix uniform location
		uniformMVP = glGetUniformLocation(program, "MVP");

		// Done with program for now
		glUseProgram(0);
	}

	void Shader::Unload()
	{
		// Delete shader program
	}

	void Shader::Bind(const Matrix4& mvp)
	{
		// Bind

		// Use shader prorgam
		glUseProgram(program);

		// Set uniforms
		glUniformMatrix4fv(uniformMVP, 1, GL_FALSE, &mvp[0][0]);

		for(unsigned i = 0; i < scalarUniformfs.size(); i++)
			glUniform1fv(scalarUniformfs[i].loc,scalarUniformfs[i].size, scalarUniformfs[i].ptr);

		// Enable attributes
		for(str_glint_it ii = attributes.begin(); ii != attributes.end(); ++ii)
			glEnableVertexAttribArray(ii->second);
	}

	void Shader::Unbind()
	{
		glUseProgram(0);
	}

	void Shader::SetTextureSrc(GLuint* texSrc)
	{


		// Use program while setting parameters
		glUseProgram(program);
		// Get texture uniform location
		*texSrc = glGetUniformLocation(program, "Tex0");
		glUniform1i(*texSrc, 0);

		// Done with program for now
		glUseProgram(0);		
	}

	void Shader::SetAttribute(std::string attrName)
	{
		glUseProgram(program);

		GLint loc = glGetAttribLocation(program,attrName.c_str());
		if(loc == -1)
		{
			std::cout << "Requested shader attribute "<<attrName<<" not found in shader (or name is prefixed with gl_  - dont do that...)\n";
		}
		else
		{
			attributes[attrName] = loc;
		}
		glUseProgram(0);
	}

	GLint Shader::GetAttributeLocation(std::string attrName)
	{
		return attributes.find(attrName)->second; 
	}

	void Shader::SetUniformf(std::string attrName, int size, GLfloat* ptr)
	{
		glUseProgram(program);

		GLint loc = glGetUniformLocation(program,attrName.c_str());
		if(loc == -1)
		{
			std::cout << "Requested shader uniform "<<attrName<<" not found in shader (or name is prefixed with gl_  - dont do that...)\n";
		}
		else
		{
			uniformf uf;
			uf.name = attrName;
			uf.size = size;
			uf.loc = loc;
			uf.ptr = ptr;
			scalarUniformfs.push_back(uf);
		}
		glUseProgram(0);
	}

	GLint Shader::GetUniformLocation(std::string attrName)
	{
		// Check scalar float uniforms
		for(unsigned i = 0; i < scalarUniformfs.size(); i++)
			if(scalarUniformfs[i].name == attrName)
				return scalarUniformfs[i].loc;

		// Check other types of uniforms here if necessary - none currently used

		// Inform user if not found
		std::cout << "Uniform '"<<attrName<<"' not found in scalarUniformfs vector for shader.\n";
		return -1;
	}


	GLuint Shader::GetProgHandle()
	{
		return program;
	}

	GLuint Shader::LoadShader(const std::string &filename, const int shaderType)
	{
		// Load shader file as char array
		char* shaderText = FileLib::loadTextFile(filename);
		if(!shaderText)
		{
			std::cout << "No data in shader file!" << std::endl;
			return 0;
		}

		// Create shader of specific type
		GLuint shader = glCreateShader(shaderType);

		const char *ptr = shaderText;

		// Set shader source
		// Arguments: shader, num of srcs, array of ptrs to strings containing src, NULL (each string is assumed to be null terminated)
		glShaderSource(shader, 1, &ptr, NULL);

		// Shader data is not needed any more
		delete shaderText;

		// Compile shader
		GLint compiled;
		glCompileShader(shader);

		// Check if shader compiled correctly
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

		// If so, indicate, if not get shader compilation log and cout
		if(compiled)
			std::cout << "shader compilation success: " + filename << std::endl;
		else
		{
			std::cout << "compilation fail." << std::endl;

			int infoLogLen = 0;
			int charsWritten = 0;
			GLchar *infoLog;
		 
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);
		 
			if (infoLogLen > 0)
			{
				infoLog = new GLchar[infoLogLen];
				glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
				std::cout << "InfoLog : " << std::endl << infoLog << std::endl;
				delete [] infoLog;
			}
		}

		// Return compiled shader
		return(shader);
	}

}
