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

#include "Texture.h"

namespace previewer
{

	void Texture::SetTexture(std::string filename, GLenum type)
	{
		texType = type;

		// Load tga file for texture 
		unsigned char* imageData = FileLib::loadTGAFile(filename, width, height, bpp);

		// Check for alpha, if so set appropriate format for glteximage2d 
		format = (bpp == 32) ? GL_RGBA : GL_RGB;

		// Check width and height set properly on load
		DebugPrint("Texture: w %i, h %i, bpp %i\n",width, height, bpp);

		// Gl generate texture
		glGenTextures(1, &texID);

		// Bind
		glBindTexture(texType, texID);

		// Set texture parameters
		glTexParameteri(texType, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(texType, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		// Create texture from loaded data. args: target/level/internal format/width/height/border/format/type/data
		glTexImage2D(texType, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, imageData);

		glBindTexture(texType, 0);

	}

	void Texture::SetTexture(GLuint& newTex, GLenum type)
	{
		texID = newTex;
		texType = type;
	}
	
	// Set texture from input data buffer
	void Texture::SetTexture(void* data, int _width, int _height, GLenum _texType, GLenum _format, GLenum _dataType)
	{
		texType = _texType;
		width = _width;
		height = _height;
		format = _format;
		dataType = _dataType;

		// Gl generate texture
		glGenTextures(1, &texID);


		// Bind
		glBindTexture(texType, texID);
		// Set texture parameters
		glTexParameteri(texType, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(texType, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(texType, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(texType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		// Create texture from loaded data. args: target/level/internal format/width/height/border/format/type/data
		glTexImage2D(texType, 0, format, width, height, 0, format, dataType, data);
		glBindTexture(texType, 0);
	}

	void Texture::ReplaceTextureData(void* data, int _width, int _height)
	{
		glBindTexture(texType, texID);
		glTexSubImage2D(texType, 0, 0, 0, width, height, format, dataType, data);
		glBindTexture(texType, 0);
	}

	void Texture::Bind()
	{
		// use specific IDs rather than gl_texture0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(texType, texID);
	}	

	void Texture::Unbind()
	{
		glBindTexture(texType, 0);
	}

	int Texture::GetWidth()
	{
		return width;
	}

	int Texture::GetHeight()
	{
		return height;
	}

}