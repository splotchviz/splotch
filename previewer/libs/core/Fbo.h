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

#include <iostream>
#include "GL/gl.h"

class Fbo {
public:
	Fbo();
	~Fbo();

	void Load(int width, int height);
	void Update(int width, int height);

	void Bind();
	void Unbind();

	GLuint GetTexID();

private:

	GLuint FBOTexId;
	GLuint FBOId;
	GLuint RBOId;
};