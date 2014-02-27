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

#include "Fbo.h"

Fbo::Fbo()
{

}

Fbo::~Fbo()
{

}

void Fbo::Load(int width, int height)
{
	    //create texture object to render to
	    glGenTextures(1, &FBOTexId);
	    glBindTexture(GL_TEXTURE_2D, FBOTexId);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
	    glBindTexture(GL_TEXTURE_2D, 0);
	 		

	    //create framebuffer object 
	    glGenFramebuffersEXT(1, &FBOId);
	    glBindFramebufferEXT(GL_FRAMEBUFFER, FBOId);


	    //create render buffer object 
	    glGenRenderbuffersEXT(1, &RBOId);
		glBindRenderbufferEXT(GL_RENDERBUFFER, RBOId);
		glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glBindRenderbufferEXT(GL_RENDERBUFFER, 0);


	    //attach a texture to FBO color attachment point
	    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBOTexId, 0);


	    //attach a renderbuffer to FBO depth attachment point
	    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RBOId);


		GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
		if(status == GL_FRAMEBUFFER_COMPLETE)
			std::cout << "framebuffer creation complete" << std::endl;
		else std::cout << "framebuffer error! " << status << std::endl;

	    //unbind
	    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
}

void Fbo::Update(int width, int height)
{
		glBindTexture(GL_TEXTURE_2D, FBOTexId);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
	    glBindTexture(GL_TEXTURE_2D, 0);

		glBindFramebufferEXT(GL_FRAMEBUFFER, FBOId);

		glBindRenderbufferEXT(GL_RENDERBUFFER, RBOId);
		glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
		glBindRenderbufferEXT(GL_RENDERBUFFER, 0);

	    //attach a texture to FBO color attachment point
	    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBOTexId, 0);


	    //attach a renderbuffer to FBO depth attachment point
	    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RBOId);


		GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
		if(status == GL_FRAMEBUFFER_COMPLETE)
			std::cout << "framebuffer update complete" << std::endl;
		else std::cout << "framebuffer error! " << status << std::endl;

	    //unbind
	    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
}

void Fbo::Bind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER, FBOId);
}

void Fbo::Unbind()
{
	glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
}

GLuint Fbo::GetTexID()
{
	return FBOTexId;
}


