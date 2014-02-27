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
 
#include "PP_FBO.h"
#include "previewer/Previewer.h"

namespace previewer
{
	void PP_FBO::Load(const ParticleData& pData)
	{
		//Store passed in data and bounding box
		particleList = pData.GetParticleList();
		dataBBox = pData.GetBoundingBox();

		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();

		// Create the VBO for particle drawing
		genVBO();

		PrintOpenGLError();

		// Create material
		material = new PP_ParticleMaterial();

		// Load shader program, with geometry shader
		material->Load(ParticleSimulation::GetExePath()+"previewer/data/shaders/PP_FBO", true);

		// Set up rest of material
		material->SetBlend(true);
		material->SetBlendSrc(GL_SRC_ALPHA);
		material->SetBlendDst(GL_ONE);
		material->SetTexture(true);
		material->LoadTexture(ParticleSimulation::GetExePath()+"previewer/data/textures/particle.tga", GL_TEXTURE_2D);

		// Set up brightness + smoothing length uniforms
		brightness = pData.GetParameterBrightness();
		// Ensure unused elements up to tenth are 1 (static size 10 array in shader)
		if(brightness.size()<10)
			brightness.resize(10,1);

		material->SetShaderUniformf("inBrightness", 10, (float*)&brightness[0]);

		radial_mod = pData.GetRadialMod();
		material->SetShaderUniformf("inRadialMod", 1, (float*)&radial_mod);

		smoothingLength = pData.GetParameterSmoothingLength();
		if(smoothingLength.size()<10)
			smoothingLength.resize(10,0);

		material->SetShaderUniformf("inSmoothingLength",10,(float*)&smoothingLength[0]);

		//brightmod = splotchParams->find<int>("ptypes",1);

		// Set shader attribute arrays
		material->SetShaderAttribute("inPosition");
		material->SetShaderAttribute("inColor");
		material->SetShaderAttribute("inRadius");
		material->SetShaderAttribute("inType");

		// Load and position camera_xa
		camera.SetPerspectiveProjection(ParticleSimulation::GetFOV(), ParticleSimulation::GetAspectRatio(), 1, 200000);

		// Check if recalculation is required
		bool recalc = splotchParams->find<bool>("pv_recalc_cam",true);
		if(recalc)
			camera.Create(dataBBox);
		else
		{
			vec3f lookat(splotchParams->find<double>("lookat_x"),splotchParams->find<double>("lookat_y"),splotchParams->find<double>("lookat_z"));
			vec3f sky(splotchParams->find<double>("sky_x",0),splotchParams->find<double>("sky_y",0),splotchParams->find<double>("sky_z",1));
			vec3f campos(splotchParams->find<double>("camera_x",0),splotchParams->find<double>("camera_y",0),splotchParams->find<double>("camera_z",1));

			camera.Create(campos,lookat,(sky*=-1));
		}

		camera.SetMainCameraStatus(true);

		// Set up passthrough FBO
		DebugPrint("Before passthrough FBO Setup");
		PrintOpenGLError();

		Fbo_Passthrough.Load(ParticleSimulation::GetXRes(), ParticleSimulation::GetYRes());

		DebugPrint("After passthrough FBO Setup");
		PrintOpenGLError();

		GLuint fboPTTex = Fbo_Passthrough.GetTexID();

		fboPTMaterial = new PP_ParticleMaterial();
		fboPTMaterial->Load(ParticleSimulation::GetExePath()+"previewer/data/shaders/FBO_Passthrough", false);	
		fboPTMaterial->SetTexture(true);
		fboPTMaterial->LoadTexture(fboPTTex, GL_TEXTURE_2D);

		// Set up ToneMapping FBO
		DebugPrint("Before ToneMapping FBO Setup");
		PrintOpenGLError();

		 Fbo_ToneMap.Load(ParticleSimulation::GetXRes(), ParticleSimulation::GetYRes());

		DebugPrint("After ToneMapping FBO Setup");
		PrintOpenGLError();

		 GLuint fboTMTex = Fbo_ToneMap.GetTexID();

		 fboTMMaterial = new PP_ParticleMaterial();
		 fboTMMaterial->Load(ParticleSimulation::GetExePath()+"previewer/data/shaders/FBO_ToneMap", false);
		 fboTMMaterial->SetTexture(true);
		 fboTMMaterial->LoadTexture(fboTMTex, GL_TEXTURE_2D);

		// Set identity matrix for drawin rtt quad to screen
		ident.identity();

		glEnable(GL_PROGRAM_POINT_SIZE_EXT);

		// GLfloat* glf = new GLfloat;
		// glGetFloatv(GL_SMOOTH_POINT_SIZE_RANGE, glf);
		// std::cout << "point size range: " << glf[0] << " " << glf[1] << std::endl;

		// GLint value;
		// glGetIntegerv(GL_MAX_ELEMENTS_VERTICES, &value);
		// std::cout << "Max recommended points for draw elements: " << value << std::endl;

		//std::cout << "point size granularity: " << glGetFloatv(GL_SMOOTH_POINT_SIZE_GRANULARITY) << std::endl;
	}

	void PP_FBO::Draw()
	{

		Clear(0.2,0.2,0.2,1.0);
		// Draw scene into passthrough FBO
		Fbo_Passthrough.Bind();

		glClearColor(0.7, 0.1, 0.1, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		DrawGeom();

		Fbo_Passthrough.Unbind();

		// Draw passthrough FBO into ToneMapping FBO
		Fbo_ToneMap.Bind();

		glClearColor(0.1, 0.7, 0.1, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Bind material
		fboPTMaterial->Bind(ident);

		glBegin(GL_QUADS);

		    glTexCoord2f(0,0);  glVertex3f(-1, -1, 0);
		    glTexCoord2f(0,1);  glVertex3f(-1, 1, 0);
		    glTexCoord2f(1,1);  glVertex3f(1, 1, 0);
		    glTexCoord2f(1,0);  glVertex3f(1, -1, 0);

		glEnd();

		fboPTMaterial->Unbind();

		Fbo_ToneMap.Unbind();

		// Draw ToneMapping FBO to screen
		glEnable(GL_SCISSOR_TEST);

		glViewport(0,0, ParticleSimulation::GetSimWindowWidth(),ParticleSimulation::GetSimWindowHeight());
		glScissor(0,0, ParticleSimulation::GetSimWindowWidth(),ParticleSimulation::GetSimWindowHeight());
		Clear(0.2,0.2,0.2,1.0);

		// Set 3d rendering viewport size 
		float viewXmin = ParticleSimulation::GetRenderXMin();
		float viewYmin = ParticleSimulation::GetRenderYMin();

		// Set 3d render viewport to scissor and clear
		glViewport(viewXmin,viewYmin, ParticleSimulation::GetRenderWidth(), ParticleSimulation::GetRenderHeight());
		glScissor(viewXmin,viewYmin, ParticleSimulation::GetRenderWidth(),ParticleSimulation::GetRenderHeight());

		glClearColor(1.0, 0.1, 0.1, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Bind material
		fboTMMaterial->Bind(ident);

		glBegin(GL_QUADS);

		    glTexCoord2f(0,0);  glVertex3f(-1, -1, 0);
		    glTexCoord2f(0,1);  glVertex3f(-1, 1, 0);
		    glTexCoord2f(1,1);  glVertex3f(1, 1, 0);
		    glTexCoord2f(1,0);  glVertex3f(1, -1, 0);

		glEnd();

		fboTMMaterial->Unbind();

		glDisable(GL_SCISSOR_TEST);

	}

	void PP_FBO::Unload()
	{

	}

	void PP_FBO::Update()
	{
		// Reset camera projection, and update frame buffer objects
		camera.SetPerspectiveProjection(ParticleSimulation::GetFOV(), ParticleSimulation::GetAspectRatio(), 1, 200000);
		Fbo_Passthrough.Update(ParticleSimulation::GetXRes(), ParticleSimulation::GetYRes());
		Fbo_ToneMap.Update(ParticleSimulation::GetXRes(), ParticleSimulation::GetYRes());
	}

	void PP_FBO::OnKeyPress(Event ev)
	{
		// Movement * seconds per frame for uniform movement regardless of framerate 
		// (* 100 / *0.1 to make scale similar for setters move and rotation)
		if(ev.keyID == "w")
				camera.MoveForward(ParticleSimulation::GetMoveSpeed()*100*WindowManager::GetElapsedTime());

		else if(ev.keyID == "a")
				camera.MoveRight(-ParticleSimulation::GetMoveSpeed()*100*WindowManager::GetElapsedTime());

		else if(ev.keyID == "s")
				camera.MoveForward(-ParticleSimulation::GetMoveSpeed()*100*WindowManager::GetElapsedTime());

		else if(ev.keyID == "d")
				camera.MoveRight(ParticleSimulation::GetMoveSpeed()*100*WindowManager::GetElapsedTime());

		else if(ev.keyID == "q")
				camera.MoveUpward(ParticleSimulation::GetMoveSpeed()*50*WindowManager::GetElapsedTime());

		else if(ev.keyID == "e")
				camera.MoveUpward(-ParticleSimulation::GetMoveSpeed()*50*WindowManager::GetElapsedTime());

		else if(ev.keyID == "i")
				camera.RotateAroundTarget(dataBBox.centerPoint, 0.0,ParticleSimulation::GetRotationSpeed()*0.1*WindowManager::GetElapsedTime(),0.0);

		else if(ev.keyID == "j")
				camera.RotateAroundTarget(dataBBox.centerPoint, ParticleSimulation::GetRotationSpeed()*0.1*WindowManager::GetElapsedTime(),0.0,0.0);

		else if(ev.keyID == "k")
				camera.RotateAroundTarget(dataBBox.centerPoint, 0.0,-ParticleSimulation::GetRotationSpeed()*0.1*WindowManager::GetElapsedTime(),0.0);

		else if(ev.keyID == "l")
				camera.RotateAroundTarget(dataBBox.centerPoint, -ParticleSimulation::GetRotationSpeed()*0.1*WindowManager::GetElapsedTime(),0.0,0.0);

	}

	void PP_FBO::OnMotion(Event ev)
	{
		//account for mouse moving around screen between clicks or going off the render screen
		if( (ev.mouseX > (mouseMotionX + 10.f)) || (ev.mouseX < (mouseMotionX - 10.f)) || 
			(ev.mouseX > ParticleSimulation::GetRenderXMax()) || (ev.mouseX < ParticleSimulation::GetRenderXMin()) ||
			 (ev.mouseY > ParticleSimulation::GetRenderYMax()) || (ev.mouseY < ParticleSimulation::GetRenderYMin())  )

			mouseMotionX = ev.mouseX;
		if( (ev.mouseY > (mouseMotionY + 10.f)) || (ev.mouseY < (mouseMotionY - 10.f)) ||
		 	(ev.mouseY > ParticleSimulation::GetRenderYMax()) || (ev.mouseY < ParticleSimulation::GetRenderYMin()) ||
		 	(ev.mouseX > ParticleSimulation::GetRenderXMax()) || (ev.mouseX < ParticleSimulation::GetRenderXMin()) )
			mouseMotionY = ev.mouseY;

		float xRot = (mouseMotionX - ev.mouseX)*0.1f; //*time elapsed in frame
		float yRot = (mouseMotionY - ev.mouseY)*0.1f;

		camera.Rotate(xRot, yRot, 0.f);

		mouseMotionX = ev.mouseX;
		mouseMotionY = ev.mouseY;
	} 

	void PP_FBO::genVBO()
	{

		// Generate Vertex Buffer Object with space reserved for data, then insert interleaved data.
		int bufferSize = (particleList.size()*sizeof(particle_sim));
		std::cout << "buffer size: " << bufferSize << std::endl;
		glGenBuffers(1, &VBO);	
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STATIC_DRAW); //pass 0 to reserve space
		glBufferSubData(GL_ARRAY_BUFFER, 0, bufferSize, &particleList[0]); //enum,offset,size,start
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// Set intervals for drawing.
		byteInterval = (int)sizeof(particle_sim);
	}

	void PP_FBO::DrawGeom()
	{
		// Set 3d rendering viewport minimums (0 for drawing to texture)
		float viewXmin = 0;
		float viewYmin = 0;

		// Set 3d render viewport to scissor and clear
		glViewport(viewXmin,viewYmin, ParticleSimulation::GetXRes(), ParticleSimulation::GetYRes());
		glScissor(viewXmin,viewYmin, ParticleSimulation::GetXRes(),ParticleSimulation::GetYRes());

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Bind material
		material->Bind(camera.GetMVPMatrix());

		// Bind VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		// Set pointers to vertex data, colour data, r+I data and type data
    	glVertexAttribPointer(material->GetAttributeLocation("inPosition"), 3, GL_FLOAT, GL_TRUE, byteInterval, (void*)sizeof(particleList[0].e));
    	glVertexAttribPointer(material->GetAttributeLocation("inColor"), 3, GL_FLOAT, GL_TRUE, byteInterval, (void*)0);
    	glVertexAttribPointer(material->GetAttributeLocation("inRadius"), 1, GL_FLOAT, GL_TRUE, byteInterval, (void*)(sizeof(particleList[0].x)*6));
    	glVertexAttribPointer(material->GetAttributeLocation("inType"), 1, GL_UNSIGNED_SHORT, GL_FALSE, byteInterval, (void*)(sizeof(particleList[0].x)*8));

	   	// Draw
	    glDrawArrays(GL_POINTS, 0, particleList.size());

		// Unbind vbo
		glBindBuffer(GL_ARRAY_BUFFER, 0);		

		// Unbind material
		material->Unbind();

		//Check for error
		//PrintOpenGLError();
	}

}

