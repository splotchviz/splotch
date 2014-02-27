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

#include "FF_VBO.h"
#include "previewer/Previewer.h"

namespace previewer
{
	void FF_VBO::Load(const ParticleData& pData)
	{
		//Store passed in data and data bounding box
		particleList = pData.GetParticleList();
		dataBBox = pData.GetBoundingBox();

		DebugPrint("Generating VBO...");
		// Create the display list for particle drawing
		genVBO();

		DebugPrint("Generated");
		// Set up GUI

		// Set up material
		material = new FF_ParticleMaterial();
		material->Load();
		material->SetBlend(true);
		material->SetBlendSrc(GL_SRC_ALPHA);
		material->SetBlendDst(GL_ONE);

		// Load and position camera
		camera.SetPerspectiveProjection(ParticleSimulation::GetFOV(), ParticleSimulation::GetAspectRatio(), 1, 200000);

		// Check if recalculation is required
		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();
		bool recalc = splotchParams->find<bool>("pv_recalc_cam",true);
		if(recalc)
			camera.Create(dataBBox);
		else
		{
			vec3f lookat(splotchParams->find<double>("lookat_x"),splotchParams->find<double>("lookat_y"),splotchParams->find<double>("lookat_z"));
			vec3f sky(splotchParams->find<double>("sky_x",0),splotchParams->find<double>("sky_y",0),splotchParams->find<double>("sky_z",1));
			vec3f campos(splotchParams->find<double>("camera_x",0),splotchParams->find<double>("camera_y",0),splotchParams->find<double>("camera_z",1));
			// sky *= -1 as splotch sky is inverted compared to ours
			camera.Create(campos,lookat,(sky*=-1));
		}

		camera.SetMainCameraStatus(true);
		
	}

	void FF_VBO::Draw()
	{
		glEnable(GL_SCISSOR_TEST);

		glScissor(0,0, ParticleSimulation::GetSimWindowWidth(),ParticleSimulation::GetSimWindowHeight());
		Clear(0.2,0.2,0.2,1.0);
		
		// Set 3d rendering viewport size 
		float viewXmin = ParticleSimulation::GetRenderXMin();
		float viewYmin = ParticleSimulation::GetRenderYMin();

		// Set 3d render viewport to scissor and clear
		glViewport(viewXmin,viewYmin, ParticleSimulation::ParticleSimulation::GetRenderWidth(), ParticleSimulation::GetRenderHeight());
		glScissor(viewXmin,viewYmin, ParticleSimulation::GetRenderWidth(),ParticleSimulation::GetRenderHeight());

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Set mMode
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();

		// Set Projection
		Matrix4 projection = camera.GetProjectionMatrix();
		glLoadMatrixf(&projection[0][0]);

		// Switch mModes and prepare to draw
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glPushMatrix();
		
		// Load cameras view matrix
		Matrix4 view = camera.GetViewMatrix();
		glLoadMatrixf(&view[0][0]);

		// Bind material
		material->Bind();

		// Bind VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		// Set vertex/colour pointers
  		glVertexPointer( 3, GL_FLOAT, 0, 0 );
 		glColorPointer(  3, GL_FLOAT, 0, (void*)(vertices.size()*sizeof(vec3f)) );

 		// Enable client states
	    glEnableClientState(GL_VERTEX_ARRAY);
	    glEnableClientState(GL_COLOR_ARRAY);

	   	// Draw Particles
	    glDrawArrays(GL_TRIANGLES, 0, vertices.size() );

	    //Disable client states
	    glDisableClientState(GL_VERTEX_ARRAY);
	    glDisableClientState(GL_COLOR_ARRAY);

		// Unbind vbo
		glBindBuffer(GL_ARRAY_BUFFER, 0);		

		// Unbind material
		material->Unbind();

		glPopMatrix();

		//Check for error
		//PrintOpenGLError();

		// Draw labels
		glDisable(GL_SCISSOR_TEST);
	}

	void FF_VBO::Unload()
	{

	}

	void FF_VBO::Update()
	{
		camera.SetPerspectiveProjection(ParticleSimulation::GetFOV(), ParticleSimulation::GetAspectRatio(), 1, 200000);		
	}

	void FF_VBO::OnKeyPress(Event ev)
	{
		// Movement * telapsed for uniform movement regardless of framerate 
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

	void FF_VBO::OnMotion(Event ev)
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

	void FF_VBO::genVBO()
	{

	vertices.resize(particleList.size()*12);
	colours.resize(particleList.size()*12);
	int i = 0;
	for(int j = 0; j < particleList.size();j++, i += 12)
	{
		for(int k = i; k < (i + 12); k++)
		{
			colours[k].x = particleList[j].e.r;
			colours[k].y = particleList[j].e.g;
			colours[k].z = particleList[j].e.b;

			colours[k] *= 0.01;
		}

		float radius = 70;//(colours[j].x * colours[j].y) / 2;
	
		vertices[i].x = particleList[j].x;
		vertices[i].y = particleList[j].y + radius;
		vertices[i].z = particleList[j].z;

		vertices[i+1].x = particleList[j].x - radius;
		vertices[i+1].y = particleList[j].y - radius;
		vertices[i+1].z = particleList[j].z;

		vertices[i+2].x = particleList[j].x;
		vertices[i+2].y = particleList[j].y;
		vertices[i+2].z = particleList[j].z + radius;

		vertices[i+3].x = particleList[j].x - radius;
		vertices[i+3].y = particleList[j].y - radius;
		vertices[i+3].z = particleList[j].z;

		vertices[i+4].x = particleList[j].x + radius;
		vertices[i+4].y = particleList[j].y - radius;
		vertices[i+4].z = particleList[j].z;

		vertices[i+5].x = particleList[j].x;
		vertices[i+5].y = particleList[j].y;
		vertices[i+5].z = particleList[j].z + radius;

		vertices[i+6].x = particleList[j].x + radius;
		vertices[i+6].y = particleList[j].y - radius;
		vertices[i+6].z = particleList[j].z;

		vertices[i+7].x = particleList[j].x;
		vertices[i+7].y = particleList[j].y + radius;
		vertices[i+7].z = particleList[j].z;

		vertices[i+8].x = particleList[j].x;
		vertices[i+8].y = particleList[j].y;
		vertices[i+8].z = particleList[j].z + radius;

		vertices[i+9].x = particleList[j].x - radius;
		vertices[i+9].y = particleList[j].y - radius;
		vertices[i+9].z = particleList[j].z;

		vertices[i+10].x = particleList[j].x + radius;
		vertices[i+10].y = particleList[j].y - radius;
		vertices[i+10].z = particleList[j].z;

		vertices[i+11].x = particleList[j].x;
		vertices[i+11].y = particleList[j].y + radius;
		vertices[i+11].z = particleList[j].z;
	}

	// Generate Vertex Buffer Object with space reserved for data, then insert data.

	int bufferSize = (vertices.size() + colours.size())*sizeof(vec3f);

	glGenBuffers(1, &VBO);	
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STATIC_DRAW); //pass 0 to reserve space
	glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size()*sizeof(vec3f), &vertices[0].x); //enum,offset,size,start
	glBufferSubData(GL_ARRAY_BUFFER, vertices.size()*sizeof(vec3f), colours.size()*sizeof(vec3f), &colours[0].x);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	}

}

