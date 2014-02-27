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


#include "ParticleSimulation.h"
// Need previewer include, in cpp to avoid circular deps
#include "previewer/Previewer.h"

namespace previewer
{
	// Static decls
	int ParticleSimulation::pSimWindowWidth = 0;
	int ParticleSimulation::pSimWindowHeight = 0;

	int ParticleSimulation::renderWidth = 0;
	int ParticleSimulation::renderHeight = 0;
	int ParticleSimulation::renderXMin = 0;
	int ParticleSimulation::renderXMax = 0;
	int ParticleSimulation::renderYMin = 0;
	int ParticleSimulation::renderYMax = 0;
	float ParticleSimulation::aspectRatio = 0;
	int ParticleSimulation::fieldOfView = 0;

	int ParticleSimulation::moveSpeed = 200;
	int ParticleSimulation::rotationSpeed = 200;
	bool ParticleSimulation::rendererUpdated = true;
	int ParticleSimulation::xres = 0;
	int ParticleSimulation::yres = 0;

	bool ParticleSimulation::firstRun = true;
	IRenderer* ParticleSimulation::renderer;
	ParticleData ParticleSimulation::particles;
	std::string ParticleSimulation::exepath = "";


	void ParticleSimulation::Load(std::string _exepath)
	{
		DebugPrint("Particle Simulation being loaded with parameters");

		// Store path of executable
		exepath = _exepath;

		// Initial setup
		viewingImage = false;

		// Get resolution from parameter file
		paramfile* splotchParams =  Previewer::parameterInfo.GetParamFileReference();

		xres = splotchParams->find<int>("xres",800);
		yres = splotchParams->find<int>("yres",xres);
		aspectRatio = (float)xres/(float)yres;

		// Get fov from params, default to 30 if unavailable, user can change within app.
		fieldOfView = splotchParams->find<int>("fov",30);

      	// Setup render screen size parameters (force update)
		Update(true);

		// Load based on parameter file
		// Load particle data
		particles.Load();

		// Create instance of appropriate renderer, as specified in the makefile
#if defined RENDER_FF_VBO
		renderer = new FF_VBO();
#elif defined RENDER_PP_GEOM
		renderer = new PP_GEOM();
#elif defined RENDER_PP_FBO
		renderer = new PP_FBO();
#elif defined RENDER_PP_FBOF
		renderer = new PP_FBOF();
#endif
		DebugPrint("Render reference has been created");

		// Load renderer and pass particle data
		renderer->Load(particles);

		DebugPrint("Renderer has Loaded Particle Data");

		// Setup other renderer stuff here

	}

	void ParticleSimulation::Run()
	{
		// Update (do not force)
		Update(false);

		//DebugPrint("Running Particle Simulation frame");

		// Draw splotch created image if (viewingImage), otherwise draw simulation
		if(viewingImage)
			renderer->DrawImage(renderXMin, renderYMin, renderWidth, renderHeight);
		else
			renderer->Draw();
		
		//PrintOpenGLError();

	}

	void ParticleSimulation::Unload()
	{
		DebugPrint("Unloading Particle Simulation");
	}

	void ParticleSimulation::Update(bool forced)
	{
		int newWidth = WindowManager::GetSimulationWidth();
		int newHeight = WindowManager::GetSimulationHeight();

		// If the screen has changed, or it is the first run, update necessary variables		
		if(forced || (pSimWindowWidth != newWidth) || (pSimWindowHeight != newHeight))
		{
			// Get dimensions of particle simulation window
			pSimWindowWidth = newWidth;
			pSimWindowHeight = newHeight;

			// Compute dimensions and location of render viewport 
			if(aspectRatio>1) // xres>yres
			{
				renderWidth = pSimWindowWidth;
				renderHeight = (float)pSimWindowWidth / aspectRatio;
				renderXMin = 0;
				renderXMax = renderWidth;
				renderYMin = (pSimWindowHeight-renderHeight)/2;
				renderYMax = renderYMin + renderHeight;
			}
			else if(aspectRatio<1) // yres>xres
			{
				renderHeight = pSimWindowHeight;
				renderWidth = (float)pSimWindowHeight * aspectRatio;
				renderXMin = (pSimWindowWidth-renderWidth)/2;
				renderXMax = renderXMin + renderWidth;
				renderYMin = 0;
				renderYMax = renderHeight;
			}
			else //xres==yres
			{
				if(pSimWindowWidth>pSimWindowHeight)
				{
					renderWidth = pSimWindowHeight;
					renderHeight = pSimWindowHeight;	
				}
				else
				{
					renderWidth = pSimWindowWidth;
					renderHeight = pSimWindowWidth;	
				}			

				renderXMin = 0;
				renderXMax = renderWidth;
				renderYMin = 0;
				renderYMax = renderHeight;						
			}

			// Force renderer update
			rendererUpdated = false;
		}

		// If it is not the first run and renderer needs updating, update the renderer
		if(!firstRun && !rendererUpdated)
		{
			std::cout << "render update" << std::endl;
			renderer->Update();
			rendererUpdated = true;
		}

		firstRun = false;
	}

	int ParticleSimulation::GetRenderWidth()
	{
		return renderWidth;
	}

	int ParticleSimulation::GetRenderHeight()
	{
		return renderHeight;
	}
	
	int ParticleSimulation::GetRenderXMin()
	{
		return renderXMin;
	}

	int ParticleSimulation::GetRenderYMin()
	{
		return renderYMin;
	} 

	int ParticleSimulation::GetRenderXMax()
	{
		return renderXMax;
	}

	int ParticleSimulation::GetRenderYMax()
	{
		return renderYMax;
	}

	int ParticleSimulation::GetSimWindowWidth()
	{
		return pSimWindowWidth;
	}

	int ParticleSimulation::GetSimWindowHeight()
	{
		return pSimWindowHeight;
	}

	float ParticleSimulation::GetAspectRatio()
	{
		return aspectRatio;
	}

	int ParticleSimulation::GetFOV()
	{
		return fieldOfView;
	}

	void ParticleSimulation::SetFOV(int newFOV)
	{
		fieldOfView = newFOV;
		// Force renderer update
		rendererUpdated = false;
	}


	void ParticleSimulation::SetMoveSpeed(int newSpeed)
	{
		moveSpeed = newSpeed;
	}

	int ParticleSimulation::GetMoveSpeed()
	{
		return moveSpeed;
	}

	void ParticleSimulation::SetRotationSpeed(int newSpeed)
	{
		rotationSpeed = newSpeed;
	}

	int ParticleSimulation::GetRotationSpeed()
	{
		return rotationSpeed;
	}

	void ParticleSimulation::ReloadColourData()
	{

		particles.ReloadColourData();

		// Reload instance of appropriate renderer, as specified in the makefile
#if defined RENDER_FF_VBO
		renderer = new FF_VBO();
#elif defined RENDER_PP_GEOM
		renderer = new PP_GEOM();
#elif defined RENDER_PP_FBO
		renderer = new PP_FBO();
#elif defined RENDER_PP_FBOF
		renderer = new PP_FBOF();
#endif
		
		renderer->Load(particles);
	}

	void ParticleSimulation::SetPalette(std::string paletteFilename, int particleType)
	{
		particles.SetPalette(paletteFilename, particleType);
	}

	std::string ParticleSimulation::GetPalette(int particleType)
	{
		return particles.GetPalette(particleType);
	}
	void ParticleSimulation::SetXRes(int newxres, bool updateParams)
	{
		xres = newxres;
		UpdateResolution(updateParams);
	}

	void ParticleSimulation::SetYRes(int newyres, bool updateParams)
	{
		yres = newyres;
		UpdateResolution(updateParams);
	}

	int ParticleSimulation::GetXRes()
	{
		return xres;
	}

	int ParticleSimulation::GetYRes()
	{
		return yres;
	}

	void ParticleSimulation::UpdateResolution(bool updateParams)
	{
		std::cout << "xres: " << xres << " yres: " << yres << std::endl; 

		// Update aspect ratio
		aspectRatio = (float)xres/(float)yres;

		std::cout << "aspect ratio: " << aspectRatio << std::endl; 

		// Force update
		Update(true);

		if(updateParams)
		{
			//write to parameter file 
			paramfile* splotchParams =  Previewer::parameterInfo.GetParamFileReference();

			splotchParams->setParam<int>("xres",xres);
			splotchParams->setParam<int>("yres",yres);
		}
	}

	Camera& ParticleSimulation::GetCameraReference()
	{
		return renderer->GetCameraReference();
	}

	void ParticleSimulation::ViewImage(std::string file)
	{
		// check image exists first!
		viewingImage = true;
		renderer->LoadImage(file);

		// Set temporary resolutions for drawing image and do not update parameter with these.
		SetXRes(renderer->GetImageWidth(), false);
		SetYRes(renderer->GetImageHeight(), false);
	}

	void ParticleSimulation::StopViewingImage()
	{
		// Stop viewing image
		viewingImage = false;

		// Restore correct resolution from param file
		paramfile* splotchParams =  Previewer::parameterInfo.GetParamFileReference();
		SetXRes(splotchParams->find<int>("xres"), false);
		SetYRes(splotchParams->find<int>("yres"), false);
	}

	void ParticleSimulation::SetRenderBrightness(int type, float b)
	{
		renderer->SetRenderBrightness(type, b);
	}

	float ParticleSimulation::GetRenderBrightness(int type)
	{
		return renderer->GetRenderBrightness(type);
	}

	void ParticleSimulation::SetSmoothingLength(int type, float sl)
	{
		renderer->SetSmoothingLength(type, sl);
	}

	float ParticleSimulation::GetSmoothingLength(int type)
	{
		return renderer->GetSmoothingLength(type);
	}

	void ParticleSimulation::ResetCamera()
	{
		renderer->ResetCamera();
	}

	std::string ParticleSimulation::GetExePath()
	{
		return exepath;
	}


}