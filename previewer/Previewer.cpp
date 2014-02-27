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
 
#include "Previewer.h"

namespace previewer
{
	Parameter Previewer::parameterInfo;

	void Previewer::Load(std::string paramFilePath)
	{
		// Get path of executable
		int ret;
		pid_t pid; 
		std::string exepath;

		#ifndef SPLOTCHMAC
		char pathbuf[1024];
		readlink("/proc/self/exe", pathbuf, 1024);
		// Remove executable name from path
		int len = 0;
		exepath = std::string(pathbuf);
		for(unsigned i = exepath.length()-1; i > 0; i--)
			if(exepath[i] == '/')
			{
				len = i;
				break;
			}
		exepath = exepath.substr(0,len+1);

		#else
		char pathbuf[PROC_PIDPATHINFO_MAXSIZE];
		pid = getpid();
		ret = proc_pidpath (pid, pathbuf, sizeof(pathbuf));
		if ( ret <= 0 ) 
		{
			std::cout << "Could not get path of exe.\n";
			std::cout << "PID: " << pid << std::endl;
			exit(0);
		} 
		else 
		{
			// Remove executable name from path
			int len = 0;
			exepath = std::string(pathbuf);
			for(unsigned i = exepath.length()-1; i > 0; i--)
				if(exepath[i] == '/')
				{
					len = i;
					break;
				}
			exepath = exepath.substr(0,len+1);
		}
		
		#endif


		DebugPrint("Previewer Loaded with Parameter File");
		DebugPrint("Parameter File:", paramFilePath);

		// Set application terminating flag
		isApplicationTerminating = false;

		// Go ahead and create local instance
		DebugPrint("Previewer::Load has indicated parameter file does exist");

		// Setup the parameter system
		parameterInfo.Load(paramFilePath);

		// Tell the particle simulation to load data
		particleSim.Load(exepath);

		// Load the animation simulation system
		animationSim.Load(exepath);

		return;
	}

	void Previewer::Run()
	{
		//DebugPrint("Previewer Running");
		WindowManager::UpdateFPS();

		// Run each controller
		parameterInfo.Run();
		particleSim.Run();
		animationSim.Run();

		return;
	}

	void Previewer::Unload()
	{
		DebugPrint("Previewer Unloading");

		// Unload all the components
		animationSim.Unload();
		particleSim.Unload();
		parameterInfo.Unload();

		DebugPrint("Preview Unloaded");

		return;
	}

	void Previewer::OnQuitApplication(Event)
	{
		DebugPrint("Terminating Application");

		isApplicationTerminating = true;
	}

	void Previewer::TriggerOnButtonPressEvent(std::string buttonPressed, float xPosition, float yPosition)
	{
		// Trigger On Button Press Event
		Event event;

		// Set the event type to button press
		event.eventType = evButtonPress;

		// Store the button pressed as a string
		event.keyID = buttonPressed;

		// Store cursor location at button click
		event.mouseX = xPosition;
		event.mouseY = yPosition;

		// Store translated cursor at button click
		event.translatedMouseX = xPosition;
		event.translatedMouseY = screenSize.y - yPosition; // (swap axis)

		// Call the event handler
		OnButtonPressEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnButtonReleaseEvent(std::string buttonReleased, float xPosition, float yPosition)
	{
		// Trigger on Button Release Event
		Event event;

		// Set the event type to button release
		event.eventType = evButtonRelease;

		// Store the button released as a string
		event.keyID = buttonReleased;

		// Store cursor location at button click
		event.mouseX = xPosition;
		event.mouseY = yPosition;

		// Store translated cursor at button click
		event.translatedMouseX = xPosition;
		event.translatedMouseY = screenSize.y - yPosition; // (swap axis)

		// Call the event handler
		OnButtonReleaseEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnExposedEvent()
	{
		// Trigger on Exposed Event
		Event event;

		// Set event type to exposed
		event.eventType = evExposed;

		// Call the event handler
		OnExposedEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnKeyPressEvent(std::string keyPressed)
	{
		// Trigger on Key Press Event
		Event event;

		// Set event type to key press
		event.eventType = evKeyPress;

		// Mouse element is not used
		event.mouseX = 0;
		event.mouseY = 0;

		// Setup the event
		event.keyID = keyPressed;

		// Call the event handler
		OnKeyPressEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnKeyReleaseEvent(std::string keyReleased)
	{
		// Trigger on Key Release Event
		Event event;

		// Set event type to key release
		event.eventType = evKeyRelease;

		// Mouse element is not used
		event.mouseX = 0;
		event.mouseY = 0;

		// Setup the event
		event.keyID = keyReleased;

		// Call the event handler
		OnKeyReleaseEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnMotionEvent(float xPosition, float yPosition)
	{
		// Trigger on Motion Event
		Event event;

		// Set the event type and keyID to mouse motion
		event.eventType = evMouseMotion;
		event.keyID = "Motion";

		// Store the new cursor position
		event.mouseX = xPosition;
		event.mouseY = yPosition;
		
		// Store translated cursor at button click
		event.translatedMouseX = xPosition;
		event.translatedMouseY = screenSize.y - yPosition; // (swap axis);

		OnMotionEvent::CallEvent(event);

		return;
	}

	void Previewer::TriggerOnQuitApplicationEvent()
	{
		// Trigger on Quit Application Event
		Event event;

		// Set the event type to quit the application
		event.eventType = evQuitApplication;

		// Call the event handler
		OnQuitApplicationEvent::CallEvent(event);

		return;
	}

	void Previewer::SetRenderWidth(int width)
	{
		screenSize.x = width;

		WindowManager::SetSimulationSize(screenSize.x, screenSize.y);
	}
	void Previewer::SetRenderHeight(int height)
	{
		screenSize.y = height;

		WindowManager::SetSimulationSize(screenSize.x, screenSize.y);
	}
	void Previewer::SetRenderSize(int width, int height)
	{
		screenSize.x = width;
		screenSize.y = height;

		WindowManager::SetSimulationSize(screenSize.x, screenSize.y);
	}
	void Previewer::SetRenderPositionX(int x)
	{
		screenPosition.x = x;

		WindowManager::SetSimulationPosition(screenPosition.x, screenPosition.y);
	}
	void Previewer::SetRenderPositionY(int y)
	{
		screenPosition.y = y;

		WindowManager::SetSimulationPosition(screenPosition.x, screenPosition.y);
	}
	void Previewer::SetRenderPosition(int x, int y)
	{
		screenPosition.x = x;
		screenPosition.y = y;

		WindowManager::SetSimulationPosition(screenPosition.x, screenPosition.y);
	}

	void Previewer::WriteParameterFile(std::string outpath)
	{
		// Update parameter values
		Camera cam = particleSim.GetCameraReference();
		vec3f camPos = cam.GetCameraPosition();
		vec3f camLookAt = cam.GetLookAt();
		vec3f camUp =  cam.GetUpVector();
		paramfile* param = parameterInfo.GetParamFileReference();
		param->setParam<double>("camera_x",camPos.x);
		param->setParam<double>("camera_y",camPos.y);
		param->setParam<double>("camera_z",camPos.z);
		// Use campos - lookat as our lookat is normalised camera-relative coords 
		// and splotch expects non-normalised world-relative coords. 
		param->setParam<double>("lookat_x",(camPos.x - camLookAt.x));
		param->setParam<double>("lookat_y",(camPos.y - camLookAt.y));
		param->setParam<double>("lookat_z",(camPos.z - camLookAt.z));
		// Invert sky vector as previewer and splotch camera setups are slightly different
		param->setParam<double>("sky_x",(camUp.x * -1));
		param->setParam<double>("sky_y",(camUp.y * -1));
		param->setParam<double>("sky_z",(camUp.z * -1));

		// Get render brightness values, for max 10 ptypes
	 	for(unsigned i = 0; i < std::min(param->find<int>("ptypes", 1), 10); i++)
	 	{
	 		param->setParam<float>("brightness"+Utils::ToString(i), particleSim.GetRenderBrightness(i));
	 		param->setParam<float>("size_fix"+Utils::ToString(i), particleSim.GetSmoothingLength(i));
	 	}
		// Get Parameter to write out the values
		parameterInfo.WriteParameterFile(outpath);
	}

	void Previewer::WriteSplotchAnimationFile(std::string outpath)
	{
		animationSim.WriteSplotchAnimFile(outpath);
	}

	void Previewer::ReloadColourData()
	{
		particleSim.ReloadColourData();
	}

	void Previewer::SetPalette(std::string paletteFilename, int particleType)
	{
		particleSim.SetPalette(paletteFilename, particleType);
	}

	std::string Previewer::GetPalette(int particleType)
	{
		return particleSim.GetPalette(particleType);
	}

	double Previewer::GetFPS()
	{
		return WindowManager::GetFPS();
	}

	void Previewer::SetAnimationPoint(int time)
	{
		animationSim.SetTime(time);
		animationSim.SetPoint();
	}

	void Previewer::RemoveAnimationPoint()
	{
		animationSim.RemovePoint();
	}

	void Previewer::SetMovementSpeed(int speed)
	{
		particleSim.SetMoveSpeed(speed);
	}

	void Previewer::SetRotationSpeed(int speed)
	{
		particleSim.SetRotationSpeed(speed);
	}

	void Previewer::PreviewAnimation()
	{
		animationSim.Preview();
	}

	void Previewer::SaveAnimationFile(std::string filename)
	{
		animationSim.SetFilename(filename);
		animationSim.WriteAnimFile();
	}

	void Previewer::LoadAnimationFile(std::string filename)
	{
		animationSim.SetFilename(filename);
		animationSim.LoadAnimFile();
	}

	void Previewer::UnloadAnimationFile()
	{
		animationSim.UnloadCurrentAnim();
	}

	void Previewer::SetXRes(int xRes)
	{
		particleSim.SetXRes(xRes, true);
	}

	void Previewer::SetYRes(int yRes)
	{
		particleSim.SetYRes(yRes, true);
	}

	void Previewer::SetFOV(int fov)
	{
		particleSim.SetFOV(fov);
	}

	void Previewer::Interpolate()
	{
		animationSim.Interpolate();
	}

	void Previewer::SetCameraInterpolation(int newInterp)
	{
		animationSim.SetCameraInterpolation(newInterp);
	}

	void Previewer::SetLookatInterpolation(int newInterp)
	{
		animationSim.SetLookatInterpolation(newInterp);
	}

	void Previewer::ViewImage(std::string file)
	{
		particleSim.ViewImage(file);
	}

	void Previewer::StopViewingImage()
	{
		particleSim.StopViewingImage();
	}

	void Previewer::SetRenderBrightness(int type, float b)
	{
		particleSim.SetRenderBrightness(type,b);
	}

	float Previewer::GetRenderBrightness(int type)
	{
		return particleSim.GetRenderBrightness(type);
	}

	void Previewer::SetSmoothingLength(int type, float sl)
	{
		particleSim.SetSmoothingLength(type,sl);
	}

	float Previewer::GetSmoothingLength(int type)
	{
		return particleSim.GetSmoothingLength(type);
	}

	void Previewer::SetParameter(std::string name, std::string value)
	{
		paramfile* param = parameterInfo.GetParamFileReference();
		param->setParam<std::string>(name,value);		
	}


	// Note this is only useful for displaying parameters. They are not converted to their original 
	// format, they are kept as string
	std::string Previewer::GetParameter(std::string name)
	{
		paramfile* param = parameterInfo.GetParamFileReference();
		return param->find<std::string>(name);		
	}

	void Previewer::ResetCamera()
	{
		particleSim.ResetCamera();
	}

}