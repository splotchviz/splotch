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

#include "AnimationSimulation.h"
#include "previewer/Previewer.h"
namespace previewer
{
	// Static decls
	float AnimationSimulation::fps = 30;
	int AnimationSimulation::timeValue = 0;
	std::string AnimationSimulation::exepath = "";

	void AnimationSimulation::Load(std::string _exepath)
	{
		DebugPrint("Animation Simulation has been loaded");

		exepath = _exepath;
		// Reset frame and demo status
		currentFrame = 0;
		isActive = false;
		demoPlaying = false;
		fileReady = false;
		cameraInterpType = CUBIC;
		lookatInterpType = LINEAR;

		// Default filename
		currentFile = "animation";

		//IInterpolation* interp = new CubicCurveInterpolation(0.0, 10.0, 0.0, 100.0);
		animation.Load();
	}

	void AnimationSimulation::Run()
	{
		//DebugPrint("Running Animation Simulation Frame");
		if(demoPlaying)
			PlayDemo();

	}

	void AnimationSimulation::Unload()
	{
		DebugPrint("Unloading Animation Simulation");
	}

	void AnimationSimulation::PlayDemo()
	{

		DebugPrint("Demo is playing, frame:", currentFrame);
		DebugPrint("Out of total: ", totalFrames);

		// Set previewer settings relative to animation path current frame
		vec3f newLookAt = vec3f(0.0,0.0,0.0);

			for(unsigned i = 0; i < animationPaths.size() ; i++)
			{
				if(animationPaths[i].GetComponent() == "camera")
				{
					DebugPrint("camera component detected - animpath: ", (int)i );
					if(animationPaths[i].GetParameter() == "camera_x")
					{
						DebugPrint("parameter posX");
						CameraAction::CallMoveCameraAction(vec3f(std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str()), 0, 0), XCOORD);
					}
					else if(animationPaths[i].GetParameter() == "camera_y")
					{
						DebugPrint("parameter posY");
						CameraAction::CallMoveCameraAction(vec3f(0, std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str()), 0), YCOORD);
					}
					else if(animationPaths[i].GetParameter() == "camera_z")
					{
						DebugPrint("parameter posZ");
						CameraAction::CallMoveCameraAction(vec3f(0, 0, std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str())), ZCOORD);
					}
					else if(animationPaths[i].GetParameter() == "lookat_x")
					{
						DebugPrint("parameter lookAtX");
						newLookAt.x = std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str());
					}
					else if(animationPaths[i].GetParameter() == "lookat_y")
					{
						DebugPrint("parameter lookAtY");
						newLookAt.y = std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str());
					}
					else if(animationPaths[i].GetParameter() == "lookat_z")
					{
						DebugPrint("parameter lookAtZ");
						newLookAt.z = std::atof(animationPaths[i].GetInterpolatedDataPointList()[currentFrame].value.c_str());
					}
				}
			}

			// If lookat has been changed, set new lookat (as target rather than normalised directional vec)
			if(newLookAt.x != 0.0 || newLookAt.y != 0.0 || newLookAt.z != 0.0)
			{
				CameraAction::CallSetCameraLookAtAction(newLookAt);
			}

			currentFrame++;

			// Reset current frame at end of demo to play again
			if(currentFrame >= totalFrames)
			{
				currentFrame = 0;
				demoPlaying = false;
			}
	}

	// Write animation file
	void AnimationSimulation::WriteAnimFile()
	{
		if(!fileReady)
		{
			std::cout << "You need to create an animation before you can write it to file!" << std::endl;
			return;
		}
		else
			std::cout << "Writing File" << std::endl;

		// Write file

		if(!animationPaths.empty())
		{
			AnimationData::WriteAnimDataFile(currentFile, animationPaths);
		}
		else 
			std::cout << "Animation paths list is empty..." << std::endl;

	}

	// Load animation file
	void AnimationSimulation::LoadAnimFile()
	{
		DebugPrint("Loading file");
		if(animation.LoadAnimDataFile(currentFile))
		{
			DebugPrint("File Loaded");
			animationPaths = animation.GetPaths();
			DebugPrint("Got Paths");
			totalFrames = animationPaths[0].GetInterpolatedDataPointList().size();
			DebugPrint("Got total framecount: ", totalFrames);
			//fileReady = true;
		}
		else
			std::cout << "File loading failed." << std::endl;

	}

	void AnimationSimulation::WriteSplotchAnimFile(std::string filename)
	{
		// Check there is an animation ready to be written
		if(!fileReady)
		{
			std::cout << "You need to create an animation before you can write it to file!" << std::endl;
			return;
		}

		std::cout << "Writing splotch animation file ..." << std::endl;

		// Create requested file
		std::string filepath = exepath+"previewer/data/splotchAnimationPaths/"+filename+".txt";
		std::ofstream file;

		// Open file for writing
		file.open( filepath.c_str(), std::ios::out | std::ios::trunc);
		if(file.fail()) 
		{
			std::cout << "Could not open path file for writing '"+filepath+"'." << std::endl;
			return;
		}

		std::cout << "File successfully opened" << std::endl;

		// Write list of parameters on first line
		for(unsigned i = 0; i < animationPaths.size(); ++i)
		{
			file << animationPaths[i].GetParameter() << " ";
		}

		// Add fidx
		file << "fidx"; 

		std::cout << "Written params to first line" << std::endl;

		// New line...
		file << "\n";

		// Get list of interpolated points and reformat them for splotch 
		// Ie invert sky vector and change lookat from normalised vector to target location
		std::vector<DataPointList> dplVec; 


		for(unsigned i = 0; i < animationPaths.size(); i++)
		{
			// Invert sky vector for splotch
			// Check which path we are on
			std::string parameter = animationPaths[i].GetParameter();

			// Get it's data point list
			DataPointList dpl = animationPaths[i].GetInterpolatedDataPointList();

			// Check if the path is path of the sky vector's z component
			if(parameter == "sky_x" || parameter == "sky_y" || parameter == "sky_z")
			{
				// If so, invert all the values in the path
				for (unsigned j = 0; j < dpl.size(); j++)
				{
					float invert = (Utils::atof(dpl[j].value) * -1);
					dpl[j].value = Utils::ToString(invert);
				}
			}

			// Convert lookat from normalised vector to target location
			if(parameter == "lookat_x")
			{
				// Find corresponding camera path
				for(unsigned j = 0; j < animationPaths.size(); j++)
				{	
					// Replace lookat with camera - lookat
					std::string parameter2 = animationPaths[j].GetParameter();
					if(parameter2 == "camera_x")
					{
						DataPointList cameraDpl = animationPaths[j].GetInterpolatedDataPointList();
						for(unsigned k = 0; k < dpl.size(); k++)
							dpl[k].value = Utils::ToString(Utils::atof(cameraDpl[k].value) - Utils::atof(dpl[k].value));
					}
				}			
			}
			else if(parameter == "lookat_y")
			{

				// Find corresponding camera path
				for(unsigned j = 0; j < animationPaths.size(); j++)
				{	
					// Replace lookat with camera - lookat
					std::string parameter2 = animationPaths[j].GetParameter();
					if(parameter2 == "camera_y")
					{
						DataPointList cameraDpl = animationPaths[j].GetInterpolatedDataPointList();
						for(unsigned k = 0; k < dpl.size(); k++)
							dpl[k].value = Utils::ToString(Utils::atof(cameraDpl[k].value) - Utils::atof(dpl[k].value));
					}				
				}	
			}
			else if(parameter == "lookat_z")
			{
				// Find corresponding camera path
				for(unsigned j = 0; j < animationPaths.size(); j++)
				{	
					// Replace lookat with camera - lookat
					std::string parameter2 = animationPaths[j].GetParameter();
					if(parameter2 == "camera_z")
					{
						DataPointList cameraDpl = animationPaths[j].GetInterpolatedDataPointList();
						for(unsigned k = 0; k < dpl.size(); k++)
							dpl[k].value = Utils::ToString(Utils::atof(cameraDpl[k].value) - Utils::atof(dpl[k].value));
					}				
				}	
			}

			// Push dpl into vector
			dplVec.push_back(dpl);
		}

		std::cout << "Got interpolated + reformatted data point list" << std::endl;

		// Get fidx from parameter file
		paramfile* splotchParams =  Previewer::parameterInfo.GetParamFileReference();

		double fidx = splotchParams->find<double>("fidx",0.0);

		// Write list of values for every element in the vector
		for(unsigned i = 0; i < dplVec[0].size(); i++)
		{
			// For each member write a value for each path in the dplVec
			for(unsigned j = 0; j < dplVec.size(); j++)
			{

				file << "     " << dplVec[j][i].value;
			}

			// Write current file index and skip to next line (unless last line)
			file << "     " << fidx;

			if(i!=dplVec[0].size()-1) 
				file << "\n";
		}

		std::cout << "Splotch animation file '"+filepath+"' written." << std::endl;
	}

	void AnimationSimulation::SetAnimationPoint(std::string dataPoint, std::string component, std::string parameter, int interpType)
	{
		// Add point to list
		DebugPrint("Adding Point");
		animation.AddPoint(dataPoint, component, parameter, interpType, timeValue);
	}

	void AnimationSimulation::Interpolate()
	{
			// Interpolate and store interpolated paths
			DebugPrint("Animation Simulation: Interpolating");
			animation.InterpolatePaths();
			animationPaths = animation.GetPaths();
			totalFrames = animation.GetTotalFrames();
			fileReady = true;
	}

	void AnimationSimulation::SetFilename(std::string newName)
	{
		currentFile = newName;
	}

	std::string AnimationSimulation::GetFilename()
	{
		return currentFile;
	}

	void AnimationSimulation::SetFPS(float newFPS)
	{
		fps = newFPS;
	}

	float AnimationSimulation::GetFPS()
	{
		return fps;
	}

	void AnimationSimulation::SetTime(int newTime)
	{
		timeValue = newTime;
	}

	float AnimationSimulation::GetTime()
	{
		return timeValue;
	}

	void AnimationSimulation::SetPoint()
	{
		DebugPrint("Setting points for all animation elements");

		DebugPrint("Setting Animation Point: camposx");
		vec3f campos = vec3f(0.0,0.0,0.0);
		CameraAction::CallGetCameraPosAction(campos);
		SetAnimationPoint(Utils::ToString(campos.x), "camera", "camera_x", cameraInterpType);
		DebugPrint("Setting Animation Point: camposy type: ", cameraInterpType);
		campos = vec3f(0.0,0.0,0.0);
		CameraAction::CallGetCameraPosAction(campos);
		SetAnimationPoint(Utils::ToString(campos.y), "camera", "camera_y", cameraInterpType);
		DebugPrint("Setting Animation Point: camposz");
		campos = vec3f(0.0,0.0,0.0);
		CameraAction::CallGetCameraPosAction(campos);
		SetAnimationPoint(Utils::ToString(campos.z), "camera", "camera_z", cameraInterpType);
		DebugPrint("Setting animation points: lookat");
		vec3f camLookAt;
		CameraAction::CallGetCameraLookAtAction(camLookAt);
		SetAnimationPoint(Utils::ToString(camLookAt.x), "camera", "lookat_x", lookatInterpType);
		SetAnimationPoint(Utils::ToString(camLookAt.y), "camera", "lookat_y", lookatInterpType);
		SetAnimationPoint(Utils::ToString(camLookAt.z), "camera", "lookat_z", lookatInterpType);	
		vec3f camUpVec;
		CameraAction::CallGetCameraUpAction(camUpVec);
		SetAnimationPoint(Utils::ToString(camUpVec.x), "camera", "sky_x", lookatInterpType);
		SetAnimationPoint(Utils::ToString(camUpVec.y), "camera", "sky_y", lookatInterpType);
		SetAnimationPoint(Utils::ToString(camUpVec.z), "camera", "sky_z", lookatInterpType);	
	}

	void AnimationSimulation::RemovePoint()
	{
			// Remove last set point for all paths set in "SetPoint" method
			animation.RemovePoint("camera", "camera_x");
			animation.RemovePoint("camera", "camera_y");
			animation.RemovePoint("camera", "camera_z");
			animation.RemovePoint("camera", "lookat_x");
			animation.RemovePoint("camera", "lookat_y");
			animation.RemovePoint("camera", "lookat_z");
			animation.RemovePoint("camera", "sky_x");
			animation.RemovePoint("camera", "sky_y");
			animation.RemovePoint("camera", "sky_z");
	}

	void AnimationSimulation::Preview()
	{
		// Play demo on keypress
		if(!demoPlaying && fileReady)
			demoPlaying = true;
	}

	void AnimationSimulation::UnloadCurrentAnim()
	{
		fileReady = false;
	}

	void AnimationSimulation::SetCameraInterpolation(int newInterp)
	{
		cameraInterpType = newInterp;
	}

	void AnimationSimulation::SetLookatInterpolation(int newInterp)
	{
		lookatInterpType = newInterp;
	}
}
