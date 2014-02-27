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

#include "AnimationData.h"

namespace previewer
{
	AnimationPathList AnimationData::paths;

	float AnimationData::GetMaxAnimationTime()
	{
		float max = 0;

		for(unsigned i = 0; i < paths.size(); i++)
		{
			max = (((paths[i].GetDataPointList()).back().time > max) ? (paths[i].GetDataPointList()).back().time : max);
		}

		return (max);
	}

	void AnimationData::Load()
	{

	}

	void AnimationData::InterpolatePaths()
	{
		for(unsigned i = 0; i < paths.size(); i++)
		{
			paths[i].InterpolatePoints();
		}
	}

	AnimationPathList AnimationData::GetPaths()
	{
		return paths;
	}

	int AnimationData::GetTotalFrames()
	{
		return (GetMaxAnimationTime() * 30);
	} 

	void AnimationData::AddPoint(std::string dataValue, std::string component, std::string parameter, int interpType, int timeValue)
	{
		// Create datapoint
		DataPoint dp;
		dp.value = dataValue;
		dp.interpType = interpType;
		dp.time = timeValue;

		// Check if a path of this type already exists, if so add datapoint
		bool found = false;

		for(unsigned i = 0; i < paths.size(); i++)
		{
			if(paths[i].GetComponent() == component && paths[i].GetParameter() == parameter)
			{
				found = true;
				paths[i].AddPoint(dp);
			}
		}

		// Create path if previousely non-existant
		if(!found)
		{
			AnimationPath newPath(component, parameter);
			newPath.AddPoint(dp);
			paths.push_back(newPath);
		}		
	}

	void AnimationData::RemovePoint(std::string component, std::string parameter)
	{
		// Check if a path of this type already exists, if so remove datapoint
		bool found = false;

		for(unsigned i = 0; i < paths.size(); i++)
		{
			if(paths[i].GetComponent() == component && paths[i].GetParameter() == parameter)
			{
				found = true;
				paths[i].RemovePoint();
			}
		}

		// Notify user if no path of that type exists
		if(!found)
			std::cout << "No path found matching component: " << component << " and parameter: " << parameter << std::endl;	
	}


	bool AnimationData::LoadAnimDataFile(std::string filename)
	{
		// // Open file, check for errors
		std::string filepath = ParticleSimulation::GetExePath()+"previewer/data/previewerAnimationPaths/"+filename;
		std::ifstream file( filepath.c_str(), std::ifstream::in );

		if(file.fail())
		 {
		  std::cout << "Animation file "+filename+" not found." << std::endl;
		  return false;
		}

		//Clear paths
		paths.erase(paths.begin(), paths.end());

		while(!file.eof())
		{
			// Temp storage
			AnimationPath newPath("","");
			// Get line
			std::string line;
			bool pathFound = false;
			while(getline(file,line))
			{
				// Check for '#' to indicate new path
				if(line[0] == '#')
				{
					// If this is not the first path, store current path and clear to start new path
					if(pathFound == true)
					{
						paths.push_back(newPath);
						for(int i = 0; i < newPath.Size()+2; i++)
							newPath.RemovePoint();
					}
					// Then get path info
					pathFound = true;
					int pathNum;
					char scratch;
					char component[20];
					char parameter[20];

					int ret = sscanf(line.c_str(), "%c %i %s %s", &scratch, &pathNum, component, parameter);
					if(ret != 4)
					{
						std::cout << "Animation file path info invalid..." << std::endl;
						return false;
					}
					std::string cmp(component);
					std::string prm(parameter);
					newPath.SetComponent(cmp);
					newPath.SetParameter(prm);
					std::cout << "Path number: " << pathNum << "component: " << newPath.GetComponent() << " parameter: " << newPath.GetParameter() << std::endl;
				}
				// If no initial valid path was found there was a problem with file...
				else if(pathFound == false)
				{
					if(paths.size() < 1)
					{
						// If no initial path has been found the file is incorrect
						std::cout << " Animation file is incorrect, please refer to previewer document for more info, or recreate path" << std::endl;
						return false;
					}
					else
						break;
				}
				else
				{
					// Read xyz data and add to current path
					float t;
					char v[10];
					int iType;

					int ret = sscanf(line.c_str(), "%f %s %i", &t, v, &iType);
					if(ret != 3)
					{
						std::cout << "Animation file path data invalid..." << std::endl;
						return false;
					}
					else
					{	
						std::string val(v);
						DataPoint dp;
						dp.time = t;
						dp.value = v;
						dp.interpType = iType;

						newPath.AddPoint(dp);

						std::cout << "Added point" << std::endl;				
					}					
				}
			}
		}
		return true;
	}

	void AnimationData::WriteAnimDataFile(std::string filename, AnimationPathList& animationPaths)
	{
		std::cout << "Writing path file ..." << std::endl;

		std::string filepath = ParticleSimulation::GetExePath()+"previewer/data/previewerAnimationPaths/"+filename;
		std::ofstream file;
		file.open( filepath.c_str(), std::ios::out | std::ios::trunc);	

		if(file.fail()) 
		{
			std::cout << "Could not open path file for writing '"+filepath+"'." << std::endl;
			return;
		}

		for(unsigned i = 0; i < animationPaths.size(); ++i)
		{
			file << "# " << i << "    " << animationPaths[i].GetComponent() << "    " << animationPaths[i].GetParameter() << std::endl;

			DataPointList dpl = animationPaths[i].GetDataPointList();

			for(unsigned j = 0; j < dpl.size(); j++)
			{
				file << dpl[j].time << "    " << dpl[j].value << "    " << dpl[j].interpType << std::endl;
			}

		}
		std::cout << "Animation file '"+filepath+"' written." << std::endl;		
	}

}



	// bool AnimationData::LoadAnimDataFile(std::string filename)
	// {
	// 	// // Open file, check for errors
	// 	std::string filepath = "previewer/data/previewerAnimationPaths/"+filename;
	// 	std::ifstream file( filepath.c_str(), std::ifstream::in );

	// 	if(file.fail())
	// 	 {
	// 	  std::cout << "Animation file "+filename+" not found." << std::endl;
	// 	  return false;
	// 	}

	// 	DebugPrint("Opened File");

	// 	//Clear paths
	// 	paths.erase(paths.begin(), paths.end());

	// 	while(file)
	// 	{
	// 		// Temp storage
	// 		int pathNum;
	// 		AnimationPath newPath("","");

	// 		// Get line as string
	// 		std::string s;
	// 		if(!std::getline(file, s, '@'))
	// 		{
	// 			DebugPrint("Couldnt get line ending in @");
	// 			std::cout << "Wrong format animation file, or reading finished" << std::endl;
	// 			return false;
	// 		}

	// 		// Create stringstream of line, get path number, component and parameter
	// 		std::stringstream line(s);
	// 		std::string element;
	// 		if(!std::getline(line, element, '#'))
	// 		{
	// 			DebugPrint("Couldnt get line ending in # for pathnum, component and parameter");
	// 			std::cout << "Wrong format animation file, or reading finished" << std::endl;
	// 			return false;
	// 		}
	// 		else 
	// 		{
	// 			// Temp storage
	// 	 		std::vector<std::string> data;

	// 			//fill animation path with component and param
	// 			std::stringstream ssElement(element);
	// 			for(int i = 0; i < 3; i++)
	// 			{	
	// 				// Get pathNum, component and param
	// 				std::string subElement;
	// 				if(!std::getline(ssElement, subElement, ','))
	// 				{
	// 					std::cout << "path size: " << paths.size() << std::endl;
	// 					std::cout << "Reading animation file finished!" << std::endl;
	// 					return true;
	// 				}	
	// 				else
	// 				{
	// 					data.push_back(subElement);
	// 				}		
	// 			}

	// 			// Store them
	// 			pathNum = Utils::atoi(data[0]);
	// 			newPath.SetComponent(data[1]);
	// 			newPath.SetParameter(data[2]);
	// 		}

	// 		std::cout << "Path number: " << pathNum << "component: " << newPath.GetComponent() << " parameter: " << newPath.GetParameter() << std::endl;

	// 		while(line)
	// 		{
	// 			//std::cout << "this happened" << std::endl;
	// 			std::string singleElement;
	// 			// Temp storage
	// 	 		std::vector<std::string> data;

	// 			if(!std::getline(line, singleElement, '#'))
	// 			{
	// 			}
	// 			else 
	// 			{
	// 			//fill display list with data

	// 				//get data
	// 				std::stringstream ssElement(singleElement);
	// 				for(int i = 0; i < 3; i++)
	// 				{	
	// 					// Get pathNum, component and param
	// 					std::string subElement;
	// 					if(!std::getline(ssElement, subElement, ','))
	// 					{
	// 						return true;
	// 					}	
	// 					else
	// 					{
	// 						data.push_back(subElement);
	// 					}		
	// 				}

	// 				DataPoint dp;
	// 				dp.time = Utils::atof(data[0]);
	// 				dp.value = data[1];
	// 				dp.interpType = Utils::atoi(data[2]);

	// 				newPath.AddPoint(dp);

	// 				std::cout << "Added point" << std::endl;
	// 			}				
	// 		}

	// 		paths.push_back(newPath);
	// 		std::cout << "added path" << std::endl;
	// 	}
	// 	return false;
	// }

	// void AnimationData::WriteAnimDataFile(std::string filename, AnimationPathList& animationPaths)
	// {
	// 	std::cout << "Writing path file ..." << std::endl;

	// 	std::string filepath = "previewer/data/previewerAnimationPaths/"+filename;
	// 	std::ofstream file;
	// 	// ios::out is assumed!
	// 	file.open( filepath.c_str(), std::ios::out | std::ios::trunc);
	// 	if(file.fail()) 
	// 	{
	// 		std::cout << "Could not open path file for writing '"+filepath+"'." << std::endl;
	// 		return;
	// 	}

	// 	for(uint i = 0; i < animationPaths.size(); ++i)
	// 	{
	// 		file << i << "," << animationPaths[i].GetComponent() << "," << animationPaths[i].GetParameter() << ",#";

	// 		DataPointList dpl = animationPaths[i].GetDataPointList();

	// 		for(uint j = 0; j < dpl.size(); j++)
	// 		{
	// 			file << dpl[j].time << "," << dpl[j].value << "," << dpl[j].interpType << ",#";
	// 		}

	// 		// Mark end of current path with @
	// 		file << "@\n";

	// 	}
	// 	std::cout << "Animation file '"+filepath+"' written." << std::endl;
	// }