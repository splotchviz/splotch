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

#include "Parameter.h"

namespace previewer
{
		void Parameter::Load(std::string filename)
		{
			DebugPrint("Loading parameter file supplied");
			DebugPrint("Parameter File:", filename);

			paramFileReference = paramfile(filename, false);

			DebugPrint("Splotch Parameter File Read Completed");
		}

		void Parameter::Load()
		{
			DebugPrint("Loading parameter with no file supplied");
		}

		void Parameter::Run()
		{
			//DebugPrint("Running parameter frame");
		}

		void Parameter::Unload()
		{
			DebugPrint("Unloading parameter system");
		}

		paramfile* Parameter::GetParamFileReference()
		{
			return &paramFileReference;
		}

		void Parameter::WriteParameterFile(std::string fileName)
		{
			DebugPrint("Writing Parameter file");

			// Get Out File Stream and Open the File
			std::ofstream file;
			file.open(fileName.c_str(), std::ios::trunc);

			if(file.is_open())
			{
				// Get parameters
				params_type params = GetParamFileReference()->getParams();

				// Loop through and output all parameters
				for(params_type::iterator it = params.begin(); it != params.end(); it++)
				{
					file << it->first << "=" << it->second << std::endl;
				}

				// Close the file
				file.close();
			}
		}

		void Parameter::SetFilePath(std::string)
		{

		}
}