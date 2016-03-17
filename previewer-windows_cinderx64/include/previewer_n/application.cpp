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

#include "application.h"

namespace previewer
{
	Parameter Previewer::parameterInfo;

	previewer::ParticleData Previewer::Load()
	{
		
		char pathbuf[MAX_PATH];
		GetModuleFileNameA(NULL, pathbuf, MAX_PATH);
		// Remove executable name from path
		int len = 0;
		std::string exepath(pathbuf);
		
		for (unsigned i = exepath.length() - 1; i > 0; i--)
		{
			if (exepath.at(i) == '\\')
			{
				exepath.erase(exepath.begin() + i + 1, exepath.end());
				break;
			}
		}
		
		// Tell the particle simulation to load data
		//paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();
		particles.Load();

		return particles;
	}

	paramfile* Previewer::LoadParameterFile(std::string paramFilePath)
	{		
		parameterInfo.Load(paramFilePath);
		return parameterInfo.GetParamFileReference();
	}
}