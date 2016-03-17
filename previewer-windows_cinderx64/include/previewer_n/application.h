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

#ifndef SPLOTCH_PREVIEWER_PREVIEWER
#define SPLOTCH_PREVIEWER_PREVIEWER

// Member includes
#include "libs/core/Parameter.h"
#include "libs/core/ParticleData.h"
#include "cxxsupport/vec3.h"

// Usage includes
#include "cxxsupport/paramfile.h"

#include <string>
#include <stdlib.h>
#include <errno.h>
#include <Windows.h>

// Usage includes
#include "libs/core/FileLib.h"

namespace previewer
{
	// Provides the main entry point to the application from splotch. Also acts as
	// the main running code that is controlling the flow of the application
	class Previewer
	{
	public:
		ParticleData Load();
		paramfile* LoadParameterFile(std::string);


		static Parameter parameterInfo;

	private:
		ParticleData particles;
	};
}

#endif