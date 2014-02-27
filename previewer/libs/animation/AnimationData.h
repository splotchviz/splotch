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

#ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONDATA
#define SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONDATA

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "previewer/libs/animation/AnimationPath.h"
#include "previewer/libs/core/ParticleSimulation.h"
#include "previewer/libs/core/Utils.h"

//Debug include
#include "previewer/libs/core/Debug.h"

namespace previewer
{
	// Creates a type indicating a list of way points
	typedef std::vector<AnimationPath> AnimationPathList;

	// Provides functionality that loads and saves animation data
	// from files in to the simulation format
	class AnimationData
	{
	public:

		static float GetMaxAnimationTime();

		void Load();

		AnimationPathList GetPaths();
		int GetTotalFrames();

		void InterpolatePaths();
		void AddPoint(std::string, std::string, std::string, int, int);
		void RemovePoint(std::string, std::string);

		bool LoadAnimDataFile(std::string);
		static void WriteAnimDataFile(std::string, AnimationPathList&);


	private:
		static AnimationPathList paths;

	};
}

#endif