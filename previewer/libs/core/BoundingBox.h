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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_BOUNDINGBOX
#define SPLOTCH_PREVIEWER_LIBS_CORE_BOUNDINGBOX

//Debug include
#include "previewer/libs/core/Debug.h"

// Member includes
#include "../../../cxxsupport/vec3.h"
#include "../../../cxxsupport/arr.h"
#include "splotch/splotchutils.h"

namespace previewer
{
	// Type definition for a list of particles
	typedef std::vector<particle_sim> ParticleList;

	// Defines a structure for a bounding box (used to setup camera for the
	// data sets)

	struct BoundingBox
	{
		float minX;
		float maxX;
		float minY;
		float maxY;
		float minZ;
		float maxZ;

		vec3f centerPoint;

		// Compute and store bounding box parameters
		void Compute(const ParticleList& pData)
		{
			arr<Normalizer<float> > minmax(3);
			for(unsigned i = 0; i < pData.size(); i++)
			{
				minmax[0].collect(pData[i].x);
				minmax[1].collect(pData[i].y);
				minmax[2].collect(pData[i].z);
			}		

			//Store and display bounding box of data
			minX = minmax[0].minv;
			maxX = minmax[0].maxv;
			minY = minmax[1].minv;
			maxY = minmax[1].maxv;
			minZ = minmax[2].minv;
			maxZ = minmax[2].maxv;

			centerPoint = vec3f((maxX + minX)/2, (maxY + minY)/2, (maxZ + minZ)/2);
		}

	};
}

#endif