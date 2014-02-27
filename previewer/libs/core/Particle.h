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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLE
#define SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLE

//Debug include
#include "previewer/libs/core/Debug.h"

#include "cxxsupport/vec3.h"
#include "kernel/colour.h"

namespace previewer
{
	// Struct to store information about 1 single particle
	// First line of file:
	// "Active:0","Active:1","Active:2","Coords:0","Coords:1","Coords:2","Intensity","Partition","PointIds","Radius","Rank","Scalars","Type"
	
	struct Particle
	{
		vec3f location;
		float dummy;
		float intensity;
		float type;
		COLOUR colour;
		float radius;
	};
}

#endif