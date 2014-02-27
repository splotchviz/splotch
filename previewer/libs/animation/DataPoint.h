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

 #ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONPARAMETER
 #define SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONPARAMETER

//Debug include
#include "previewer/libs/core/Debug.h"

#include <string>

namespace previewer
{
	// Here we define the structure of a datapoint, multiple times for each type of datapoint used plus
	// a generic string type that will be used when transporting sets of datapoints

	enum {
		LINEAR,
		CUBIC,
		CUBIC_HERMITE_KB
	};

	struct DataPoint
	{
		// Current value of the data point
		std::string value;

		// Type of interpolation to be carried out between this point and the next
		int interpType;

		// Current time (in ms) the point represents in the animation 
		float time;
	};

	struct DataPointF
	{
		// Current value of the data point
		float value;

		// Type of interpolation to be carried out between this point and the next
		int interpType;

		// Current time (in ms) the point represents in the animation 
		float time;
	};
}

#endif