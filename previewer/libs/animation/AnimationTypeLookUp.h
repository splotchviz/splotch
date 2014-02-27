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

#ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONTYPELOOKUP_H
#define	SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONTYPELOOKUP_H

#include <string>
#include <iostream>

namespace previewer
{
	enum
	{
		FLOAT,
		DOUBLE,
		VEC3FLOAT,
		VEC3DOUBLE
	};

	class AnimationTypeLookUp
	{
	public:
		static int Query(std::string, std::string);
	};
}

#endif