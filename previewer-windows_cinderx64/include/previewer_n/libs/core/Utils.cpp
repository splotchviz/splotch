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

#include "Utils.h"

namespace previewer
{
	namespace Utils
	{
		// Type conversions

		int atoi(std::string s)
		{
			std::stringstream ss;
			int i;
			ss << s;
			ss >> i;
			return i;
		}

		float atof(std::string s)
		{
			std::stringstream ss;
			float f;
			ss << s;
			ss >> f;
			return f;
		}

		std::string ToString(float floatVal)
		{
			std::stringstream strstrm;
			std::string str;
			strstrm.precision(4);
			strstrm << std::fixed << floatVal;
			strstrm >> str;
			return str;
		}

		std::string ToString(int intVal)
		{
			std::stringstream strstrm;
			std::string str;
			strstrm << intVal;
			strstrm >> str;
			return str;
		}

		std::string ToString(unsigned uintVal)
		{
			std::stringstream strstrm;
			std::string str;
			strstrm << uintVal;
			strstrm >> str;
			return str;
		}
	}
}