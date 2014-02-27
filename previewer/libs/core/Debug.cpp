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

//Debug include
#include "previewer/libs/core/Debug.h"

void DebugPrint(std::string s)
{
	#ifdef DEBUG_MODE
		std::cout << "Debug Message: " << s << std::endl;
	#endif
}

void DebugPrint(std::string s, int i)
{
	#ifdef DEBUG_MODE
		std::cout << "Debug Message: " << s << "  " << i << std::endl;
	#endif
}

void DebugPrint(std::string s, unsigned i)
{
	#ifdef DEBUG_MODE
		std::cout << "Debug Message: " << s << "  " << i << std::endl;
	#endif
}

void DebugPrint(std::string s, float f)
{
	#ifdef DEBUG_MODE
		std::cout << "Debug Message: " << s << "  " << f << std::endl;
	#endif
}

void DebugPrint(std::string s1, std::string s2)
{
	#ifdef DEBUG_MODE
		std::cout << "Debug Message: " << s1 << "  " << s2 << std::endl;
	#endif
}

void PrintOpenGLError()
{
	#ifdef DEBUG_MODE
		int ret = glGetError();
		if(ret)
			std::cout << "OpenGL Error: " << ret << std::endl;
		else
			std::cout << "No OpenGL Error" << std::endl;
	#endif
}