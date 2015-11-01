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

// Variadic messages, simple printf style functions, e.g:
// ErrorMessage("Load failed! %i", error_code_int);
// DebugPrint("string here %i", anInteger);
//
// Format:
// i int; u unsigned int; f double; s cstring; p ptr address;  
// Can use \n for new line

void ErrorMessage(const char* s, ...)
{
	std::cout << "\n\nPreviewer Error: ";
	va_list args;
	va_start(args, s);
	while(*s != '\0')
	{
		if(*s=='%')
		{
			++s;
			switch(*s)
			{
				case 'i':
					std::cout << va_arg(args, int);
				break;
				case 'u':
					std::cout << va_arg(args, unsigned int);
				break;
				case 'f':
					std::cout << va_arg(args, double);
				break;
				case 's':
					std::cout << std::string(va_arg(args, const char*));
				break;
				case 'p':
					std::cout << va_arg(args, void*);
				break;
				default:
				std::cout << "%?";
				break;
			}
		}
		else if(*s == '\n')
		{
			std::cout << std::endl;
		}
		else 
		{
			putchar(*s);
		}
		++s;
	}
	std::cout << "In file: " << __FILE__ << std::endl;
	std::cout << "At line: " << __LINE__ << std::endl;
	std::cout << "\n\n";
	exit(-1);
} 

void DebugPrint(const char* s, ...)
{
	#ifdef DEBUG_MODE
		va_list args;
		va_start(args, s);
		while(*s != '\0')
		{
			if(*s=='%')
			{
				++s;
				switch(*s)
				{
					case 'i':
						std::cout << va_arg(args, int);
					break;
					case 'u':
						std::cout << va_arg(args, unsigned int);
					break;
					case 'f':
						std::cout << va_arg(args, double);
					break;
					case 's':
						std::cout << std::string(va_arg(args, const char*));
					break;
					case 'p':
						std::cout << va_arg(args, void*);
					break;
					default:
					std::cout << "%?";
					break;
				}
			}
			else if(*s == '\n')
			{
				std::cout << std::endl;
			}
			else 
			{
				putchar(*s);
			}
			++s;
		}

	#endif
}


void PrintOpenGLError()
{
	#ifdef PREVIEWER_OPENGL
		#ifdef DEBUG_MODE
			int ret = glGetError();
			if(ret)
				std::cout << "OpenGL Error: " << ret << std::endl;
			else
				std::cout << "No OpenGL Error" << std::endl;
		#endif
	#endif
}