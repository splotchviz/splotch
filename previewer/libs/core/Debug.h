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

#ifndef SPLOTCH_PREVIEWER_DEBUG_MODE
#define SPLOTCH_PREVIEWER_DEBUG_MODE

//#define PREVIEWER_OPENGL

//Debug include
#include <cstdarg>
#include <iostream>
#include "cxxsupport/vec3.h"

#ifdef PREVIEWER_OPENGL
#include "GL/gl.h"
#endif

void ErrorMessage(const char* s, ...);
void DebugPrint(const char* s, ...);
void PrintOpenGLError();

#endif