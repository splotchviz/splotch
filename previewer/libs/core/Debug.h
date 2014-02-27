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

//Debug include
#include "previewer/libs/core/Debug.h"

#include "GL/gl.h"
#include <iostream>

void DebugPrint(std::string);
void DebugPrint(std::string, int);
void DebugPrint(std::string, unsigned);
void DebugPrint(std::string, float);
void DebugPrint(std::string, std::string);

void PrintOpenGLError();

#endif