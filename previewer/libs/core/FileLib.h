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

#ifndef FILELIB_H
#define FILELIB_H

//Debug include
#include "previewer/libs/core/Debug.h"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "kernel/colourmap.h"
#include "Parameter.h"

namespace previewer
{
	namespace FileLib
	{
		bool FileExists(const char* filepath);
		char* loadTextFile(const std::string &filename);
		unsigned char* loadTGAFile(const std::string &filename, int &width, int &height, int &bpp);
		int LoadColourPalette(Parameter param, std::vector<COLOURMAP>& colourMaps);
	}
}

#endif