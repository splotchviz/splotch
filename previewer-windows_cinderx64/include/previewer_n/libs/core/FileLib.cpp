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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_FILELIB
#define SPLOTCH_PREVIEWER_LIBS_CORE_FILELIB

#include "FileLib.h"


namespace previewer
{
	namespace FileLib{

		// Check if file exists
		bool FileExists(const char* filepath)
		{
			std::filebuf fbuf;
			fbuf.open(filepath, std::ios::in);

			if(fbuf.is_open())
			{
				fbuf.close();
				return true;
			}

			return false;
		}

		// Load a text file to a char array ptr
		char* loadTextFile(const std::string &filename)
		{
			// Open file
			std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

			// Indicate if not file found
			if(!file)
				std::cout << "shader text file not found" << std::endl;

			char* data = 0;
			int size = 0;

			// Get size of the file
			file.seekg(0,std::ios::end);
			size = file.tellg();
			file.seekg(0,std::ios::beg);

			// Indicate if file is empty
			if(!size)
			{
				std::cout << "shader file is empty!" << std::endl;
			}

			// Create buffer for data
			data = new char[size+1];

			// Read file
			file.read(data, size);

			// Null terminate char array
			data[size] = '\0';

			return data;

		}

		// This function will load a TGA image
		// Returns unsigned char pointer to data
		// Takes width/height/bpp by reference and modifies them to be the correct values
		unsigned char* loadTGAFile(const std::string &filename, int &width, int &height, int &bpp)
		{
			//open file
			std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary);

			if(!file)
				std::cout << "tga loader: TGA image file not found" << std::endl;

			unsigned char imageTypeInfo[4];
			unsigned char imageDataInfo[6];

			//read type (image type is at byte 3)
			file.seekg(0, std::ios::beg);	
			file.read( (char*)imageTypeInfo, 4);

			//read info(width/height/bpp info starts at byte 12)
			file.seekg(12, std::ios::beg);
			file.read( (char*)imageDataInfo, 6);

			//check image type is supported. Currently only type 2/3 are supported (uncompressed true-colour/greyscale)
			int colourMapPresent = imageTypeInfo[1];
			int imageType = imageTypeInfo[2];

			if (colourMapPresent || (imageType != 2 && imageType != 3))
				std::cout << "tga loader: unsupported tga type" << std::endl;

			//get height, width and bits per pixel (pixel depth)
			//stored as 2 bytes 
			//lowbyte + highbyte*256
			width = imageDataInfo[0] + imageDataInfo[1] * 256;
			height = imageDataInfo[2] + imageDataInfo[3] * 256;
			bpp = imageDataInfo[4];

			//check image has 24/32 bpp
			if(bpp != 24 && bpp != 32)
				std::cout << "tga loader: unsupported pixel depth" << std::endl;
			
			//calculate size of data in bytes (bitsPP>>3 == bytesPP)
			long dataSize = width * height * (bpp>>3);

			//create buffer for data
			unsigned char* data = new unsigned char[dataSize]; 

			//set pointer to start of image data, and read
			file.seekg(18, std::ios::beg);
			file.read( (char*)data, dataSize);

			//check if flipping is necessary
			//flipbits are first two bits after pixel depth
			bool flipX = ((imageDataInfo[4] & 0x20));
			bool flipY = ((imageDataInfo[4] & 0x10));
			
			//flip
			if(flipX)
			{
				// Horizontally byteswap pixel data
				for(int i = 0; i < height; i++)
				{
					for(int j = 0; j < width/2; j++)
					{
						// Work out which byte we are swapping
						int d1 = (i * width + j) * bpp/8;
						int d2 = ((i + 1) * width - j - 1) * bpp/8;
						// Swap that and the next 2/3 depending on bpp to swap entire pixel
						for(int k = 0; k < bpp/8; k++)
						{
							unsigned char temp = data[d1+k];
							data[d1+k] = data[d2+k];
							data[d2+k] = temp;
						}
					}
				}
			 }


			if(flipY)
			{
				// Vertically byteswap pixel data
				for(int i = 0; i < width; i++)
				{
					for(int j = 0; j < height/2; j++)
					{
						// Work out which byte we are swapping
						int d1 = (j * width + i) * bpp/8;
						int d2 = ((height - j - 1) * width + i) * bpp/8;
						// Swap that and the next 2/3 depending on bpp to swap entire pixel
						for(int k = 0; k < bpp/8; k++)
						{
							unsigned char temp = data[d1+k];
							data[d1+k] = data[d2+k];
							data[d2+k] = temp;
						}
					}
				}
			}


			//reverse colour, tga is BGR/A we want RGB/A
			for(int i = 0; i < dataSize; i += (bpp/8))
			{
				unsigned char temp = data[i];
				data[i] = data[i + 2];
				data[i + 2] = temp;
			}

			return data;	
		}


		// Load a set of splotch colour palettes
		int LoadColourPalette(Parameter param, std::vector<COLOURMAP>& colourMaps)
		{
			paramfile* splotchParams = param.GetParamFileReference();

			// Get number of particle types
			int ptypes = splotchParams->find<int>("ptypes",1);

			// Resize colourmap vec appropriately 
			colourMaps.resize(ptypes);

			// For each type
			for (int i=0; i<ptypes; i++)
			{
				// Check if colour is vector, if so there will be no colourmap to load
				if (splotchParams->find<bool>("color_is_vector"+dataToString(i),false))
					std::cout << " color of ptype " << i << " is vector, so no colormap to load ..." << std::endl;
				else
				{
					// Read palette file
					std::ifstream infile (splotchParams->find<std::string>("palette"+dataToString(i)).c_str());
					if(!infile)
					{
						std::cout << "Could not open palette file "<< splotchParams->find<std::string>("palette"+dataToString(i)).c_str() << " for " << "palette"+dataToString(i) << std::endl;
						return 0;
					}

					// Get number of colours in palette
					std::string dummy;
					int nColours;
					infile >> dummy >> dummy >> nColours;
					std::cout << " loading " << nColours << " entries of color table of ptype " << i << std::endl;

					// Load
					double step = 1./(nColours-1);
					for (int j=0; j<nColours; j++)
					{
						float rrr,ggg,bbb;
						infile >> rrr >> ggg >> bbb;
						colourMaps[i].addVal(j*step,COLOUR(rrr/255,ggg/255,bbb/255));
					}
				}

				// Sort
				colourMaps[i].sortMap();
			}

			return 1;
		}
	}
}

#endif