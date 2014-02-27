/*
 * Copyright (c) 2010-2014
 *              Marzia Rivi (1) 
 *               (1) University of Oxford
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
 */
 
#include "cuda/CuPolicy.h"

CuPolicy::CuPolicy(int xres, int yres, paramfile &params)
  {
    res.first = xres;
    res.second = yres;
    boundary_width = params.find<int>("tile_boundary_width", 8); // width of the boundary around the image tile = max particle radius
    tile_size.first = params.find<int>("tile_x-side", 8);  // x side dimension of the image tile, in terms of pixels
    tile_size.second = params.find<int>("tile_y-side", tile_size.first); // y side dimension of the image tile, in terms of pixels
    x_num_tiles = xres/tile_size.first;
    if (xres%tile_size.first) x_num_tiles++;
    y_num_tiles = yres/tile_size.second;
    if (yres%tile_size.second) y_num_tiles++;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
  //  p_blockSize = deviceProp.maxThreadsPerBlock;
    p_blockSize = 512;
    m_gridSize = deviceProp.maxGridSize[0];
    gmsize = deviceProp.totalGlobalMem;
  }

void CuPolicy::GetTileInfo(int *tile_sidex, int *tile_sidey, int *width, int *nxtiles, int *nytiles)
  {
    *tile_sidex = tile_size.first;
    *tile_sidey = tile_size.second;
    *width = boundary_width;
    *nxtiles = x_num_tiles;
    *nytiles = y_num_tiles;
  }

int CuPolicy::GetNumTiles()
{
   return x_num_tiles*y_num_tiles;
}

size_t CuPolicy::GetGMemSize() // return dimension in terms of bytes
  { 
   // int MB = 1<<20;
   // int size = gmsize/MB;
    return gmsize; 
  }

int CuPolicy::GetMaxGridSize() 
  { 
    return m_gridSize; 
  }

int CuPolicy::GetBlockSize() 
  { 
    return p_blockSize; 
  }

size_t CuPolicy::GetImageSize()
{
    size_t size = (res.first)*(res.second)*sizeof(cu_color);
    return size;
}

void CuPolicy::GetDimsBlockGrid(int n, dim3 *dimGrid, dim3 *dimBlock)
  {
    *dimBlock = dim3(p_blockSize);
    int nBlock = (n + p_blockSize - 1)/p_blockSize;
    *dimGrid =dim3(nBlock); 
    if (nBlock > m_gridSize)
      cout << "Error: dim grid = " << nBlock << "too large!" << endl;
  }
