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

 /*
CuPolicy is the class that knows the overall state of cuda application.
All 'magic numbers' are out of this class.
*/
#ifndef CUPOLICY_H
#define CUPOLICY_H

#include "cxxsupport/paramfile.h"
#include "cuda/splotch_cuda.h"

#ifdef __CUDACC__
#include <cuda.h>
#else
struct dim3;
#endif

using namespace std;

class CuPolicy
  {
  private:
    int m_gridSize, p_blockSize;
    pair <int,int> res, tile_size;
    int boundary_width, x_num_tiles, y_num_tiles;
    size_t gmsize;
  public:
    CuPolicy(int xres, int yres, paramfile &params);

    void GetTileInfo(int *tile_sidex, int *tiley, int *width, int *nxtiles, int *nytiles);
    int GetNumTiles();
    size_t GetGMemSize();
    size_t GetImageSize();
    int GetBlockSize();
    int GetMaxGridSize();
    void GetDimsBlockGrid(int n, dim3 *dimGrid, dim3 *dimBlock);
  };

#endif //CUPOLICY_H
