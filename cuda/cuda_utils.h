/*
 * Copyright (c) 2010-2014
 *              Marzia Rivi (1), Tim Dykes (2)
 *               (1) University of Oxford
 *               (2) University of Portsmouth
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

#ifndef SPLOTCH_CUDA_H
#define SPLOTCH_CUDA_H

#define SPLOTCH_CLASSIC

#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "cxxsupport/paramfile.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/string_utils.h"
#include "kernel/colour.h"
#include "kernel/transform.h"
#include "cxxsupport/arr.h"
#include "cxxsupport/colour.h"
#include "cxxsupport/mpi_support.h"
//#include "splotch/splotchutils.h"

#include "cuda/cuda_policy.h"


// Use atomics for image update instead of layered update in k_renderC2
//#define CUDA_ATOMIC_TILE_UPDATE

// Replace full implementation with atomic update
#define CUDA_FULL_ATOMICS

//  Data structs for using on device
// 'd_' means device (for device pointers)

// RGB triplet to represent a color
struct cu_color
{
  float               r,g,b;
};

// Struct to represent a splotch particle on the device
struct cu_particle_sim
{
  cu_color            e;
  float               x,y,z,r,I;
  unsigned short      type;
  bool                active;
};

// Max N types as available in GADGET format
// TODO: Allow as many types as requested
#define MAX_P_TYPE 8//('XXXX','TEMP','U','RHO','MACH','DTEG','DISS','VEL')

// Rendering paramaters to be used on the device
struct cu_param
{
  float               p[12];
  bool                projection;
  int                 xres, yres;
  float               fovfct, dist, xfac;
  float               minrad_pix;
  double              xshift, yshift;
  int                 ptypes;
  float               zmaxval, zminval;
  bool                col_vector[MAX_P_TYPE];
  float               brightness[MAX_P_TYPE];
  float               smooth_fac[MAX_P_TYPE];
  bool                log_int[MAX_P_TYPE];
  bool                log_col[MAX_P_TYPE];
  bool                asinh_col[MAX_P_TYPE];
  float               inorm_mins[MAX_P_TYPE];
  float               inorm_maxs[MAX_P_TYPE];
  float               cnorm_mins[MAX_P_TYPE];
  float               cnorm_maxs[MAX_P_TYPE];
  bool                do_logs;
  float               bfak, h2sigma, rfac;
};

// Map element for device color map
struct cu_color_map_entry
{
  float               val;
  cu_color            color;
};

// Info about colormap to pass to device
struct cu_colormap_info
{
  cu_color_map_entry  *map;
  int                 mapSize;
  int                 *ptype_points;
  int                 ptypes;
};


// Variables used by each gpu
struct cu_gpu_vars 
{
  // Core vars
  CuPolicy            *policy;
  cu_particle_sim     *d_pd;             // Device ptr to device particle data
  cu_color            *d_pic;
  int                 colormap_size;
  int                 colormap_ptypes;

  // Implementation specific
#ifndef CUDA_FULL_ATOMICS
#ifndef CUDA_ATOMIC_TILE_UPDATE
  cu_color            *d_pic1;
  cu_color            *d_pic2;
  cu_color            *d_pic3;
#endif
  int                 *d_active;         // -1=non-active particle, -2=active big particle, n=number of tile, n+1=C3 particles 
  int                 *d_index;
  int                 *d_tiles;          // Number of particles per tile
  int                 *d_tileID;         // Tile ID
#endif

};


// Context for cuda renderer
struct cu_cpu_vars
{
  cu_gpu_vars gv;
  CuPolicy* policy;
  long int len;
  long int nP;
  bool initialised;
  int xres;
  int yres;
  COLOUR* buf;
};


void      cu_Device(int devID);
void      cu_get_trans_params(cu_param &para_trans, paramfile &params, const vec3 &campos, const vec3 &lookat, vec3 &sky, const vec3 &centerpos);
void      cu_init_colormap(cu_colormap_info info, cu_gpu_vars* pgv);
int       cu_copy_particles_to_device(cu_particle_sim* h_pd, unsigned int n, cu_gpu_vars* pgv);
int       cu_range(int nP, cu_gpu_vars* pgv);
int      cu_allocate(long int nP, int ntiles, cu_gpu_vars* pgv);
int       cu_init_transform(paramfile &fparams, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, vec3 &sky, float b_brightness, bool& doLogs);

#ifndef CUDA_FULL_ATOMICS
int       cu_process(int n, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nxtiles, int nytiles);
void      cu_renderC2(int nP, int grid, int block, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nxtiles);
void      cu_indexC3(int nP, int nC3, cu_gpu_vars* pgv);
void      cu_renderC3(int nP, int nC3, int res, cu_gpu_vars* pgv);
#ifndef CUDA_ATOMIC_TILE_UPDATE
void      cu_add_images(int res, cu_gpu_vars* pgv);
#endif
#else
int cu_process(int nP, cu_gpu_vars* pgv);
void cu_render(int nP, cu_gpu_vars* pgv);
#endif

void      cu_end(cu_gpu_vars* pgv);
long int  cu_get_chunk_particle_count(cu_gpu_vars* pgv, int nTasksDev, size_t psize, int ntiles, float pfactor);
void      cu_clear_device_img(cu_gpu_vars& pgv);
#endif
