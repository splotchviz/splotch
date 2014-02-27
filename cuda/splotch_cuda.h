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
#include "cxxsupport/paramfile.h"
#include "kernel/colour.h"
#include "splotch/splotchutils.h"
class CuPolicy;

//data structs for using on device
//'d_' means device
//typedef particle_sim cu_particle_sim;

struct cu_color
  {
  float r,g,b;
  };

struct cu_particle_sim
  {
    cu_color e;
    float x,y,z,r,I;
    unsigned short type;
    bool active;
  };

#define MAX_P_TYPE 8//('XXXX','TEMP','U','RHO','MACH','DTEG','DISS','VEL')
                                        //in mid of developing only
#define MAX_EXP -20.0

struct cu_param
  {
  float p[12];
  bool  projection;
  int   xres, yres;
  float fovfct, dist, xfac;
  float minrad_pix;
  int ptypes;
  float zmaxval, zminval;
  bool col_vector[MAX_P_TYPE];
  float brightness[MAX_P_TYPE];
  bool log_int[MAX_P_TYPE];
  bool log_col[MAX_P_TYPE];
  bool asinh_col[MAX_P_TYPE];
  float inorm_mins[MAX_P_TYPE];
  float inorm_maxs[MAX_P_TYPE];
  float cnorm_mins[MAX_P_TYPE];
  float cnorm_maxs[MAX_P_TYPE];
  bool do_logs;
  float bfak, h2sigma, sigma0, rfac;
  };


struct cu_color_map_entry
  {
  float val;
  cu_color color;
  };

struct cu_colormap_info
  {
  cu_color_map_entry *map;
  int mapSize;
  int *ptype_points;
  int ptypes;
  };

struct cu_gpu_vars //variables used by each gpu
  {
  CuPolicy            *policy;
  cu_particle_sim     *d_pd;             //device_particle_data
  int		              *d_active; //  -1=non-active particle, -2=active big particle, n=number of tile, n+1=C3 particles 
  int                 *d_index;
  cu_color            *d_pic;
  cu_color            *d_pic1;
  cu_color            *d_pic2;
  cu_color            *d_pic3;
  int	      	        *d_tiles;	 //number of particles per tile
  int	      	        *d_tileID; // tile ID
  int                 colormap_size;
  int                 colormap_ptypes;
  };

//functions
int cu_init(int devID, long int nP, int ntiles, cu_gpu_vars* pgv, paramfile &fparams, const vec3 &campos, const vec3 &lookat, vec3 &sky, float b_brightness, bool& doLogs);
int cu_copy_particles_to_device(cu_particle_sim* h_pd, unsigned int n, cu_gpu_vars* pgv);
int cu_process (int n, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nxtiles, int nytiles);
int cu_range(int nP, cu_gpu_vars* pgv);
void cu_init_colormap(cu_colormap_info info, cu_gpu_vars* pgv);
void cu_render1
  (int nP, int grid, int block, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nxtiles);
void cu_indexC3(int nP, int nC3, cu_gpu_vars* pgv);
void cu_addC3(int nP, int nC3, int res, cu_gpu_vars* pgv);
void cu_add_images(int res, cu_gpu_vars* pgv);
void cu_end(cu_gpu_vars* pgv);
long int cu_get_chunk_particle_count(cu_gpu_vars* pgv, int nTasksDev, size_t psize, int ntiles, float pfactor);
void getCuTransformParams(cu_param &para_trans,
      paramfile &params, const vec3 &campos, const vec3 &lookat, vec3 &sky);

#endif
