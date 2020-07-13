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

#include "cuda/cuda_utils.h"
#include "cuda/cuda_kernel.cuh"

using namespace std;

extern __constant__ cu_param dparams;
extern __constant__ cu_color_map_entry dmap[MAXSIZE];
extern __constant__ int ptype_points[10];

#define CLEAR_MEM(p) if(p) {cudaFree(p); p=0;}

template<typename T> T findParamWithoutChange(paramfile *param, std::string &key, T &deflt)
{
  return param->param_present(key) ? param->find<T>(key) : deflt;
}

void cu_Device(int devID)
{
  cudaError_t error;
  cudaSetDevice (devID); // initialize cuda runtime
 
  if (error) printf("Error setting device\n");
}

void cu_get_trans_params(cu_param &para_trans, paramfile &params, const vec3 &campos, const vec3 &lookat, vec3 &sky, const vec3 &centerpos)
{

  // Get ranging parameters
  // If mins and maxs were not already in parameter file, cuda version of particle_normalize in 
  // scenemaker will have written them in.
  int nt = params.find<int>("ptypes",1);
  for(int t=0; t<nt; t++)
  {
      para_trans.inorm_mins[t] = params.find<float>("intensity_min"+dataToString(t));
      para_trans.inorm_maxs[t] = params.find<float>("intensity_max"+dataToString(t));
      para_trans.cnorm_mins[t] = params.find<float>("color_min"+dataToString(t));
      para_trans.cnorm_maxs[t] = params.find<float>("color_max"+dataToString(t));
  }

  int xres = params.find<int>("xres",800),
      yres = params.find<int>("yres",xres);
  float fov = params.find<double>("fov",45); //in degrees
  float fovfct = tan(fov*0.5f*degr2rad);

  sky.Normalize();
  vec3 zaxis = (centerpos-lookat).Norm();
  vec3 xaxis = crossprod (sky,zaxis).Norm();
  vec3 yaxis = crossprod (zaxis,xaxis);
  TRANSFORM trans;
  trans.Make_General_Transform
        (TRANSMAT(xaxis.x,xaxis.y,xaxis.z,
                  yaxis.x,yaxis.y,yaxis.z,
                  zaxis.x,zaxis.y,zaxis.z,
                  0,0,0));
  trans.Invert();
  TRANSFORM trans2;
  trans2.Make_Translation_Transform(-campos);
  trans2.Add_Transform(trans);
  trans=trans2;

  vec3 tcpos = trans.TransPoint(centerpos);

  bool projection = params.find<bool>("projection",true);

 float dist=(centerpos-lookat).Length();
 float xfac=0.5f*xres/(fovfct*dist);
 double xshift=-xfac*tcpos.x;
 double yshift=-xfac*tcpos.y;
  if (!projection)
   cout << " Horizontal field of fiew: " << xres/xfac << endl;

  float minrad_pix = params.find<float>("minrad_pix",1.);

  //retrieve the parameters for transformation
  for (int i=0; i<12; i++)
    para_trans.p[i] =trans.Matrix().p[i];
  para_trans.projection=projection;
  para_trans.xres=xres;
  para_trans.yres=yres;
  para_trans.fovfct=fovfct;
  para_trans.dist=dist;
  para_trans.xfac=xfac;
  para_trans.minrad_pix=minrad_pix;
  para_trans.xshift=xshift;
  para_trans.yshift=yshift;

  float h2sigma = 0.5*pow(Pi,-1./6.);
  para_trans.h2sigma=h2sigma;
#ifdef SPLOTCH_CLASSIC
  para_trans.bfak=0.5*pow(Pi,-5./6.); // 0.19261
  float rfac=0.75;
#else
  float rfac=1.f;
#endif
  para_trans.rfac=rfac;
}

int cu_allocate(long int nP, int ntiles, cu_gpu_vars* pgv)
{
 cudaError_t error;
 
  // particle vector  
  size_t size = nP * sizeof(cu_particle_sim);
  error = cudaMalloc((void**) &pgv->d_pd, size);
  if (error != cudaSuccess) 
  {
   cout << "Device Memory: particle data allocation error!" << endl;
   return 1;
  }
  cudaMemset(pgv->d_pd,0,size);

  // Image
  size = pgv->policy->GetImageSize();
  error = cudaMalloc((void**) &pgv->d_pic, size); 
  if (error != cudaSuccess)
   {
     cout << "Device Malloc: image allocation error!" << endl;
     return 1;
   }
  cudaMemset(pgv->d_pic,0,size);



#if !defined(CUDA_FULL_ATOMICS)

  // Particle active flag vector
  size = nP * sizeof(int);
  error = cudaMalloc((void**) &pgv->d_active, size);
  if (error != cudaSuccess)
   {
     cout << "Device Malloc: active flag vector allocation error!" << endl;
     return 1;
   }

  // Index buffer for C3 particles
  size = nP * sizeof(int);
  error = cudaMalloc((void**) &pgv->d_index, size);
  if (error != cudaSuccess) 
  {
   cout << "Device Memory: index buffer allocation error!" << endl;
   return 1;
  }

  // Tiles
  size = (ntiles+1)*sizeof(int);
  error = cudaMalloc((void**) &pgv->d_tiles, size);   // containing number of particles per tile
  if (error != cudaSuccess) 
  {
     cout << "Device Malloc: array tiles allocation error!" << endl;
     return 1;
  }
  // Tile IDs
  error = cudaMalloc((void**) &pgv->d_tileID, size);  // containing tiles ID
  if (error != cudaSuccess) 
  {
     cout << "Device Malloc: array tiles allocation error!" << endl;
     return 1;
  }

#if !defined(CUDA_ATOMIC_TILE_UPDATE)
  size = pgv->policy->GetImageSize();
  error = cudaMalloc((void**) &pgv->d_pic1, size); 
  if (error != cudaSuccess)
   {
     cout << "Device Malloc: image 1 allocation error!" << endl;
     return 1;
   }
  cudaMemset(pgv->d_pic1,0,size);
  error = cudaMalloc((void**) &pgv->d_pic2, size); 
  if (error != cudaSuccess)
   {
     cout << "Device Malloc: image 2 allocation error!" << endl;
     return 1;
   }
  cudaMemset(pgv->d_pic2,0,size);
  error = cudaMalloc((void**) &pgv->d_pic3, size); 
  if (error != cudaSuccess)
   {
     cout << "Device Malloc: image 3 allocation error!" << endl;
     return 1;
   }
  cudaMemset(pgv->d_pic3,0,size);
#endif 
#endif
  
  if (error != cudaSuccess)
  {
    cout << "Device Malloc: initial allocation error!" << endl;
    cout << "Error: " << cudaGetErrorString(error) << std::endl;
    return 1;
  }
  return 0;
}

int cu_init_transform(paramfile &fparams, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, vec3 &sky, float b_brightness, bool& doLogs)
{
  cudaError_t error;
  //retrieve parameters
  cu_param tparams;
  cu_get_trans_params(tparams,fparams,campos,lookat,sky,centerpos);

  tparams.zmaxval   = fparams.find<float>("zmax",1.e23);
  tparams.zminval   = fparams.find<float>("zmin",0.0);
  tparams.ptypes    = fparams.find<int>("ptypes",1);

  for(int itype=0; itype<tparams.ptypes; itype++)
  {
    tparams.brightness[itype] = fparams.find<double>("brightness"+dataToString(itype),1.);
    tparams.brightness[itype] *= b_brightness;
    tparams.smooth_fac[itype] = fparams.find<double>("smooth_factor"+dataToString(itype),1.);
    tparams.col_vector[itype] = fparams.find<bool>("color_is_vector"+dataToString(itype),false);
    tparams.log_col[itype] = fparams.find<bool>("color_log"+dataToString(itype),false);
    tparams.log_int[itype] = fparams.find<bool>("intensity_log"+dataToString(itype),false);
    tparams.asinh_col[itype] = fparams.find<bool>("color_asinh"+dataToString(itype),false);
  }

  // Check if logs have already been done by host or not
  bool dflt = true;
  std::string key = "cuda_doLogs";
  doLogs = findParamWithoutChange<bool>(&fparams, key, dflt);


  //dump parameters to device
  error = cudaMemcpyToSymbol(dparams, &tparams, sizeof(cu_param));
  if (error != cudaSuccess)
  {
    cout << "Device Malloc: parameters allocation error!" << endl;
    cout << "Error: " << cudaGetErrorString(error) << std::endl;
    cout << "If error==invalid device symbol check you are compiling correctly for seperable compilation (-dc and extra link step)" << std::endl;
    return 1;
  }
  return 0;
}



void cu_init_colormap(cu_colormap_info h_info, cu_gpu_vars* pgv)
{
  //allocate memories for colormap and ptype_points and dump host data into it
  size_t size =sizeof(cu_color_map_entry)*h_info.mapSize;
  cudaMemcpyToSymbol(dmap, h_info.map, size);
  //type
  size =sizeof(int)*h_info.ptypes;
  cudaMemcpyToSymbol(ptype_points, h_info.ptype_points, size);

  //set fields of global variable pgv->d_colormap_info
  pgv->colormap_size   = h_info.mapSize;
  pgv->colormap_ptypes = h_info.ptypes;
}

int cu_copy_particles_to_device(cu_particle_sim* h_pd, unsigned int n, cu_gpu_vars* pgv)
{
  cudaError_t error;
  size_t size = n*sizeof(cu_particle_sim);
  error = cudaMemcpy(pgv->d_pd, h_pd, size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
   cout << "Device Memory: particle data copy error!" << endl;
   printf("cudaMemcpy returned: %s\n",cudaGetErrorString(error));
   return 1;
  }
  return 0;
}

 
int cu_range(int nP, cu_gpu_vars* pgv)
{
  // Get block and grid dimensions from policy object
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(nP, &dimGrid, &dimBlock);
  
  // Ranging process
  k_range<<<dimGrid, dimBlock>>>(nP, pgv->d_pd);

  return 0;
}
 


#ifdef CUDA_FULL_ATOMICS
// --------------------------------------
// Raster/render for full atomic implementation
// --------------------------------------
int cu_process(int n, cu_gpu_vars* pgv)
{
  //Get block dim and grid dim from pgv->policy object
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(n, &dimGrid, &dimBlock);

  cudaFuncSetCacheConfig(k_process, cudaFuncCachePreferL1);
  k_process<<<dimGrid,dimBlock>>>(pgv->d_pd, n, pgv->colormap_size, pgv->colormap_ptypes);
 
  return 0;
}

void cu_render(int nP, cu_gpu_vars* pgv)
{
  // Blocks 512 wide
  // Grid wide enough for each thread to have a particle
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(nP, &dimGrid, &dimBlock);

  //printf("Grid dim: %i, Block dim: %i\n", dimGrid.x, dimBlock.x);
  // Pass in pgv->d_pd (particle data) pgv->d_pic (final picture)
  k_render<<<dimGrid,dimBlock>>>(nP, pgv->d_pd, pgv->d_pic);
}

#else
// --------------------------------------
// Raster/render etc for tiled & partial atomics
// --------------------------------------

int cu_process (int n, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nxtiles, int nytiles)
{
  //Get block dim and grid dim from pgv->policy object
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(n, &dimGrid, &dimBlock);

  cudaFuncSetCacheConfig(k_process, cudaFuncCachePreferL1);
  k_process<<<dimGrid,dimBlock>>>(pgv->d_pd, pgv->d_active, n, pgv->colormap_size, pgv->colormap_ptypes, tile_sidex, tile_sidey, width, nxtiles, nytiles);
 
  return 0;
}

#ifndef CUDA_ATOMIC_TILE_UPDATE
void cu_add_images(int res, cu_gpu_vars* pgv)
{
  //fetch grid dim and block dim and call device
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(res, &dimGrid, &dimBlock);

  cudaFuncSetCacheConfig(k_add_images, cudaFuncCachePreferL1);
  k_add_images<<<dimGrid,dimBlock>>>(res, pgv->d_pic, pgv->d_pic1, pgv->d_pic2, pgv->d_pic3);
}
#endif

void cu_renderC2(int nP, int grid, int block, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv, int tile_sidex, int tile_sidey, int width, int nytiles)
{
  //get dims from pgv->policy object first
  dim3 dimGrid = dim3(grid); 
  dim3 dimBlock = dim3(block);
  size_t SharedMem = (tile_sidex+2*width)*(tile_sidey+2*width)*sizeof(cu_color);

  cudaFuncSetCacheConfig(k_renderC2, cudaFuncCachePreferShared);
  k_renderC2<<<dimGrid, dimBlock, SharedMem>>>(nP, pgv->d_pd, pgv->d_tileID, pgv->d_tiles, pgv->d_pic,
  pgv->d_pic1, pgv->d_pic2, pgv->d_pic3, tile_sidex, tile_sidey, width, nytiles);
}

void cu_indexC3(int nP, int nC3, cu_gpu_vars* pgv)
{
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(nC3, &dimGrid, &dimBlock);
 
  k_indexC3<<<dimGrid, dimBlock>>>(nC3, pgv->d_pd+nP-nC3, pgv->d_index);
}

void cu_renderC3(int nP, int nC3, int res, cu_gpu_vars* pgv)
{
  //fetch grid dim and block dim and call device
  dim3 dimGrid, dimBlock;
  pgv->policy->GetDimsBlockGrid(res, &dimGrid, &dimBlock);

  if (nC3 > 0)
  {
    cudaThreadSynchronize();
    pgv->policy->GetDimsBlockGrid(nC3, &dimGrid, &dimBlock);
    k_renderC3<<<dimGrid,dimBlock>>>(nC3, pgv->d_index, pgv->d_pd+nP-nC3, pgv->d_pic);
  }
}

#endif

void cu_end(cu_gpu_vars* pgv)
{
  CLEAR_MEM((pgv->d_pd));
  CLEAR_MEM((pgv->d_pic));

#ifndef CUDA_FULL_ATOMICS
  CLEAR_MEM((pgv->d_active));
  CLEAR_MEM((pgv->d_index));
  CLEAR_MEM((pgv->d_tiles));
  CLEAR_MEM((pgv->d_tileID));
#ifndef CUDA_ATOMIC_TILE_UPDATE
  CLEAR_MEM((pgv->d_pic1));
  CLEAR_MEM((pgv->d_pic2));
  CLEAR_MEM((pgv->d_pic3));
#endif
#endif

  delete pgv->policy;
  cudaDeviceReset();
}

long int cu_get_chunk_particle_count(cu_gpu_vars* pgv, int nTasksDev, size_t psize, int ntiles, float pfactor)
{
   size_t gMemSize = pgv->policy->GetGMemSize();
   size_t ImSize = pgv->policy->GetImageSize();
   size_t tiles = (ntiles+1)*sizeof(int);
   size_t colormap_size = pgv->colormap_size*sizeof(cu_color_map_entry)+pgv->colormap_ptypes*sizeof(int);
   size_t spareMem = 20*(1<<20);

   // Number of imagebuffers, if atomics are not used in any way then we use 4 buffers
   int nIm = 4;
#if defined(CUDA_FULL_ATOMICS) || defined(CUDA_ATOMIC_TILE_UPDATE)
   nIm = 1;
#endif

   long int arrayParticleSize = gMemSize/nTasksDev - nIm*ImSize - 2*tiles - spareMem - colormap_size;
   long int len = (long int) (arrayParticleSize/((psize+2*sizeof(int))*pfactor)); 
   long int maxlen = (long int)pgv->policy->GetMaxGridSize() * (long int)pgv->policy->GetBlockSize();

   if (len > maxlen) len = maxlen;
   return len;
}

void cu_clear_device_img(cu_gpu_vars& pgv)
{
  int size = pgv.policy->GetImageSize();
  cudaMemset(pgv.d_pic,0,size);
}






