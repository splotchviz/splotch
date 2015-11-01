
#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include "cuda/cuda_utils.h"
#include <cstdio>

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
//#endif


//MACROs
#define Pi 3.141592653589793238462643383279502884197
#define MAXSIZE 1000

__device__ __forceinline__ void clamp (float minv, float maxv, float &val);
__device__ __forceinline__ double my_asinh (double val);
__device__ __forceinline__ cu_color get_color(int ptype, float val, int map_size, int map_ptypes);
__global__ void k_range(int nP, cu_particle_sim *p);

// Implementation specific kernels 
#if !defined(CUDA_FULL_ATOMICS) 
__global__ void k_process(cu_particle_sim *p, int *p_active, int n, int mapSize, int types, int tile_sidex, int tile_sidey, int width, int nxtiles, int nytiles);
__device__ int  pixelLocalToGlobal(int lpix, int xo, int yo, int width, int tile_sidey);
__global__ void k_renderC2(int nP, cu_particle_sim *part, int *tileId, int *tilepart, cu_color *pic, cu_color *pic1, cu_color *pic2, cu_color *pic3, int tile_sidex, int tile_sidey, int width, int nytiles);
__global__ void k_indexC3(int n, cu_particle_sim *part, int *index);
__global__ void k_renderC3(int nC3, int *index, cu_particle_sim *part, cu_color *pic);
#if !defined(CUDA_ATOMIC_TILE_UPDATE)
  __global__ void k_add_images(int n, cu_color *pic, cu_color *pic1, cu_color *pic2, cu_color *pic3);
#endif
#else
__global__ void k_process(cu_particle_sim *p, int n, int mapSize, int types);
__global__ void k_render(int nP, cu_particle_sim *part, cu_color *pic);
#endif

// check for non-active and big particles to remove from the device
struct particle_notValid
  {
    __host__ __device__ 
    bool operator()(const int flag)
    {
      return (flag < 0);
    }
  };

// check for active big particles to copy back to the host
struct reg_notValid
  {
    __host__ __device__
    bool operator()(const int flag)
    {
      return (flag==-2);
    }
  };

struct sum_op
{
  __host__ __device__
  cu_particle_sim operator()(const cu_particle_sim& p1, const cu_particle_sim& p2) const{

    cu_particle_sim sum;
    sum = p1;
    sum.e.r = p1.e.r + p2.e.r;
    sum.e.g = p1.e.g + p2.e.g;
    sum.e.b = p1.e.b + p2.e.b;

    return sum; 
   } 
};
#endif