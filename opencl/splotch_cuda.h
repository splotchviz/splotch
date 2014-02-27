
#ifndef SPLOTCH_CUDA_H
#define SPLOTCH_CUDA_H
//#define opencl2
#define OPENCL1
//#define DEBUG
//#define MEM_OPTIMISE
#ifdef VS
#define THREADFUNC DWORD WINAPI
#else
#define THREADFUNC static void*
#endif
#ifdef OPENCL1
#include <CL/cl.h>
#endif
#include <cstring>
#include "cxxsupport/paramfile.h"
#include "kernel/colour.h"
#include "splotch/splotchutils.h"
class CuPolicy;

//data structs for using on device
//'d_' means device
typedef particle_sim cu_particle_sim;

#define MAX_P_TYPE 8//('XXXX','TEMP','U','RHO','MACH','DTEG','DISS','VEL')
                                        //in mid of developing only
#define MAX_EXP -20.0

struct cu_color
  {
  float r,g,b;
  };

struct range_part
  {
 unsigned short  minx,miny,maxx,maxy;
  };

struct particle_sim2
  {
  cu_color e;
  float x,y,r,I;
  unsigned short type;
  bool active;
  
  };

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
  float rfac;
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

struct cu_particle_splotch
  {
  float x,y,r,I;
  int type;
  unsigned short maxx,minx;
  cu_color e;
  unsigned long posInFragBuf;
  };


struct cu_fragment_AeqE
  {
  float aR, aG, aB;
  };

struct cu_fragment_AneqE
  {
  float aR, aG, aB;
  float qR, qG, qB;
  };


struct cu_gpu_vars //variables used by each gpu
  {
  CuPolicy            *policy;
#ifdef OPENCL1
 // cl_mem d_pd; //particle_sim2
    cl_mem d_ps_render; //cu_particle_splotch
    cl_mem d_fbuf; //void
    cl_mem d_pic; //cu_color
    cl_mem cu_col; //cu_color_map_entry *
    cl_mem max_min;
   cl_mem ptype_points; //int *
    cl_mem par; //cu_param*
    cl_mem output_combined_image;
#endif
  cu_param param_h;
  int colormap_size;
  int colormap_ptypes;
  };

//functions

void cu_init(int devID, int nP, cu_gpu_vars* pgv, paramfile &fparams, vec3 &campos, vec3 &lookat, vec3 &sky,int pict_size);
void load_ocl_program ();
void cu_init_colormap(cu_colormap_info info, cu_gpu_vars* pgv);
void cu_copy_particles_to_render(cu_particle_splotch *p, int n, cu_gpu_vars* pgv);
void cu_render1(int nP, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv);
void cu_get_fbuf(cu_fragment_AeqE *h_fbuf, bool a_eq_e, unsigned long n, cu_gpu_vars* pgv);
void cu_get_fbuf2(cu_fragment_AneqE *h_fbuf, bool a_eq_e, unsigned long n, cu_gpu_vars* pgv);
void transform(cu_particle_sim* p, unsigned int n, cu_gpu_vars* pgv,range_part* minmax);
void cu_end (cu_gpu_vars* pgv);
int cu_get_chunk_particle_count(paramfile &params, CuPolicy* policy, size_t psize, float pfactor);
void getCuTransformParams(cu_param &para_trans, paramfile &params, vec3 &campos, vec3 &lookat, vec3 &sky);

#endif
