#ifndef SPLOTCH_CUDA2_H
#define SPLOTCH_CUDA2_H

#include "opencl/splotch_cuda.h"
#include "splotch/splotch_host.h"
#include "cxxsupport/walltimer.h"

THREADFUNC host_thread_func(void *pinfo);
THREADFUNC cu_thread_func(void *pinfo);


// struct containing thread task info
struct thread_info
  {
  int devID;                  //index of the device selected
  int startP, endP;           //start and end particles to handle
  long npart_all;             //total number of particles
#ifdef OPENCL
  cu_color *pPic;         //output image computed
#else
  arr2<COLOUR> *pPic;
#endif
  wallTimerSet times;
  int xres,yres;
  };

//some global info shared by all threads
extern paramfile       *g_params;
extern std::vector<particle_sim> particle_data; //raw data from file
extern vec3 campos, lookat, sky;
extern std::vector<COLOURMAP> amap;
extern int ptypes;
extern wallTimerSet cuWallTimers;

int check_device(int rank);
void print_device_info(int rank, int dev);

//void opencl_rendering(int mydevID, int nDev, cu_color *pic,int xres,int yres);
void opencl_rendering(int mydevID, std::vector<particle_sim> &particle, int nDev, arr2<COLOUR> &pic);
void DevideThreadsTasks(thread_info *tInfo, int nThread, bool bHostThread);
void cu_draw_chunk(void *pinfo, cu_gpu_vars* gv);
int filter_chunk(int StartP, int chunk_dim, int nParticle, int maxRegion,
                 int nFBufInCell, cu_particle_sim *d_particle_data,
                 cu_particle_splotch *cu_ps_filtered, int *End_cu_ps, 
                 int *nFragments2Render,range_part* minmax,range_part* minmax2);
void combine_chunk(int StartP, int EndP, cu_fragment_AeqE *fragBuf, cu_color *pPic,
		   int xres,int yres,range_part* minmax);
void combine_chunk2(int StartP, int EndP, cu_fragment_AneqE *fragBuf,cu_color *pPic,
		    int xres,int yres,range_part* minmax);
void setup_colormap(int ptypes, cu_gpu_vars* gv);

void GPUReport(wallTimerSet &cuTimers);
void cuda_timeReport();

#endif
