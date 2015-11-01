/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
 *              Claudio Gheller ETH-CSCS
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

#ifndef MIC_UTILS
#define MIC_UTILS

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define REUSE alloc_if(0) free_if(0)

#ifdef __INTEL_OFFLOAD
#define EVT_TARGET_MIC __attribute__((target(mic))) 
#else
#define EVT_TARGET_MIC
#endif

//#include "parameter_define.h"
/*
// Number of threadgroups
#define N_THREAD_GROUPS 6
// Number of threads per threadgroup
#define N_THREADS_PER_GROUP 40
// Dimension of square image tile
#define TILE_SIZE 40
*/

#pragma offload_attribute(push, target(mic))


// #if defined (USE_MPI)
// #include "mpi.h"
// #else
// #include <sys/time.h>
// #include <stdlib.h>
// #endif

#if defined (_OPENMP)
#include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <mathimf.h>

#include "mic/mic_arr.h"
#include "mic/mic_pod_arr.h"
#include "mic/mic_timer.h"


//#include "papi/papi_wrapper.h"


#ifdef __MIC__
#include "immintrin.h"
#endif

// Data to be offloaded to MIC for transformation
struct transform_data{
	float xfac;
	int xres;
	int yres;
	float zminval;
	float zmaxval;
	float ycorr;
	float dist;
	float fovfct;
	float bfak;
	float h2sigma;
	float sqrtpi;
	bool projection;
	float minrad_pix;
  float xshift;
  float yshift;
};

// Render paramaters
struct render_data{
	int n_thread_groups;
	int n_threads_per_group;
	int tile_size;
};

// Color map entry
struct mic_color
{
	float r,g,b;
};	

// Color map for coloring splotch particles
struct mic_color_map{
	int size;
	int* ptype_color_offset;
	mic_color* mapcolor;
	float* mapvalue;
};

// Memory structure to pass around SOA particle data 
struct mic_soa_particles{
  mic_soa_particles() : allocated(false), chunksize(0) {}
	float* er0; float* eg0; float* eb0; float* x0; float* y0; float* z0; float* r0; float* I0;
	float* er1; float* eg1; float* eb1; float* x1; float* y1; float* z1; float* r1; float* I1;
	short* type0; short* type1; 
	bool* active0; bool* active1;
	mic_color* devicePic;
  bool allocated;
  unsigned chunksize;
};

#ifdef __MIC__
union mm512f32{
	float f32[16];
	__m512  m512;
};
#endif


template<typename T> class mic_exptable
  {
  private:
    T expfac, taylorlimit;
    Array_T<T> tab1, tab2;
    enum {
      nbits=10,
      dim1=1<<nbits,
      mask1=dim1-1,
      dim2=(1<<nbits)<<nbits,
      mask2=dim2-1,
      mask3=~mask2
      };

  public:
    mic_exptable (T maxexp)
      : expfac(dim2/maxexp), tab1(dim1), tab2(dim1)
      {
      for (int m=0; m<dim1; ++m)
        {
        tab1[m]=exp(m*dim1/expfac);
        tab2[m]=exp(m/expfac);
        }
      taylorlimit = sqrt(T(2)*abs(maxexp)/dim2);
      }

    T taylorLimit() const { return taylorlimit; }

    T operator() (T arg)
      {
      int iarg= int(arg*expfac);
      if (iarg&mask3)
        return (iarg<0) ? T(1) : T(0);
      return tab1[iarg>>nbits]*tab2[iarg&mask1];
      }
    T expm1(T arg) const
      {
      if (abs(arg)<taylorlimit) return arg;
      return operator()(arg)-T(1);
      }
  };

class mic_work_distributor
  {
  private:
    int sx, sy, tx, ty, nx;

  public:
    mic_work_distributor (int sx_, int sy_, int tx_, int ty_)
      : sx(sx_), sy(sy_), tx(tx_), ty(ty_), nx((sx+tx-1)/tx) {}

    int ntiles() const
      { return nx * ((sy+ty-1)/ty); }

    void tile_info (int n, int &x0, int &x1, int &y0, int &y1) const
      {
      int ix = n%nx;
      int iy = n/nx;
      x0 = ix*tx; x1 = (x0+tx <  sx ? x0+tx : sx);
      y0 = iy*ty; y1 = (y0+ty <  sy ? y0+ty : sy);
      }
    void tile_info_idx (int n, int &ix, int &iy) const
      {
      ix = n%nx;
      iy = n/nx;
      }
  };


//   // Timer to run in offloaded code
// class mic_timer{
// public:
// 	mic_timer() {times_.setCapacity(10);}

// 	void start(std::string s) {
// 		double t = wallTime();
// 		timePair tp(s,-t);
// 		times_.push_back(tp);
// 	}

// 	void stop(std::string s) {
// 		unsigned i = 0;
// 		// Must use c_str, use of s on its own causes linker errors
// 		for(i = 0; i < times_.size(); i++){
// 			if(strcmp(times_[i].first.c_str(),s.c_str()) == 0){
// 				double d = wallTime();
// 				times_[i].second += d;
// 				break;
// 			}
// 		}
// 	}

// 	// void accumulate(){
// 	// 	for(unsigned i = 0; i < times_.size(); i++){
// 	// 		if(times_[i].second > 0)
// 	// 			for(unsigned j = i; j < times_.size(); j++){
// 	// 				if(times_[j].second > 0)
// 	// 					if(times_[i].first == times_[j].first){
// 	// 						times_[i].second += times_[j].second;
// 	// 						times_[j].second = -1;
// 	// 					}
// 	// 		}
// 	// 	}
// 	// }

// 	void print() {
// 		printf("--------- Offload Timer Report --------\n");
// 		for(unsigned i = 0; i < times_.size(); i++){
// 			if(times_[i].second >= 0)
// 				printf("%s -> %f\n", times_[i].first.c_str(), times_[i].second);
// 		}
// 		fflush(0);
// 	}

// private:
// 	struct timePair{
// 		timePair() {}
// 		timePair(std::string str, double d) { first = str; second = d;}
// 		std::string first;
// 		double second;
// 	};

// 	double wallTime(){
// 		#if defined (_OPENMP)
// 		  return omp_get_wtime();
// 		#elif defined (USE_MPI)
// 		  return MPI_Wtime();
// 		#else
// 		  struct timeval t;
// 		  gettimeofday(&t, NULL);
// 		  return t.tv_sec + 1e-6*t.tv_usec;
// 		#endif
// 	}

// 	Array_T<timePair> times_;
// };

#pragma offload_attribute(pop)

#endif

