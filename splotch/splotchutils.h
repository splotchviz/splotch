/*
 * Copyright (c) 2004-2013
 *              Martin Reinecke (1), Klaus Dolag (1)
 *               (1) Max-Planck-Institute for Astrophysics
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

#ifndef SPLOTCHUTILS_H
#define SPLOTCHUTILS_H

#include "cxxsupport/mpi_support.h"
#include "cxxsupport/paramfile.h"
#include "kernel/colour.h"
#include "cxxsupport/vec3.h"
#include "kernel/colourmap.h"
#include "cxxsupport/walltimer.h"

#ifdef CUDA
#include "cuda/cuda_utils.h"
#endif

#ifdef SPLVISIVO
#include "optionssetter.h"
#include "luteditor.h"
#endif

#include <fstream>
#include <limits.h>

#ifdef LONGIDS
#define MyIDType uint64
#define MyMaxID ULLONG_MAX
#else
#define MyIDType uint32
#define MyMaxID UINT_MAX
#endif

struct particle_sim
  {
  COLOUR e;
  float32 x,y,z,r,I;
  unsigned short type;
  bool active;
#ifdef MPI_A_NEQ_E
  unsigned short ghost;
  float32 r2;
#endif

  particle_sim (const COLOUR &e_, float32 x_, float32 y_, float32 z_, float32 r_,
                float32 I_, int type_, bool active_)
    : e(e_), x(x_), y(y_), z(z_), r(r_), I(I_), type(type_), active(active_) 
        #ifdef MPI_A_NEQ_E
        , ghost(0), r2(r_)
        #endif
     {}

  particle_sim () {}
  };

struct zcmp
  {
  bool operator()(const particle_sim &p1, const particle_sim &p2) const
    { return p1.z<p2.z; }
  };

struct vcmp1
  {
  bool operator()(const particle_sim &p1, const particle_sim &p2) const
    { return p1.e.r>p2.e.r; }
  };

struct vcmp2
  {
  bool operator()(const particle_sim &p1, const particle_sim &p2) const
    { return p1.e.r<p2.e.r; }
  };

struct hcmp
  {
  bool operator()(const particle_sim &p1, const particle_sim &p2) const
    { return p1.r>p2.r; }
};

struct render_context{
  // Camera and colormap
  vec3 campos, centerpos, lookat, sky;
  std::vector<COLOURMAP> amap;
  arr2<COLOUR> pic;
  tsize npart;
  tsize npart_all;
  float b_brightness;
  int xres;
  int yres;

#ifdef CUDA
  int mydevID = -1;
  // Number of processes using the same GPU
  int nTasksDev;     
  cu_cpu_vars cv;
#endif

#ifdef MPI_A_NEQ_E
  std::vector<KDnode<particle_sim, 3>* > active_nodes;
  std::vector<int> composite_order;
  Box<float,3> bbox;
  arr2<COLOUR> opacity_map;
#endif  
};

void update_res(paramfile& params, render_context& c);
void do_background(paramfile& params, arr2<COLOUR>& pic, const std::string& outfile);

#ifdef MPI_A_NEQ_E
bool xcomp_smaller(const particle_sim &p1, const particle_sim &p2);
bool ycomp_smaller(const particle_sim &p1, const particle_sim &p2);
bool zcomp_smaller(const particle_sim &p1, const particle_sim &p2);
float x_accessor(const particle_sim &p1);
float y_accessor(const particle_sim &p1);
float z_accessor(const particle_sim &p1);
float r_accessor(const particle_sim &p1);
float ghost_accessor(const particle_sim &p1);
void ghost_setter(particle_sim &p1);
#endif

template<typename T> struct Normalizer
  {
  T minv, maxv;

  Normalizer ()
    : minv(1e37), maxv(-1e37) {}

  void collect (T val)
    {
    using namespace std;
    minv=min(minv,val); maxv=max(maxv,val);
    }

  void collect (const Normalizer &other)
    {
    using namespace std;
    minv=min(minv,other.minv); maxv=max(maxv,other.maxv);
    }

  void normAndClamp (T &val) const
    {
    using namespace std;
    if (maxv==minv)
      val=T(1);
    else
      val = (max(minv,min(maxv,val))-minv)/(maxv-minv);
    }

  void reset ()
    {
      minv = 1e37;
      maxv = -1e37;
    }
  };

template<typename T> class exptable
  {
  private:
    T expfac, taylorlimit;
    arr<T> tab1, tab2;
    enum {
      nbits=10,
      dim1=1<<nbits,
      mask1=dim1-1,
      dim2=(1<<nbits)<<nbits,
      mask2=dim2-1,
      mask3=~mask2
      };

  public:
    exptable (T maxexp)
      : expfac(dim2/maxexp), tab1(dim1), tab2(dim1)
      {
      using namespace std;
      for (int m=0; m<dim1; ++m)
        {
        tab1[m]=exp(m*dim1/expfac);
        tab2[m]=exp(m/expfac);
        }
      taylorlimit = sqrt(T(2)*abs(maxexp)/dim2);
      }

    T taylorLimit() const { return taylorlimit; }

    T operator() (T arg) const
      {
      int iarg= int(arg*expfac);
      if (iarg&mask3)
        return (iarg<0) ? T(1) : T(0);
      return tab1[iarg>>nbits]*tab2[iarg&mask1];
      }
    T expm1(T arg) const
      {
      if (std::abs(arg)<taylorlimit) return arg;
      return operator()(arg)-T(1);
      }
  };

class work_distributor
  {
  private:
    int sx, sy, tx, ty, nx;

  public:
    work_distributor (int sx_, int sy_, int tx_, int ty_)
      : sx(sx_), sy(sy_), tx(tx_), ty(ty_), nx((sx+tx-1)/tx) {}

    int nchunks() const
      { return nx * ((sy+ty-1)/ty); }

    void chunk_info (int n, int &x0, int &x1, int &y0, int &y1) const
      {
      int ix = n%nx;
      int iy = n/nx;
      x0 = ix*tx; x1 = std::min(x0+tx, sx);
      y0 = iy*ty; y1 = std::min(y0+ty, sy);
      }
    void chunk_info_idx (int n, int &ix, int &iy) const
      {
      ix = n%nx;
      iy = n/nx;
      }
  };


void add_colorbar(paramfile &params, arr2<COLOUR> &pic,
  std::vector<COLOURMAP> &amap);

void timeReport();
void hostTimeReport(wallTimerSet &Timers);
#ifdef SPLVISIVO
void get_colourmaps (paramfile &params, std::vector<COLOURMAP> &amap, VisIVOServerOptions &opt);
#else
void get_colourmaps (paramfile &params, std::vector<COLOURMAP> &amap);
#endif
void replace_colourmap(paramfile &params, std::vector<COLOURMAP> &amap, int itype, std::string);
void check_palette_format(std::ifstream& infile);
void composite_images(render_context& c);
void colour_adjust(paramfile &params, render_context& c);
double my_asinh (double val);
bool file_present(const std::string &name);
void checkrth(float r_th, int npart, std::vector<particle_sim>& p, paramfile &params);
void checkbbox(std::vector<particle_sim>& p, paramfile& params);

// Reader IDs
enum class SimTypeId: int { NONE = -1, BIN_TABLE = 0, BIN_BLOCK = 1, GADGET = 2, ENZO = 3, GADGET_MILLENIUM = 4, BIN_BLOCK_MPI = 5,
                            MESH = 6, REGULAR_HDF5 = 7, GADGET_HDF5 = 8, /*No 9,*/ VISIVO = 10, TIPSY = 11 , H5PART = 12, RAMSES = 13, 
                            BONSAI = 14, ASCII =15, FITS = 16};
std::string simtype2str(SimTypeId);
enum FieldId: int { NONE = -1, F_X = 0, F_Y, F_Z, F_CR, F_CG, F_CB, F_R, F_I };
std::string FieldId2str(FieldId);
FieldId str2FieldId(std::string str);
#endif // SPLOTCHUTILS_H
