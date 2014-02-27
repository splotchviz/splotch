/*
 * Copyright (c) 2004-2014
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

#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "splotch/splotch_host.h"
#include "kernel/transform.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/walltimer.h"
#include "cxxsupport/sse_utils_cxx.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/openmp_support.h"

#define SPLOTCH_CLASSIC

using namespace std;

namespace host_funct {

const float32 h2sigma = 0.5*pow(pi,-1./6.);
const float32 sqrtpi = sqrt(pi);

#ifdef SPLOTCH_CLASSIC
const float32 powtmp = pow(pi,1./3.);
const float32 sigma0 = powtmp/sqrt(2*pi);
const float32 bfak=1./(2*sqrt(pi)*powtmp);
#endif

#ifdef SPLOTCH_CLASSIC
const float32 rfac=1.5*h2sigma/(sqrt(2.)*sigma0);
#else
const float32 rfac=1.;
#endif

void particle_project(paramfile &params, vector<particle_sim> &p,
  const vec3 &campos, const vec3 &lookat, vec3 sky)
  {
  int xres = params.find<int>("xres",800),
      yres = params.find<int>("yres",xres);
  float32 zmaxval = params.find<float32>("zmax",1.e23),
          zminval = params.find<float32>("zmin",0.0);
  zminval = max(0.f,zminval);

  float32 ycorr = .5f*(yres-xres);
  float32 res2 = 0.5f*xres;
  float32 fov = params.find<float32>("fov",45); //in degrees
  float32 fovfct = tan(fov*0.5f*degr2rad);
  int npart=p.size();

  sky.Normalize();
  vec3 zaxis = (lookat-campos).Norm();
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

  bool projection = params.find<bool>("projection",true);

  float32 dist = (campos-lookat).Length();
  float32 xfac = 1./(fovfct*dist);
  if (!projection)
    cout << " Horizontal field of fiew: " << 1./xfac*2. << endl;

  float32 minrad_pix = params.find<float32>("minrad_pix",1.);

#pragma omp parallel
{
  float32 xfac2=xfac;
  long m;
#pragma omp for schedule(guided,1000)
  for (m=0; m<npart; ++m)
    {
    vec3 v(p[m].x,p[m].y,p[m].z);
    v=trans.TransPoint(v);
    p[m].x=v.x; p[m].y=v.y; p[m].z=v.z;

    p[m].active = false;
    if (p[m].z<=zminval) continue;
    if (p[m].z>=zmaxval) continue;

    if (!projection)
      {
      p[m].x = res2*(p[m].x+fovfct*dist)*xfac2;
      p[m].y = res2*(p[m].y+fovfct*dist)*xfac2 + ycorr;
      }
    else
      {
      xfac2=1.f/(fovfct*p[m].z);
      p[m].x = res2*(p[m].x+fovfct*p[m].z)*xfac2;
      p[m].y = res2*(p[m].y+fovfct*p[m].z)*xfac2 + ycorr;
      }

#ifdef SPLOTCH_CLASSIC
    p[m].I *= 0.5f*bfak/p[m].r;
    p[m].r*=sqrt(2.f)*sigma0/h2sigma; //  *= 2 ;)
#else
    p[m].I *= 8.f/(pi*p[m].r*p[m].r*p[m].r); // SPH kernel normalisation
    p[m].I *= (h2sigma*sqrtpi*p[m].r); // integral through the center
#endif
    p[m].r *= res2*xfac2;

    float32 rcorr = sqrt(p[m].r*p[m].r + minrad_pix*minrad_pix)/p[m].r;
    p[m].r*=rcorr;
#ifdef SPLOTCH_CLASSIC
    p[m].I/=rcorr;
#else
    p[m].I/=rcorr*rcorr;
#endif

    float32 posx=p[m].x, posy=p[m].y;
    float32 rfacr=rfac*p[m].r;

    int minx=int(posx-rfacr+1);
    if (minx>=xres) continue;
    minx=max(minx,0);
    int maxx=int(posx+rfacr+1);
    if (maxx<=0) continue;
    maxx=min(maxx,xres);
    if (minx>=maxx) continue;
    int miny=int(posy-rfacr+1);
    if (miny>=yres) continue;
    miny=max(miny,0);
    int maxy=int(posy+rfacr+1);
    if (maxy<=0) continue;
    maxy=min(maxy,yres);
    if (miny>=maxy) continue;

    p[m].active = true;
    }
}
  }

void particle_colorize(paramfile &params, vector<particle_sim> &p,
  vector<COLOURMAP> &amap, float b_brightness)
  {
  int nt = params.find<int>("ptypes",1);
  arr<bool> col_vector(nt);
  arr<float32> brightness(nt);

  for(int t=0;t<nt;t++)
    {
    brightness[t] = params.find<float32>("brightness"+dataToString(t),1.f);
    brightness[t] *= b_brightness;
    col_vector[t] = params.find<bool>("color_is_vector"+dataToString(t),false);
    }

  int npart=p.size();

#pragma omp parallel
{
  int m;
#pragma omp for schedule(guided,1000)
  for (m=0; m<npart; ++m)
    {
    if (p[m].active)
      {
      if (!col_vector[p[m].type])
        p[m].e=amap[p[m].type].getVal_const(p[m].e.r);

      p[m].e *= p[m].I * brightness[p[m].type];

//      p[m].active = ((p[m].e.r!=0.f) || (p[m].e.g!=0.f) || (p[m].e.g!=0.f));
      }
    }
}
  }

void particle_sort(vector<particle_sim> &p, int sort_type, bool verbose)
  {
  switch(sort_type)
    {
    case 0:
      if (verbose && mpiMgr.master())
        cout << " skipped sorting ..." << endl;
      break;
    case 1:
      if (verbose && mpiMgr.master())
        cout << " sorting by z ..." << endl;
      sort(p.begin(), p.end(), zcmp());
      break;
    case 2:
      if (verbose && mpiMgr.master())
        cout << " sorting by value ..." << endl;
      sort(p.begin(), p.end(), vcmp1());
      break;
    case 3:
      if (verbose && mpiMgr.master())
        cout << " reverse sorting by value ..." << endl;
      sort(p.begin(), p.end(), vcmp2());
      break;
    case 4:
      if (verbose && mpiMgr.master())
        cout << " sorting by size ..." << endl;
      sort(p.begin(), p.end(), hcmp());
      break;
    default:
      planck_fail("unknown sorting chosen ...");
      break;
    }
  }

const int chunkdim=100;

//common interface for CPU and GPU version
void render_new (particle_sim *p, int npart, arr2<COLOUR> &pic,
  bool a_eq_e, float32 grayabsorb)
  {
  planck_assert(a_eq_e || (mpiMgr.num_ranks()==1),
    "MPI only supported for A==E so far");

  int xres=pic.size1(), yres=pic.size2();
  int ncx=(xres+chunkdim-1)/chunkdim, ncy=(yres+chunkdim-1)/chunkdim;

  arr<arr2<vector<uint32> > > idx;
  float32 rcell=sqrt(2.f)*(chunkdim*0.5f-0.5f);
  float32 cmid0=0.5f*(chunkdim-1);

  exptable<float32> xexp(-20.);
#ifdef __SSE2__
  const float32 taylorlimit=xexp.taylorLimit();
#endif

  pic.fill(COLOUR(0,0,0));

  tstack_push("Host Chunk preparation");
#pragma omp parallel
{
  arr2<vector<uint32> > lidx(ncx,ncy);
  int mythread=openmp_thread_num(),
      nthreads=openmp_num_threads();

  int64 lo, hi;
  calcShareGeneral (0, npart, nthreads, mythread, lo, hi);
  for (int64 i=lo; i<hi; ++i)
    {
    particle_sim &pp(p[i]);
    float32 rfacr = rfac*pp.r;
    if (pp.active)
      {
      int minx=max(0,int(pp.x-rfacr+1)/chunkdim);
      int maxx=min(ncx-1,int(pp.x+rfacr)/chunkdim);
      int miny=max(0,int(pp.y-rfacr+1)/chunkdim);
      int maxy=min(ncy-1,int(pp.y+rfacr)/chunkdim);
      float32 sumsq=(rcell+rfacr)*(rcell+rfacr);
      for (int ix=minx; ix<=maxx; ++ix)
        {
        float32 cx=cmid0+ix*chunkdim;
        for (int iy=miny; iy<=maxy; ++iy)
          {
          float32 cy=cmid0+iy*chunkdim;
          float32 rtot2 = (pp.x-cx)*(pp.x-cx) + (pp.y-cy)*(pp.y-cy);
          if (rtot2<sumsq)
            lidx[ix][iy].push_back(i);
          }
        }
      }
    }
#pragma omp barrier
#pragma omp single
  {
  idx.alloc(nthreads);
  }
#pragma omp barrier
#pragma omp critical (render1)
  {
  idx[mythread].swap(lidx);
  }
} // end of parallel region

  tstack_replace("Host Chunk preparation","Host Rendering proper");

  work_distributor wd (xres,yres,chunkdim,chunkdim);
#pragma omp parallel
{
  arr<float32> pre1(chunkdim);
#ifdef __SSE2__
  arr2_align<V4sf,16> lpic(chunkdim,chunkdim);
#else
  arr2<COLOUR> lpic(chunkdim,chunkdim);
#endif
  int chunk;
#pragma omp for schedule(dynamic,1)
  for (chunk=0; chunk<wd.nchunks(); ++chunk)
    {
    int x0, x1, y0, y1;
    wd.chunk_info(chunk,x0,x1,y0,y1);
    int x0s=x0, y0s=y0;
    x1-=x0; x0=0; y1-=y0; y0=0;
    lpic.fast_alloc(x1-x0,y1-y0);
#ifdef __SSE2__
    lpic.fill(V4sf(0.));
#else
    lpic.fill(COLOUR(0,0,0));
#endif
    int cx, cy;
    wd.chunk_info_idx(chunk,cx,cy);
    for (tsize t=0; t<idx.size(); ++t)
      {
      const vector<uint32> &v(idx[t][cx][cy]);

      for (tsize m=0; m<v.size(); ++m)
        {
        const particle_sim &pp(p[v[m]]);
        float32 rfacr=pp.r*rfac;
        float32 posx=pp.x, posy=pp.y;
        posx-=x0s; posy-=y0s;
        int minx=int(posx-rfacr+1);
        minx=max(minx,x0);
        int maxx=int(posx+rfacr+1);
        maxx=min(maxx,x1);
        int miny=int(posy-rfacr+1);
        miny=max(miny,y0);
        int maxy=int(posy+rfacr+1);
        maxy=min(maxy,y1);

        float32 radsq = rfacr*rfacr;
        float32 sigma = h2sigma*pp.r;
        float32 stp = -1.f/(sigma*sigma);

        COLOUR a(-pp.e.r,-pp.e.g,-pp.e.b);
#ifdef __SSE2__
        V4sf va(a.r,a.g,a.b,0.f);
#endif

        for (int y=miny; y<maxy; ++y)
          pre1[y]=xexp(stp*(y-posy)*(y-posy));

        if (a_eq_e)
          {
          for (int x=minx; x<maxx; ++x)
            {
            float32 dxsq=(x-posx)*(x-posx);
            float32 dy=sqrt(radsq-dxsq);
            int miny2=max(miny,int(posy-dy+1)),
                maxy2=min(maxy,int(posy+dy+1));
            float32 pre2 = xexp(stp*dxsq);
            for (int y=miny2; y<maxy2; ++y)
              {
              float32 att = pre1[y]*pre2;
#ifdef __SSE2__
              lpic[x][y]+=va*att;
#else
              lpic[x][y].r += att*a.r;
              lpic[x][y].g += att*a.g;
              lpic[x][y].b += att*a.b;
#endif
              }
            }
          }
        else
          {
          COLOUR q(pp.e.r/(pp.e.r+grayabsorb),
                  pp.e.g/(pp.e.g+grayabsorb),
                  pp.e.b/(pp.e.b+grayabsorb));
#ifdef __SSE2__
          float32 maxa=max(abs(a.r),max(abs(a.g),abs(a.b)));
          V4sf vq(q.r,q.g,q.b,0.f);
#endif

          for (int x=minx; x<maxx; ++x)
            {
            float32 dxsq=(x-posx)*(x-posx);
            float32 dy=sqrt(radsq-dxsq);
            int miny2=max(miny,int(posy-dy+1)),
                maxy2=min(maxy,int(posy+dy+1));
            float32 pre2 = xexp(stp*dxsq);
            for (int y=miny2; y<maxy2; ++y)
              {
              float32 att = pre1[y]*pre2;
#ifdef __SSE2__
              if ((maxa*att)<taylorlimit)
                lpic[x][y]+=(lpic[x][y]-vq)*va*att;
              else
                {
                V4sf::Tu tmp;
                tmp.v=lpic[x][y].v;
                tmp.d[0] += xexp.expm1(att*a.r)*(tmp.d[0]-q.r);
                tmp.d[1] += xexp.expm1(att*a.g)*(tmp.d[1]-q.g);
                tmp.d[2] += xexp.expm1(att*a.b)*(tmp.d[2]-q.b);
                lpic[x][y]=tmp.v;
                }
#else
              lpic[x][y].r += xexp.expm1(att*a.r)*(lpic[x][y].r-q.r);
              lpic[x][y].g += xexp.expm1(att*a.g)*(lpic[x][y].g-q.g);
              lpic[x][y].b += xexp.expm1(att*a.b)*(lpic[x][y].b-q.b);
#endif
              }
            }
          }
        } // for particle
      }
    for (int ix=0;ix<x1;ix++)
      for (int iy=0;iy<y1;iy++)
#ifdef __SSE2__
        {
        COLOUR &c(pic[ix+x0s][iy+y0s]); float32 dum;
        lpic[ix][iy].writeTo(c.r,c.g,c.b,dum);
        }
#else
        pic[ix+x0s][iy+y0s]=lpic[ix][iy];
#endif
    } // for this chunk
} // #pragma omp parallel

  tstack_pop("Host Rendering proper");
  }
}

using namespace host_funct;

void host_rendering (paramfile &params, vector<particle_sim> &particles,
  arr2<COLOUR> &pic, const vec3 &campos, const vec3 &lookat, const vec3 &sky,
  vector<COLOURMAP> &amap, float b_brightness, tsize npart_all)
  {
    bool master = mpiMgr.master();
    tsize npart = particles.size();
  //tsize npart_all = npart;
  //mpiMgr.allreduce (npart_all,MPI_Manager::Sum);

// -------------------------------------
// ----------- Transforming ------------
// -------------------------------------
  tstack_push("3D transform");
  if (master)
    cout << endl << "host: applying geometry (" << npart_all << ") ..." << endl;
  particle_project(params, particles, campos, lookat, sky);
  tstack_replace("3D transform","Particle coloring");

// ------------------------------------
// ----------- Coloring ---------------
// ------------------------------------
  if (master)
    cout << endl << "host: calculating colors (" << npart_all << ") ..." << endl;
  particle_colorize(params, particles, amap, b_brightness);
  tstack_pop("Particle coloring");

#if 0
// ------------------------------------
// -- Eliminating inactive particles --
// ------------------------------------
  tstack_push("Particle elimination");
  if (master)
    cout << endl << "host: eliminating inactive particles ..." << endl;
  tdiff i1=0, i2=particles.size()-1;
  while (true)
    {
    while (i1<=i2 && particles[i1].active) ++i1;
    while (i1<=i2 && !particles[i2].active) --i2;
    if (i1>=i2) break;
    swap(particles[i1],particles[i2]);
    }
  npart=i2+1;
  particles.resize(npart);
  npart_all=npart;
  mpiMgr.allreduce (npart_all,MPI_Manager::Sum);
  if (master)
    cout << npart_all << " particles left" << endl;
  tstack_pop("Particle elimination");
#endif

// --------------------------------
// ----------- Sorting ------------
// --------------------------------
  tstack_push("Particle sorting");
  if (!params.find<bool>("a_eq_e",true))
    {
    if (master)
      (mpiMgr.num_ranks()>1) ?
        cout << endl << "host: applying local sort ..." << endl :
        cout << endl << "host: applying sort (" << npart << ") ..." << endl;
    int sort_type = params.find<int>("sort_type",1);
    particle_sort(particles,sort_type,true);
    }
  tstack_pop("Particle sorting");

// ------------------------------------
// ----------- Rendering ---------------
// ------------------------------------
  if (master)
    cout << endl << "host: rendering (" << npart_all << "/" << npart_all << ")..." << endl;

  bool a_eq_e = params.find<bool>("a_eq_e",true);
  float32 grayabsorb = params.find<float32>("gray_absorption",0.2);

  tstack_push("Rendering");
  render_new (&(particles[0]),particles.size(),pic,a_eq_e,grayabsorb);
  tstack_pop("Rendering");
  }
