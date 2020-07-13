/*
 * Copyright (c) 2015
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

#ifndef MPI_A_NEQ_E

#include <memory>

#include "splotch/new_renderer.h"
#include "splotch/splotch_host.h"
#include "cxxsupport/geom_utils.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/walltimer.h"

using namespace std;

#define SPLOTCH_CLASSIC

namespace {

const double h2sigma = 0.5*pow(pi,-1./6.); // 0.413
const double sqrtpi = sqrt(pi); // 1.77245
#ifdef SPLOTCH_CLASSIC
const double bfak=0.5*pow(pi,-5./6.); // 0.19261
#endif

struct partg
  {
  vec3 c;
  double r;
  };

class Projector
  {
  public:
    virtual vec3 getPixDir (int ix, int iy) const = 0;
    virtual void reduceList(int ix0, int ix1, int iy0, int iy1,
      const vector<partg> &p, const vector<uint32> &ilist,
      vector<uint32> &ilist2) const = 0;
    virtual double pixRad() const = 0;
  };

class PlaneProjector: public Projector
  {
  private:
    vec3 p0, dpx, dpy;

    vec3 getPixPos (double x, double y) const
      { return p0+dpx*x+dpy*y; }

  public:
    PlaneProjector (const vec3 &p0_, const vec3 &dpx_, const vec3 &dpy_)
      : p0(p0_), dpx(dpx_), dpy(dpy_) {}

    virtual vec3 getPixDir (int ix, int iy) const
      { return (p0+dpx*ix+dpy*iy).Norm(); }

    virtual void reduceList(int ix0, int ix1, int iy0, int iy1,
      const vector<partg> &p, const vector<uint32> &ilist,
      vector<uint32> &ilist2) const
      {
      double x0=ix0,x1=ix1,y0=iy0,y1=iy1;
      if (ix1==ix0+1) x1+=0.001;
      if (iy1==iy0+1) y1+=0.001;
      ilist2.clear();
      vec3 corner[4];
      corner[0] = getPixPos(x0,y0);
      corner[1] = getPixPos(x1-1,y0);
      corner[2] = getPixPos(x1-1,y1-1);
      corner[3] = getPixPos(x0,y1-1);
      vec3 normal[4];
      normal[0]=crossprod(corner[0],corner[1]).Norm();
      normal[1]=crossprod(corner[1],corner[2]).Norm();
      normal[2]=crossprod(corner[2],corner[3]).Norm();
      normal[3]=crossprod(corner[3],corner[0]).Norm();
      for (tsize i=0; i<ilist.size(); ++i)
        {
        const partg &pg(p[ilist[i]]);
        if ((pg.r>dotprod(pg.c,normal[0])) &&
            (pg.r>dotprod(pg.c,normal[1])) &&
            (pg.r>dotprod(pg.c,normal[2])) &&
            (pg.r>dotprod(pg.c,normal[3])))
          ilist2.push_back(ilist[i]);
        }
     }

    virtual double pixRad() const
      {
      double screendist=std::abs(dotprod(p0,crossprod(dpx,dpy).Norm()));
      return min(dpx.Length(),dpy.Length())/screendist;
      }
  };

class DomeProjector: public Projector
  {
  private:
    vec3 zenith, x, y;
    double dp;

  public:
    DomeProjector (const vec3 &zenith_, const vec3 &east_, int res_)
      {
      dp = 1./res_;
      zenith = zenith_.Norm();
      y = crossprod(zenith,east_).Norm();
      x = crossprod(y,zenith).Norm();
      }

    vec3 igetPixDir (double px, double py) const
      {
      double r=sqrt(px*px+py*py), xr=1./r;
      double theta=halfpi*r;
      double sthxr=sin(theta)*xr;
      return x*(sthxr*px)+y*(sthxr*py)+zenith*cos(theta);
      }
    virtual vec3 getPixDir (int ix, int iy) const
      {
      double px=2.*dp*(ix+.5)-1., py=2.*dp*(iy+.5)-1.;
      return igetPixDir(px,py);
      }

    virtual void reduceList(int ix0, int ix1, int iy0, int iy1,
      const vector<partg> &p, const vector<uint32> &ilist,
      vector<uint32> &ilist2) const
      {
#if 0
        cout << ix0 <<  " " << ix1 << " " << iy0 << " " << iy1 << endl;
//ilist2=ilist; return;
      ilist2.clear();
      double x0=2.*dp*(ix0+.5)-1., x1=2.*dp*(ix1-.5)-1.;
      double y0=2.*dp*(iy0+.5)-1., y1=2.*dp*(iy1-.5)-1.;
      if (ix1==ix0+1) x1+=0.001*dp;
      if (iy1==iy0+1) y1+=0.001*dp;
double xc=0.5*(x0+x1),yc=0.5*(y0+y1);
double maxang=0;
vec3 cdir=igetPixDir(xc,yc).Norm();
maxang=max(maxang,v_angle(igetPixDir(x0,y0).Norm(),cdir));
maxang=max(maxang,v_angle(igetPixDir(x0,y1).Norm(),cdir));
maxang=max(maxang,v_angle(igetPixDir(x1,y0).Norm(),cdir));
maxang=max(maxang,v_angle(igetPixDir(x1,y1).Norm(),cdir));
cout << "maxang = " << maxang << endl;
      for (tsize i=0; i<ilist.size(); ++i)
        {
        const partg &pg(p[ilist[i]]);
//      if (dotprod(pg.c,cdir)<-0.5) continue;
        double ang=maxang+atan2(pg.r,sqrt((1.+pg.r)*(1.-pg.r)));
        ang=min(pi,ang);
        if (dotprod(pg.c.Norm(),cdir)<cos(ang))
          continue;
        ilist2.push_back(ilist[i]);
        }
#else
      double x0=2.*dp*(ix0+.5)-1., x1=2.*dp*(ix1-.5)-1.;
      double y0=2.*dp*(iy0+.5)-1., y1=2.*dp*(iy1-.5)-1.;
      if (ix1==ix0+1) x1+=0.001*dp;
      if (iy1==iy0+1) y1+=0.001*dp;
      bool centered=((x0*x1)<=0) && ((y0*y1)<=0); // area contains zenith
      double amaxx=max(abs(x0),abs(x1)), amaxy=max(abs(y0),abs(y1));
      double aminx=min(abs(x0),abs(x1)), aminy=min(abs(y0),abs(y1));
      double thetamax=halfpi*sqrt(amaxx*amaxx+amaxy*amaxy);
      double thetamin=0; // correct in centered case;
      if (!centered)
        {
        if (x0*x1<=0) thetamin=halfpi*aminy;
        else if (y0*y1<=0) thetamin=halfpi*aminx;
        else thetamin = halfpi*sqrt(aminx*aminx+aminy*aminy);
        }
      double cthetamin=cos(thetamin),cthetamax=cos(thetamax);
      vec3 c (x*((x0+x1)*.5)+y*((y0+y1)*.5));
      vec3 n[4];
      n[0]=(x*y0-y*x0).Norm();
      n[1]=(x*y1-y*x0).Norm();
      n[2]=(x*y0-y*x1).Norm();
      n[3]=(x*y1-y*x1).Norm();
      vec3 nx0=n[0],nx1=n[1];
      double curmin=1.5;
      for (tsize i=0; i<4; ++i)
        for (tsize j=i+1; j<4; ++j)
          if (dotprod(n[i],n[j])<curmin)
            {
            curmin=dotprod(n[i],n[j]);
            nx0=n[i]; nx1=n[j];
            }
      if (dotprod(c,nx0)>0) nx0.Flip();
      if (dotprod(c,nx1)>0) nx1.Flip();
      ilist2.clear();
      for (tsize i=0; i<ilist.size(); ++i)
        {
        const partg &pg(p[ilist[i]]);
        // halfspace check
        if (pg.r<-dotprod(zenith,pg.c))
          continue;
        // phi check
        if (!centered)
          {
          if ((pg.r<dotprod(pg.c,nx0)) ||
              (pg.r<dotprod(pg.c,nx1)))
            continue;
          }
        // theta check, still very slow
        double cpthetac=dotprod(zenith,pg.c);
        if ((cpthetac>cthetamin)||(cpthetac<cthetamax))
          {
          double pthetac=v_angle(zenith,pg.c);
          double dptheta=atan2(pg.r,sqrt((1.+pg.r)*(1.-pg.r)));
          if ((pthetac+dptheta<thetamin)||(pthetac-dptheta>thetamax))
            continue;
          }
        ilist2.push_back(ilist[i]);
        }
#endif
      }

    virtual double pixRad() const
      { return dp*pi; }
  };

typedef shared_ptr< vector<uint32> > ilsp;

struct rjob
  {
  tsize x0,x1,y0,y1;
  ilsp pilist;
  rjob() {}
  rjob (int x0_, int x1_, int y0_, int y1_, const ilsp &pilist_)
    : x0(x0_), x1(x1_), y0(y0_), y1(y1_), pilist(pilist_) {}
  };

void subrender4(const Projector &projector, int fact,
  const vector<partg> &p, const vector<COLOUR> &pod,
  const ilsp pilist, arr2<COLOUR> &pic)
  {
  planck_assert(pod.size()==p.size(),"inconsistency");
  tsize xres=pic.size1(), yres=pic.size2();
  exptable<double> xexp(-20.);
  vector<rjob> jobs;
  jobs.push_back(rjob(0,xres,0,yres,pilist));
  int nworking=0;

#pragma omp parallel
{
  arr2<COLOUR> lpic;
  rjob j;
  bool done=false;
  do
    {
    bool have_job=false;
#pragma omp critical (splotch_render)
{//cout << jobs.size() << " " << nworking << endl;
    if (!jobs.empty()) // fetch the last job in the queue
      {
      j=jobs.back();
      jobs.pop_back();
      have_job=true;
      ++nworking;
      done=false;
      }
    else // currently no jobs in the queue
      done = (nworking==0); // if no other thread is working, we can stop
}
    while (have_job)
      {
      ilsp pilist2(new vector<uint32>());
      if ((j.x0==0)&&(j.x1==xres)&&(j.y0==0)&&(j.y1==yres)) // whole image
        pilist2=j.pilist;
      else
        projector.reduceList(j.x0*fact,(j.x1-1)*fact+1,j.y0*fact,
          (j.y1-1)*fact+1,p,*j.pilist,*pilist2);
      const vector<uint32> &ilist2(*pilist2);
      if (ilist2.empty())
        {
        for (tsize ix=j.x0; ix<j.x1; ++ix)
          for (tsize iy=j.y0; iy<j.y1; ++iy)
            pic[ix][iy]=COLOUR(0,0,0);
#pragma omp critical (splotch_render)
{
        --nworking;
}
        have_job=false;
        }
      else
        {
        tsize nx=j.x1-j.x0,ny=j.y1-j.y0;
        if ((nx<10)&&(ny<10)) // small enough, do rendering
          {
          lpic.fast_alloc(nx,ny);
          for (tsize ix=j.x0; ix<j.x1; ++ix)
            {
            for (tsize iy=j.y0; iy<j.y1; ++iy)
              {
              vec3 dir = projector.getPixDir(ix*fact,iy*fact);
              COLOUR od(0.,0.,0.); // accumulated optical depth in this pixel
              for (tsize i=0; i<ilist2.size(); ++i)
                {
                double r=p[ilist2[i]].r;
                vec3 c=p[ilist2[i]].c;
                double cosang=dotprod(dir,c);
              if (cosang<0) continue;
                double casq=cosang*cosang;
                double crsq=(1.-r)*(1.+r);
                if (casq<crsq) continue; // no intersection
                double sasq=(1.-cosang)*(1.+cosang);
                double fract=sasq/(r*r*h2sigma*h2sigma);
                od+=pod[ilist2[i]]*(-xexp(-fract));
                }
              lpic[ix-j.x0][iy-j.y0]=od;
              }
            }
          for (tsize ix=j.x0; ix<j.x1; ++ix)
            for (tsize iy=j.y0; iy<j.y1; ++iy)
              pic[ix][iy]=lpic[ix-j.x0][iy-j.y0];
#pragma omp critical (splotch_render)
{
        --nworking;
}
          have_job=false;
          }
        else // need to subdivide
          {
#pragma omp critical (splotch_render)
{
          int xm=(j.x0+j.x1)>>1, ym=(j.y0+j.y1)>>1;
          if (nx>sqrt(2.)*ny)
            {
            jobs.push_back(rjob(j.x0,xm,j.y0,j.y1,pilist2));
            j.x0=xm; j.pilist=pilist2;
            }
          else if (ny>sqrt(2.)*nx)
            {
            jobs.push_back(rjob(j.x0,j.x1,j.y0,ym,pilist2));
            j.y0=ym; j.pilist=pilist2;
            }
          else
            {
            jobs.push_back(rjob(j.x0,xm,j.y0,ym,pilist2));
            jobs.push_back(rjob(j.x0,xm,ym,j.y1,pilist2));
            jobs.push_back(rjob(xm,j.x1,j.y0,ym,pilist2));
            j.x0=xm; j.y0=ym; j.pilist=pilist2;
            }
}
          } // subdivision
        } // ilist2 not empty
      } // while (have_job)
    } while (!done);
}
  }

//FIXME: needs tuning
void add_overlap2(const Projector &projector, int fact,
  const vector<partg> &p, const vector<COLOUR> &pod, arr2<COLOUR> &pic)
  {
  planck_assert(pod.size()==p.size(),"inconsistency");
  tsize xres=pic.size1(), yres=pic.size2();

  exptable<double> xexp(-20.);

  for (tsize ix=0; ix<xres; ++ix)
    {
    for (tsize iy=0; iy<yres; ++iy)
      {
      vec3 dir = projector.getPixDir(ix*fact, iy*fact);
      COLOUR od(0.,0.,0.); //optical depth
      for (tsize i=0; i<p.size(); ++i)
        {
        double r=p[i].r;
        vec3 c=p[i].c;
        double cosang=dotprod(dir,c);
        double sasq=(1.-cosang)*(1.+cosang);
        double fract=sasq/(r*r*h2sigma*h2sigma);
        od += 0.5*(erfc(-cosang/(r*h2sigma)))*pod[i]*(-xexp(-fract));
        }
      pic[ix][iy]+=od;
      }
    }
  }

void real_host_rendering(const Projector &projector,
  vector<partg> &p, vector<COLOUR> &pod, double minrad_pix, arr2<COLOUR> &pic)
  {
  planck_assert(pod.size()==p.size(),"inconsistency");
  tsize xres=pic.size1(),yres=pic.size2();
  double pixrad=projector.pixRad();
tstack_push("inner prep");

  // normalization
tstack_push("norm");
#pragma omp parallel for
  for (tsize i=0; i<p.size(); ++i)
    {
    double xl=1./p[i].c.Length();
    p[i].c*=xl;
    p[i].r*=xl;
    double rcorr =
      sqrt(p[i].r*p[i].r + minrad_pix*minrad_pix*pixrad*pixrad)/p[i].r;
    p[i].r*=rcorr;
    pod[i]*=(1./(rcorr*rcorr));
    }
tstack_pop("norm");

  // extract particles overlapping with camera (r>1) for special treatment
  // discard particles not overlapping with the screen
  vector<partg> po;
  vector<COLOUR> podo;
  {
  vector<uint32> ilist(p.size()), ilist2;
  for (tsize i=0; i<p.size(); ++i)
    ilist[i]=i;
  projector.reduceList(0,xres,0,yres,p,ilist,ilist2);
  tsize idx=0;
  for (tsize i=0; i<ilist2.size(); ++i)
    {
    if (p[ilist2[i]].r>1)
      {
      po.push_back(p[ilist2[i]]);
      podo.push_back(pod[ilist2[i]]);
      }
    else
      {
      p[idx]=p[ilist2[i]];
      pod[idx]=pod[ilist2[i]];
      ++idx;
      }
    }
  //cout << "Reduced particle list from " << p.size() << " to " << idx << endl;
  //cout << "Found " << po.size() << " particles overlapping with camera" << endl;
  p.resize(idx); p.shrink_to_fit();
  pod.resize(idx); pod.shrink_to_fit();
  }
  {
  // redistribute particles
  int nranks=mpiMgr.num_ranks();
  if (nranks>1)
    {
    arr<int> numin(nranks), numout;
    for (tsize i=0; i<mpiMgr.num_ranks(); ++i)
      {
      int64 lo, hi;
      calcShareGeneral(0,p.size(),nranks,i,lo,hi);
      numin[i]=hi-lo;
      }
    {
    vector<partg> p2;
    mpiMgr.all2allv_easy_typeless(p,numin,p2,numout);
    p.swap(p2);
    }
    {
    vector<COLOUR> pod2;
    mpiMgr.all2allv_easy_typeless(pod,numin,pod2,numout);
    pod.swap(pod2);
    }
    }
  }

  const double minpixperrad=15;
  double maxfac=1./(minpixperrad*pixrad);
  maxfac=min(maxfac,min(xres,yres)/10.);
  int nsteps=0;
  while ((1<<nsteps)<maxfac)
    ++nsteps;
  nsteps=max(nsteps-1,0);
  //cout << "nsteps = " << nsteps << endl;
  pic.dealloc();
tstack_pop("inner prep");
  for (int fact=1<<nsteps; fact>0; fact>>=1)
    {
tstack_push("loop prep");
//cout << fact << endl;
    int xdim=xres,ydim=yres;
    for (int fact2=fact; fact2>1; fact2>>=1)
      {
      xdim=(xdim>>1)+1;
      ydim=(ydim>>1)+1;
      }
    arr2<COLOUR> xpic(xdim,ydim);

    double minrad=minpixperrad*pixrad*fact;
    double maxrad=min(1.,2*minrad);
    if (fact==1) minrad=0.;
    if (fact==1<<nsteps) maxrad=1;
    ilsp pilist(new vector<uint32>());
    for (uint32 i=0; i<p.size(); ++i)
      if ((p[i].r>minrad)&&(p[i].r<=maxrad))
        pilist->push_back(i);
tstack_pop("loop prep");
    subrender4(projector, fact, p, pod, pilist, xpic);
tstack_push("overlap");
    if (fact==1<<nsteps)
      add_overlap2(projector, fact, po, podo, xpic);
tstack_pop("overlap");
tstack_push("adding");
    if (fact!=1<<nsteps) // add coarser contributions
#pragma omp parallel for
      for (int ix=0; ix<xdim; ++ix)
        for (int iy=0; iy<ydim; ++iy)
          xpic[ix][iy] +=(pic[ix>>1][iy>>1]+pic[(ix+1)>>1][iy>>1]
                         +pic[ix>>1][(iy+1)>>1]+pic[(ix+1)>>1][(iy+1)>>1])*0.25;
    pic.swap(xpic);
tstack_pop("adding");
    }
  }

} // unnamed namespace

void host_rendering_new (paramfile &params, vector<particle_sim> &particles,
  arr2<COLOUR> &pic, const vec3 &campos, const vec3 &centerpos,
  const vec3 &lookat, const vec3 &sky0,
  vector<COLOURMAP> &amap, float b_brightness, tsize npart_all)
  {
  tstack_push("Rendering");
tstack_push("prep");
    bool master = mpiMgr.master();
    tsize npart = particles.size();
    tsize npart_use = 0;
vec3 sky=sky0;
float32 zmaxval = params.find<float32>("zmax",1.e23);
#pragma omp parallel for reduction(+:npart_use)
  for (tsize m=0; m<particles.size(); ++m)
    {
  #ifdef SPLOTCH_CLASSIC
    particles[m].I *= 0.5f*bfak/particles[m].r;
    particles[m].r *= 2;
  #else
    particles[m].I *= 8.f/(pi*particles[m].r*particles[m].r*particles[m].r); // SPH kernel normalisation
    particles[m].I *= (h2sigma*sqrtpi*particles[m].r); // integral through the center
  #endif

vec3 this_pos(particles[m].x,particles[m].y,particles[m].z);
float32 dist = (centerpos-this_pos).Length();

if(dist < zmaxval)
  {
   particles[m].active = true;
   npart_use++;
  }
else
   particles[m].active = false;

    }

  if (master)
    cout << endl << "host: calculating colors (" << npart_all << ") ..." << endl;
  host_funct::particle_colorize(params, particles, amap, b_brightness);

  bool a_eq_e = params.find<bool>("a_eq_e",true);
  planck_assert(a_eq_e,"inconsistent");

  tsize npart_use_all = npart_use;
  mpiMgr.allreduce (npart_use_all,MPI_Manager::Sum);
  if (master)
    cout << "host: -> restricting to (" << npart_use_all << ") ..." << endl;

  vector<partg> p(npart_use);
  vector<COLOUR> pod(npart_use);
  tsize j=0;
  for (tsize i=0; i<particles.size(); ++i)
    {
      if(particles[i].active)
	{
	  p[j].c=vec3(particles[i].x,particles[i].y,particles[i].z)-campos;
	  p[j].r=particles[i].r;
	  pod[j]=particles[i].e;
	  j++;
	}
    }

particles.resize(0); particles.shrink_to_fit();

  bool dome = params.find<bool>("dome",false);
  unique_ptr<Projector> projector;
  if (dome)
    {
    tsize res = params.find<int>("xres",800);
    planck_assert(res==pic.size1(),"inconsistency");
    planck_assert(res==pic.size2(),"inconsistency");
    projector.reset(new DomeProjector(lookat-campos, sky, res));
    }
  else
    {
    tsize xres = params.find<int>("xres",800),
          yres = params.find<int>("yres",xres);
    planck_assert(xres==pic.size1(),"inconsistency");
    planck_assert(yres==pic.size2(),"inconsistency");
    double fov = params.find<double>("fov",45); //in degrees
    double fovfct = tan(fov*0.5f*degr2rad);
    sky.Normalize();
    vec3 zaxis = (centerpos-lookat).Norm();
    vec3 xaxis = crossprod (sky,zaxis).Norm();
    vec3 yaxis = crossprod (zaxis,xaxis).Norm();
    // [xyz]axis now form a right-handed orthonormal system
    vec3 dpx=xaxis*(fovfct/(0.5*xres));
    vec3 dpy=yaxis*(fovfct/(0.5*xres/*sic!*/));
    vec3 newcenter=(centerpos-campos)*(1./(centerpos-lookat).Length());
    vec3 corner=newcenter-zaxis-dpx*(0.5*xres)-dpy*(0.5*yres);
    projector.reset(new PlaneProjector(corner+dpx*.5+dpy*.5, dpx, dpy));
    }
  double minrad_pix = params.find<double>("minrad_pix",1.);
tstack_pop("prep");
  real_host_rendering(*projector, p, pod, minrad_pix, pic);
  if (dome)
    {
    tstack_push("post");
    tsize res = params.find<int>("xres",800);
    double xf=1./(res-1);
#pragma omp parallel for schedule (dynamic, 30)
    for (tsize i=0; i<=res/2; ++i)
      {
      double x=(i+0.5)*xf;
      x=2*x-1;
      for (tsize j=0; j<res; ++j)
        {
        double y=(j+0.5)*xf;
        y=2*y-1;
        if (x*x+y*y>1.)
          pic[i][j]=pic[i][res-1-j]=pic[res-1-i][res-1-j]=pic[res-1-i][j]
            = COLOUR(0.,0.,0.);
        else
          break;
        }
      }
    tstack_pop("post");
    }
  tstack_pop("Rendering");
  }
#endif