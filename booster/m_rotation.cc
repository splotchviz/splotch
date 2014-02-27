/*
 * Copyright (c) 2004-2010
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
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "kernel/transform.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/walltimer.h"
#include "cxxsupport/sse_utils_cxx.h"
#include "cxxsupport/string_utils.h"
#include "booster/mesh_vis.h"

#define SPLOTCH_CLASSIC

using namespace std;


const float32 h2sigma = 0.5*pow(pi,-1./6.);
const float32 sqrtpi = sqrt(pi);
const float32 sqrt2 = sqrt(2.0);

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

float gauss_weight(float dist, float sigma)
{

    float sigma2 = sigma*sigma;
    float norm = 1.0/(2.0*pi);
    norm = 1.0;
    float weight;
    float dist2=dist*dist;

    weight = norm * exp(-dist2/(2.0*sigma2));

    return weight;
}

void m_rotation(paramfile &params, Mesh_vis ** p, Mesh_dim MeshD, const vec3 &campos, const vec3 &lookat, vec3 sky)
  {
  int xres = params.find<int>("xres",800),
      yres = params.find<int>("yres",xres);
  float32 zmaxval = params.find<float32>("zmax",1.e23),
          zminval = params.find<float32>("zmin",0.0);
  zminval = max(0.f,zminval);


  float32 fov = params.find<float32>("fov",45); //in degrees
  float32 fovfct = tan(fov*0.5f*degr2rad);
  int npart=MeshD.ncell;

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

//  define wigthing factors along line of sight (sigmadd) and orthogonal to it (sigmavv)
//  float diag = MeshD.Lx*MeshD.Lx+MeshD.Ly*MeshD.Ly+MeshD.Lz*MeshD.Lz;
//  diag = pow(diag,1.0/3.0);
//  float sigmadd = 0.5*diag;
//  float sigmavv = diag;

  long m;
  for (m=0; m<npart; ++m)
   { 
    (*p)[m].active = true;
    (*p)[m].weight = 0.0;
    vec3 v((*p)[m].posx,(*p)[m].posy,(*p)[m].posz);
    v=trans.TransPoint(v);

    float diag;
    diag = MeshD.Dx*MeshD.Dx+MeshD.Dy*MeshD.Dy+MeshD.Dz*MeshD.Dz;
    diag = sqrt(diag);

// check if the projected cell is active

    if (v.z<=zminval) {(*p)[m].active = false; continue;};
    if (v.z>=zmaxval) {(*p)[m].active = false; continue;};
 
    float effvx = v.x - (fabs(v.x)/v.x)*diag;
    float effvy = v.y - (fabs(v.y)/v.y)*diag;
    float effvv2 = effvx*effvx + effvy*effvy;
    float effvv = sqrt(effvv2);
    float effdd = sqrt2 * v.z * fovfct;

    if (effvv > effdd){(*p)[m].active = false; continue;};

    float quality_factor = params.find<float>("quality_factor",1.0);
    (*p)[m].weight = quality_factor;

// weight othogonal to the line of sight
    float sigma_eff = 1.0*effdd;
    (*p)[m].weight *= gauss_weight(effvv,sigma_eff);
// weight along the line of sight
    sigma_eff = dist;
    (*p)[m].weight *= gauss_weight(v.z,sigma_eff);
    }

/*
    for (m=0; m<npart; ++m)
    {

       if((*p)[m].active == true)cout << (*p)[m].posx << " " << (*p)[m].posy << " " << (*p)[m].posz << endl;

    }
*/

}
