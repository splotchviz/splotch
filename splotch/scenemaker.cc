/**
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
#include <fstream>

#include "splotch/scenemaker.h"
#include "splotch/splotchutils.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/walltimer.h"
#include "cxxsupport/cxxutils.h"
#include "cxxsupport/datatypes.h"
#include "reader/reader.h"
#include "booster/mesh_vis.h"

#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

using namespace std;

void sceneMaker::particle_normalize(vector<particle_sim> &p, bool verbose)
  const
  {
  // how many particle types are there
  int nt = params.find<int>("ptypes",1);
  arr<bool> col_vector(nt),log_int(nt),log_col(nt),asinh_col(nt);
  arr<Normalizer<float32> > intnorm(nt), colnorm(nt), sizenorm(nt);

  // Get data from parameter file
  for(int t=0; t<nt; t++)
    {
    log_int[t] = params.find<bool>("intensity_log"+dataToString(t),false);
    log_col[t] = params.find<bool>("color_log"+dataToString(t),false);
    asinh_col[t] = params.find<bool>("color_asinh"+dataToString(t),false);
    col_vector[t] = params.find<bool>("color_is_vector"+dataToString(t),false);
    }

  int npart=p.size();
  tstack_push("minmax");
#pragma omp parallel
{
  // FIXME: the "+20" serves as protection against false sharing,
  // should be done more elegantly
  arr<Normalizer<float32> > inorm(nt+20), cnorm(nt+20), rnorm(nt+20);
  int m;
#ifdef CUDA
  // In cuda version logs are performed on device
  #pragma omp for schedule(guided,1000)
  for (m=0; m<npart; ++m)
    {
    int t=p[m].type;

    rnorm[t].collect(p[m].r);

    if (log_int[t])
      {
      if (p[m].I>0)
        inorm[t].collect(p[m].I);
      else
        p[m].I = -38;
      }
    else
      inorm[t].collect(p[m].I);

    if (log_col[t])
      {
      if (p[m].e.r>0)
        cnorm[t].collect(p[m].e.r);
      else
        p[m].e.r = -38;
      }
    else
      {
      if(asinh_col[t])
        p[m].e.r = my_asinh(p[m].e.r);

      cnorm[t].collect(p[m].e.r);
      }

    if (col_vector[t])
      {
      if (asinh_col[t])
        {
        p[m].e.g = my_asinh(p[m].e.g);
        p[m].e.b = my_asinh(p[m].e.b);
        }

      cnorm[t].collect(p[m].e.g);
      cnorm[t].collect(p[m].e.b);
      }
    }

#pragma omp critical
  for (int t=0; t<nt; t++)
    {
    if (log_int[t])
      {
      if (inorm[t].minv > 0)
        inorm[t].minv = log10(inorm[t].minv);

      if (inorm[t].maxv > 0)
        inorm[t].maxv = log10(inorm[t].maxv);
      }

    if (log_col[t])
      {
      if(cnorm[t].minv > 0)
        cnorm[t].minv = log10(cnorm[t].minv);

      if(cnorm[t].maxv > 0)
        cnorm[t].maxv = log10(cnorm[t].maxv);
      }
    }

  for (int t=0; t<nt; t++)
    {
    intnorm[t].collect(inorm[t]);
    colnorm[t].collect(cnorm[t]);
    sizenorm[t].collect(rnorm[t]);
    }
}

#else

#pragma omp for schedule(guided,1000)
  for (m=0; m<npart; ++m) // do log calculations if requested
    {
    int t=p[m].type;

    rnorm[t].collect(p[m].r);

    if (log_int[t])
      {
      if(p[m].I > 0)
        {
        p[m].I = log10(p[m].I);
        inorm[t].collect(p[m].I);
        }
      else
        p[m].I = -38;
      }
    else
      inorm[t].collect(p[m].I);

    if (log_col[t])
      {
      if(p[m].e.r > 0)
        {
        p[m].e.r = log10(p[m].e.r);
        cnorm[t].collect(p[m].e.r);
        }
      else
        p[m].e.r =-38;
      }
    else
      {
      if (asinh_col[t])
        p[m].e.r = my_asinh(p[m].e.r);
      cnorm[t].collect(p[m].e.r);
      }

    if (col_vector[t])
      {
      if (log_col[t])
        {
        p[m].e.g = log10(p[m].e.g);
        p[m].e.b = log10(p[m].e.b);
        }
      if (asinh_col[t])
        {
        p[m].e.g = my_asinh(p[m].e.g);
        p[m].e.b = my_asinh(p[m].e.b);
        }
      cnorm[t].collect(p[m].e.g);
      cnorm[t].collect(p[m].e.b);
      }
    }
#pragma omp critical
  for(int t=0; t<nt; t++)
    {
    intnorm[t].collect(inorm[t]);
    colnorm[t].collect(cnorm[t]);
    sizenorm[t].collect(rnorm[t]);
    }
  }
#endif
  for(int t=0; t<nt; t++)
    {
    mpiMgr.allreduce(intnorm[t].minv,MPI_Manager::Min);
    mpiMgr.allreduce(colnorm[t].minv,MPI_Manager::Min);
    mpiMgr.allreduce(sizenorm[t].minv,MPI_Manager::Min);
    mpiMgr.allreduce(intnorm[t].maxv,MPI_Manager::Max);
    mpiMgr.allreduce(colnorm[t].maxv,MPI_Manager::Max);
    mpiMgr.allreduce(sizenorm[t].maxv,MPI_Manager::Max);

    if (verbose && mpiMgr.master())
      {
      cout << " For particles of type " << t << ":" << endl;
      cout << " From data: " << endl;
      cout << " Color Range:     " << colnorm[t].minv << " (min) , " <<
           colnorm[t].maxv << " (max) " << endl;
      cout << " Intensity Range: " << intnorm[t].minv << " (min) , " <<
           intnorm[t].maxv << " (max) " << endl;
      cout << " Size Range: " << sizenorm[t].minv << " (min) , " <<
	    sizenorm[t].maxv << " (max) " << endl;
      }
    // Set the range and clamp values in parameter file so we can find them again later
    params.setParam("range_min"+dataToString(t), colnorm[t].minv);
    params.setParam("range_max"+dataToString(t), colnorm[t].maxv);

    if(params.param_present("intensity_min"+dataToString(t)))
      intnorm[t].minv = params.find<float>("intensity_min"+dataToString(t));

    if(params.param_present("intensity_max"+dataToString(t)))
      intnorm[t].maxv = params.find<float>("intensity_max"+dataToString(t));

    if(params.param_present("color_min"+dataToString(t)))
      colnorm[t].minv = params.find<float>("color_min"+dataToString(t));

    if(params.param_present("color_max"+dataToString(t)))
      colnorm[t].maxv = params.find<float>("color_max"+dataToString(t));

    params.setParam("clamp_min"+dataToString(t), colnorm[t].minv);
    params.setParam("clamp_max"+dataToString(t), colnorm[t].maxv);

    if (verbose && mpiMgr.master())
      {
      cout << " Restricted to: " << endl;
      cout << " Color Range:     " << colnorm[t].minv << " (min) , " <<
           colnorm[t].maxv << " (max) " << endl;
      cout << " Intensity Range: " << intnorm[t].minv << " (min) , " <<
           intnorm[t].maxv << " (max) " << endl;
      }
    }

  tstack_pop("minmax");

#ifdef CUDA

  // Write max/mins to param file to be used in cuda norm/clamping
  for(int t=0; t<nt; t++)
    {
    params.setParam("intensity_min"+dataToString(t), intnorm[t].minv);
    params.setParam("intensity_max"+dataToString(t), intnorm[t].maxv);
    params.setParam("color_min"+dataToString(t), colnorm[t].minv);
    params.setParam("color_max"+dataToString(t), colnorm[t].maxv);
    }

  params.setParam("cuda_doLogs", true);

#else
  if (mpiMgr.master())
    cout << " Host normalization and clamping" << endl;
  tstack_push("clamp");

#pragma omp parallel
{
  int m;
#pragma omp for schedule(guided,1000)
  for(m=0; m<npart; ++m)
    {
    int t=p[m].type;
#ifndef NO_I_NORM
    intnorm[t].normAndClamp(p[m].I);
#endif
    colnorm[t].normAndClamp(p[m].e.r);
    if (col_vector[t])
      {
      colnorm[t].normAndClamp(p[m].e.g);
      colnorm[t].normAndClamp(p[m].e.b);
      }
    }
}
  tstack_pop("clamp");

#endif
  } // END particle_normalize

// Higher order interpolation would be:
// Time between snapshots (cosmology!)
//    dt=(z2t(h1.redshift)-z2t(h0.redshift))*0.7
// Velocity factors:
//    v_unit1=v_unit/l_unit/sqrt(h1.time)*dt
//    v_unit0=v_unit/l_unit/sqrt(h0.time)*dt
// Delta_v (cosmology)
//    vda=2*(x1-x0)-(v0*v_unit0+v1*v_unit1)
// Delta_t (0..1)
//     t=FLOAT(k)/FLOAT(nint) == frac (!)
// Interpolated positions:
//    x=x0+v0*v_unit0*t+0.5*(v1*v_unit1-v0*v_unit0+vda)*t^2
// Interpolated velocities:
//    v=v0+t*(v1-v0)

// booster main variables

Mesh_vis * Mesh = NULL;
Mesh_dim MeshD;

void sceneMaker::particle_interpolate(vector<particle_sim> &p,double frac) const
  {
  if (mpiMgr.master())
    cout << "particle_interpolate() : Time 1,2 = "
         << time1 << "," << time2 << endl << flush;

  releaseMemory(p);

  double v_unit1, v_unit2;
  if (interpol_mode>1)
    {
    double h = params.find<double>("hubble",0.7);
    double O = params.find<double>("omega",0.3);
    double L = params.find<double>("lambda",0.7);
    double mparsck = 3.0856780e+24;
    double l_unit = params.find<double>("l_unit",3.0856780e+21);
    double v_unit = params.find<double>("v_unit",100000.00);
    double t1 = log(sqrt(L/O*time1*time1*time1)
      +sqrt((L/O*time1*time1*time1)+1))/1.5/sqrt(L)/h/1e7*mparsck;
    double t2 = log(sqrt(L/O*time2*time2*time2)
      +sqrt((L/O*time2*time2*time2)+1))/1.5/sqrt(L)/h/1e7*mparsck;
    double dt = (t2 - t1) * h;
    v_unit1=v_unit/l_unit/sqrt(time1)*dt;
    v_unit2=v_unit/l_unit/sqrt(time2)*dt;
    //    cout << "Times: " << time1 << " " << time2 << " " << t1 << " " << t2 << " " << v_unit1 << " " << v_unit2 << endl;
    }

  vector<pair<MyIDType,MyIDType> > v;
  v.reserve(min(p1.size(),p2.size()));
    {
    MyIDType i1=0,i2=0;
    while(i1<p1.size() || i2<p2.size())
      {
      if (i1>=p1.size()) // reached the end of list 1, new particle appears
        {
        v.push_back(pair<MyIDType,MyIDType>(MyMaxID,idx2[i2]));
        i2++;
        }
      else if (i2>=p2.size()) // reached the end of list 2, particle disappears
        {
        v.push_back(pair<MyIDType,MyIDType>(idx1[i1],MyMaxID));
        i1++;
        }
      else // still particles in both lists
        {
        if (id1[idx1[i1]]==id2[idx2[i2]]) // particle evolves
        v.push_back(pair<MyIDType,MyIDType>(idx1[i1++],idx2[i2++]));
        else if (id1[idx1[i1]]<id2[idx2[i2]]) // particle disappears
          {
          v.push_back(pair<MyIDType,MyIDType>(idx1[i1],MyMaxID));
          i1++;
          }
        else if (id1[idx1[i1]]>id2[idx2[i2]]) // new particle appears
          {
          v.push_back(pair<MyIDType,MyIDType>(MyMaxID,idx2[i2]));
          i2++;
          }
        }
      }
    }

  tsize npart=v.size();
  p.resize(npart);

  bool periodic = params.find<bool>("periodic",true);
  double boxhalf = boxsize / 2;

#pragma omp parallel
{
  int i;
#pragma omp for schedule(guided,1000)
  for (i=0; i<int(npart); ++i)
    {
    MyIDType i1=v[i].first, i2=v[i].second;
    /*
    planck_assert (p1[i1].type==p2[i2].type,
      "interpolate: cannot interpolate between different particle types!");
    */
    particle_sim part1,part2;
    if(i1 < MyMaxID) part1=p1[i1]; else part1=p2[i2];
    if(i2 < MyMaxID) part2=p2[i2]; else part2=p1[i1];

    vec3f x1(part1.x,part1.y,part1.z), x2(part2.x,part2.y,part2.z);
    if (periodic)
      {
      if (abs(x2.x-x1.x) > boxhalf)
         (x2.x>x1.x) ? x2.x -= boxsize : x2.x += boxsize;
      if (abs(x2.y-x1.y) > boxhalf)
         (x2.y>x1.y) ? x2.y -= boxsize : x2.y += boxsize;
      if (abs(x2.z-x1.z) > boxhalf)
         (x2.z>x1.z) ? x2.z -= boxsize : x2.z += boxsize;
      }
    vec3f pos;
    if (interpol_mode>1)
      {
      vec3f v1(0,0,0),v2(0,0,0);
      if (i1 < MyMaxID && i2 < MyMaxID)
        {
        v1 = vel1[i1];
        v2 = vel2[i2];
        }
      if (i1 == MyMaxID)
        {
        v1 = v2 = vel2[i2];
        x1 = x2 - v1 / (0.5 * (v_unit1 + v_unit2));
        }
      if (i2 == MyMaxID)
        {
        v1 = v2 = vel1[i1];
        x2 = x1 + v1 / (0.5 * (v_unit1 + v_unit2));
        }
      if (interpol_mode == 2)           // polynomial interpolation
        {
        pos = x1 + (x2-x1)*3*frac*frac
                 - (x2-x1)*2*frac*frac*frac
                 + v1*v_unit1*frac
                 - (v1*2*v_unit1+v2*v_unit2)*frac*frac
                 + (v1*v_unit1+v2*v_unit2)*frac*frac*frac;
        }
      else                              // orbital interpolation
        {
        double mypos[3];
        for(int k=0;k<3;k++)
          {
          double myx1=0,myx2=0,myv1=0,myv2=0;
          if (k==0)
            {
            myx1=x1.x;
            myx2=x2.x;
            myv1=v1.x;
            myv2=v2.x;
            }
          if (k==1)
            {
            myx1=x1.y;
            myx2=x2.y;
            myv1=v1.y;
            myv2=v2.y;
            }
          if (k==2)
            {
            myx1=x1.z;
            myx2=x2.z;
            myv1=v1.z;
            myv2=v2.z;
            }

          // Interpolation on a eliptic orbit : x = a0 + a1*cos(a3*dt) + a2*sin(a3*dt)
          // we need to find the zero point of f(a) = (dv/dx)*(1-cos(a)) = a*sin(a)
          // or equivalent the solution of a/tan(0.5*a) = dv/dx
          double dvdx = (myv1*v_unit1+myv2*v_unit2) / (myx2-myx1);
          if(dvdx > 1.99)
            dvdx = 1.99;
          // produce a scaled version of a/tan(0.5*a) which can be simple inverted
          // (function is almost symmetric to the diagonal in the coordinate system)
          double xx=6.25;
          double yy=abs(xx/tan(0.5*xx));
          double dvdx_scale=(dvdx+yy)/(yy+2)*xx;
          double a_guess,a_found;
          double correction;

          if(dvdx >= 2)
            {
            a_guess = 12.5 / dvdx;
            double myp0 = 0.28596449, myp1 = -2.3100819;
            correction = pow(myp0 * dvdx , myp1);
            a_found = a_guess - correction + 2 * M_PI;
            }
          else
            {
            if(dvdx_scale >= xx)
              {
              a_guess = a_found = 1e-8;
              correction = 0;
              }
            else
              {
              a_guess = (dvdx_scale/tan(0.5*dvdx_scale)+yy)/(yy+2)*xx;
              // we have a complicated polynomial fit to do a fist correction to the result
              double myp0 = -0.0063529879 , myp1 = 0.42990545,  myp2 = -0.015337119, myp3 = -0.017165266,
                     myp4 =  0.16812639   , myp5 = 0.062027583, myp6 =  1.8925764;
              correction = myp0 * exp(myp5*pow(a_guess,myp6)) * pow(a_guess,myp1) *
                           (a_guess - 3.2518792) * pow(abs(a_guess - 3.2518792),myp2) *
                           (a_guess - 5.8155169) * pow(abs(a_guess - 5.8155169),myp3) *
                           (a_guess - 6.25) * pow(abs(a_guess - 6.25),myp4);
              a_found = a_guess + correction;
              }
            }
          // Finally do some newton-raphson steps to improve our result
          long iter=0;
          while((abs(dvdx - a_found/tan(0.5*a_found)) > 1e-6) && (iter < 10))
            {
            double da_found = -1 * (dvdx * (1-cos(a_found)) - a_found * sin(a_found)) /
                                  ((dvdx-1)*sin(a_found) - a_found * cos(a_found));
            if(a_found + da_found < 0)
              a_found = a_found*0.95;
            else
              a_found = a_found + da_found;
            iter++;
            if(iter > 6)
              {
              cout << "Iter: " << iter << " "
                    << dvdx << " "
                    << dvdx_scale << " "
                    << a_guess << " "
                    << a_guess + correction << " "
                    << a_found << " "
                    << dvdx - a_found/tan(0.5*a_found) << endl;
              }
            }
            // For the cray compiler downgrade this to a warning because theres something suspicious going on
            // FIXME: investigate futher..
         if(iter >= 10)
          #ifndef _CRAYC
           planck_fail("could not find zero point for interpolation fit !");        
          #else
            cout << "Warning: particle interpolate() could not find zero point for interpolation fit\n";
          #endif
          // Now find the other aprameters
          double a0,a1,a2=myv1*v_unit1/a_found;
          if (abs(sin(a_found)) < 1e-6)
            a1=(myx1-myx2+a2*sin(a_found))/(1-cos(a_found));
          else
            a1=(myv1*v_unit1*cos(a_found)-myv2*v_unit2)/(a_found*sin(a_found));
          a0=myx1-a1;
          // Now we can finally interpolate the positions
          mypos[k] = a0 + a1*cos(a_found*frac) + a2*sin(a_found*frac);
          }
        pos.x = mypos[0];
        pos.y = mypos[1];
        pos.z = mypos[2];
        }
      }
    else
      pos = x1*(1.-frac) + x2*frac;


    if (i1 < MyMaxID && i2 < MyMaxID)
      p[i]=particle_sim(p1[i1].e*(1.-frac)+p2[i2].e*frac,
        pos.x,pos.y,pos.z,
        (1-frac) * p1[i1].r  + frac*p2[i2].r,
        (1-frac) * p1[i1].I  + frac*p2[i2].I,
        p1[i1].type,p1[i1].active);
    if (i1 == MyMaxID)
      p[i]=particle_sim(p2[i2].e,
        pos.x,pos.y,pos.z,
        p2[i2].r,
        frac*p2[i2].I,
        p2[i2].type,p2[i2].active);
    if (i2 == MyMaxID)
      p[i]=particle_sim(p1[i1].e,
        pos.x,pos.y,pos.z,
        p1[i1].r,
        (1-frac)*p1[i1].I,
        p1[i1].type,p1[i1].active);
    }
}

  params.setParam("time", dataToString((1.-frac)*time1 + frac*time2));
  if ((redshift1>0.0) && (redshift2>0.0))
    params.setParam("redshift",dataToString((1.-frac)*redshift1+frac*redshift2));

  if (mpiMgr.master())
    cout << "particle_interpolate() : p1.size(), p2.size(), p.size() : "
        << p1.size() << ", " << p2.size() << ", " << p.size() << endl << flush;
  }


sceneMaker::sceneMaker (paramfile &par)
  : cur_scene(-1), params(par), snr1_now(-1), snr2_now(-1)
  {
  string outfile = params.find<string>("outfile");

  // do nothing if we are only analyzing ...
  if (params.find<bool>("AnalyzeSimulationOnly",false))
    {
    string outfilen = outfile+intToString(0,4);
    scenes.push_back(scene(outfilen,false,false));
    return;
    }

  string geometry_file = params.find<string>("scene_file","");
  interpol_mode = params.find<int>("interpolation_mode",0);
  if (geometry_file=="")
    {
    string outfilen = outfile+intToString(0,4);
    #ifdef CLIENT_SERVER
    if(params.find<bool>("server",false))
       scenes.push_back(scene(outfilen,true,false));
    else
    #endif
    scenes.push_back(scene(outfilen,false,false));
    
    }
  else
    {
    ifstream inp(geometry_file.c_str());
    planck_assert (inp, "could not open scene file '" + geometry_file +"'");
    int current_scene = params.find<int>("scene_start",0);
    int scene_incr = params.find<int>("scene_incr",1);
    int last_scene = params.find<int>("last_scene",1000000);

    string line;
    getline(inp, line);
    double tmpDbl;
    if (sscanf(line.c_str(),"%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
          &tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl,&tmpDbl)==10)
      {
      //      cerr << "DEBUG: old geometry file format detected." << endl;
      line="camera_x camera_y camera_z lookat_x lookat_y lookat_z sky_x sky_y sky_z fidx";
      inp.seekg(0, ios_base::beg);
      }

    vector<string> sceneParameterKeys, sceneParameterValues;
    split(line, sceneParameterKeys);

    if (mpiMgr.master())
      {
      cout << endl << "The following parameters are dynamically modified by the scene file: " << endl;
      for(unsigned int i=0;i<sceneParameterKeys.size();i++)
        cout << sceneParameterKeys[i] << " ";
      cout << endl << flush;
      }

    for (int i=0; i<current_scene; ++i)
      getline(inp, line);

    while (getline(inp, line) && current_scene < last_scene)
      {
      paramfile scnpar;
      scnpar.setVerbosity(false);
      string outfilen = outfile+intToString(current_scene,4);
      split(line, sceneParameterValues);

      planck_assert(sceneParameterKeys.size()==sceneParameterValues.size(),
        "ERROR in scene file detected, please check!  Quitting.");
      for (unsigned int i=0; i<sceneParameterKeys.size(); i++)
        scnpar.setParam(sceneParameterKeys[i], sceneParameterValues[i]);

      double fidx = scnpar.find<double>("fidx",params.find<double>("fidx",0.));

      bool reuse=false;
      if (scenes.size()>0)
        {
        double fidxold = scenes[scenes.size()-1].sceneParameters.find<double>
          ("fidx",params.find<double>("fidx",0.));
        if (approx(fidx,fidxold))
          scenes[scenes.size()-1].keep_particles=reuse=true;
        }

      scenes.push_back(scene(scnpar,outfilen,false,reuse));
      current_scene += scene_incr;
      for (int i=0; i<scene_incr-1; ++i)
        getline(inp, line);
      }
    }

  bool do_panorama = params.find<bool>("panorama",false);
  for (tsize m=0; m<scenes.size(); ++m)
    do_panorama = do_panorama || scenes[m].sceneParameters.find<bool>("panorama",false);
  bool do_stereo = params.find<double>("EyeSeparation",0)!=0.;
  for (tsize m=0; m<scenes.size(); ++m)
    do_stereo = do_stereo || (scenes[m].sceneParameters.find<double>("EyeSeparation",0)!=0.);
  planck_assert(!(do_panorama&&do_stereo), "inconsistent parameters");
  if (do_stereo)
    {
    vector<scene> sc_orig;
    sc_orig.swap(scenes);
    for (tsize i=0; i<sc_orig.size(); ++i)
      {
      scenes.push_back(sc_orig[i]);
      scenes.push_back(sc_orig[i]);
      scene &sa = scenes[scenes.size()-2], &sb = scenes[scenes.size()-1];
      paramfile &spa(sa.sceneParameters), &spb(sb.sceneParameters);
      sa.keep_particles=true;
// MR: I think the next line is not necessary
//      sb.keep_particles=false;
      sb.reuse_particles=true;
      double eye_separation = degr2rad * spa.find<double>("EyeSeparation",
        params.find<double>("EyeSeparation",0));

      vec3 lookat(spa.find<double>("lookat_x",params.find<double>("lookat_x")),
                  spa.find<double>("lookat_y",params.find<double>("lookat_y")),
                  spa.find<double>("lookat_z",params.find<double>("lookat_z")));
      vec3 campos(spa.find<double>("camera_x",params.find<double>("camera_x")),
                  spa.find<double>("camera_y",params.find<double>("camera_y")),
                  spa.find<double>("camera_z",params.find<double>("camera_z")));
      vec3 sky(spa.find<double>("sky_x",params.find<double>("sky_x",0)),
               spa.find<double>("sky_y",params.find<double>("sky_y",0)),
               spa.find<double>("sky_z",params.find<double>("sky_z",1)));

      vec3 view = lookat - campos;

      vec3 right = crossprod (view,sky);

      double distance = eye_separation * view.Length();

      spa.setParam("center_x",campos.x);
      spa.setParam("center_y",campos.y);
      spa.setParam("center_z",campos.z);
      spb.setParam("center_x",campos.x);
      spb.setParam("center_y",campos.y);
      spb.setParam("center_z",campos.z);

      vec3 campos_r = campos + right / right.Length() * distance*0.5;
      spa.setParam("camera_x",campos_r.x);
      spa.setParam("camera_y",campos_r.y);
      spa.setParam("camera_z",campos_r.z);

      vec3 campos_l = campos - right / right.Length() * distance*0.5;
      spb.setParam("camera_x",campos_l.x);
      spb.setParam("camera_y",campos_l.y);
      spb.setParam("camera_z",campos_l.z);

      sa.outname = "right_"+sa.outname;
      sb.outname = "left_"+sb.outname;
      }
    }
  if (do_panorama)
    {
    const double camx_ofs[]= { 1, 0,-1, 0, 0, 0 };
    const double camy_ofs[]= { 0, 1, 0,-1, 0, 0 };
    const double camz_ofs[]= { 0, 0, 0, 0, 1,-1 };
    const double sky_x[]= { 0, 0, 0, 0,-1, 1 };
    const double sky_z[]= { 1, 1, 1, 1, 0, 0 };
    const char * prefix[] = { "1_", "2_", "3_", "4_", "5_", "6_" };
    vector<scene> sc_orig;
    sc_orig.swap(scenes);
    for (tsize i=0; i<sc_orig.size(); ++i)
      {
      for (tsize j=0; j<6; ++j)
        {
        scenes.push_back(sc_orig[i]);
        scene &sc(scenes.back());
        paramfile &sp(sc.sceneParameters);
        if (j<5) sc.keep_particles=true;
        if (j>0) sc.reuse_particles=true;

        sp.setParam("fov",90.);
        vec3 cam(sp.find<double>("camera_x",params.find<double>("camera_x")),
                 sp.find<double>("camera_y",params.find<double>("camera_y")),
                 sp.find<double>("camera_z",params.find<double>("camera_z")));

        sp.setParam("lookat_x",cam.x+camx_ofs[j]);
        sp.setParam("lookat_y",cam.y+camy_ofs[j]);
        sp.setParam("lookat_z",cam.z+camz_ofs[j]);
        sp.setParam("sky_x",sky_x[j]);
        sp.setParam("sky_y",0.);
        sp.setParam("sky_z",sky_z[j]);
        sc.outname = string(prefix[j])+sc.outname;
        }
      }
    }
  }

void sceneMaker::fetchFiles(vector<particle_sim> &particle_data, double fidx)
  { 

  if (scenes[cur_scene].reuse_particles)
    {
      // For CUDA we dont need to restore the data as only GPU copy was modified
#ifndef CUDA
    tstack_push("Data copy");
#ifdef MPI_A_NEQ_E
    if(mpiMgr.num_ranks() > 1)  tree.restore_data(tree_orig);
    else                        particle_data=p_orig;
#else
        //printf("Reusing particle data, p_orig.size(): %lu\n", p_orig.size());
    particle_data=p_orig;
#endif
    tstack_pop("Data copy");
#endif
    return;
    }
  tstack_push("Input");
  if (mpiMgr.master())
    cout << endl << "reading data ..." << endl;
  int simtype = params.find<int>("simtype");
  int spacing = params.find<double>("snapshot_spacing",1);
  int snr1_guess = int(fidx/spacing)*spacing, snr2_guess=snr1_guess+spacing;
  int snr1 = params.find<int>("snapshot_base1",snr1_guess);
  int snr2 = params.find<int>("snapshot_base2",snr2_guess);
  double frac=params.find<float>("frac",(fidx-snr1)/(snr2-snr1));

  tstack_push("Reading");

  if(file!=NULL) 
  {
    delete file; 
    file = NULL;
  }
  
  switch (simtype)
    {
    case 0:
      bin_reader_tab(params,particle_data);
      break;
    case 1:
      bin_reader_block(params,particle_data);
      break;
    case 2:
      if (interpol_mode>0) // Here only the two data sets are prepared, interpolation will be done later
        {
        if (mpiMgr.master())
          cout << "Loaded file1: " << snr1_now << " , file2: " << snr2_now
               << " , interpol fraction: " << frac << endl
               << " (needed files : " << snr1 << " , " << snr2 << ")" << endl;
        if (snr1==snr2_now)
          {
          if (mpiMgr.master())
            cout << " old2 = new1!" << endl;
          p1.swap(p2);
          id1.swap(id2);
          idx1.swap(idx2);
          vel1.swap(vel2);
          time1 = time2;
          snr1_now = snr1;
          }
        if (snr1_now!=snr1)
          {
          if (mpiMgr.master())
            cout << " reading new1 " << snr1 << endl;

          gadget_reader(params,interpol_mode,p1,id1,vel1,snr1,time1,boxsize);
          redshift1=-1.0;
          mpiMgr.barrier();
          tstack_replace("Reading","Particle index generation");
          buildIndex(id1.begin(),id1.end(),idx1);
          tstack_replace("Particle index generation","Reading");

          snr1_now = snr1;
          }
        if (snr2_now!=snr2)
          {
          if (mpiMgr.master())
            cout << " reading new2 " << snr2 << endl;
          gadget_reader(params,interpol_mode,p2,id2,vel2,snr2,time2,boxsize);
          mpiMgr.barrier();
          tstack_replace("Reading","Particle index generation");
          buildIndex(id2.begin(),id2.end(),idx2);
          tstack_replace("Particle index generation","Fetch remote particles");
          redshift2 = -1.0;
          snr2_now = snr2;

          MpiFetchRemoteParticles();
          mpiMgr.barrier();
          tstack_replace("Fetch remote particles","Reading");
          }
        }
      else
        {
        double time;
        gadget_reader(params,interpol_mode,particle_data,id1,vel1,snr1,time,boxsize);
        // extend the parameter list to be able to output the information later to the frame-specific logfile
        params.setParam("time", dataToString(time));
        }
      break;
    case 3:
      {
#ifdef HDF5
      long num_cells = enzo_reader(params,particle_data);
#else
      planck_fail("Need HDF5 to read enzo files!  Quitting.");
#endif
      break;
      }
    case 4:
      {
      double dummy;
      gadget_millenium_reader(params,particle_data,0,&dummy);
      break;
      }
    case 5:
#if defined(USE_MPIIO)
      {
      float maxr, minr;
      bin_reader_block_mpi(params,particle_data, &maxr, &minr, mpiMgr.rank(), mpiMgr.num_ranks());
      }
#else
      planck_fail("mpi reader not available in non MPI compiled version!");
#endif
      break;
    case 6:
      mesh_reader(params,particle_data);
      break;
#ifdef HDF5
    case 7:
    {
#ifdef CLIENT_SERVER
      if(!file)
        file = new HDF5File();
      file->read(params, particle_data);
#else
      hdf5_reader(params,particle_data);
#endif
      break;
    }
    case 8:
      // GADGET HDF5 READER
      if (interpol_mode>0) // Here only the two data sets are prepared, interpolation will be done later
        {
        cout << "Loaded file1: " << snr1_now << " , file2: " << snr2_now << " , interpol fraction: " << frac << endl;
        cout << " (needed files : " << snr1 << " , " << snr2 << ")" << endl;
        if (snr1==snr2_now)
          {
          cout << " old2 = new1!" << endl;
          p1.swap(p2);
          id1.swap(id2);
          idx1.swap(idx2);
          vel1.swap(vel2);
          time1 = time2;
          redshift1 = redshift2;
          snr1_now = snr1;
          }
        if (snr1_now!=snr1)
          {
          cout << " reading new1 " << snr1 << endl;
          gadget_hdf5_reader(params,interpol_mode,p1,id1,vel1,snr1,time1,redshift1,boxsize);
          mpiMgr.barrier();
          tstack_replace("Input","Particle index generation");
          buildIndex(id1.begin(),id1.end(),idx1);
          tstack_replace("Particle index generation","Input");
          snr1_now = snr1;
          }
        if (snr2_now!=snr2)
          {
          cout << " reading new2 " << snr2 << endl;
          gadget_hdf5_reader(params,interpol_mode,p2,id2,vel2,snr2,time2,redshift2,boxsize);
          mpiMgr.barrier();
          tstack_replace("Input","Particle index generation");
          buildIndex(id2.begin(),id2.end(),idx2);
          tstack_replace("Particle index generation","Input");
          snr2_now = snr2;

          MpiFetchRemoteParticles();
          mpiMgr.barrier();
          }
        }
      else
        {
        double time, redshift;
        gadget_hdf5_reader(params,interpol_mode,particle_data,id1,vel1,0,time,redshift,boxsize);
        // extend the parameter list to be able to output the information later to the frame-specific logfile
        params.setParam("time", dataToString(time));
        params.setParam("redshift", dataToString(redshift));
        }
      break;
#endif
#ifdef SPLVISIVO
    case 10:
      if(!visivo_reader(params,particle_data,opt))
        planck_fail("Invalid read data ...");
      break;
#endif
    case 11:
      {
        if (interpol_mode>0) // Here only the two data sets are prepared, interpolation will be done later
        {
          if (mpiMgr.master())
            cout << "Loaded file1: " << snr1_now << " , file2: " << snr2_now
                 << " , interpol fraction: " << frac << endl
                 << " (needed files : " << snr1 << " , " << snr2 << ")" << endl;
          if (snr1==snr2_now)
            {
            if (mpiMgr.master())
              cout << " old2 = new1!" << endl;
            p1.swap(p2);
            id1.swap(id2);
            idx1.swap(idx2);
            time1 = time2;
            snr1_now = snr1;
            }
          if (snr1_now!=snr1)
            {
            if (mpiMgr.master())
              cout << " reading new1 " << snr1 << endl;
    	       tipsy_reader(params, p1, id1, snr1);

            mpiMgr.barrier();
            tstack_replace("Reading","Particle index generation");
            buildIndex(id1.begin(),id1.end(),idx1);
            tstack_replace("Particle index generation","Reading");

            snr1_now = snr1;
            }
          if (snr2_now!=snr2)
            {
            if (mpiMgr.master())
              cout << " reading new2 " << snr2 << endl;

          	  tipsy_reader(params, p2, id2, snr2);
        	  
        	  mpiMgr.barrier();
                  tstack_replace("Reading","Particle index generation");
                  buildIndex(id2.begin(),id2.end(),idx2);
                  tstack_replace("Particle index generation","Fetch remote particles");

            snr2_now = snr2;

            MpiFetchRemoteParticles();
            mpiMgr.barrier();
            tstack_replace("Fetch remote particles","Reading");
            }
        }
        else
        {
          tipsy_reader(params,particle_data, id1, 0);
        }
      }
      break;
#ifdef HDF5
    case 12:
      //h5part_reader(params,particle_data);
      planck_fail("h5part reader currently inactive");
      break;
#endif
    case 13:
      ramses_reader(params,particle_data);
      break;
#ifdef USE_MPI
    case 14:
      bonsai_reader(params,particle_data);
      break;
#endif
    case 15:
      ascii_reader(params,particle_data);
      break;
#ifdef FITS
    case 16:
      fits_reader(params,particle_data);
      break;
#endif
    default:
      planck_fail("Invalid simtype!");
      break;
    }
  mpiMgr.barrier();
  tstack_pop("Reading");
  tstack_pop("Input");

  if (interpol_mode>0)
    {
    if (mpiMgr.master())
      cout << "Interpolating between " << p1.size() << " and " <<
           p2.size() << " particles ..." << endl;
    tstack_push("Time interpolation");
    particle_interpolate(particle_data,frac);
    tstack_pop("Time interpolation");
    }

#ifdef CUDA // ranging is done on GPU

  // Check for maxes and mins in parameter file
  int nt = params.find<int>("ptypes",1);
  bool found = true;
  for (int t=0; t<nt; t++)
    found &= ( params.param_present("intensity_min"+dataToString(t))
             & params.param_present("intensity_max"+dataToString(t))
             & params.param_present("color_min"+dataToString(t))
             & params.param_present("color_max"+dataToString(t)));

  // If maxes and mins are not specified then run host ranging to determine these
  if (!found)
    {
    tstack_push("Particle ranging");
    tsize npart_all = particle_data.size();
    mpiMgr.allreduce (npart_all,MPI_Manager::Sum);
    if (mpiMgr.master())
      cout << endl << "host: ranging values (" << npart_all << ") ..." << endl;
    particle_normalize(particle_data, true);
    tstack_pop("Particle ranging");
    }

#else
  tstack_push("Particle ranging");
  tsize npart_all = particle_data.size();
  mpiMgr.allreduce (npart_all,MPI_Manager::Sum);
  if (mpiMgr.master())
    cout << endl << "host: ranging values (" << npart_all << ") ..." << endl;
  particle_normalize(particle_data, true);
  tstack_pop("Particle ranging");
#endif

#ifdef MPI_A_NEQ_E
  if(mpiMgr.num_ranks()>1)
  {
    tstack_push("Tree build");
    // Construct KD tree
    tree.build(&particle_data[0], particle_data.size(), false, 0, mpiMgr.rank(), mpiMgr.num_ranks());
    tstack_replace("Tree build","Tree redistribute");
    tree.redistribute_data();
    tstack_pop("Tree redistribute");
    std::vector<particle_sim>().swap(particle_data);
    //tree.status();
    //tree.all_leaves_status();
    tree.box();
    // tree copy too 
    if(scenes[cur_scene].keep_particles) tree.backup_data(tree_orig);    
  }
  else
  {
#endif
// For cuda we dont need a copy of the particles as they are modified on GPU
#ifndef CUDA
    if (scenes[cur_scene].keep_particles) p_orig = particle_data;
#endif
#ifdef MPI_A_NEQ_E
  }
#endif

// boost initialization

  bool boost = params.find<bool>("boost",false);
  if(boost)
    {
    if (mpiMgr.master())
      cout << "Boost setup..." << endl;
    mesh_creator(particle_data, &Mesh, &MeshD);
    randomizer(particle_data, Mesh, MeshD);
    }
  }

bool sceneMaker::getNextScene (vector<particle_sim> &particle_data,
  vector<particle_sim> &r_points, vec3 &campos, vec3 &centerpos, vec3 &lookat, vec3 &sky,
  string &outfile)
  {
  if (tsize(++cur_scene) >= scenes.size()) return false;

  tstack_push("Scene update");
  const scene &scn=scenes[cur_scene];

  // patch the params object with parameter values relevant to the current scene
  map<string,string> sceneParameters=scn.sceneParameters.getParams();
  for (map<string,string>::const_iterator it=sceneParameters.begin(); it!=sceneParameters.end(); ++it)
    params.setParam(it->first, it->second);


  outfile=scn.outname;
  double fidx=params.find<double>("fidx",0);
  fetchFiles(particle_data,fidx);

  // Fetch the values from the param object which may have been altered by the scene file or copied from the opt object.
  campos = vec3(params.find<double>("camera_x"), params.find<double>("camera_y"), params.find<double>("camera_z"));
  lookat = vec3(params.find<double>("lookat_x"), params.find<double>("lookat_y"), params.find<double>("lookat_z"));
  sky = vec3(params.find<double>("sky_x", 0), params.find<double>("sky_y", 0), params.find<double>("sky_z", 1));
  if (params.param_present("center_x"))
    centerpos = vec3(params.find<double>("center_x"), params.find<double>("center_y"), params.find<double>("center_z"));
  else
    centerpos = campos; 

  if (params.find<bool>("periodic",true))
    {
    tstack_push("Box Wrap");
    int npart = particle_data.size();
    double boxhalf = boxsize / 2;

    if(mpiMgr.master())
      cout << " doing parallel box wrap " << boxsize << endl;
#pragma omp parallel
{
    int m;
#pragma omp for schedule(guided,1000)
    for (m=0; m<npart; ++m)
      {
      if(particle_data[m].x - lookat.x > boxhalf)
        particle_data[m].x -= boxsize;
      if(lookat.x - particle_data[m].x > boxhalf)
        particle_data[m].x += boxsize;
      if(particle_data[m].y - lookat.y > boxhalf)
        particle_data[m].y -= boxsize;
      if(lookat.y - particle_data[m].y > boxhalf)
        particle_data[m].y += boxsize;
      if(particle_data[m].z - lookat.z > boxhalf)
        particle_data[m].z -= boxsize;
      if(lookat.z - particle_data[m].z > boxhalf)
        particle_data[m].z += boxsize;
      }
}
    tstack_pop("Box Wrap");

  }

  // Let's try to boost!!!
  bool boost = params.find<bool>("boost",false);
  if(boost)
    {
    if (mpiMgr.master())
      cout << "Boost!!!" << endl;
    m_rotation(params, &Mesh, MeshD, campos, lookat, sky);
    p_selector(particle_data, Mesh, MeshD, r_points);
    }

  // dump information on the currently rendered image into a log file in *scene file format*

  if ( mpiMgr.master() && params.find<bool>("print_logfile", true))
    {
    string logFileName;
    logFileName.assign(outfile);
    logFileName.append(".log");
    ofstream logFile(logFileName.c_str());
    map<string,string> paramsMap;
    paramsMap = params.getParams();
    for(map<string,string>::iterator it=paramsMap.begin(); it!=paramsMap.end(); ++it)
      logFile << it->first << "=" << it->second << endl;
    logFile.close();
    }
  tstack_pop("Scene update");
  return true;
  }

void sceneMaker::updateCurrentScene (vector<particle_sim> &particle_data, bool new_data)
  { 

  tstack_push("Scene update");
  double fidx=params.find<double>("fidx",0);
  // If there is no new data, reuse the particles
  scenes[cur_scene].keep_particles = true;
  scenes[cur_scene].reuse_particles = !new_data;
  fetchFiles(particle_data,fidx);
  tstack_pop("Scene update");
  }

namespace {

template<typename T> void comm_helper
  (const vector<vector<MyIDType> > &idx_send, vector<T> &buf)
  {
  tsize sendsize=0, ntasks=mpiMgr.num_ranks();
  arr<int> sendcnt(ntasks), recvcnt;
  for (tsize i=0; i<ntasks; ++i)
    {
    sendcnt[i]=idx_send[i].size();
    sendsize+=idx_send[i].size();
    }

  vector<T> sendbuf(sendsize);
  tsize ofs=0;
  for (tsize i=0; i<ntasks; ++i)
    for (int j=0; j<sendcnt[i]; ++j)
      sendbuf[ofs++]=buf[idx_send[i][j]];
  releaseMemory(buf);
  mpiMgr.all2allv_easy_typeless(sendbuf,sendcnt,buf,recvcnt);
  }

} // unnamed namespace

// MpiFetchRemoteParticles() rearranges the *1 particles so that they match *2
void sceneMaker::MpiFetchRemoteParticles ()
  {
  int ntasks = mpiMgr.num_ranks();
  if (ntasks==1) return;
  int mytask = mpiMgr.rank();

  vector<MyIDType> id_needed(idx2.size()); // IDs needed on task t_req

  for (tsize i=0; i<idx2.size(); ++i)
    id_needed[i]=id2[idx2[i]];

#if 0
  // make sure that id_needed and id1 are sorted (debugging only)
  for (tsize i=1; i<id_needed.size(); ++i)
    planck_assert(id_needed[i-1]<id_needed[i],"id_needed not ordered");
  for (tsize i=1; i<idx1.size(); ++i)
    planck_assert(id1[idx1[i-1]]<id1[idx1[i]],"id1 not ordered");
#endif

  vector<vector<MyIDType> > idx_send (ntasks); // what gets sent where
  vector<bool> sent(idx1.size(),false);

  for (int tc=0; tc<ntasks; ++tc) // circular data exchange
    {
    int t_req=(mytask+ntasks-tc)%ntasks; // task requesting particles
    tsize i1=0, i2=0, i2n=0;
    while ((i1<idx1.size()) && (i2<id_needed.size()))
      {
      if (id_needed[i2]==id1[idx1[i1]]) // needed and available
        {
        idx_send[t_req].push_back(idx1[i1]);
        sent[i1]=true;
        i1++; i2++;
        }
      else if (id_needed[i2]<id1[idx1[i1]]) // needed but not available
        id_needed[i2n++]=id_needed[i2++]; // compress id_needed
      else // available but not needed
        i1++;
      }
    id_needed.resize(i2n); // shrink id_needed, reduces communication volume
    if (tc<(ntasks-1)) // rotate id_needed by 1 task
      mpiMgr.sendrecv_realloc(id_needed,
        (mytask+1)%ntasks,(mytask+ntasks-1)%ntasks);
    }
  releaseMemory(id_needed);

  // keep particles that are not needed anywhere on their original task
  for (tsize i1=0; i1<idx1.size(); ++i1)
    if (!sent[i1]) idx_send[mytask].push_back(idx1[i1]);

  comm_helper(idx_send,id1);
  comm_helper(idx_send,p1);
  if (interpol_mode>1) comm_helper(idx_send,vel1);

  releaseMemory(idx1);
  buildIndex(id1.begin(), id1.end(), idx1);

#if 0
  // make sure that id1 is sorted (debugging only)
  for (tsize i=1; i<idx1.size(); ++i)
    planck_assert(id1[idx1[i-1]]<id1[idx1[i]],"id1 not ordered");
#endif
  }

#ifdef MPI_A_NEQ_E
void sceneMaker::treeInit()
{
       // Setup KD tree
    sMaker.tree.add_smaller_than_function(xcomp_smaller);
    sMaker.tree.add_smaller_than_function(ycomp_smaller);
    sMaker.tree.add_smaller_than_function(zcomp_smaller);
    sMaker.tree.add_accessor(x_accessor);
    sMaker.tree.add_accessor(y_accessor);
    sMaker.tree.add_accessor(z_accessor);
    sMaker.tree.add_accessor(r_accessor);
    sMaker.tree.add_accessor(ghost_accessor);

    int depth = mpiMgr.num_ranks() > 1 ? ceil(log2(mpiMgr.num_ranks())) : 0;
    sMaker.tree.set_ghosts(true/*, ghost_setter*/, depth);
    // Max depth such that we have at least one leaf per node
    sMaker.tree.set_max_depth(depth);
    sMaker.tree.split_by_longest_axis();
}

 Box<float,3> bbox treeBoundingBox(render_context& rc)
 {
  rc.active_nodes = tree.active_node_list();
  if(rc.active_nodes.size() != 1)
    planck_fail("MPI_A_NEQ_E: Only supporting 1 node per rank so far (i.e. num ranks must be power of 2)");
  // Get bounding box from tree to give to renderer
  return sMaker.tree.node_box_to_raw(rc.active_nodes[0]);  
 }
  

void treeUpdateOpacityRes(arr2<COLOUR>& pic, arr2<COLOUR>& opac);
{
  int x = pic.size1();
  int y = pic.size2();
  if(x != opac.size1() || y != opac.size2())
    opac.alloc(x, y);
}
#endif 

#ifdef CLIENT_SERVER
// Unload data forcing deallocation
void sceneMaker::unloadData(bool force_dealloc)
{
  for(auto&& i : scenes) 
   i.keep_particles = true;
  for(auto&& i : scenes) 
   i.reuse_particles = false;

 if(force_dealloc)
 {
  // Force deallocation
   std::vector<particle_sim>().swap(p_orig);
   if(interpol_mode>0)
   {
     std::vector<particle_sim>().swap(p1);
     std::vector<particle_sim>().swap(p2);
     std::vector<MyIDType>().swap(id1);
     std::vector<MyIDType>().swap(id2);
     std::vector<MyIDType>().swap(idx1);
     std::vector<MyIDType>().swap(idx2);
   }
 }
 else
 {
    p_orig.clear();
   if(interpol_mode>0)
   {
     p1.clear();
     p2.clear();
     id1.clear();
     id2.clear();
     idx1.clear();
     idx2.clear();
   }
 }

 if(file!=NULL){
  delete file;
  file = NULL;
 }

}

#endif
bool sceneMaker::is_final_scene()
{
  return (cur_scene == scenes.size() - 1) ? true : false;
}
