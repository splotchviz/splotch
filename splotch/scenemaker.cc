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
#ifndef NEW_MPISTUFF
#include <limits>
#endif

using namespace std;

void sceneMaker::particle_normalize(vector<particle_sim> &p, bool verbose)
  const
  {
  // how many particle types are there
  int nt = params.find<int>("ptypes",1);
  arr<bool> col_vector(nt),log_int(nt),log_col(nt),asinh_col(nt);
  arr<Normalizer<float32> > intnorm(nt), colnorm(nt);

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
  arr<Normalizer<float32> > inorm(nt+20), cnorm(nt+20);
  int m;
#ifdef CUDA
  // In cuda version logs are performed on device
  for (m=0; m<npart; ++m)
    {
    int t=p[m].type;

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
    }
}

#else

#pragma omp for schedule(guided,1000)
  for (m=0; m<npart; ++m) // do log calculations if requested
    {
    int t=p[m].type;

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
    }
  }
#endif
  for(int t=0; t<nt; t++)
    {
    mpiMgr.allreduce(intnorm[t].minv,MPI_Manager::Min);
    mpiMgr.allreduce(colnorm[t].minv,MPI_Manager::Min);
    mpiMgr.allreduce(intnorm[t].maxv,MPI_Manager::Max);
    mpiMgr.allreduce(colnorm[t].maxv,MPI_Manager::Max);

    if (verbose && mpiMgr.master())
      {
      cout << " For particles of type " << t << ":" << endl;
      cout << " From data: " << endl;
      cout << " Color Range:     " << colnorm[t].minv << " (min) , " <<
           colnorm[t].maxv << " (max) " << endl;
      cout << " Intensity Range: " << intnorm[t].minv << " (min) , " <<
           intnorm[t].maxv << " (max) " << endl;
      }

    if(params.param_present("intensity_min"+dataToString(t)))
      intnorm[t].minv = params.find<float>("intensity_min"+dataToString(t));

    if(params.param_present("intensity_max"+dataToString(t)))
      intnorm[t].maxv = params.find<float>("intensity_max"+dataToString(t));

    if(params.param_present("color_min"+dataToString(t)))
      colnorm[t].minv = params.find<float>("color_min"+dataToString(t));

    if(params.param_present("color_max"+dataToString(t)))
      colnorm[t].maxv = params.find<float>("color_max"+dataToString(t));

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
    }

  vector<pair<MyIDType,MyIDType> > v;
  v.reserve(min(p1.size(),p2.size()));
    {
    MyIDType i1=0,i2=0;
    while(i1<p1.size() && i2<p2.size())
      {
      if (id1[idx1[i1]]==id2[idx2[i2]])
        //  if(p1[idx1[i1]].type==p2[idx2[i2]].type)
        v.push_back(pair<MyIDType,MyIDType>(idx1[i1++],idx2[i2++]));
      else if (id1[idx1[i1]]<id2[idx2[i2]])
	{
	  v.push_back(pair<MyIDType,MyIDType>(idx1[i1],MyMaxID));
	  i1++;
	}
      else if (id1[idx1[i1]]>id2[idx2[i2]])
	{
	  v.push_back(pair<MyIDType,MyIDType>(MyMaxID,idx2[i2]));
	  i2++;
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
	vec3f v1,v2;
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
	vec3f vda = (x2-x1)*2. - (v1*v_unit1 + v2*v_unit2);
	pos = x1 + v1*(v_unit1*frac)
          + (v2*v_unit2 - v1*v_unit1 + vda)*(frac*frac*0.5);
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
    scenes.push_back(scene(outfilen,false,false));
    }
  else
    {
    ifstream inp(geometry_file.c_str());
    planck_assert (inp, "could not open scene file '" + geometry_file +"'");
    int current_scene = params.find<int>("scene_start",0);
    int scene_incr = params.find<int>("scene_incr",1);

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

    while (getline(inp, line))
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

  bool do_stereo = params.find<double>("EyeSeparation",0)!=0.;
  for (tsize m=0; m<scenes.size(); ++m)
    do_stereo = do_stereo || (scenes[m].sceneParameters.find<double>("EyeSeparation",0)!=0.);
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

      vec3 right = crossprod (sky,view);

      double distance = eye_separation * view.Length();

      vec3 campos_r = campos - right / right.Length() * distance*0.5;
      spa.setParam("camera_x",campos_r.x);
      spa.setParam("camera_y",campos_r.y);
      spa.setParam("camera_z",campos_r.z);

      vec3 campos_l = campos + right / right.Length() * distance*0.5;
      spb.setParam("camera_x",campos_l.x);
      spb.setParam("camera_y",campos_l.y);
      spb.setParam("camera_z",campos_l.z);

      sa.outname = "left_"+sa.outname;
      sb.outname = "right_"+sb.outname;
      }
    }
  }

void sceneMaker::fetchFiles(vector<particle_sim> &particle_data, double fidx)
  {
  if (scenes[cur_scene].reuse_particles)
    {
    particle_data=p_orig;
    return;
    }
  tstack_push("Input");
  if (mpiMgr.master())
    cout << endl << "reading data ..." << endl;
  int simtype = params.find<int>("simtype");
  int spacing = params.find<double>("snapshot_spacing",1);
  int snr1 = int(fidx/spacing)*spacing, snr2=snr1+spacing;
  double frac=(fidx-snr1)/spacing;

  tstack_push("Reading");

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
#ifndef NEW_MPISTUFF
          if ((mpiMgr.num_ranks()>1) // <-- only makes sense with true MPI runs
              &&
              params.find<bool>("mpi_interpolation_reread_data",false)) // <-- saves some memory at the expense of re-reading the dataset p1 (formerly p2)
            {
            // re-read the data set since no backup copy can be expected to exist in memory
              if (mpiMgr.master())
                cout << " re-reading new1 " << snr1 << endl;
            gadget_reader(params,interpol_mode,p1,id1,vel1,snr1,time1,boxsize);
            redshift1=-1.0;
            mpiMgr.barrier();
            tstack_replace("Reading","Particle index generation");
            buildIndex(id1.begin(),id1.end(),idx1);
            tstack_replace("Particle index generation","Reading");
            }
          else
            {
            // MPI and non-MPI default case
            tstack_replace("Reading","Fetch remote particles");
            mpiMgr.barrier();
            MpiStripRemoteParticles();
            mpiMgr.barrier();
            //
            p1.clear();   p1.swap(p2);
            id1.clear();  id1.swap(id2);
            idx1.clear(); idx1.swap(idx2);
            vel1.clear(); vel1.swap(vel2);
            tstack_replace("Fetch remote particles","Reading");
            time1 = time2;
            }
#else
          p1.swap(p2);
          id1.swap(id2);
          idx1.swap(idx2);
          vel1.swap(vel2);
          time1 = time2;
#endif
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
      hdf5_reader(params,particle_data);
      break;
    case 8:
      // GADGET HDF5 READER
      if (interpol_mode>0) // Here only the two data sets are prepared, interpolation will be done later
        {
        cout << "Loaded file1: " << snr1_now << " , file2: " << snr2_now << " , interpol fraction: " << frac << endl;
        cout << " (needed files : " << snr1 << " , " << snr2 << ")" << endl;
        if (snr1==snr2_now)
          {
          cout << " old2 = new1!" << endl;
#ifdef NEW_MPISTUFF
          p1.swap(p2);
          id1.swap(id2);
          idx1.swap(idx2);
          vel1.swap(vel2);
          time1 = time2;
          redshift1 = redshift2;
#else
          if ((mpiMgr.num_ranks()>1) && params.find<bool>("mpi_interpolation_reread_data",false))
            {
            // re-read the data set since no backup copy exists in memory
            cout << " re-reading new1 " << snr1 << endl;
            gadget_hdf5_reader(params,interpol_mode,p1,id1,vel1,snr1,time1,redshift1,boxsize);
            mpiMgr.barrier();
            tstack_replace("Input","Particle index generation");
            buildIndex(id1.begin(),id1.end(),idx1);
            tstack_replace("Particle index generation","Input");
            }
          else
            {
            // MPI and non-MPI default case
            mpiMgr.barrier();
            MpiStripRemoteParticles();
            mpiMgr.barrier();
            //
            p1.clear();   p1.swap(p2);
            id1.clear();  id1.swap(id2);
            idx1.clear(); idx1.swap(idx2);
            vel1.clear(); vel1.swap(vel2);
            //
            time1 = time2;
            redshift1 = redshift2;
            }
#endif
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
      tipsy_reader(params,particle_data);
      break;
#ifdef HDF5
    case 12:
      h5part_reader(params,particle_data);
      break;
#endif
    case 13:
      ramses_reader(params,particle_data);
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

  if (scenes[cur_scene].keep_particles) p_orig = particle_data;

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
  vector<particle_sim> &r_points, vec3 &campos, vec3 &lookat, vec3 &sky,
  string &outfile)
  {
  if (tsize(++cur_scene) >= scenes.size()) return false;

  const scene &scn=scenes[cur_scene];

  // patch the params object with parameter values relevant to the current scene
  map<string,string> sceneParameters=scn.sceneParameters.getParams();
  for (map<string,string>::const_iterator it=sceneParameters.begin(); it!=sceneParameters.end(); ++it)
    params.setParam(it->first, it->second);

  // Fetch the values from the param object which may have been altered by the scene file or copied from the opt object.
  campos=vec3(params.find<double>("camera_x"),params.find<double>("camera_y"),params.find<double>("camera_z"));
  lookat=vec3(params.find<double>("lookat_x"),params.find<double>("lookat_y"),params.find<double>("lookat_z"));
  sky   =vec3(params.find<double>("sky_x",0), params.find<double>("sky_y",0), params.find<double>("sky_z",1));

  outfile=scn.outname;
  double fidx=params.find<double>("fidx",0);

  fetchFiles(particle_data,fidx);

  if (params.find<bool>("periodic",true))
    {
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
  if ( mpiMgr.master() )
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

  return true;
  }

#ifdef NEW_MPISTUFF

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

  for (int tc=0; tc<ntasks; ++tc) // circular data exchange
    {
    int t_req=(mytask+ntasks-tc)%ntasks; // task requesting particles
    tsize i1=0, i2=0, i2n=0;
    while ((i1<idx1.size()) && (i2<id_needed.size()))
      {
      if (id_needed[i2]==id1[idx1[i1]]) // needed and available
        {
        idx_send[t_req].push_back(idx1[i1]);
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

#else

/**
 * Routines for MPI parallel interpolation below.  Testing and optimization needed.  (Klaus Reuter, RZG)
 */

// MpiFetchRemoteParticles() adds particles from remote processes to p2
void sceneMaker::MpiFetchRemoteParticles ()
{
  if (mpiMgr.num_ranks()==1)
    return;

  bool debug_msg   = params.find<bool>("mpi_interpolation_debug_msg",false);
  bool patch_data  = params.find<bool>("mpi_interpolation_patch_data",true);
  bool reread_data = params.find<bool>("mpi_interpolation_reread_data",false);

  if (debug_msg)
  {
    cout << "MpiFetchRemoteParticles() : BEGIN" << endl << flush;
    cout << "MpiFetchRemoteParticles() : Determining required particles ..." << endl << flush;
  }

  if (patch_data)
  {
    // save the number of particles which are initially in p2
    // this information is needed in MpiStripRemoteParticles()
    numberOfLocalParticles=p2.size();
  }
  else
  {
    // create a backup copy of p2 and the related data structures
    if (!reread_data)
    {
      p2Backup   = p2;
      id2Backup  = id2;
      idx2Backup = idx2;
      vel2Backup = vel2;
    }
  }

  // data structures ("q2") which collect data and are used to patch p2 and id2
  vector<MyIDType>      idQ2;   idQ2.clear();
  vector<particle_sim>  q2;     q2.clear();
  vector<vec3f>         velQ2;  velQ2.clear();

  vector<MyIDType> requiredRemoteParticleIds;
  requiredRemoteParticleIds.clear();

  if (patch_data)
  {
    //search for the particles that are *really* needed
    //
    tsize i1=0;
    tsize i2=0;
    tsize nZeros=0;
    //
    while(i1<p1.size() && i2<p2.size())
    {
      if (id1[idx1[i1]]==id2[idx2[i2]])
      {
        // particle is present in both, p1 and p2
        if (id1[idx1[i1]]==0)
          nZeros++;
        i1++;
        i2++;
      }
      else if (id1[idx1[i1]]<id2[idx2[i2]])
      {
        // particle is present in p1 but not in p2
        if (id1[idx1[i1]]>0)
          requiredRemoteParticleIds.push_back( id1[idx1[i1]] );
        i1++;
      }
      else if (id1[idx1[i1]]>id2[idx2[i2]])
      {
        // particle is present in p2 but not in p1
        i2++;
      }
    }
    //
    if (debug_msg)
      cout << "MpiFetchRemoteParticles() : nZeros : " << nZeros << endl << flush;
  }
  else
  {
    // "dumb" method: set all required Ids to id1
    requiredRemoteParticleIds=id1;
  }


  int numberOfRequiredRemoteParticleIds;
  numberOfRequiredRemoteParticleIds = requiredRemoteParticleIds.size();

  if (debug_msg)
    cout << "MpiFetchRemoteParticles() : Required particles : " << numberOfRequiredRemoteParticleIds << endl << flush;


  //
  // In each MPI rank, we now have:
  //  requiredRemoteParticleIds         : list of particle IDs which are not in p2 and needed from other MPI processes
  //  numberOfRequiredRemoteParticleIds : the number of particles needed from other MPI processes
  //
  // We now loop over all ranks, where iRank is the rank
  // which shall receive required data from all other ranks
  // the vectors q2 and idQ2 collect the data and the indices
  //
  for (int iRank=0; iRank<mpiMgr.num_ranks(); iRank++)
  {

    if (debug_msg)
      cout << "MpiFetchRemoteParticles() : Redistributing required particles -- begin, iRank=" << iRank << endl << flush;

    if (iRank==mpiMgr.rank())
    {
      idQ2.clear();
      q2.clear();
      velQ2.clear();
    }


    // communicate the number of particles iRank needs to all other ranks
    MyIDType nParticles;
    if (iRank==mpiMgr.rank())
      nParticles=numberOfRequiredRemoteParticleIds;
    mpiMgr.bcastRaw(&nParticles, 1, iRank);


    // communicate the actual particle Ids iRank needs to all other ranks
    vector<MyIDType> needParticleIds;
    needParticleIds.clear();
    //
    if (iRank==mpiMgr.rank())
      needParticleIds=requiredRemoteParticleIds;
    else
      needParticleIds.resize(nParticles);
    //
    mpiMgr.bcastRaw(&needParticleIds[0], nParticles/*no sizeof() needed here!!!*/, iRank);


    // in each rank, we build an index for the particles iRank needs
    vector<MyIDType> needIdx;
    needIdx.clear();
    buildIndex(needParticleIds.begin(),needParticleIds.end(),needIdx);


    // now we need to find out if the current rank has some of the particles needed by iRank
    int haveNParticleIds;   haveNParticleIds = 0;
    // save particle data if it is found
    vector<MyIDType>     haveParticleId;    haveParticleId.clear();
    vector<particle_sim> haveParticleData;  haveParticleData.clear();
    vector<vec3f>        haveParticleVel;   haveParticleVel.clear();
    //
    if (   (!patch_data)
        || ((patch_data)&&(iRank!=mpiMgr.rank())))
    {
      if (debug_msg)
        cout << "MpiFetchRemoteParticles() : Redistributing required particles -- searching, iRank=" << iRank << endl << flush;

      tsize i1=0, i2=0;
      while(i1<needParticleIds.size() && i2<p2.size())
      {
        if (needParticleIds[needIdx[i1]]==id2[idx2[i2]])
        {
          // particle is present in both, needParticleIds *and* id2
          haveParticleId.push_back(   id2[idx2[i2]] );
          haveParticleData.push_back(  p2[idx2[i2]] );
          if (interpol_mode >1)
            haveParticleVel.push_back(  vel2[idx2[i2]] );
          haveNParticleIds++;
          i1++;
          i2++;
        }
        else if (needParticleIds[needIdx[i1]]<id2[idx2[i2]])
        {
          // particle is present in needParticleIds but not in id2
          i1++;
        }
        else if (needParticleIds[needIdx[i1]]>id2[idx2[i2]])
        {
          // particle is present in id2 but not in needParticleIds
          i2++;
        }
      }
    }
    mpiMgr.barrier();


    // now we need to communicate back to iRank
    // the number of particles each rank has to offer
    vector<int> numberOfParticlesFromRank;
    numberOfParticlesFromRank.clear();
    {
      arr<int> numPartTmp;
      numPartTmp.alloc(mpiMgr.num_ranks());
      // for convenience, we use gather_m which supports the arr datatype
      mpiMgr.gather_m(haveNParticleIds, numPartTmp, iRank);
      //
      if (iRank==mpiMgr.rank())
      {
        for (int jRank=0; jRank<mpiMgr.num_ranks(); jRank++)
        {
          numberOfParticlesFromRank.push_back( numPartTmp[jRank] );
        }
        if (debug_msg)
        {
          cout << "MpiFetchRemoteParticles() : numberOfParticlesFromRank, jRank" << endl;
          for (int jRank=0; jRank<mpiMgr.num_ranks(); jRank++)
          {
            cout << "   " << numberOfParticlesFromRank[jRank] << ", " << jRank << endl;
          }
          cout << flush;
        }
      }
      numPartTmp.dealloc();
    }
    mpiMgr.barrier();


    // now we need to actually communicate the particle data and the Ids back to iRank
    //
    if (debug_msg)
      cout << "MpiFetchRemoteParticles() : Redistributing required particles -- sending particle IDs, iRank=" << iRank << endl << flush;

    // exchange particle IDs
    for (int jRank=0; jRank<mpiMgr.num_ranks(); jRank++)
    {
      if (iRank!=jRank)
      {
        if (iRank==mpiMgr.rank())
        {
          // receive particle Ids
          vector<MyIDType> idQ2Tmp;
          idQ2Tmp.resize( numberOfParticlesFromRank[jRank] );
          // receive particle_sim objects from rank jRank
          mpiMgr.recvRawVoid(&(idQ2Tmp[0]), NAT_CHAR, numberOfParticlesFromRank[jRank]*sizeof(MyIDType), jRank);
          // append these objects to q2
          for (vector<MyIDType>::iterator it=idQ2Tmp.begin(); it!=idQ2Tmp.end(); ++it)
          {
            idQ2.push_back(*it);
          }
          idQ2Tmp.clear();
        }
        else if (jRank==mpiMgr.rank())
        {
          // send particle Ids
          mpiMgr.sendRawVoid(&(haveParticleId[0]), NAT_CHAR, haveParticleId.size()*sizeof(MyIDType), iRank);
        }
      }
      else // (iRank==jRank)
      {
        if ((!patch_data) && (iRank==mpiMgr.rank()))
          idQ2.insert(idQ2.end(), haveParticleId.begin(), haveParticleId.end());
      }
      mpiMgr.barrier();
    } // end of particle id exchange loop

    if (debug_msg)
      cout << "MpiFetchRemoteParticles() : Redistributing required particles -- sending particle data, iRank=" << iRank << endl << flush;

    // exchange particle data (particle_sim)
    for (int jRank=0; jRank<mpiMgr.num_ranks(); jRank++)
    {
      if (iRank!=jRank)
      {
        if (iRank==mpiMgr.rank())
        {
          // receive particle data
          vector<particle_sim> q2Tmp;
          q2Tmp.resize( numberOfParticlesFromRank[jRank] );
          // receive particle_sim objects from rank jRank
          tsize nBytes=numberOfParticlesFromRank[jRank]*sizeof(particle_sim);
          //
          // TODO : cut MPI message into pieces
          planck_assert(nBytes < std::numeric_limits<int>::max(), "Ooops, too many elements in MPI message.");
          //
          tsize source=jRank;
          if (debug_msg)
            cout << "nBytes, source : " << nBytes << ", " << source << endl << flush;
          //
          mpiMgr.recvRawVoid(&(q2Tmp[0]), NAT_CHAR, nBytes, source);
          // append these objects to q2
          for (vector<particle_sim>::iterator it=q2Tmp.begin(); it!=q2Tmp.end(); ++it)
          {
            q2.push_back(*it);
          }
          q2Tmp.clear();
        }
        else if (jRank==mpiMgr.rank())
        {
          // send particle data
          tsize nBytes=haveParticleData.size()*sizeof(particle_sim);
          //
          // TODO : cut MPI message into pieces
          planck_assert(nBytes<std::numeric_limits<int>::max(), "Ooops, too many elements in MPI message.");
          //
          tsize target=iRank;
          if (debug_msg)
            cout << "nBytes, target : " << nBytes << ", " << target << endl << flush;
          //
          mpiMgr.sendRawVoid(&(haveParticleData[0]), NAT_CHAR, nBytes, target);
        }
      }
      else // (iRank==jRank)
      {
        if ((!patch_data) && (iRank==mpiMgr.rank()))
          q2.insert(q2.end(), haveParticleData.begin(), haveParticleData.end());
      }
      mpiMgr.barrier();
    } // end of particle data exchange loop

    // exchange particle velocities
    if (interpol_mode>1)
    {
      for (int jRank=0; jRank<mpiMgr.num_ranks(); jRank++)
      {
        if (iRank!=jRank)
        {
          if (iRank==mpiMgr.rank())
          {
            // receive particle data
            vector<vec3f> velQ2Tmp;
            velQ2Tmp.resize( numberOfParticlesFromRank[jRank] );
            // receive particle_sim objects from rank jRank
            tsize nBytes=numberOfParticlesFromRank[jRank]*sizeof(vec3f);
            //
            // TODO : cut MPI message into pieces
            planck_assert(nBytes < std::numeric_limits<int>::max(), "Ooops, too many elements in MPI message.");
            //
            tsize source=jRank;
            if (debug_msg)
              cout << "nBytes, source : " << nBytes << ", " << source << endl << flush;
            //
            mpiMgr.recvRawVoid(&(velQ2Tmp[0]), NAT_CHAR, nBytes, source);
            // append these objects to velQ2
            for (vector<vec3f>::iterator it=velQ2Tmp.begin(); it!=velQ2Tmp.end(); ++it)
            {
              velQ2.push_back(*it);
            }
            velQ2Tmp.clear();
          }
          else if (jRank==mpiMgr.rank())
          {
            // send particle data
            tsize nBytes=haveParticleVel.size()*sizeof(vec3f);
            //
            // TODO : cut MPI message into pieces
            planck_assert(nBytes<std::numeric_limits<int>::max(), "Ooops, too many elements in MPI message.");
            //
            tsize target=iRank;
            if (debug_msg)
              cout << "nBytes, target : " << nBytes << ", " << target << endl << flush;
            //
            mpiMgr.sendRawVoid(&(haveParticleVel[0]), NAT_CHAR, nBytes, target);
          }
        }
        else // (iRank==jRank)
        {
          if ((!patch_data) && (iRank==mpiMgr.rank()))
            velQ2.insert(velQ2.end(), haveParticleVel.begin(), haveParticleVel.end());
        }
        mpiMgr.barrier();
      } // end of particle velocities exchange loop
    }

    if (debug_msg)
      cout << "MpiFetchRemoteParticles() : end of iRank loop ..." << endl << flush;

  } // end of the outer loop over iRank

  mpiMgr.barrier();

  // Status: On each MPI process, the vectors
  //   idQ2 hold the particle Ids for "remote" particles,
  //   q2   holds the particle data - " -
  if (debug_msg)
  {
    cout << "MpiFetchRemoteParticles() : appending remote particles ..." << endl;
    cout << "p1Size, id1Size  : " << p1.size() << ", " << id1.size()  << endl;
    cout << "p2Size, id2Size  : " << p2.size() << ", " << id2.size()  << endl;
    cout << "q2Size, idQ2Size : " << q2.size() << ", " << idQ2.size() << endl << flush;
  }

  if (patch_data)
  {
    for (vector<MyIDType>::iterator it=idQ2.begin(); it!=idQ2.end(); ++it)
    {
      id2.push_back(*it);
    }
    for (vector<particle_sim>::iterator it=q2.begin(); it!=q2.end(); ++it)
    {
      p2.push_back(*it);
    }
    if (interpol_mode > 1)
    {
      for (vector<vec3f>::iterator it=velQ2.begin(); it!=velQ2.end(); ++it)
      {
        vel2.push_back(*it);
      }
    }
  }
  else
  {
    id2.swap(idQ2);
    p2.swap(q2);
    vel2.swap(velQ2);
  }

  idQ2.clear();
  q2.clear();
  velQ2.clear();

  if (debug_msg)
    cout << "MpiFetchRemoteParticles() : recreating index idx2 ..." << endl << flush;

  tstack_replace("Fetch remote particles","Particle index generation");
  idx2.clear();
  buildIndex(id2.begin(), id2.end(), idx2);
  tstack_replace("Particle index generation","Fetch remote particles");

  mpiMgr.barrier();

  if (debug_msg)
    cout << "MpiFetchRemoteParticles() : END" << endl << flush;

  return;
}

// MpiStripRemoteParticles() removes particles from p2
void sceneMaker::MpiStripRemoteParticles ()
{
  if (mpiMgr.num_ranks()==1)
    return;

  bool debug_msg  = params.find<bool>("mpi_interpolation_debug_msg",false);
  bool patch_data = params.find<bool>("mpi_interpolation_patch_data",true);
  bool reread_data= params.find<bool>("mpi_interpolation_reread_data",false);

  if (reread_data)
  {
    p2.clear();
    id2.clear();
    idx2.clear();
    vel2.clear();
  }
  else
  {
    if (patch_data)
    {
      if (debug_msg)
          cout << "MpiStripRemoteParticles() : removing remote data ..." << endl << flush;

      p2.erase(  p2.begin() + numberOfLocalParticles,  p2.end());
      id2.erase(id2.begin() + numberOfLocalParticles, id2.end());
      if (interpol_mode > 1)
        vel2.erase(vel2.begin() + numberOfLocalParticles, vel2.end());

      if (debug_msg)
        cout << "MpiStripRemoteParticles() : regenerating idx2 ..." << endl << flush;

      tstack_replace("Fetch remote particles","Particle index generation");
      idx2.clear();
      buildIndex(id2.begin(), id2.end(), idx2);
      tstack_replace("Particle index generation","Fetch remote particles");
    }
    else
    {
      p2.swap(p2Backup);
      id2.swap(id2Backup);
      idx2.swap(idx2Backup);
      vel2.swap(vel2Backup);
      p2Backup.clear();
      id2Backup.clear();
      idx2Backup.clear();
      vel2Backup.clear();
    }
  }

  return;
}

#endif
