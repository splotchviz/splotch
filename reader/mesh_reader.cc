/*
 * Copyright (c) 2004-2014
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
#include <vector>
#include <string>

#include "cxxsupport/arr.h"
#include "cxxsupport/sort_utils.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

using namespace std;

namespace {

void mesh_reader_prep (paramfile &params, bifstream &inp, arr<int> &qty_idx,
  int &nfields, int64 &mybegin, int64 &npart, float *rrr, int64 &npart_total)
  {
  /* qty_idx characterize the mesh according 
     to the following standard:
     qty_idx[0] = x size (number of cells)
     qty_idx[1] = y size (number of cells)
     qty_idx[2] = z size (number of cells)
     qty_idx[3] = dummy -> substituted by rrr which must be float
     qty_idx[4] = intensity
     qty_idx[5] = color 1 (R)
     qty_idx[6] = color 2 (G)
     qty_idx[7] = color 3 (B)
     qty_idx[8] = ordering (0=C, 1=Fortran)
  */

  bool doswap = params.find<bool>("swap_endian",true);
  string datafile = params.find<string>("infile");
  inp.open (datafile.c_str(),doswap);
  nfields = params.find<int>("num_blocks");

  qty_idx.alloc(9);
  qty_idx[0] = params.find<int>("x");
  qty_idx[1] = params.find<int>("y");
  qty_idx[2] = params.find<int>("z");
  qty_idx[4] = params.find<int>("I",-1)-1;
  qty_idx[5] = params.find<int>("C1")-1;
  qty_idx[6] = params.find<int>("C2",-1)-1;
  qty_idx[7] = params.find<int>("C3",-1)-1;
  qty_idx[8] = params.find<int>("order",0);
  npart_total = qty_idx[0]*qty_idx[1]*qty_idx[2];
  *rrr = params.find<float>("r",1.0);

  if (mpiMgr.master())
    {
    cout << "Input data file name: " << datafile << endl;
    cout << "Number of columns " << nfields << endl;
    cout << "Number of mesh cells " << npart_total << endl;
    }
  int64 myend;
  mpiMgr.calcShare (0, npart_total, mybegin, myend);
  npart = myend-mybegin;
  }

void mesh_reader_finish (vector<particle_sim> &points)
  {
  float minr=1e30;
  float minx=1e30;
  float miny=1e30;
  float minz=1e30;
  float maxr=-1e30;
  float maxx=-1e30;
  float maxy=-1e30;
  float maxz=-1e30;
  for (tsize i=0; i<points.size(); ++i)
    {
    //points[i].active = 1;
    points[i].type=0;
    minr = min(minr,points[i].r);
    maxr = max(maxr,points[i].r);
    minx = min(minx,points[i].x);
    maxx = max(maxx,points[i].x);
    miny = min(miny,points[i].y);
    maxy = max(maxy,points[i].y);
    minz = min(minz,points[i].z);
    maxz = max(maxz,points[i].z);
    }
  mpiMgr.allreduce(maxr,MPI_Manager::Max);
  mpiMgr.allreduce(minr,MPI_Manager::Min);
  mpiMgr.allreduce(maxx,MPI_Manager::Max);
  mpiMgr.allreduce(minx,MPI_Manager::Min);
  mpiMgr.allreduce(maxy,MPI_Manager::Max);
  mpiMgr.allreduce(miny,MPI_Manager::Min);
  mpiMgr.allreduce(maxz,MPI_Manager::Max);
  mpiMgr.allreduce(minz,MPI_Manager::Min);
#ifdef DEBUG
  cout << "MIN, MAX --> " << minr << " " << maxr << endl;
  cout << "MIN, MAX --> " << minx << " " << maxx << endl;
  cout << "MIN, MAX --> " << miny << " " << maxy << endl;
  cout << "MIN, MAX --> " << minz << " " << maxz << endl;
#endif
  }

} // unnamed namespace

void mesh_reader (paramfile &params, vector<particle_sim> &points)
  {
  bifstream inp;
  float rrr;
  int nfields;
  int64 mybegin, npart, npart_total;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "MESH BINARY DATA" << endl;
  mesh_reader_prep (params, inp, qty_idx, nfields, mybegin, npart, &rrr, npart_total);

  points.resize(npart);
  arr<float> buffer(npart);

  for (tsize qty=4; qty<qty_idx.size()-1; ++qty)
    {
    if(qty_idx[qty]<0) continue;

    inp.seekg(sizeof(float)*(qty_idx[qty]*npart_total+mybegin),ios::beg);
    inp.get(&buffer[0],npart);

#define CASEMACRO__(num,str) \
    case num: \
      for (int64 i=0; i<npart; ++i) \
        points[i].str = buffer[i]; \
      break;

    switch(qty)
      {
      CASEMACRO__(0,x)
      CASEMACRO__(1,y)
      CASEMACRO__(2,z)
      CASEMACRO__(3,r)
      CASEMACRO__(4,I)
      CASEMACRO__(5,e.r)
      CASEMACRO__(6,e.g)
      CASEMACRO__(7,e.b)
      }
    }

#undef CASEMACRO__

//set intensity if not read
  if (qty_idx[4]<0)
    for (int64 i=0; i<npart; ++i) points[i].I=0.5;

//set smoothing length: assumed constant for all volume
  for (int64 i=0; i<npart; ++i) points[i].r=rrr;

//set coordinates: now be careful to ordering!
  int dimx = qty_idx[0];
  int dimy = qty_idx[1];
  int dimz = qty_idx[2];
  if (qty_idx[8] == 1)
    swap(dimx,dimz);

  for(int64 i=0; i<npart; ++i)
    {
    int64 iaux = i + mybegin;
    int i1 = iaux/(dimx*dimy);
    int res = iaux%(dimx*dimy);
    int i2 = res/dimx;
    int i3 = res%dimx;
    points[i].x = float(i3);
    points[i].y = float(i2);
    points[i].z = float(i1);
    }

  mesh_reader_finish (points);
  }
