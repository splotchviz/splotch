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


// This reader allows to read only 1D (particles) or 3D (regular grids) datasets

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

#ifdef HDF5
#include <hdf5.h>

using namespace std;

namespace {

void hdf5_reader_prep (paramfile &params, hid_t * inp, int64 &npart, int64 * start, string * field)
{

  hid_t       file_id, dataset_id;  /* HDF5 handles */
  hid_t       dataset_space, nrank;
  herr_t      status;
  float raux;
  float smooth_factor = params.find<float>("smooth_factor",1.0);
  string datafile = params.find<string>("infile");

  file_id = H5Fopen(datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  *inp = file_id;

  dataset_id = H5Dopen(file_id,field[0].c_str());
  dataset_space = H5Dget_space(dataset_id);
  nrank = H5Sget_simple_extent_ndims(dataset_space);
  hsize_t * s_dims    = new hsize_t [nrank];
  hsize_t * s_maxdims = new hsize_t [nrank];

  H5Sget_simple_extent_dims(dataset_space, s_dims, s_maxdims);
  H5Dclose(dataset_id);

  int64 npart_tot, mybegin, myend;
  npart_tot = s_dims[0];
  mpiMgr.calcShare (0, npart_tot, mybegin, myend);
  npart = myend-mybegin;
  *start = mybegin;
}

void hdf5_reader_finish (vector<particle_sim> &points)
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
//#ifdef DEBUG
  cout << "MIN, MAX --> " << minr << " " << maxr << endl;
  cout << "MIN, MAX --> " << minx << " " << maxx << endl;
  cout << "MIN, MAX --> " << miny << " " << maxy << endl;
  cout << "MIN, MAX --> " << minz << " " << maxz << endl;
//#endif
  }

} // unnamed namespace

void galaxy_reader (paramfile &params, vector<particle_sim> &points)
{
  hid_t file_id, dataset_id;
  hid_t dataset_space;
  float rrr;
  int nfields;
  string * field;
  int64 start_local;
  int64 mybegin, npart;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "GALAXY DATA" << endl;

  int number_of_fields = 10; 
  field = new string[number_of_fields];
  
  field[0] = "Xpos";
  field[1] = "Ypos";
  field[2] = "Zpos";
  field[3] = "Red";
  field[4] = "Green";
  field[5] = "Blue";
  field[6] = "Radius";
  field[7] = "Intensity";
  field[8] = "Type";

  qty_idx.alloc(number_of_fields);
  qty_idx[0] = 1;
  qty_idx[1] = 1;
  qty_idx[2] = 1;
  qty_idx[3] = 1;
  qty_idx[4] = 1;
  qty_idx[5] = 1;
  qty_idx[6] = -1;
  qty_idx[7] = -1;
  qty_idx[8] = 1;

  hdf5_reader_prep (params, &file_id, npart, &start_local, field);
  points.resize(npart);

  float * fbuffer;
  unsigned short * ubuffer;

  fbuffer = new float[npart];
  ubuffer = new unsigned short[npart];

//NOW HDF5 READ STUFF

  int rank = 1;
  hsize_t * start     = new hsize_t [rank];
  hsize_t * stride    = new hsize_t [rank];
  hsize_t * count     = new hsize_t [rank];
  hsize_t * block     = new hsize_t [rank];

  start[0]  = (hsize_t)start_local;
  stride[0] = 1;
  count[0]  = npart;
  block[0]  = 1;
  
  for (int i=0; i<9; i++)
  {
     hid_t memoryspace;
     if(qty_idx[i] >= 0)
     {
      cout << "working on " << i << endl;
      dataset_id = H5Dopen(file_id,field[i].c_str());
      dataset_space = H5Dget_space(dataset_id);
      H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, start, stride, count, block);
      memoryspace = H5Screate_simple(rank, count, count);
      H5Dread(dataset_id, H5T_NATIVE_FLOAT, memoryspace, dataset_space, H5P_DEFAULT, fbuffer);
     }

#define CASEMACRO__(num,str,noval) \
    case num: \
        if (qty_idx[num]>=0) \
          for (int64 ii=0; ii<npart; ++ii) points[ii].str = fbuffer[ii]; \
        else \
          for (int64 ii=0; ii<npart; ++ii) points[ii].str = noval; \
        break;


    switch(i)
      {
      CASEMACRO__(0,x,1.0)
      CASEMACRO__(1,y,1.0)
      CASEMACRO__(2,z,1.0)
      CASEMACRO__(3,e.r,0.0)
      CASEMACRO__(4,e.g,0.0)
      CASEMACRO__(5,e.b,0.0)
      CASEMACRO__(6,r,0.00001)
      CASEMACRO__(7,I,0.5)
      case 8:
        for (int64 ii=0; ii<npart; ++ii) points[ii].type = (unsigned int)fbuffer[ii];
        break;
      }

     if(i<5 || i>8)
     {
      H5Dclose(dataset_id);
      H5Sclose(memoryspace);
      H5Sclose(dataset_space);
     }
  } 
  delete [] fbuffer;

  for (int64 ii=0; ii<npart; ++ii) points[ii].active = true; 

  H5Fclose(file_id);
  
  hdf5_reader_finish (points);
}

#else

using namespace std;

void hdf5_reader (paramfile &params, vector<particle_sim> &points)
{
    cout << "HDF5 I/O not supported... Exiting... " << endl;
}
#endif
