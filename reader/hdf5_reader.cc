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

#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "splotch/splotchutils.h"

using namespace std;
#ifdef HDF5
#include <hdf5.h>

namespace {

void hdf5_reader_prep (paramfile &params, hid_t * inp, arr<int> &qty_idx,
  int &nfields, int64 &npart, float * rrr, string * field, int * rank, int64 * start)
  {
  /* qty_idx characterize the mesh according 
     to the following standard:
     qty_idx[0] = x size (number of cells)
     qty_idx[1] = y size (number of cells)
     qty_idx[2] = z size (number of cells)
  */

  hid_t       file_id, dataset_id;  /* identifiers */
  herr_t      status;
  float raux;
  string datafile = params.find<string>("infile");

  qty_idx.alloc(5);
  raux = params.find<float>("smooth_param",0.0);

    cout << "FIELD NAME -> " << field[0].c_str() << endl; 
  int use_field = 0;
  if(field[0].compare("-1") == 0)use_field=3;

  hid_t dataset_space, nrank;

  file_id = H5Fopen(datafile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  *inp = file_id;
  dataset_id = H5Dopen(file_id,field[use_field].c_str());
  dataset_space = H5Dget_space(dataset_id);
  nrank = H5Sget_simple_extent_ndims(dataset_space);
  //cout << "SPACE DIM = " << nrank << endl;
  *rank = nrank;
  hsize_t * s_dims    = new hsize_t [nrank];
  hsize_t * s_maxdims = new hsize_t [nrank];
  H5Sget_simple_extent_dims(dataset_space, s_dims, s_maxdims);
  //cout << "DIMENSIONS = " << s_dims[0] << " " << s_dims[1] << " " << s_dims[2]  << endl;
  H5Dclose(dataset_id);

  if(s_dims[0]<mpiMgr.num_ranks())
    {
      cout << "Too many processors. For this problem maximum number of processors is " 
           << s_dims[0] << "... Exiting\n";
      exit(100);
    } 

  int64 dimaux = (int)s_dims[0]/mpiMgr.num_ranks();
  qty_idx[0] = dimaux;
  if(mpiMgr.rank() == mpiMgr.num_ranks()-1) qty_idx[0] = s_dims[0]-(qty_idx[0]*(mpiMgr.num_ranks()-1));
  int64 npart_total = qty_idx[0];
  if(nrank == 3)
  {
    qty_idx[1] = (int)s_dims[1];
    qty_idx[2] = (int)s_dims[2];
    npart_total = qty_idx[0]*qty_idx[1]*qty_idx[2];
  }

  npart = npart_total;
  *rrr = raux;
  *start = dimaux * mpiMgr.rank();

  if (mpiMgr.master())
    {
    cout << "Input data file name: " << datafile << endl;
    cout << "Number of mesh cells " << npart_total << endl;
    }

  }

void hdf5_reader_finish (vector<particle_sim> &points, float thresh)
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
    points[i].e.r+=thresh;
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

void hdf5_reader (paramfile &params, vector<particle_sim> &points)
  {
  hid_t file_id, dataset_id;
  float rrr;
  int nfields;
  string * field;
  int64 start_local;
  int rank;
  int64 mybegin, npart;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "HDF5 DATA" << endl;

  int number_of_fields = 8; 
  field = new string[number_of_fields];
  
  field[0] = params.find<string>("x", "-1");
  field[1] = params.find<string>("y", "-1");
  field[2] = params.find<string>("z", "-1");
  field[3] = params.find<string>("C1");
  field[4] = params.find<string>("C2", "-1");
  field[5] = params.find<string>("C3", "-1");
  field[6] = params.find<string>("r", "-1");
  field[7] = params.find<string>("I", "-1");
  float thresh = params.find<float>("thresh", 0.0);

  hdf5_reader_prep (params, &file_id, qty_idx, nfields, npart, &rrr, field, &rank, &start_local);
  hid_t dataset_space;
  hsize_t * start     = new hsize_t [rank];
  hsize_t * stride    = new hsize_t [rank];
  hsize_t * count     = new hsize_t [rank];
  hsize_t * block     = new hsize_t [rank];

  start[0]  = (hsize_t)start_local;
  if(rank == 3)
  {
    start[1]  = 0;
    start[2]  = 0; 
  }
  for(int k=0; k<rank; k++)
  {
    stride[k] = NULL;
    count[k]  = (hsize_t)qty_idx[k];
    block[k]  = NULL;
  }

//#ifdef DEBUG
  cout << mpiMgr.rank() << " - - - - " << start[0] << endl;
  cout << mpiMgr.rank() << " npart - - - - " << npart << endl;
  cout << mpiMgr.rank() << " - - - - " << qty_idx[0] << " " << qty_idx[1] << " " << qty_idx[2] << endl;
//#endif

  points.resize(npart);

// read fields

  for (tsize qty=0; qty<number_of_fields; ++qty)
    {
    if(field[qty].compare("-1") == 0) continue;
    float * buffer = new float [npart];
    cout << "FIELD NAME " << field[qty].c_str() << endl; 

//NOW HDF READ STUFF

// set the hyperslab to be read
  dataset_id = H5Dopen(file_id,field[qty].c_str());
  dataset_space = H5Dget_space(dataset_id);  
  H5Sselect_hyperslab(dataset_space, H5S_SELECT_SET, start, NULL, count, NULL); 
// prepare the memory space
  hid_t memoryspace = H5Screate_simple(rank, count, count); 

  H5Dread(dataset_id, H5T_NATIVE_FLOAT, memoryspace, dataset_space, H5P_DEFAULT, buffer);

  H5Dclose(dataset_id);
  H5Sclose(memoryspace);
  H5Sclose(dataset_space);
    
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
      CASEMACRO__(3,e.r)
      CASEMACRO__(4,e.g)
      CASEMACRO__(5,e.b)
      CASEMACRO__(6,r)
      }

    delete [] buffer;
    }

#undef CASEMACRO__

//set intensity if not read
  if(field[6].compare("-1") == 0)
    for (int64 i=0; i<npart; ++i) points[i].I=0.5;

//set smoothing length: assumed constant for all volume
  if(field[7].compare("-1") == 0)
    for (int64 i=0; i<npart; ++i) points[i].r=rrr;

    if(field[0].compare("-1") == 0)
    {
//set coordinates: HDF5 files are ALWAYS written in C ordering
    int dimx = qty_idx[0];
    int dimy = qty_idx[1];
    int dimz = qty_idx[2];

    for(int64 i=0; i<npart; ++i)
      {
        //int64 iaux = i + mybegin;
        int64 iaux = i;
        int i1 = iaux/(dimz*dimy);  
        int res = iaux%(dimz*dimy);
        int i2 = res/dimz;
        int i3 = res%dimz;
        points[i].x = (float)(i1+start_local);
        points[i].y = (float)i2;
        points[i].z = (float)i3;
        //if(points[i].e.r > 1e-10)cout << mpiMgr.rank() << " " << i << " " << points[i].x << " " << points[i].y << " " << points[i].z << " " <<  points[i].e.r << "\n";
        
      }
// end if
      }

  hdf5_reader_finish (points, thresh);
  }
#else

void hdf5_reader (paramfile &params, vector<particle_sim> &points)
  {
    cout << "HDF5 I/O not supported... Exiting... " << endl;
  }
#endif
