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


/**
 *  Reader for Gadget HDF5 output (started from <gadget_reader.cc>).
 *
 *  TODO  - implement improved error handling of HDF5 calls
 *        - merge with <gadget_reader.cc> as a significant amount
 *          of code is shared?
 *
 *                                               (Klaus Reuter, RZG)
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "cxxsupport/arr.h"
#include "cxxsupport/sort_utils.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

using namespace std;

#ifdef HDF5

// This reader uses the 1.8.x API and will not work with earlier versions.
#ifdef H5_USE_16_API
#undef H5_USE_16_API
#endif

#include <hdf5.h>


// helper function to read a single attribute from a HDF5 group, typically from "Header"
//
void read_hdf5_group_attribute( hid_t hdf5_group, const char *attrName, void *attribute, hid_t hdf5_datatype )
{
  hid_t   hdf5_attribute, hdf5_dataSpace;
  int     rank;
  hsize_t sdim[64];
  //
  hdf5_attribute = H5Aopen_name(hdf5_group, attrName);
  hdf5_dataSpace = H5Aget_space(hdf5_attribute);
  rank           = H5Sget_simple_extent_ndims(hdf5_dataSpace);
  H5Sget_simple_extent_dims(hdf5_dataSpace, sdim, NULL);
  H5Aread(hdf5_attribute, hdf5_datatype, attribute);
  H5Sclose(hdf5_dataSpace);
  H5Aclose(hdf5_attribute);
}


int gadget_read_hdf5_header(hid_t hdf5_file, unsigned int *npart, double &time, double &redshift, 
                            unsigned int *nparttotal, double &boxsize, paramfile &params)
{
  hid_t  hdf5_header;
  herr_t hdf5_status;
  double tmpDbl;

  hdf5_header = H5Gopen(hdf5_file, "/Header", H5P_DEFAULT);

  read_hdf5_group_attribute(hdf5_header, "NumPart_ThisFile", npart,      H5T_NATIVE_UINT);
  read_hdf5_group_attribute(hdf5_header, "Time_GYR",         &time,      H5T_NATIVE_DOUBLE);
  read_hdf5_group_attribute(hdf5_header, "Redshift",         &redshift,  H5T_NATIVE_DOUBLE);
  read_hdf5_group_attribute(hdf5_header, "NumPart_Total",    nparttotal, H5T_NATIVE_UINT);
  read_hdf5_group_attribute(hdf5_header, "BoxSize",          &boxsize,   H5T_NATIVE_DOUBLE);

  // read information necessary for the high order interpolation
  read_hdf5_group_attribute(hdf5_header, "HubbleParam",      &tmpDbl,    H5T_NATIVE_DOUBLE);
  params.setParam("hubble",dataToString(tmpDbl));
  read_hdf5_group_attribute(hdf5_header, "Omega0",           &tmpDbl,    H5T_NATIVE_DOUBLE);
  params.setParam("omega",dataToString(tmpDbl));
  read_hdf5_group_attribute(hdf5_header, "OmegaLambda",      &tmpDbl,    H5T_NATIVE_DOUBLE);
  params.setParam("lambda",dataToString(tmpDbl));

  hdf5_status = H5Gclose(hdf5_header);

  return 0;
}


// function to read data (==arrays) from an HDF5 table
//
void readHDF5DataArray( hid_t group_id, const char * arrayName, hid_t hdf5Type, void *dataArray )
{
  hid_t dataSetId;
  int nColumns;

  // if arrayName does not exist, H5Dopen writes an error message to <stderr>
  // and returns a value < 0
  dataSetId = H5Dopen(group_id, arrayName, H5P_DEFAULT);

  if (dataSetId<0)
  {
    return;
  }

  hid_t dataSpace;
  int   dataSetRank;
  dataSpace   = H5Dget_space(dataSetId);
  dataSetRank = H5Sget_simple_extent_ndims(dataSpace);
  herr_t status;
  hsize_t dataSetDims[64];
  status = H5Sget_simple_extent_dims(dataSpace, dataSetDims, NULL);
  status = H5Sclose(dataSpace);

  if (dataSetRank==1)
  {
    nColumns=1;   // scalar field
  }
  else if(dataSetRank==2)
  {
    nColumns=3;   // vector field
    if (nColumns!=dataSetDims[1])
      throw "readHDF5DataArray: dataSetDims[1]!=3";
  }
  else
  {
    throw "readHDF5DataArray: unknown HDF5 array format";
  }

  if (!((hdf5Type==H5T_NATIVE_UINT)||(hdf5Type==H5T_NATIVE_FLOAT)))
  {
    throw "readHDF5DataArray: wrong HDF5 data type";
  }

  H5Dread(dataSetId, hdf5Type, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataArray);

  H5Dclose(dataSetId);
}




void gadget_hdf5_reader(paramfile &params, int interpol_mode,
                        vector<particle_sim> &p, vector<MyIDType> &id, vector<vec3f> &vel, int snr,
                        double &time, double &redshift, double &boxsize)
{
  int numfiles = params.find<int>("numfiles",1);
  int readparallel = params.find<int>("readparallel",1);
  int ptypes = params.find<int>("ptypes",1);
  int ptype_found = -1, ntot = 1;

  string infilename = params.find<string>("infile");
  string snapdir    = params.find<string>("snapdir",string(""));
  string datadir    = params.find<string>("datadir",string(""));
  string filename;

  int ThisTask=mpiMgr.rank(),NTasks=mpiMgr.num_ranks();
  arr<int> ThisTaskReads(NTasks), DataFromTask(NTasks);
  arr<long> NPartThisTask(NTasks);


  // --- construct the filename (prefix) for the data files ---
  //     (uses the variable filename as temporary variable)
  filename.clear();
  // (1) add a data directory, if present
  //     e.g. "/data/"
  if (datadir.size()>0)
    filename += datadir+"/";
  //
  // (2) below the data directory, add a snapshot directory
  //     which is numbered by definition
  //     e.g. "/data/snapshot_000/"
  if (snapdir.size()>0)
  {
    filename += snapdir+intToString(snr,3)+"/";
  }
  //
  // (3) add a number to the filename, which is done always based on snr
  //     e.g. "/data/snapshot_000/snap_000"
  filename += infilename+intToString(snr,3);
  infilename = filename;
  filename.clear();
  // ---


  planck_assert(numfiles >= readparallel,
    "Number of files must be larger or equal number of parallel reads ...");
  planck_assert(numfiles%readparallel == 0,
    "Number of files must be a multiple of number of parallel reads ...");
  planck_assert(NTasks >= readparallel,
    "Number of tasks must be larger or equal number of parallel reads ...");
  planck_assert(NTasks%readparallel == 0,
    "Number of tasks must be a multiple of number of parallel reads ...");

  // Figure out who will read what
  int NTaskPerRead=NTasks/readparallel;
  int NFilePerRead=numfiles/readparallel;
  int SendingTask=-1;

  for(int i=0; i<NTasks; i++)
  {
    if(i%NTaskPerRead == 0 && (readparallel > 1 || i == 0))
    {
      ThisTaskReads[i]=i/NTaskPerRead*NFilePerRead;
      DataFromTask[i]=-1;
      SendingTask=i;
    }
    else
    {
      ThisTaskReads[i]=-1;
      DataFromTask[i]=SendingTask;
    }
  }

  if(mpiMgr.master())
  {
    {
      cout << "NOTE: Splotch was compiled to use ";
      if (sizeof(MyIDType)==4)
        cout << "32";
      else if (sizeof(MyIDType)==8)
        cout << "64";
      cout << "bit ParticleIDs." << endl;
    }
    int itask=0;
    for(int rt=0; rt<readparallel; rt++)
    {
      long NPartThisReadTask = 0;
      for(int f=0; f<NFilePerRead; f++)
      {
        int file=rt*NFilePerRead+f;
        unsigned int npartthis[6],nparttotal[6];
        filename=infilename;

        if (numfiles>1)
          filename+="."+dataToString(file);

        filename+=".hdf5";


        hid_t file_id;
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);
        H5Fclose(file_id);

        if((rt==0 && f==0) || !params.find<bool>("AnalyzeSimulationOnly"))
        {
          cout << "    Timestamp from file: t=" << time << endl;
          cout << "    Redshift from file:  z=" << redshift << endl;
        }

        if(rt==0 && f==0)
        {
          cout << "    Total number of particles in file :" << endl;
          cout << "    Type 0 (gas):   " << nparttotal[0] << endl;
          cout << "    Type 1 (dm):    " << nparttotal[1] << endl;
          cout << "    Type 2 (bndry): " << nparttotal[2] << endl;
          cout << "    Type 3 (bndry): " << nparttotal[3] << endl;
          cout << "    Type 4 (stars): " << nparttotal[4] << endl;
          cout << "    Type 5 (BHs):   " << nparttotal[5] << endl;
          if(params.find<bool>("AnalyzeSimulationOnly"))
          {
            if(nparttotal[0] > 0)
            {
              cout << "To visualize the gas particles add/change the following lines to the parameter file:" << endl;
              cout << "ptypes=1" << endl;
              cout << "AnalyzeSimulationOnly=FALSE" << endl;
              cout << "ptype0=0" << endl;
              cout << "size_fix0=0" << endl;
              cout << "size_label0=HSML" << endl;
              cout << "color_label0=U" << endl;
              ptype_found = 0;
              ptypes = 1;
              ntot = nparttotal[0];
            }
            else
            {
              if(nparttotal[1] > 0)
              {
                cout << "To vizualize the dm particles add/change following lines to the parameter file:" << endl;
                cout << "ptypes=1" << endl;
                cout << "AnalyzeSimulationOnly=FALSE" << endl;
                cout << "ptype0=1" << endl;
                cout << "color_label0=VEL" << endl;
                cout << "color_present0=63" << endl;
                cout << "color_is_vector0=TRUE" << endl;
                cout << "color_log0=FALSE" << endl;
                cout << "color_asinh0=TRUE" << endl;
                ptype_found = 1;
                ptypes = 1;
                ntot = nparttotal[1];
              }
            }
          }
        }
        for(int itype=0; itype<ptypes; itype++)
        {
          int type = params.find<int>("ptype"+dataToString(itype),0);
          if(params.find<bool>("AnalyzeSimulationOnly") && ptype_found >= 0)
            type = ptype_found;
          NPartThisReadTask += npartthis[type];
        }
      }
      long SumPartThisReadTask = 0;
      for(int t=0; t<NTaskPerRead-1; t++)
      {
        NPartThisTask[itask] = NPartThisReadTask / NTaskPerRead;
        SumPartThisReadTask += NPartThisReadTask / NTaskPerRead;
        itask++;
      }
      NPartThisTask[itask] = NPartThisReadTask - SumPartThisReadTask;
      itask++;
    }
  }


  mpiMgr.bcast(NPartThisTask,0);
  mpiMgr.bcast(boxsize,0);

  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
  {
    cout << " Reading " << numfiles << " files by " << readparallel << " tasks ... " << endl;
    cout << " Task " << ThisTask << "/" << NTasks << endl;
    cout << " NTaskPerRead/NFilePerRead " << NTaskPerRead << "," << NFilePerRead << endl;
    cout << " ThisTaskReads";
    for(int i=0; i<NTasks; i++)
      cout << ',' << ThisTaskReads[i];
    cout << endl;
    cout << " DataFromTask";
    for(int i=0; i<NTasks; i++)
      cout << ',' << DataFromTask[i];
    cout << endl;

    cout << " NPartThis";
    for(int i=0; i<NTasks; i++)
      cout  << ',' << NPartThisTask[i];
    cout << endl;
  }

  long npart=NPartThisTask[ThisTask],nmax=0;
  p.resize(npart);
  if (interpol_mode>0)
    id.resize(npart);
  if (interpol_mode>1)
    vel.resize(npart);

  for(int i=0; i<NTasks; i++)
    if(NPartThisTask[i] > nmax)
      nmax = NPartThisTask[i];

  arr<float> v1_tmp(nmax), v2_tmp(nmax), v3_tmp(nmax);
  arr<uint32> i1_tmp(nmax);

  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
    cout << " Reading positions ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
  {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0; f<NFilePerRead; f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;

//      if (numfiles>1)
//        filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
      if (numfiles>1)
        filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      filename+=".hdf5";

      if(!params.find<bool>("AnalyzeSimulationOnly"))
        cout << " Task: " << ThisTask << " reading file " << filename << endl;

      hid_t file_id;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);


      for(int itype=0; itype<ptypes; itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        if(params.find<bool>("AnalyzeSimulationOnly") && ptype_found >= 0)
          type = ptype_found;

        arr<float32> ftmp(3*npartthis[type]);

        // If there are 0 particles of species "type" there's also no
        // HDF5 group named "/PartType{type}" which would lead to error
        // messages from H5Gopen below.
        if(npartthis[type]==0) continue;

        string GroupName;
        GroupName.assign("/PartType");
        GroupName.append(dataToString(type));

        hid_t group_id;
        group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
        readHDF5DataArray(group_id, "Coordinates", H5T_NATIVE_FLOAT, (void*) &ftmp[0]);
        H5Gclose(group_id);

        for(int m=0; m<npartthis[type]; ++m)
        {
          if(ThisTask == ToTask)
          {
            p[ncount].x=ftmp[3*m];
            p[ncount].y=ftmp[3*m+1];
            p[ncount].z=ftmp[3*m+2];
            p[ncount].type=itype;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
            {
              ToTask++;
              ncount=0;
            }
          }
          else
          {
            v1_tmp[ncount]=ftmp[3*m];
            v2_tmp[ncount]=ftmp[3*m+1];
            v3_tmp[ncount]=ftmp[3*m+2];
            i1_tmp[ncount] = itype;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
            {
              mpiMgr.sendRaw(&v1_tmp[0], NPartThisTask[ToTask], ToTask);
              mpiMgr.sendRaw(&v2_tmp[0], NPartThisTask[ToTask], ToTask);
              mpiMgr.sendRaw(&v3_tmp[0], NPartThisTask[ToTask], ToTask);
              mpiMgr.sendRaw(&i1_tmp[0], NPartThisTask[ToTask], ToTask);
              ToTask++;
              ncount=0;
            }
          }
        }
        LastType=type;
      }
      H5Fclose(file_id);
    }
    planck_assert(ncount == 0,"Some particles were left when reading positions ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v2_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v3_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&i1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
    {
      p[m].x=v1_tmp[m];
      p[m].y=v2_tmp[m];
      p[m].z=v3_tmp[m];
      p[m].type=i1_tmp[m];
    }
  }

  if(params.find<bool>("AnalyzeSimulationOnly"))
  {
    arr<Normalizer<float32> > posnorm(3);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
    {
      posnorm[0].collect(p[m].x);
      posnorm[1].collect(p[m].y);
      posnorm[2].collect(p[m].z);
    }

    mpiMgr.allreduce(posnorm[0].minv,MPI_Manager::Min);
    mpiMgr.allreduce(posnorm[1].minv,MPI_Manager::Min);
    mpiMgr.allreduce(posnorm[2].minv,MPI_Manager::Min);
    mpiMgr.allreduce(posnorm[0].maxv,MPI_Manager::Max);
    mpiMgr.allreduce(posnorm[1].maxv,MPI_Manager::Max);
    mpiMgr.allreduce(posnorm[2].maxv,MPI_Manager::Max);
    if(mpiMgr.master())
    {
      if(ptype_found == 1)
      {
        double dx=(posnorm[0].maxv - posnorm[0].minv);
        double dy=(posnorm[1].maxv - posnorm[1].minv);
        double dz=(posnorm[2].maxv - posnorm[2].minv);
        double l;
        l = pow(dx * dy * dz / ntot,1./3.) / 5;
        cout << "size_fix0=" << l << endl;
      }
      cout << "camera_x=" << (posnorm[0].minv + posnorm[0].maxv)/2 << endl;
      cout << "camera_y=" << (posnorm[1].minv + posnorm[1].maxv)/2 + (posnorm[1].maxv - posnorm[1].minv) *1.5 << endl;
      cout << "camera_z=" << (posnorm[2].minv + posnorm[2].maxv)/2 << endl;
      cout << "lookat_x=" << (posnorm[0].minv + posnorm[0].maxv)/2 << endl;
      cout << "lookat_y=" << (posnorm[1].minv + posnorm[1].maxv)/2 << endl;
      cout << "lookat_z=" << (posnorm[2].minv + posnorm[2].maxv)/2 << endl;
      cout << "sky_x=0" << endl;
      cout << "sky_y=0" << endl;
      cout << "sky_z=1" << endl;
      cout << "fov=30" << endl;
      cout << "pictype=0" << endl;
      cout << "outfile=demo.tga" << endl;
      cout << "xres=800" << endl;
      cout << "colorbar=TRUE" << endl;
      if(ptype_found == 0)
        cout << "palette0=palettes/OldSplotch.pal" << endl;
      cout << "brightness0=2.0" << endl;
    }
    ptypes=0;
    p.resize(0);
  }


  if (interpol_mode>0)
  {
    if (interpol_mode>1)
    {
      if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
        cout << " Reading velocities ..." << endl;
      if(ThisTaskReads[ThisTask] >= 0)
      {
        int ToTask=ThisTask;
        long ncount=0;

        for(int f=0; f<NFilePerRead; f++)
        {
          unsigned int npartthis[6],nparttotal[6];
          int LastType=-1;

          filename=infilename;

//          if (numfiles>1)
//            filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
          if (numfiles>1)
            filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
          filename+=".hdf5";


          hid_t file_id;
          file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
          gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);

          for(int itype=0; itype<ptypes; itype++)
          {
            int type = params.find<int>("ptype"+dataToString(itype),0);

            arr<float32> ftmp(3*npartthis[type]);
            // infile.get(&ftmp[0],ftmp.size());

            if(npartthis[type]==0) continue;

            string GroupName;
            GroupName.assign("/PartType");
            GroupName.append(dataToString(type));
            hid_t group_id;
            group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
            readHDF5DataArray(group_id, "Velocity", H5T_NATIVE_FLOAT, (void*) &ftmp[0]);
            H5Gclose(group_id);

            for(int m=0; m<npartthis[type]; ++m)
            {
              if(ThisTask == ToTask)
              {
                vel[ncount].x=ftmp[3*m];
                vel[ncount].y=ftmp[3*m+1];
                vel[ncount].z=ftmp[3*m+2];
                ncount++;
                if(ncount == NPartThisTask[ToTask])
                {
                  ToTask++;
                  ncount=0;
                }
              }
              else
              {
                v1_tmp[ncount]=ftmp[3*m];
                v2_tmp[ncount]=ftmp[3*m+1];
                v3_tmp[ncount]=ftmp[3*m+2];
                ncount++;
                if(ncount == NPartThisTask[ToTask])
                {
                  mpiMgr.sendRaw(&v1_tmp[0], NPartThisTask[ToTask], ToTask);
                  mpiMgr.sendRaw(&v2_tmp[0], NPartThisTask[ToTask], ToTask);
                  mpiMgr.sendRaw(&v3_tmp[0], NPartThisTask[ToTask], ToTask);
                  ToTask++;
                  ncount=0;
                }
              }
            }
            LastType=type;
          }
          H5Fclose(file_id);
        }
        planck_assert(ncount == 0,"Some particles were left when reading positions ...");
      }
      else
      {
        mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
        mpiMgr.recvRaw(&v2_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
        mpiMgr.recvRaw(&v3_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
        for (int m=0; m<NPartThisTask[ThisTask]; ++m)
        {
          vel[m].x=v1_tmp[m];
          vel[m].y=v2_tmp[m];
          vel[m].z=v3_tmp[m];
        }
      }
    }


    if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
      cout << " Reading ids ..." << endl;
    if(ThisTaskReads[ThisTask] >= 0)
    {
      int ToTask=ThisTask;
      long ncount=0;

      for(int f=0; f<NFilePerRead; f++)
      {
        unsigned int npartthis[6],nparttotal[6];

        int LastType=-1;
        filename=infilename;

//        if (numfiles>1)
//          filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
        if (numfiles>1)
          filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
        filename+=".hdf5";

        hid_t file_id;
        file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);

        for(int itype=0; itype<ptypes; itype++)
        {
          int type = params.find<int>("ptype"+dataToString(itype),0);

          //arr<MyIDType> ftmp(npartthis[type]);
          // TODO: implement support for long IDs
          arr<uint32> ftmp(npartthis[type]);

          if(npartthis[type]==0) continue;

          string GroupName;
          GroupName.assign("/PartType");
          GroupName.append(dataToString(type));
          hid_t group_id;
          group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
          readHDF5DataArray(group_id, "ParticleIDs", H5T_NATIVE_UINT, (void*) &ftmp[0]);
          H5Gclose(group_id);

          if(type == 4)
          {
            cout << " WARNING: Patching IDs for star particles!" << endl;
            // TODO  - think about a better way to make star IDs unique
            //       - what if there are more than 2 Billion non-star particles?
            // max for uint32 is 4.294.967.295
            for(unsigned int m=0; m<ftmp.size(); m++)
              ftmp[m] = ftmp[m] + 2000000000;
          }

          for(int m=0; m<npartthis[type]; ++m)
          {
            if(ThisTask == ToTask)
            {
              id[ncount]=ftmp[m];
              ncount++;
              if(ncount == NPartThisTask[ToTask])
              {
                ToTask++;
                ncount=0;
              }
            }
            else
            {
              i1_tmp[ncount]=ftmp[m];
              ncount++;
              if(ncount == NPartThisTask[ToTask])
              {
                mpiMgr.sendRaw(&i1_tmp[0], NPartThisTask[ToTask], ToTask);
                ToTask++;
                ncount=0;
              }
            }
          }
          LastType=type;
        }
        H5Fclose(file_id);
      }
      planck_assert(ncount == 0,"Some particles were left when reading IDs ...");
    }
    else
    {
      mpiMgr.recvRaw(&i1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
      for (int m=0; m<NPartThisTask[ThisTask]; ++m)
        id[m]=i1_tmp[m];
    }
  }


  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
    cout << " Reading smoothing ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
  {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0; f<NFilePerRead; f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;

//      if (numfiles>1)
//        filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
      if (numfiles>1)
        filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      filename+=".hdf5";

      hid_t file_id;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);

      for(int itype=0; itype<ptypes; itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        float fix_size = params.find<float>("size_fix"+dataToString(itype),1.0);
        float size_fac = params.find<float>("size_fac"+dataToString(itype),1.0);
        string label_size = params.find<string>("size_label"+dataToString(itype),"XXXX");

        arr<float32> ftmp(npartthis[type]);

        if(npartthis[type]==0) continue;

        if (!(fix_size > 0.0))
        {
          string GroupName;
          GroupName.assign("/PartType");
          GroupName.append(dataToString(type));
          hid_t group_id;
          group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
          readHDF5DataArray(group_id, label_size.c_str(), H5T_NATIVE_FLOAT, (void*) &ftmp[0]);
          H5Gclose(group_id);
        }

        for (int m=0; m<npartthis[type]; ++m)
        {
          if(ThisTask == ToTask)
          {
            p[ncount++].r = (fix_size==0.0) ? ftmp[m]*size_fac : fix_size;
            if(ncount == NPartThisTask[ToTask])
            {
              ToTask++;
              ncount=0;
            }
          }
          else
          {
            v1_tmp[ncount++] = (fix_size==0.0) ? ftmp[m]*size_fac : fix_size;
            if(ncount == NPartThisTask[ToTask])
            {
              mpiMgr.sendRaw(&v1_tmp[0], NPartThisTask[ToTask], ToTask);
              ToTask++;
              ncount=0;
            }
          }
        }
        LastType=type;
      }
      H5Fclose(file_id);
    }
    planck_assert(ncount == 0,"Some particles were left when reading sizes ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].r=v1_tmp[m];
  }

  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
    cout << " Reading colors ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
  {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0; f<NFilePerRead; f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;

//      if (numfiles>1)
//        filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
      if (numfiles>1)
        filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      filename+=".hdf5";

      hid_t file_id;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);

      for(int itype=0; itype<ptypes; itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);

        string label_col = params.find<string>("color_label"+dataToString(itype),"XXXX");
        bool col_vector = params.find<bool>("color_is_vector"+dataToString(itype),false);
        float col_fac = params.find<float>("color_fac"+dataToString(itype),1.0);

        int read_col=1;//=gadget_find_block(infile,label_col);

        tsize fnread=0;
        if (read_col>0)
          fnread=npartthis[type];
        if ((read_col>0) && col_vector)
          fnread=3*npartthis[type];

        arr<float32> ftmp(fnread);

        if(npartthis[type]==0) continue;

        string GroupName;
        GroupName.assign("/PartType");
        GroupName.append(dataToString(type));
        hid_t group_id;
        group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
        readHDF5DataArray(group_id, label_col.c_str(), H5T_NATIVE_FLOAT, (void*) &ftmp[0]);
        H5Gclose(group_id);


        for (int m=0; m<npartthis[type]; ++m)
        {
          if(ThisTask == ToTask)
          {
            if (read_col > 0)
            {
              tsize ofs = col_vector ? 3*m : m;
              p[ncount].e.r = ftmp[ofs]*col_fac;
              if(col_vector)
              {
                p[ncount].e.g = ftmp[ofs+1]*col_fac;
                p[ncount].e.b = ftmp[ofs+2]*col_fac;
              }
            }
            else
              p[ncount].e.Set(1,1,1);

            ncount++;
            if(ncount == NPartThisTask[ToTask])
            {
              ToTask++;
              ncount=0;
            }
          }
          else
          {
            if (read_col > 0)
            {
              tsize ofs = col_vector ? 3*m : m;
              v1_tmp[ncount] = ftmp[ofs]*col_fac;
              if(col_vector)
              {
                v2_tmp[ncount] = ftmp[ofs+1]*col_fac;
                v3_tmp[ncount] = ftmp[ofs+2]*col_fac;
              }
            }
            else
              v1_tmp[ncount] = v2_tmp[ncount] = v3_tmp[ncount] = 1;

            ncount++;
            if(ncount == NPartThisTask[ToTask])
            {
              mpiMgr.sendRaw(&v1_tmp[0], NPartThisTask[ToTask], ToTask);
              mpiMgr.sendRaw(&v2_tmp[0], NPartThisTask[ToTask], ToTask);
              mpiMgr.sendRaw(&v3_tmp[0], NPartThisTask[ToTask], ToTask);
              ToTask++;
              ncount=0;
            }
          }
        }
        LastType=type;
      }
      H5Fclose(file_id);
    }
    planck_assert(ncount == 0,"Some particles were left when reading colors ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v2_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v3_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].e.Set(v1_tmp[m],v2_tmp[m],v3_tmp[m]);
  }




  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
    cout << " Reading intensity ..." << endl;

  if(ThisTaskReads[ThisTask] >= 0)
  {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0; f<NFilePerRead; f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;

//      if (numfiles>1)
//        filename+="."+dataToString(ThisTaskReads[ThisTask]+f)+".hdf5";
      if (numfiles>1)
        filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      filename+=".hdf5";

      hid_t file_id;
      file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      gadget_read_hdf5_header(file_id, npartthis, time, redshift, nparttotal, boxsize, params);

      for(int itype=0; itype<ptypes; itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);

        string label_int = params.find<string>("intensity_label"+dataToString(itype),"XXXX");
        // int read_int=gadget_find_block(infile,label_int);
        int read_int=1; //=gadget_find_block(infile,label_int);

        arr<float32> ftmp(npartthis[type]);

        if(npartthis[type]==0) continue;

        string GroupName;
        GroupName.assign("/PartType");
        GroupName.append(dataToString(type));

        hid_t group_id;
        group_id = H5Gopen(file_id, GroupName.c_str(), H5P_DEFAULT);
        readHDF5DataArray(group_id, label_int.c_str(), H5T_NATIVE_FLOAT, (void*) &ftmp[0]);
        H5Gclose(group_id);

        for (int m=0; m<npartthis[type]; ++m)
        {
          if(ThisTask == ToTask)
          {
            p[ncount++].I = (read_int>0) ? ftmp[m] : 1;
            if(ncount == NPartThisTask[ToTask])
            {
              ToTask++;
              ncount=0;
            }
          }
          else
          {
            v1_tmp[ncount++] = (read_int>0) ? ftmp[m] : 1;
            if(ncount == NPartThisTask[ToTask])
            {
              mpiMgr.sendRaw(&v1_tmp[0], NPartThisTask[ToTask], ToTask);
              ToTask++;
              ncount=0;
            }
          }
        }
        LastType=type;
      }
      H5Fclose(file_id);
    }
    planck_assert(ncount == 0,"Some Particles where left when reading Colors ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].I=v1_tmp[m];
  }
}

#else
// No HDF5 support ...

void gadget_hdf5_reader(paramfile &params, int interpol_mode,
                        vector<particle_sim> &p, vector<MyIDType> &id, vector<vec3f> &vel, int snr,
                        double &time, double &redshift, double &boxsize)
{
  cout << "Splotch was built without support for GADGET HDF5 I/O.  Exiting..." << endl;
}

#endif
