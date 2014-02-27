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

#include "cxxsupport/arr.h"
#include "cxxsupport/sort_utils.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

using namespace std;

int gadget_find_block (bifstream &file,const string &label)
{
  int i;
  int32 blocksize=0, blksize;
  char blocklabel[5]={"    "};

  file.clear();
  file.rewind();

  while(!file.eof() && blocksize == 0)
  {
    file >> blksize;
    if(file.eof())
    {
      blksize=-1;
      break;
    }
    if(blksize != 8)
    {
      file.clear();
      file.rewind();
      file.flipSwap();
      file >> blksize;
      planck_assert(blksize==8,
          "wrong structure in GADGET file: " +dataToString(blksize));
    }
    file.get(blocklabel,4);
    blocklabel[4] = '\0';
    i=3;
    while (blocklabel[i]==' ')
    {
      blocklabel[i]='\0';
      i--;
    }
    file >> blocksize;
    file.skip(4);
    if (label!=blocklabel)
    {
      file.skip(blocksize);
      blocksize=0;
    }
  }
  if(file.eof()) file.clear();
  return(blocksize-8);
}

int gadget_read_header(bifstream &file, unsigned int *npart, double &time, unsigned int *nparttotal, double &boxsize, paramfile &params, float sample_factor)
{
  bool doSample = params.find<bool>("sampler",false);
  double h,O,L;
  int blocksize = gadget_find_block (file,"HEAD");
  planck_assert (blocksize>0, "Header block not found");
  file.skip(4);
  // num particles in this particular file
  file.get(npart,6);
  if(doSample)
    for(int i = 0; i < 6; i++)
      npart[i] /= sample_factor;

  file.skip(6*8);
  file >> time;
  file.skip(8+4+4);
  // num particles in whole dump
  file.get(nparttotal,6);
  if(doSample)
    for(int i = 0; i < 6; i++)
      nparttotal[i] /= sample_factor;

  file.skip(4+4);
  file >> boxsize;
  file >> O;
  params.setParam("omega",dataToString(O));
  file >> L;
  params.setParam("lambda",dataToString(L));
  file >> h;
  params.setParam("hubble",dataToString(h));
  return blocksize;
}

void gadget_reader(paramfile &params, int interpol_mode,
    vector<particle_sim> &p, vector<MyIDType> &id, vector<vec3f> &vel, int snr,
    double &time, double &boxsize)
{
  int numfiles = params.find<int>("numfiles",1);
  bool doswap = params.find<bool>("swap_endian",false);
  //string infilename = params.find<string>("infile");
  int readparallel = params.find<int>("readparallel",1);
  int ptypes = params.find<int>("ptypes",1);
  int ptype_found = -1, ntot = 1;

  // Sampling, this cannot be done while interpolating...
  // Sample factor is read in as percentage then converted to factor
  bool doSample = params.find<bool>("sampler",false);
  float sample_factor = 1;
  if(doSample && !interpol_mode)
  { 
    // If no factor, use 100% sampling ie no sample.
    sample_factor = params.find<float>("sample_factor",100);
    if((sample_factor > 0) && (sample_factor <= 100))
    {
      sample_factor = 100/sample_factor;
      if(sample_factor < 2)
      {
        cout << "No sampling occuring." << endl;
        cout << "Data is sampled in fractions, valid sample factors are 1/2, 1/3, 1/4, 1/5 etc" << endl;
        cout << "Expressed as a percentage in the parameter file, ie. 50, 33, 25, 20 etc" << endl;
      }
    }
    else
    {
      cout << "Sample factor given is not a valid percentage, no sampling occuring" << endl;
      cout << "Data is sampled in fractions, valid sample factors are 1/2, 1/3, 1/4, 1/5 etc" << endl;
      cout << "Expressed as a percentage in the parameter file, ie. 50, 33, 25, 20 etc" << endl;
      sample_factor = 1; 
    }
  }
  else if(doSample)
    cout << " Cannot sample and interpolate. No sampling occuring" << std::endl;

  string infilename = params.find<string>("infile");
  string snapdir    = params.find<string>("snapdir",string(""));
  //  cout << snapdir << endl;
  string datadir    = params.find<string>("datadir",string(""));
  string filename;

  //string filename;
  bifstream infile;

  int ThisTask=mpiMgr.rank(),NTasks=mpiMgr.num_ranks();
  arr<int> ThisTaskReads(NTasks), DataFromTask(NTasks);
  arr<long> NPartThisTask(NTasks);

  //  infilename += intToString(snr,3);
  //  if (params.find<bool>("snapdir",false))
  //    infilename = "snapdir_"+intToString(snr,3)+"/"+infilename;


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

  for(int i=0;i<NTasks;i++)
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
    int itask=0;
    for(int rt=0;rt<readparallel;rt++)
    {
      long NPartThisReadTask = 0;
      for(int f=0;f<NFilePerRead;f++)
      {
        int file=rt*NFilePerRead+f;
        unsigned int npartthis[6],nparttotal[6];
        filename=infilename;
        if(numfiles>1) filename+="."+dataToString(file);
        infile.open(filename.c_str(),doswap);
        planck_assert (infile,"could not open input file! <" + filename + ">");
        // Read header gets numparticles of each type for both this particular file (npartthis) and 
        // the whole dump (nparttotal)
        gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);
        infile.close();
        //	if((rt==0 && f==0) || !params.find<bool>("AnalyzeSimulationOnly"))
        //	  cout << "    Timestamp from file : t=" << time << endl;
        // Print useful info on first file read only
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
              cout << "To vizualize the gas particles add/change the following lines to the parameter file:" << endl;
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
        // Find out how many particles this read task will read
        // Dependant on requested types (from param file) and availability of data in file
        for(int itype=0;itype<ptypes;itype++)
        {
          int type = params.find<int>("ptype"+dataToString(itype),0);
          if(params.find<bool>("AnalyzeSimulationOnly") && ptype_found >= 0)
            type = ptype_found;
          NPartThisReadTask += npartthis[type];
        }
      }
      // Work out how many particles each individual task will read for this read task
      // Check difference in sum of particles actually read and number of particles intended
      // to be read for this read task
      long SumPartThisReadTask = 0;
      for(int t=0;t<NTaskPerRead-1;t++)
      {
        NPartThisTask[itask] = NPartThisReadTask / NTaskPerRead;
        SumPartThisReadTask += NPartThisReadTask / NTaskPerRead;
        itask++;
      }
      NPartThisTask[itask] = NPartThisReadTask - SumPartThisReadTask;
      itask++;
    }
  }

  // Broadcast num particles and rank of broadcast root to all tasks
  mpiMgr.bcast(NPartThisTask,0);
  mpiMgr.bcast(boxsize,0);
  mpiMgr.bcast(time,0);

  // Master outputs some useful data
  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
  {
    cout << " Reading " << numfiles << " files by " << readparallel << " tasks ... " << endl;
    cout << " Task " << ThisTask << "/" << NTasks << endl;
    cout << " NTaskPerRead/NFilePerRead " << NTaskPerRead << "," << NFilePerRead << endl;
    cout << " ThisTaskReads";
    for(int i=0;i<NTasks;i++)
      cout << ',' << ThisTaskReads[i];
    cout << endl;
    cout << " DataFromTask";
    for(int i=0;i<NTasks;i++)
      cout << ',' << DataFromTask[i];
    cout << endl;

    cout << " NPartThis";
    for(int i=0;i<NTasks;i++)
      cout  << ',' << NPartThisTask[i];
    cout << endl;
  }

  long npart=NPartThisTask[ThisTask],nmax=0;
  p.resize(npart);
  if (interpol_mode>0)
    id.resize(npart);
  if (interpol_mode>1)
    vel.resize(npart);

  for(int i=0;i<NTasks;i++)
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

    for(int f=0;f<NFilePerRead;f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int present=1+2+4+8+16+32;
      int LastType=-1;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      //    if(!params.find<bool>("AnalyzeSimulationOnly"))
      //	cout << " Task: " << ThisTask << " reading file " << filename << endl;
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);
      gadget_find_block(infile,"POS");
      infile.skip(4);
      for(int itype=0;itype<ptypes;itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        if(params.find<bool>("AnalyzeSimulationOnly") && ptype_found >= 0)
          type = ptype_found;
        for(int s=LastType+1; s<type; s++)
          if(npartthis[s]>0 && (1<<s & present))
            infile.skip(4*3*npartthis[s]*sample_factor);
        arr<float32> ftmp(3*npartthis[type]*sample_factor);
        infile.get(&ftmp[0],ftmp.size());
        for(unsigned int m=0; m<npartthis[type]*sample_factor; m+=sample_factor)
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
      infile.close();
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

  // If we are interpolating we need to read IDs
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

        for(int f=0;f<NFilePerRead;f++)
        {
          unsigned int npartthis[6],nparttotal[6];
          int present=1+2+4+8+16+32;
          int LastType=-1;
          filename=infilename;
          if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
          infile.open(filename.c_str(),doswap);
          planck_assert (infile,"could not open input file! <" + filename + ">");
          gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);
          gadget_find_block(infile,"VEL");
          infile.skip(4);
          for(int itype=0;itype<ptypes;itype++)
          {
            int type = params.find<int>("ptype"+dataToString(itype),0);
            for(int s=LastType+1; s<type; s++)
              if(npartthis[s]>0 && (1<<s & present))
                infile.skip(4*3*npartthis[s]);
            arr<float32> ftmp(3*npartthis[type]);
            infile.get(&ftmp[0],ftmp.size());
            for(unsigned int m=0; m<npartthis[type]; ++m)
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
          infile.close();
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

      for(int f=0;f<NFilePerRead;f++)
      {
        unsigned int npartthis[6],nparttotal[6];
        int present=1+2+4+8+16+32;
        int LastType=-1;
        filename=infilename;
        if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
        infile.open(filename.c_str(),doswap);
        planck_assert (infile,"could not open input file! <" + filename + ">");
        gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);
        string label_id = params.find<string>("id_label","ID");
        gadget_find_block(infile,label_id);
        infile.skip(4);
        for(int itype=0;itype<ptypes;itype++)
        {
          int type = params.find<int>("ptype"+dataToString(itype),0);
          for(int s=LastType+1; s<type; s++)
            if(npartthis[s]>0 && (1<<s & present))
              infile.skip(sizeof(MyIDType)*npartthis[s]);
          arr<MyIDType> ftmp(npartthis[type]);
          infile.get(&ftmp[0],ftmp.size());

          //cout << "ParticleID CHECK : " << ftmp[0] << endl << flush;

          tstack_replace("Reading","Patching particle IDs");
          if(type == 0)
          {
	    if(mpiMgr.master())
	      cout << " WARNING: Patching IDs for gas particles!" << endl;
 
            MyIDType bits=1;
            if (sizeof(MyIDType)==4)
            { // IDs are 32 bit unsigned integers
              bits=(bits<<30)-1;
              for(unsigned int m=0; m<ftmp.size(); m++)
                ftmp[m] = ftmp[m] & bits; // remove upper 2 bits of 32bit value 2^30-1
            }
            else
            { // IDs are 64 bit unsigned integers
	      for(unsigned int m=0; m<ftmp.size(); m++)
		ftmp[m] = ftmp[m] & ((((long long) 1) << 60) - 1);
            }
          }
          else if(type == 4)
          {
	    if(mpiMgr.master())
	      cout << " WARNING: Patching IDs for star particles!" << endl;
            MyIDType bits=1;
            if (sizeof(MyIDType)==4)
            { // IDs are 32 bit unsigned integers
              bits=bits<<29;
              for(unsigned int m=0; m<ftmp.size(); m++)
                ftmp[m] = ftmp[m] + bits;   // adding 2^29
            }
            else
            { // IDs are 64 bit unsigned integers
              bits=bits<<60;
              for(unsigned int m=0; m<ftmp.size(); m++)
                ftmp[m] = ftmp[m] + (((long long) 1) << 60);   // adding 2^60
            }
          }

	  tstack_replace("Patching particle IDs","Reading");

          for(unsigned int m=0; m<npartthis[type]; ++m)
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
        infile.close();
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

    for(int f=0;f<NFilePerRead;f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);

      for(int itype=0;itype<ptypes;itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        float fix_size = params.find<float>("size_fix"+dataToString(itype),1.0);
        float size_fac = params.find<float>("size_fac"+dataToString(itype),1.0);
        string label_size = params.find<string>("size_label"+dataToString(itype),"XXXX");
        if (fix_size == 0.0)
        {
          gadget_find_block(infile,label_size);
          infile.skip(4);
          int present = params.find<int>("size_present"+dataToString(itype),type);
          for(int s=LastType+1; s<type; s++)
            if(npartthis[s]>0 && (1<<s & present))
              infile.skip(4*npartthis[s]*sample_factor);
        }
        arr<float32> ftmp(npartthis[type]*sample_factor);
        if (fix_size == 0.0) infile.get(&ftmp[0],ftmp.size());
        for (unsigned int m=0; m<npartthis[type]*sample_factor; m+=sample_factor)
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
      infile.close();
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

    for(int f=0;f<NFilePerRead;f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);

      for(int itype=0;itype<ptypes;itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        string label_col = params.find<string>("color_label"+dataToString(itype),"XXXX");
        bool col_vector = params.find<bool>("color_is_vector"+dataToString(itype),false);
        float col_fac = params.find<float>("color_fac"+dataToString(itype),1.0);
        int read_col=gadget_find_block(infile,label_col);
        if (read_col > 0)
        {
          infile.skip(4);
          int present = params.find<int>("color_present"+dataToString(itype),type);
          for(int s=LastType+1; s<type; s++)
            if(npartthis[s]>0 && (1<<s & present))
            {
              int nskip=npartthis[s]*sample_factor;
              if(col_vector)
                nskip *=3;
              infile.skip(4*nskip);
            }
        }
        else
          if(mpiMgr.master())
            cout << " Cannot find color field <" << label_col << "> ..." << endl;
        tsize fnread=0;
        if (read_col>0) fnread=npartthis[type]*sample_factor;
        if ((read_col>0) && col_vector) fnread=3*npartthis[type]*sample_factor;
        arr<float32> ftmp(fnread);
        infile.get(&ftmp[0],ftmp.size());
        for (unsigned int m=0; m<npartthis[type]*sample_factor; m+=sample_factor)
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
      infile.close();
    }
    planck_assert(ncount == 0,"Some particles were left when reading colors ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v2_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    mpiMgr.recvRaw(&v3_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (unsigned int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].e.Set(v1_tmp[m],v2_tmp[m],v3_tmp[m]);
  }


  if(mpiMgr.master() && !params.find<bool>("AnalyzeSimulationOnly"))
    cout << " Reading intensity ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
  {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0;f<NFilePerRead;f++)
    {
      unsigned int npartthis[6],nparttotal[6];
      int LastType=-1;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_read_header(infile,npartthis,time,nparttotal,boxsize,params,sample_factor);
      for(int itype=0;itype<ptypes;itype++)
      {
        int type = params.find<int>("ptype"+dataToString(itype),0);
        string label_int = params.find<string>("intensity_label"+dataToString(itype),"XXXX");
        int read_int=gadget_find_block(infile,label_int);
        if (read_int > 0)
        {
          infile.skip(4);
          int present = params.find<int>("intensity_present"+dataToString(itype),type);
          for(int s=LastType+1; s<type; s++)
            if(npartthis[s]>0 && (1<<s & present))
              infile.skip(4*npartthis[s]*sample_factor);
        }
        else
          if(mpiMgr.master() && itype==0 && f==0)
            cout << " Cannot find intensity field <" << label_int << "> ..." << endl;
        arr<float32> ftmp(npartthis[type]*sample_factor);
        if (read_int>0) infile.get(&ftmp[0],ftmp.size());
        for (unsigned int m=0; m<npartthis[type]*sample_factor; m+=sample_factor)
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
      infile.close();
    }
    planck_assert(ncount == 0,"Some particles where left when reading Colors ...");
  }
  else
  {
    mpiMgr.recvRaw(&v1_tmp[0], NPartThisTask[ThisTask], DataFromTask[ThisTask]);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].I=v1_tmp[m];
  }
}
