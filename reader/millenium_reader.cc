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


#ifdef USE_MPI
#include "mpi.h"
#endif
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>

#include "cxxsupport/arr.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

using namespace std;

#define TAG_POSX        11
#define TAG_POSY        12
#define TAG_POSZ        13
#define TAG_SIZE        14
#define TAG_INT         15
#define TAG_COL1        16
#define TAG_COL2        17
#define TAG_COL3        18
#define TAG_TYPE        19
#define TAG_ID          20


void gadget_plain_read_header(bifstream &file, int *npart, double *time)
  {
  int length1,length2;

  file.clear();
  file.rewind();

  file >> length1;
  file.get(npart,6);
  file.skip(6*8);
  file >> *time;
  file.skip(8+8+6*4);

  file.skip(256 - 6*4 - 6*8 - 3*8 - 6*4);
  file >> length2;
  if(length1!=length2)
    planck_fail("Header is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=256 || length2!=256)
    planck_fail("Header length is not 256 ! ("+dataToString(length1)+","+dataToString(length2)+")");
  }

void gadget_find_pos(bifstream &file, int* npart)
  {
  double time;

  // Skipping Header
  gadget_plain_read_header(file, npart, &time);
  }

void gadget_find_vel(bifstream &file, int* npart)
  {
  int length1,length2,ntot=0;

  // Jump to Positions
  gadget_find_pos(file, npart);
  // Skipping Positions
  file >> length1;
  for(int i=0;i<6;i++)
    if(npart[i] > 0)
      ntot+=npart[i];
  file.skip(3*4*ntot);
  file >> length2;
  if(length1!=length2)
    planck_assert(false,"Position skip is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=3*4*ntot || length2!=3*4*ntot)
    planck_assert(false,"Position skip length is not "+dataToString(3*4*ntot)+" ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(file.eof()) file.clear();
  }

void gadget_find_id(bifstream &file, int* npart)
  {
  int length1,length2,ntot=0;

  // Jump to Velocities
  gadget_find_vel(file, npart);
  // Skipping Velocities
  file >> length1;
  for(int i=0;i<6;i++)
    if(npart[i] > 0)
      ntot+=npart[i];
  file.skip(3*4*ntot);
  file >> length2;
  if(length1!=length2)
    planck_assert(false,"Velocity skip is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=3*4*ntot || length2!=3*4*ntot)
    planck_assert(false,"Velocity skip length is not "+dataToString(3*4*ntot)+" ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(file.eof()) file.clear();
  }

void gadget_find_mass(bifstream &file, int* npart)
  {
  int length1,length2,ntot=0,lid=4;

  // Jump to IDs
  gadget_find_id(file, npart);
  // Skipping IDs
  file >> length1;
  for(int i=0;i<6;i++)
    if(npart[i] > 0)
      ntot+=npart[i];
  if(length1 > lid*ntot)
    lid=8;
  file.skip(lid*ntot);
  file >> length2;
  if(length1!=length2)
    planck_assert(false,"ID skip is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=lid*ntot || length2!=lid*ntot)
    planck_assert(false,"ID skip length is not "+dataToString(lid*ntot)+" ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(file.eof()) file.clear();
  }

void gadget_find_hsml(bifstream &file, int* npart)
  {
  // Jump to Mass
  gadget_find_mass(file, npart);
  // No mass entry to skip!
  }

void gadget_find_density(bifstream &file, int* npart)
  {
  int length1,length2,ntot=0;

  // Jump to HSML
  gadget_find_hsml(file, npart);
  // Skipping HSML
  file >> length1;
  for(int i=0;i<6;i++)
    if(npart[i] > 0)
      ntot+=npart[i];
  file.skip(4*ntot);
  file >> length2;
  if(length1!=length2)
    planck_assert(false,"HSML skip is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=4*ntot || length2!=4*ntot)
    planck_assert(false,"HSML skip length is not "+dataToString(4*ntot)+" ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(file.eof()) file.clear();
  }

void gadget_find_veldisp(bifstream &file, int* npart)
  {
  int length1,length2,ntot=0;

  // Jump to Density
  gadget_find_density(file, npart);
  // Skipping Density
  file >> length1;
  for(int i=0;i<6;i++)
    if(npart[i] > 0)
      ntot+=npart[i];
  file.skip(4*ntot);
  file >> length2;
  if(length1!=length2)
    planck_assert(false,"Density skip is not matched ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(length1!=4*ntot || length2!=4*ntot)
    planck_assert(false,"Density skip length is not "+dataToString(4*ntot)+" ! ("+dataToString(length1)+","+dataToString(length2)+")");
  if(file.eof()) file.clear();
  }


void gadget_millenium_reader(paramfile &params, vector<particle_sim> &p, int /*snr*/, double *time)
  {
  int numfiles = params.find<int>("numfiles",1);
  bool doswap = params.find<bool>("swap_endian",true);
  string infilename = params.find<string>("infile");
  int readparallel = params.find<int>("readparallel",1);
  int ptypes = params.find<int>("ptypes",1);

  string filename;
  bifstream infile;

  int ThisTask=mpiMgr.rank(),NTasks=mpiMgr.num_ranks();

  vector<int> ThisTaskReads(NTasks),DataFromTask(NTasks);
  vector<long> NPartThisTask(NTasks);

#ifdef USE_MPI
  MPI_Status status;
#endif

  if(mpiMgr.master())
    {
    planck_assert(numfiles >= readparallel,
      "Number of files must be larger or equal number of parallel reads ...");
    planck_assert(numfiles%readparallel == 0,
      "Number of files must be a multiple of number of parallel reads ...");
    planck_assert(NTasks >= readparallel,
      "Number of tasks must be larger or equal number of parallel reads ...");
    planck_assert(NTasks%readparallel == 0,
      "Number of tasks must be a multiple of number of parallel reads ...");
    }

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
        int npartthis[6];
        filename=infilename;
        if(numfiles>1) filename+="."+dataToString(file);
        infile.open(filename.c_str(),doswap);
        planck_assert (infile,"could not open input file! <" + filename + ">");
        gadget_plain_read_header(infile,npartthis,time);
        infile.close();
        for(int itype=0;itype<ptypes;itype++)
          {
          int type = params.find<int>("ptype"+dataToString(itype),0);
          NPartThisReadTask += npartthis[type];
          }
        }
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
    cout << " Timestamp from file : " << *time << endl;
    }

#ifdef USE_MPI
  MPI_Bcast(&NPartThisTask[0], NTasks, MPI_LONG, 0, MPI_COMM_WORLD);
#endif

  if(mpiMgr.master())
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

  for(int i=0;i<NTasks;i++)
    if(NPartThisTask[i] > nmax)
      nmax = NPartThisTask[i];

  vector <float> fdummy;
  vector <int> idummy;

  arr<float> v1_tmp(nmax),v2_tmp(nmax),v3_tmp(nmax);
  arr<int> i1_tmp(nmax);

  if(mpiMgr.master())
    cout << " Reading positions ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
    {
    int ToTask=ThisTask;
    //int NPartThis=NPartThisTask[ThisTask];
    long ncount=0;

    for(int f=0;f<NFilePerRead;f++)
      {
      int npartthis[6];
      int present=1+2+4+8+16+32;
      int LastType=0;
      if(numfiles>1) filename=infilename+"."+dataToString(ThisTaskReads[ThisTask]+f);
      else           filename=infilename;
      cout << " Task: " << ThisTask << " reading file " << filename << endl;
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_find_pos(infile,npartthis);
      infile.skip(4);
      for(int itype=0;itype<ptypes;itype++)
        {
        int type = params.find<int>("ptype"+dataToString(itype),1);
        for(int s=LastType+1; s<type; s++)
          if(npartthis[s]>0 && (1<<s & present))
            infile.skip(4*3*npartthis[s]);

        fdummy.resize(npartthis[type]*3);
        idummy.resize(npartthis[type]);
        infile.get(&fdummy[0],npartthis[type]*3);
        for(int nread=0,m=0; m<npartthis[type]; ++m)
          {
          if(ThisTask == ToTask)
            {
            p[ncount].x=fdummy[nread++];
            p[ncount].y=fdummy[nread++];
            p[ncount].z=fdummy[nread++];
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
#ifdef USE_MPI
            v1_tmp[ncount] = fdummy[nread++];
            v2_tmp[ncount] = fdummy[nread++];
            v3_tmp[ncount] = fdummy[nread++];
            i1_tmp[ncount] = itype;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              MPI_Ssend(&v1_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_POSX, MPI_COMM_WORLD);
              MPI_Ssend(&v2_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_POSY, MPI_COMM_WORLD);
              MPI_Ssend(&v3_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_POSZ, MPI_COMM_WORLD);
              MPI_Ssend(&i1_tmp[0], NPartThisTask[ToTask], MPI_INT, ToTask, TAG_TYPE, MPI_COMM_WORLD);
              ToTask++;
              ncount=0;
              }
#else
            planck_fail("Should not be executed without MPI support !!!");
#endif
            }
          /*
          planck_assert(nread < npartthis[type]*3,"Running out of read buffer ("+dataToString(nread)+
                        "/"+dataToString(npartthis[type]*3)+") ...");
          */
          }
        LastType=type;
        }
      infile.close();
      }
    planck_assert(ncount==0,"Some Particles were left when reading positions ("+dataToString(ncount)+")...");
    }
  else
    {
#ifdef USE_MPI
    MPI_Recv(&v1_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_POSX, MPI_COMM_WORLD, &status);
    MPI_Recv(&v2_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_POSY, MPI_COMM_WORLD, &status);
    MPI_Recv(&v3_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_POSZ, MPI_COMM_WORLD, &status);
    MPI_Recv(&i1_tmp[0], NPartThisTask[ThisTask], MPI_INT, DataFromTask[ThisTask], TAG_TYPE, MPI_COMM_WORLD, &status);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      {
      p[m].x=v1_tmp[m];
      p[m].y=v2_tmp[m];
      p[m].z=v3_tmp[m];
      p[m].type=i1_tmp[m];
      }
#else
    planck_fail("Should not be executed without MPI support !!!");
#endif
    }

  cout << "   -> " << p[0].x << "," << p[0].y << "," << p[0].z << endl;
  cout << "   -> " << p[npart-1].x << "," << p[npart-1].y << "," << p[npart-1].z << endl;

  if(mpiMgr.master())
    cout << " Reading smoothing ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
    {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0;f<NFilePerRead;f++)
      {
      int npartthis[6];
      int LastType=0;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      gadget_plain_read_header(infile,npartthis,time);

      for(int itype=0;itype<ptypes;itype++)
        {
        int type = params.find<int>("ptype"+dataToString(itype),1);
        float fix_size = params.find<float>("size_fix"+dataToString(itype),1.0);
        float size_fac = params.find<float>("size_fac"+dataToString(itype),1.0);
        if (fix_size == 0.0)
          {
          gadget_find_hsml(infile,npartthis);
          infile.skip(4);
          int present = params.find<int>("size_present"+dataToString(itype),type);
          for(int s=LastType+1; s<type; s++)
            if(npartthis[s]>0 && (1<<s & present))
              infile.skip(4*npartthis[s]);
          }
        if (fix_size == 0.0)
          {
          fdummy.resize(npartthis[type]);
          infile.get(&fdummy[0],npartthis[type]);
          }
        for (int nread=0,m=0; m<npartthis[type]; ++m)
          {
          if(ThisTask == ToTask)
            {
            if (fix_size == 0.0)
              p[ncount].r = fdummy[nread++] * size_fac;
            else
              p[ncount].r = fix_size;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              ToTask++;
              ncount=0;
              }
            }
          else
            {
#ifdef USE_MPI
            if (fix_size == 0.0)
              v1_tmp[ncount] = fdummy[nread++] * size_fac;
            else
              v1_tmp[ncount] = fix_size;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              MPI_Ssend(&v1_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_SIZE, MPI_COMM_WORLD);
              ToTask++;
              ncount=0;
              }
#else
            planck_fail("Should not be executed without MPI support !!!");
#endif
            }
          }
        LastType=type;
        infile.close();
        }
      }
    planck_assert(ncount==0,"Some Particles were left when reading sizes ...");
    }
  else
    {
#ifdef USE_MPI
    MPI_Recv(&v1_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_SIZE, MPI_COMM_WORLD, &status);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].r=v1_tmp[m];
#else
    planck_fail("Should not be executed without MPI support !!!");
#endif
    }

  cout << "   -> " << p[0].r << endl;
  cout << "   -> " << p[npart-1].r << endl;


  if(mpiMgr.master())
    cout << " Reading colors ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
    {
    int ToTask=ThisTask;
    long ncount=0;

    for(int f=0;f<NFilePerRead;f++)
      {
      int npartthis[6];
      int LastType=0;
      filename=infilename;
      if(numfiles>1) filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      for(int itype=0;itype<ptypes;itype++)
        {
        int type = params.find<int>("ptype"+dataToString(itype),1);
        int col = params.find<int>("color_field"+dataToString(itype),0);
        bool col_vector = params.find<bool>("color_is_vector"+dataToString(itype),false);
        float col_fac = params.find<float>("color_fac"+dataToString(itype),1.0);
        int read_col = 1;

        switch(col)
          {
          case 0:
            gadget_find_vel(infile,npartthis);
            planck_assert(col_vector,"Color type "+dataToString(col)+" has to be declared as a vector !!!");
            break;
          case 1:
            gadget_find_density(infile,npartthis);
            planck_assert(!col_vector,"Color type "+dataToString(col)+" has not to be declared as a vector !!!");
            break;
          case 2:
            gadget_find_veldisp(infile,npartthis);
            planck_assert(!col_vector,"Color type "+dataToString(col)+" has not to be declared as a vector !!!");
            break;
          default:
            planck_assert(false,"Color type "+dataToString(col)+" not known !!!");
            break;
          }
        infile.skip(4);
        int present = params.find<int>("color_present"+dataToString(itype),0);
        for(int s=LastType+1; s<type; s++)
          if(npartthis[s]>0 && (1<<s & present))
            infile.skip(4*npartthis[s] * (col_vector ? 3:1));
        if (read_col > 0)
          {
          if(col_vector)
            {
            fdummy.resize(npartthis[type]*3);
            infile.get(&fdummy[0],npartthis[type]*3);
            }
          else
            {
            fdummy.resize(npartthis[type]);
            infile.get(&fdummy[0],npartthis[type]);
            }
          }
        for (int nread=0,m=0; m<npartthis[type]; ++m)
          {
          if(ThisTask == ToTask)
            {
            if (read_col > 0)
              {
              p[ncount].e.r = fdummy[nread++] * col_fac;
              if(col_vector)
                {
                p[ncount].e.g = fdummy[nread++] * col_fac;
                p[ncount].e.b = fdummy[nread++] * col_fac;
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
#ifdef USE_MPI
            if (read_col > 0)
              {
              v1_tmp[ncount] = fdummy[nread++] * col_fac;
              if(col_vector)
                {
                v2_tmp[ncount] = fdummy[nread++] * col_fac;
                v3_tmp[ncount] = fdummy[nread++] * col_fac;
                }
              }
            else
              v1_tmp[ncount] = v2_tmp[ncount] = v3_tmp[ncount] = 1;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              MPI_Ssend(&v1_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_COL1, MPI_COMM_WORLD);
              MPI_Ssend(&v2_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_COL2, MPI_COMM_WORLD);
              MPI_Ssend(&v3_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_COL3, MPI_COMM_WORLD);
              ToTask++;
              ncount=0;
              }
#else
            planck_assert(false,"Should not be executed without MPI support !!!");
#endif
            }
          }
        LastType=type;
        }
      infile.close();
      }
    planck_assert(ncount == 0,"Some Particles were left when reading colors ...");
    }
  else
    {
#ifdef USE_MPI
    MPI_Recv(&v1_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_COL1, MPI_COMM_WORLD, &status);
    MPI_Recv(&v2_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_COL2, MPI_COMM_WORLD, &status);
    MPI_Recv(&v3_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_COL3, MPI_COMM_WORLD, &status);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].e.Set(v1_tmp[m],v2_tmp[m],v3_tmp[m]);
#else
    planck_fail("Should not be executed without MPI support !!!");
#endif
    }

  cout << "   -> " << p[0].e.r << "," << p[0].e.g << "," << p[0].e.b << endl;
  cout << "   -> " << p[npart-1].e.r << "," << p[npart-1].e.g << "," << p[npart-1].e.b << endl;

 if(mpiMgr.master())
    cout << " Reading intensity ..." << endl;
  if(ThisTaskReads[ThisTask] >= 0)
    {
    int ToTask=ThisTask;
    //int NPartThis=NPartThisTask[ThisTask];
    long ncount=0;

    for(int f=0;f<NFilePerRead;f++)
      {
      int npartthis[6];
      int LastType=0;
      filename=infilename;
      filename+="."+dataToString(ThisTaskReads[ThisTask]+f);
      infile.open(filename.c_str(),doswap);
      planck_assert (infile,"could not open input file! <" + filename + ">");
      for(int itype=0;itype<ptypes;itype++)
        {
        int type = params.find<int>("ptype"+dataToString(itype),1);
        int intensity = params.find<int>("intensity_field"+dataToString(itype),0);
        float intensity_fac = params.find<float>("intensity_fac"+dataToString(itype),1.0);
        int read_intensity = 1;

        switch(intensity)
          {
          case 0:
            read_intensity = 0;
            break;
          case 1:
            gadget_find_density(infile,npartthis);
            break;
          case 2:
            gadget_find_veldisp(infile,npartthis);
            break;
          default:
            planck_assert(false,"Intensity type "+dataToString(intensity)+" not known !!!");
            break;
          }
        infile.skip(4);
        int present = params.find<int>("intensity_present"+dataToString(itype),0);
        for(int s=LastType+1; s<type; s++)
          if(npartthis[s]>0 && (1<<s & present))
            infile.skip(4*npartthis[s]);
        if (read_intensity > 0)
          {
          fdummy.resize(npartthis[type]);
          infile.get(&fdummy[0],npartthis[type]);
          }
        for (int nread=0,m=0; m<npartthis[type]; ++m)
          {
          if(ThisTask == ToTask)
            {
            if (read_intensity > 0)
              p[ncount].I = fdummy[nread++] * intensity_fac;
            else
              p[ncount].I = 1;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              ToTask++;
              ncount=0;
              }
            }
          else
            {
#ifdef USE_MPI
            if (read_intensity > 0)
              v1_tmp[ncount] = fdummy[nread++] * intensity_fac;
            else
              v1_tmp[ncount] = 1;
            ncount++;
            if(ncount == NPartThisTask[ToTask])
              {
              MPI_Ssend(&v1_tmp[0], NPartThisTask[ToTask], MPI_FLOAT, ToTask, TAG_INT, MPI_COMM_WORLD);
              ToTask++;
              ncount=0;
              }
#else
            planck_fail("Should not be executed without MPI support !!!");
#endif
            }
          }
        LastType=type;
        }
      infile.close();
      }
    planck_assert(ncount == 0,"Some Particles were left when reading intensities ...");
    }
  else
    {
#ifdef USE_MPI
    MPI_Recv(&v1_tmp[0], NPartThisTask[ThisTask], MPI_FLOAT, DataFromTask[ThisTask], TAG_INT, MPI_COMM_WORLD, &status);
    for (int m=0; m<NPartThisTask[ThisTask]; ++m)
      p[m].I=v1_tmp[m];
#else
    planck_assert(false,"Should not be executed without MPI support !!!");
#endif
    }

  cout << "   -> " << p[0].I << endl;
  cout << "   -> " << p[npart-1].I << endl;
  }
