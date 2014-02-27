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
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

using namespace std;

float smooth_factor = 1.0;

namespace {

void bin_reader_prep (paramfile &params, bifstream &inp, arr<int> &qty_idx,
  int &nfields, int64 &mybegin, int64 &npart, int64 &npart_total)
  {
  /* qty_idx represents the index for the various quantities according
     to the following standard:
     qty_idx[0] = x coord
     qty_idx[1] = y coord
     qty_idx[2] = z coord
     qty_idx[3] = r coord
     qty_idx[4] = intensity
     qty_idx[5] = color 1 (R)
     qty_idx[6] = color 2 (G)
     qty_idx[7] = color 3 (B) */

  bool doswap = params.find<bool>("swap_endian",true);
  smooth_factor = params.find<float>("smooth_factor",1.0);
  string datafile = params.find<string>("infile");
  inp.open (datafile.c_str(),doswap);
  planck_assert (inp,"could not open input file '" + datafile + "'");
  nfields = params.find<int>("num_blocks");

  qty_idx.alloc(8);
  qty_idx[0] = params.find<int>("x")-1;
  qty_idx[1] = params.find<int>("y")-1;
  qty_idx[2] = params.find<int>("z")-1;
  qty_idx[3] = params.find<int>("r",-1)-1;
  qty_idx[4] = params.find<int>("I",-1)-1;
  qty_idx[5] = params.find<int>("C1")-1;
  qty_idx[6] = params.find<int>("C2",-1)-1;
  qty_idx[7] = params.find<int>("C3",-1)-1;

  if (mpiMgr.master())
    {
    cout << "Input data file name: " << datafile << endl;
    cout << "Number of columns " << nfields << endl;
    cout << "x  value in column " << qty_idx[0] << endl;
    cout << "y  value in column " << qty_idx[1] << endl;
    cout << "z  value in column " << qty_idx[2] << endl;
    cout << "r  value in column " << qty_idx[3] << endl;
    cout << "I  value in column " << qty_idx[4] << endl;
    cout << "C1 value in column " << qty_idx[5] << endl;
    cout << "C2 value in column " << qty_idx[6] << endl;
    cout << "C3 value in column " << qty_idx[7] << endl;
    }
  inp.seekg(0,ios::end);
  npart_total = inp.tellg()/(sizeof(float)*nfields);
  inp.seekg(0,ios::beg);
  int64 myend;
  mpiMgr.calcShare (0, npart_total, mybegin, myend);
  npart = myend-mybegin;

  }

void bin_reader_finish (vector<particle_sim> &points, bifstream &inp)
  {
  float minr=1e30;
  float maxr=-1e30;
  for (tsize i=0; i<points.size(); ++i)
    {
    points[i].type=0;
    minr = min(minr,points[i].x);
    maxr = max(maxr,points[i].x);
    }
  mpiMgr.allreduce(maxr,MPI_Manager::Max);
  mpiMgr.allreduce(minr,MPI_Manager::Min);
  //cout << ">>>>>>>> : " << minr << " , " << maxr << endl;
  inp.close();
  }

} // unnamed namespace

/* In this case we expect the file to be written as a binary table
   xyzrabcdexyzrabcdexyzrabcde...xyzrabcde */
void bin_reader_tab (paramfile &params, vector<particle_sim> &points)
  {
  bifstream inp;
  int nfields;
  int64 mybegin, npart, npart_total;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "TABULAR BINARY FILE" << endl;
  bin_reader_prep (params, inp, qty_idx, nfields, mybegin, npart, npart_total);

  inp.skip(mybegin*sizeof(float)*nfields);

  arr<float> buffer(nfields);
  points.resize(npart);

  bool have_c2c3 = (qty_idx[6]>=0) && (qty_idx[7]>=0);
  for(int64 i=0; i<npart; ++i)
    {
    inp.get(&buffer[0],nfields);

    points[i].x = buffer[qty_idx[0]];
    points[i].y = buffer[qty_idx[1]];
    points[i].z = buffer[qty_idx[2]];
    points[i].r = (qty_idx[3]>=0) ? smooth_factor*buffer[qty_idx[3]] : smooth_factor;
    points[i].I = (qty_idx[4]>=0) ? buffer[qty_idx[4]] : 0.5;
    points[i].e.r = (qty_idx[5]>=0) ? buffer[qty_idx[5]] : 1.0;
    //points[i].e.r = buffer[qty_idx[5]];
    points[i].e.g = have_c2c3 ? buffer[qty_idx[6]] : 0.0;
    points[i].e.b = have_c2c3 ? buffer[qty_idx[7]] : 0.0;
    }

  bin_reader_finish (points, inp);
  }

/* In this case we expect the file to be written as
   xxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyy...TTTTTTTTTTTTTTTTTTTT */
void bin_reader_block (paramfile &params, vector<particle_sim> &points)
  {
  bifstream inp;
  int nfields;
  int64 mybegin, npart, npart_total;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "BLOCK BINARY FILE" << endl;
  bin_reader_prep (params, inp, qty_idx, nfields, mybegin, npart, npart_total);

  points.resize(npart);
  arr<float> buffer(npart);

  for (tsize qty=0; qty<qty_idx.size(); ++qty)
    {
    if (qty_idx[qty]>=0)
      {
      inp.seekg(sizeof(float)*(qty_idx[qty]*npart_total+mybegin),ios::beg);
      inp.get(&buffer[0],npart);
      }

#define CASEMACRO__(num,str,noval,ss) \
      case num: \
        if (qty_idx[num]>=0) \
          for (int64 i=0; i<npart; ++i) points[i].str = ss*buffer[i]; \
        else \
          for (int64 i=0; i<npart; ++i) points[i].str = noval; \
        break;

    switch(qty)
      {
      CASEMACRO__(0,x,0,1.0)
      CASEMACRO__(1,y,0,1.0)
      CASEMACRO__(2,z,0,1.0)
      CASEMACRO__(3,r,smooth_factor,smooth_factor)
      CASEMACRO__(4,I,.5,1.0)
      CASEMACRO__(5,e.r,1.0,1.0)
      CASEMACRO__(6,e.g,0,1.0)
      CASEMACRO__(7,e.b,0,1.0)
      }
    }

#undef CASEMACRO__

  bin_reader_finish (points, inp);
  }
