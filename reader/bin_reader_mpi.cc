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


#if defined(USE_MPI) && defined(USE_MPIIO)
#ifdef USE_MPI
#include "mpi.h"
#endif
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>

#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "kernel/colour.h"

using namespace std;

#include "splotch/splotchutils.h"
#include "partition.h"

#define SWAP_4(x) ( ((x) << 24) | \
		(((x) << 8) & 0x00ff0000) | \
		(((x) >> 8) & 0x0000ff00) | \
		((x) >> 24) )
#define SWAP_FLOAT(x) (*(unsigned int *)&(x)=SWAP_4(*(unsigned int *)&(x)))

long bin_reader_block_mpi(paramfile &params, vector<particle_sim> &points, 
		          float *maxr, float *minr, int mype, int npes)
{
/*
In this case we expect the file to be written as 
xxxxxxxxxx 
xxxxxxxxxx 
yyyyyyyyyy
yyyyyyyyyy
...
TTTTTTTTTT
TTTTTTTTTT

which_fields represents the block position inside the file according
to the following standard:
which_fields[0] = x coord
which_fields[1] = y coord
which_fields[2] = z coord
which_fields[3] = r coord
which_fields[4] = intensity
which_fields[5] = color 1 (R)
which_fields[6] = color 2 (G)
which_fields[7] = color 3 (B)
*/

   FILE * pFile;
   FILE * auxFile;
   float * dataarray;
   float * destarray;
   
   long total_size=0;
   long pe_size;
   long field_size;
   float minradius=1e30;
   float maxradius=-1e30;
// offset could be a input parameter
   MPI_Offset offset = 0;
   long stride;

   int n_load_fields = 8;
   int * which_fields = new int [n_load_fields];
   long totalsize;
   long totalsize_f;
   long last_pe_adding;

   bool doswap = params.find<bool>("swap_endian",true);   
   string datafile = params.find<string>("infile");
   int num_of_fields = params.find<int>("num_blocks",1);   

   which_fields[0] = params.find<int>("x",-1);
   which_fields[1] = params.find<int>("y",-1);
   which_fields[2] = params.find<int>("z",-1);
   which_fields[3] = params.find<int>("r",-1);
   which_fields[4] = params.find<int>("I",-1);
   which_fields[5] = params.find<int>("C1",-1);
   which_fields[6] = params.find<int>("C2",-1);
   which_fields[7] = params.find<int>("C3",-1);

   if(mype == 0)
   {
     cout << "BLOCK BINARY FILE\n";
     
     cout << "Input data file name: " << datafile << endl;
     cout << "Number of blocks " << num_of_fields << endl;
     cout << "x block (1 - " << num_of_fields << "), " << which_fields[0] << endl;
     cout << "y block (2 - " << num_of_fields << "), " << which_fields[1] << endl;
     cout << "z block (3 - " << num_of_fields << "), " << which_fields[2] << endl;
     cout << "r block (4 - " << num_of_fields << "), " << which_fields[3] << endl;
     cout << "I block (5 - " << num_of_fields << "), " << which_fields[4] << endl;
     cout << "C1 block (6 - " << num_of_fields << "), " << which_fields[5] << endl;
     cout << "C2 block (7 - " << num_of_fields << "), " << which_fields[6] << endl;
     cout << "C3 block (8 - " << num_of_fields << "), " << which_fields[7] << endl;

     pFile = fopen(datafile.c_str(), "rb");
     fseek (pFile, 0, SEEK_END);
     totalsize = ftell (pFile);
     fclose(pFile);
     // number of elements (of each variable) for a processor
     field_size = totalsize/(sizeof(float)*num_of_fields);
   } 
   
   MPI_Bcast(&field_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);

   int nsize_global[2];
   nsize_global[0] = num_of_fields;
   nsize_global[1] = field_size;

   long div_size = (long)(field_size / npes);
   last_pe_adding = field_size-div_size*npes;
   pe_size = div_size;
   if(mype == npes-1)pe_size += last_pe_adding;
#ifdef DEBUG
   cout << "-----------------> " << pe_size << "\n";
#endif
   int nsize[2];
   nsize[0] = nsize_global[0];
   nsize[1] = pe_size;  

   int start_size_global[2];
   start_size_global[0] = 0;
   start_size_global[1] = mype*div_size;

 //  long pe_size_orig = pe_size;
   points.resize(pe_size);
   float *readarray; 
   readarray = new float [num_of_fields*pe_size];

   cout << "DATAFILE INSIDE " << datafile << "     " << mype << "\n";

#ifdef DEBUG
   cout << "Reading " << n_load_fields << " fields for " << pe_size << " particles\n";
#endif

   MPI_Ajo_read(MPI_COMM_WORLD, mype, datafile.c_str(), 2, 
		  nsize_global, nsize, start_size_global, MPI_FLOAT, readarray, offset);


// MPI_Ajo_write(MPI_COMM_WORLD, mype,"verifica.dat", 2, 
//		  nsize_global, nsize, start_size_global, MPI_FLOAT, readarray, offset);
   if(doswap)
       for(long index=0; index<num_of_fields*pe_size; index++) SWAP_FLOAT(readarray[index]);
   
   for(int n_fields=0; n_fields<n_load_fields; n_fields++)
   {
     int n_fields_eff = which_fields[n_fields]-1;
     if(which_fields[n_fields] < 0)continue;

  /*   stride=sizeof(float)*(n_fields_eff*field_size+pe_size_orig*mype)+offset;

     infile.rewind();
     infile.skip(stride);
     for(long index=0; index<pe_size; index++)infile >> readarray[index];
  */
     
     switch(n_fields)
     {
     case 0:
       for(long index=0; index<pe_size; index++)
                 points[index].x=readarray[n_fields_eff*pe_size+index];
       break;
     case 1:
       for(long index=0; index<pe_size; index++)
                 points[index].y=readarray[n_fields_eff*pe_size+index];
       break;
     case 2:
       for(long index=0; index<pe_size; index++)
                 points[index].z=readarray[n_fields_eff*pe_size+index];
       break;
     case 3:
       for(long index=0; index<pe_size; index++)
                 points[index].r=readarray[n_fields_eff*pe_size+index];
       break;
     case 4:
       for(long index=0; index<pe_size; index++)
                 points[index].I=readarray[n_fields_eff*pe_size+index];
       break;
     case 5:
       for(long index=0; index<pe_size; index++)
                 points[index].e.r=readarray[n_fields_eff*pe_size+index];
       break;
     case 6:
       for(long index=0; index<pe_size; index++)
                 points[index].e.g=readarray[n_fields_eff*pe_size+index];
       break;
     case 7:
       for(long index=0; index<pe_size; index++)
                 points[index].e.b=readarray[n_fields_eff*pe_size+index];
       break;
     }
     for(long index=0; index<pe_size; index++)
     {
       points[index].type=0;
       float smooth = points[index].r;

       minradius = (minradius <= smooth ? minradius : smooth);
       maxradius = (maxradius >= smooth ? maxradius : smooth);
     }

   }
 //  infile.close();


   if(which_fields[4] < 0)
       for(long index=0; index<pe_size; index++)points[index].I=0.5;

   //maxradius = 1.0;
   *maxr=maxradius;
   *minr=minradius;

   MPI_Allreduce(&maxradius, maxr, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&minradius, minr, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
   
   delete [] which_fields;
   delete [] readarray;
   return pe_size;
}


#endif // USE_MPI,USE_MPIIO
