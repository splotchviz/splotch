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


#ifdef HDF5
#ifdef USE_MPI
#include "mpi.h"
#endif
#include <cstdio>
#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>
#include "hdf5.h"
#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"


using namespace std;

long enzo_reader (paramfile &params, std::vector<particle_sim> &points)
{

   FILE * pFile;
   FILE * auxFile;
   FILE * hdf5File;
   char datafilename[1000];
   char outputfilename[1000];
   string groupprefix;
   string hgroup;
   string completename;
   long np,ng;
   int nrank = 3;
   int nghost = 3;
   int nfiles;
   double leftside[nrank];
   double rightside[nrank];
   int lbox[nrank];
   int rbox[nrank];
   double lxbox[nrank];
   double rxbox[nrank];
   int sbox[nrank];
   int nleft[nrank];
   int nright[nrank];
   int nsize[nrank];
   const int numberoffields = 11;
   int gridid;
   char groupid[100];
   long gcounter=0;
   float sigma;
   float *maxr;
    float *minr;

   int ngx[nrank];
   int vngx[nrank];
   double dxvngx[nrank];
   double rdxvngx[nrank];
   double boxsize[nrank];
   double hboxsize[nrank];
   double boxcenter[nrank];
   double basesize[nrank];
   double lintersect[nrank];
   double rintersect[nrank];
   double leftcorner[nrank];
   double rightcorner[nrank];
   int lnintersect[nrank];
   int rnintersect[nrank];
   int lbintersect[nrank];
   int rbintersect[nrank];
   basesize[0] = 1.0;
   basesize[1] = 1.0;
   basesize[2] = 1.0;
   int maxlevel;
   int boxcells[nrank];

   float * dataarray;
   float * destarray;
   
   long total_size=0;
   long total_size_old=0;
   float minradius=1e30;
   float maxradius=-1e30;

   string fieldsnames [numberoffields];

   int mype=0;
   int npes=1;

#ifdef USE_MPI
   MPI_Comm_rank(MPI_COMM_WORLD, &mype);
   MPI_Comm_size(MPI_COMM_WORLD, &npes);
#endif

// load parameters from parameter file

   float  smooth_factor = params.find<float>("smooth_factor",1.0);
   int  red = params.find<int>("red",-1);
   int  green = params.find<int>("green",-1);
   int  blue = params.find<int>("blue",-1);
   int  intensity = params.find<int>("I",-1);
 
   groupprefix = "/Grid";

   //for(int i=0; i<numberoffields; i++)sf[i]=-1;

   fieldsnames[0] = "/Dark_Matter_Density";
   fieldsnames[1] = "/Density";
   fieldsnames[2] = "/Temperature";
   fieldsnames[3] = "/x-velocity";
   fieldsnames[4] = "/y-velocity";
   fieldsnames[5] = "/z-velocity";
   fieldsnames[6] = "/GasEnergy";
   fieldsnames[7] = "/TotalEnergy";
   fieldsnames[8] = "/Bx";
   fieldsnames[9] = "/By";
   fieldsnames[10] = "/Bz";


#ifdef MESHMERGER
   std::vector< vector <float> > colors;

// define the quantities to be stored in the 
   int * color = new int [numberoffields];
   color[0] = params.find<int>("color0",-1);
   color[1] = params.find<int>("color1",-1);
   color[2] = params.find<int>("color2",-1);
   color[3] = params.find<int>("color3",-1);
   color[4] = params.find<int>("color4",-1);
   color[5] = params.find<int>("color5",-1);
   color[6] = params.find<int>("color6",-1);
   color[7] = params.find<int>("color7",-1);
   color[8] = params.find<int>("color8",-1);
   color[9] = params.find<int>("color9",-1);
   color[10] = params.find<int>("color10",-1);

// set the output filename
   string outfile = params.find<string>("output_file");

// create file in parallel
   MPI_Info info = MPI_INFO_NULL;
   //MPI_Info_create(&info);
   //hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
   hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
   H5Pset_fapl_mpio(fapl_id, MPI_COMM_WORLD, info);
   hid_t output_file = H5Fcreate(outfile.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, fapl_id);
   H5Pclose(fapl_id);

#endif



   string hierarchyname = params.find<string>("hierarchy_file");
   int sf[numberoffields];
   string selfield[numberoffields];
   sf[0] = red;
   sf[1] = green;
   sf[2] = blue;
   sf[3] = intensity;

   ngx[0] = params.find<int>("nx",16);
   ngx[1] = params.find<int>("ny",16);
   ngx[2] = params.find<int>("nz",16);
   leftcorner[0] = params.find<float>("leftcornerx",0.0);
   leftcorner[1] = params.find<float>("leftcornery",0.0);
   leftcorner[2] = params.find<float>("leftcornerz",0.0);
   rightcorner[0] = params.find<float>("rightcornerx",1.0);
   rightcorner[1] = params.find<float>("rightcornery",1.0);
   rightcorner[2] = params.find<float>("rightcornerz",1.0);

   maxlevel = params.find<int>("levels",1);
   maxlevel--;

   int number_of_fields2read;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// WARNING X and Z are swapped!!!!!!!!!!!
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ifdef USEMPIOLD
   MPI_Bcast(&number_of_fields2read, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(sf, number_of_fields2read, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&hierarchyname[0], 1000, MPI_CHAR, 0, MPI_COMM_WORLD);
   MPI_Bcast(ngx, 3, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(leftcorner, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(rightcorner, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(&maxlevel, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif


// level 0 resolution:

   double dxbase[nrank];
   for (int j=0; j<nrank; j++)dxbase[j] = 1.0/float(ngx[j]);

// half box size

   float fmaxlevel = (float)maxlevel;
   for (int i=0; i<nrank; i++) {
	boxsize[i] = rightcorner[i] - leftcorner[i];
	hboxsize[i] = 0.5 * boxsize[i];
   }

// calculate virtual global grid size and resolution at the output level

   for (int i=0; i<nrank; i++) 
   {
       vngx[i] = ngx[i] * (int) pow(2.0f, fmaxlevel); 
       dxvngx[i] = 1.0 / (double)vngx[i];
       rdxvngx[i] = 1.0 / dxvngx[i];
#ifdef DEBUG   
       printf("Virtual Size: %d %d\n", i, vngx[i]);
#endif
   }

//

   double boxcoarseL[3];
   double boxcoarseR[3];
   int ndims_aux[3];

   for (int i=0; i<nrank; i++)
   {
       boxcoarseL[i] = leftcorner[i];
       boxcoarseR[i] = rightcorner[i];
       ndims_aux[i] = ((int)(boxsize[i]*(double)ngx[i]))*(int)pow(2.0f, fmaxlevel);
       //ndims_aux[i] = ((int)(boxsize[i]*(double)ngx[i])+1)*(int)pow(2.0f, fmaxlevel);
   }


// esitmate the size of the output box at the output resolution
// notice that the whole box has size basesize=1.0

   int boxstart_cell[nrank];
   int boxend_cell[nrank];
   for (int i=0; i<nrank; i++)
   {

// box references in output resolution cells

        boxstart_cell[i]  = (int)((boxcoarseL[i]/basesize[i]) * vngx[i]);
	if(boxstart_cell[i] < 0) boxstart_cell[i] = 0;
        boxend_cell[i]    = (int)((boxcoarseR[i]/basesize[i]) * vngx[i])-1;
	if(boxend_cell[i] >= vngx[i]) boxend_cell[i] = vngx[i]-1;
        boxcells[i] = boxend_cell[i]-boxstart_cell[i]+1;

// box references in universe (0-1) units

        lxbox[i] = boxcoarseL[i];
        if(lxbox[i] < 0.0) lxbox[i] = 0.0;
        rxbox[i] = boxcoarseR[i];
        if(rxbox[i] > basesize[i]) rxbox[i] = basesize[i];

#ifdef DEBUG   
        printf("Cells in the BOX: %d %d\n", i, boxcells[i]);
        printf("BOX starts at: %d %d\n", i, boxstart_cell[i]);
        printf("BOX ends at: %d %d\n", i, boxend_cell[i]);
#endif
   }  

// calculate total size and prepare HDF5 quantities and output file

   ng = 1;
   for (int j=0; j<nrank; j++) ng*=boxcells[j];

   
// create all DATASETS

   hid_t obj_id;
   hid_t dataspace;
   hid_t memoryspace;
#ifdef SP5
   long long * start = new long long [nrank];
#endif
#ifdef BCX
   hsize_t * start = new hsize_t [nrank];
#endif
#ifdef CLX
   hssize_t * start = new hssize_t [nrank];
#endif
   hsize_t * stride = new hsize_t [nrank];
   hsize_t * count  = new hsize_t [nrank];
   hsize_t * block  = new hsize_t [nrank];
   long sourcesize;
   long destinationsize;
   hid_t source_id;
   hid_t source_obj;
   hid_t source_space;

#ifdef SP5
   long long * s_start = new long long [nrank];
#endif
#ifdef BCX
   hsize_t * s_start = new hsize_t [nrank];
#endif
#ifdef CLX
   hssize_t * s_start = new hssize_t [nrank];
#endif
   hsize_t * s_stride = new hsize_t [nrank];
   hsize_t * s_count  = new hsize_t [nrank];
   hsize_t * s_block  = new hsize_t [nrank];
   hsize_t * s_dims   = new hsize_t [nrank];
   hsize_t * s_maxdims   = new hsize_t [nrank];

// create all the datasets in the HDF5 output file

// build the datasets

   pFile = fopen (hierarchyname.c_str(), "r");
   fscanf (pFile, "%d", &nfiles);

// check if the number of processors is less than the number of files

   if(nfiles < npes)
   {
      printf("Only the case with a number of MPI processes less or equal than files is supported\n");
      printf("NPES = %d, Nfiles = %d, aborting...\n", npes, nfiles);
      exit(11);
   }

   int acont=0;
   long countt=0;
   double dxxx[nrank];
   double rdxxx[nrank];
   double levelbox;
   int nlevelbox;
   int checking = 0;

   int respe=0;
   int modfiles = int(nfiles%npes);
   int modpe = int(modfiles/(mype+1));
   if(modpe >= 1)respe=1;

   int chunk_pe = int(nfiles/npes)+respe;
   int istart_pe = mype*chunk_pe + modfiles;
   if(respe == 1) istart_pe = mype*chunk_pe;
   int iend_pe   = istart_pe+chunk_pe;
   //if(mype == npes-1 && npes > 1)iend_pe=iend_pe+(int)nfiles%npes; 

#ifdef DEBUG
   printf("PE %d STARTS FROM %d END AT %d\n", mype, istart_pe, iend_pe);
#endif

#ifdef MESHMERGER
// Preprocessing stage

   long pesize = 0;
   for (int i=0; i<nfiles; i++)
     {

        for (int j=0; j<nrank; j++) fscanf(pFile, "%d", &lbox[2-j]);
        for (int j=0; j<nrank; j++) fscanf(pFile, "%d", &rbox[2-j]);
        for (int j=0; j<nrank; j++) fscanf(pFile, "%lf", &leftside[2-j]);
        for (int j=0; j<nrank; j++) fscanf(pFile, "%lf", &rightside[2-j]);
        fscanf (pFile, "%s", datafilename);
        if(i < istart_pe || i >= iend_pe)continue;

        pesize += (rbox[2]-lbox[2]+1)*(rbox[1]-lbox[1]+1)*(rbox[0]-lbox[0]+1);

     }

// calculate global data size 
   long totalsize;
   MPI_Allreduce(&pesize, &totalsize, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
   //cout << " MYPE = " << mype << " SIZE IS = " << totalsize << endl;
// send to next processor the starting point
   long * sizearray = new long [npes];
   long * sizearray_tot = new long [npes];
   for (int jj=0; jj<npes; jj++)sizearray[jj]=0;
   for (int jj=0; jj<npes; jj++)sizearray_tot[jj]=0;
   sizearray[mype] = pesize;

   MPI_Allreduce(sizearray, sizearray_tot, npes, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

   long startpesize = 0;
   long endpesize = 0;
   for (int jj=0; jj<mype; jj++)startpesize += sizearray_tot[jj];
   endpesize = startpesize+pesize-1;

   //cout << "MYPE = " << mype << " LEFT = " << startpesize << " RIGHT = " << pesize-1 << endl;

// create the full dataspace
   hid_t outspace;
   hid_t * outobj_id = new hid_t [numberoffields+3];
   int outrank = 1;
   hsize_t * outdims = new hsize_t [outrank];
   outdims[0] = totalsize;
   outspace = H5Screate_simple (outrank, outdims, NULL);

// define memspace (chuncks)
   hid_t memspace;
   hsize_t * memdim = new hsize_t [outrank];
   memdim[0] = pesize;
   memspace = H5Screate_simple(outrank, memdim, NULL);

// set the hyperslab
   hsize_t * pecount = new hsize_t [outrank];
   hsize_t * peoffset = new hsize_t [outrank];
   pecount[0] = pesize;
   peoffset[0] = startpesize;
   H5Sselect_hyperslab(outspace, H5S_SELECT_SET, peoffset, NULL, pecount, NULL);


// create parallel transfer properties 
   fapl_id = H5Pcreate(H5P_DATASET_XFER);
   H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_INDEPENDENT);

// create the dataspace for the dataset
///   hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
///   H5Pset_chunk(plist_id, outrank, pecount);

// create coordinates entry
   outobj_id[0] = H5Dcreate1(output_file, "coordx", H5T_NATIVE_FLOAT, outspace, H5P_DEFAULT);//, H5P_DEFAULT, H5P_DEFAULT);//, plist_id, H5P_DEFAULT);
   outobj_id[1] = H5Dcreate1(output_file, "coordy", H5T_NATIVE_FLOAT, outspace, H5P_DEFAULT);//, H5P_DEFAULT, H5P_DEFAULT);//, plist_id, H5P_DEFAULT);
   outobj_id[2] = H5Dcreate1(output_file, "coordz", H5T_NATIVE_FLOAT, outspace, H5P_DEFAULT);//, H5P_DEFAULT, H5P_DEFAULT);//, plist_id, H5P_DEFAULT);
   outobj_id[3] = H5Dcreate1(output_file, "resolution", H5T_NATIVE_FLOAT, outspace, H5P_DEFAULT);//, H5P_DEFAULT, H5P_DEFAULT);//, plist_id, H5P_DEFAULT);

// create fields entry
   int activefields=0;
   for (int ifields=0; ifields<numberoffields; ifields++)
   {
      if(color[ifields] != -1)
      {
        outobj_id[activefields+4] = H5Dcreate1(output_file, fieldsnames[color[ifields]].c_str(), H5T_NATIVE_FLOAT, 
                                              outspace, H5P_DEFAULT);//, H5P_DEFAULT, H5P_DEFAULT);//, plist_id, H5P_DEFAULT);
        activefields++;
      }
   }
   colors.resize(activefields+4);
///   H5Pclose(plist_id);

   fseek (pFile , sizeof(nfiles) , SEEK_SET );
     
// allocate the vectors
   for (int ii=0; ii<activefields+4; ii++)colors[ii].resize(pesize);

#endif

//// MAIN LOOP

   long xcounter = 0;
   long ccounter = 0;
   long ccounter0 = 0;
   float * smooth = new float [nfiles];

   for (int i=0; i<nfiles; i++)
//   for (int i=0; i<1; i++)
     {
 
        acont++; 
        //printf("Merging block %d\n",i);
        checking = 0;

        char buffer[100];
        int iaux = 10000001+i;
        sprintf(buffer,"%s%d", groupprefix.c_str(),iaux);
        hgroup = buffer;
        string aux0="0";
        hgroup.replace(5,1,aux0);
        

// read the i-th slab

        int grididaux;

        // NOTICE that due to hierarchy file box inverse mapping z and x are swapped:
        // 0 --> 2
        // 1 --> 1
        // 2 --> 0

        //fscanf(pFile, "%d", &gridid);
        for (int j=0; j<nrank; j++) fscanf(pFile, "%d", &lbox[2-j]);
        for (int j=0; j<nrank; j++) fscanf(pFile, "%d", &rbox[2-j]);
	for (int j=0; j<nrank; j++) fscanf(pFile, "%lf", &leftside[2-j]);
	for (int j=0; j<nrank; j++) fscanf(pFile, "%lf", &rightside[2-j]);
        fscanf (pFile, "%s", datafilename);

        for (int j=0; j<nrank; j++)
	{
            lbox[j] -= 3;
            rbox[j] -= 3;
	    sbox[j] = rbox[j]-lbox[j]+1; 
            dxxx[j] = (rightside[j]-leftside[j])/((double)sbox[j]);
	    rdxxx[j] = 1.0 / dxxx[j];
 
        }
// assume same resolution in all directions
        smooth[i] = dxxx[0];
       

        levelbox = log10(dxbase[0]/dxxx[0])/log10(2.0);
        nlevelbox = (int)levelbox;
        
        //for(int j=0; j<nrank; j++)
        //printf("LEFT and RIGHT %lf >= %lf or  %lf <= %lf --> checking=1\n", leftside[j], rxbox[j],  rightside[j], lxbox[j]);


        //for(int j=0; j<nrank; j++)printf(">>>>>>>> %f <?  %f, %f >? %f\n", rightside[j], lxbox[j], leftside[j], rxbox[j]);
        for(int j=0; j<nrank; j++)if(rightside[j] <= lxbox[j] || leftside[j] >= rxbox[j])checking=1;

#ifdef DEBUG   
	if(checking)printf("Skipping block %d, not intersecting selected region...\n", i) ;
#endif
	if(checking)continue;

// next iteration if block is not assigned to the processor

        if(i < istart_pe || i >= iend_pe)continue;

        printf("BOX = %d, LEVEL = %d\n", i, nlevelbox);
        if(nlevelbox > maxlevel) continue;

        sourcesize = 1;
        destinationsize = 1;

	for(int j=0; j<nrank; j++) 
        {

          float raux;
#ifdef DEBUG   
          printf("CHECK CORRECTNESS %f %f %f %f\n", lxbox[j],leftside[j],rxbox[j],rightside[j]);
#endif

// intersection region in world coordinates

	  lintersect[j] = (leftside[j] >= lxbox[j])?leftside[j]:lxbox[j]; 
	  rintersect[j] = (rightside[j] <= rxbox[j])?rightside[j]:rxbox[j]; 

// intersection region in slab coordinates

	  raux = ((lintersect[j]-leftside[j])*rdxxx[j]);
	  lnintersect[j] = (int)raux;
          raux = ((rintersect[j]-leftside[j])*rdxxx[j])-1;
          rnintersect[j] = (int)raux;

// intersection region in box coordinates

	  float powconv = pow(2.0f, (float)(maxlevel-nlevelbox));

	  raux = ((lintersect[j]-lxbox[j])*rdxvngx[j]);
	  lbintersect[j] = (int)raux;
          //////raux = ((rintersect[j]-lxbox[j])*rdxvngx[j])-1;
          raux = lbintersect[j] + (float)(rnintersect[j]-lnintersect[j]+1)*powconv;
	  rbintersect[j] = (int)raux - 1; 

// calculate auxiliary read array size

          destinationsize *= (long)((float)(rnintersect[j]-lnintersect[j]+1)*powconv);
          sourcesize *= (rnintersect[j]-lnintersect[j]+1);
	  

	}

#ifdef DEBUG   
        printf("SOURCESIZE = %ld\n", sourcesize);
        printf("DESTSIZE   = %ld\n", destinationsize);

        for(int j=0; j<nrank; j++)printf("------> INTERSECT LEFT ABS: %f\n", lintersect[j]);
        for(int j=0; j<nrank; j++)printf("------> INTERSECT RIGHT ABS: %f\n", rintersect[j]);
        for(int j=0; j<nrank; j++)printf("------> SLAB INTERSECT LEFT: %d\n", lnintersect[j]);
        for(int j=0; j<nrank; j++)printf("------> SLAB INTERSECT RIGHT: %d\n", rnintersect[j]);
        for(int j=0; j<nrank; j++)printf("------> BOX INTERSECT LEFT: %d\n", lbintersect[j]);
        for(int j=0; j<nrank; j++)printf("------> BOXINTERSECT RIGHT: %d\n", rbintersect[j]);

/*

        printf ("%f, %f, %f\n", leftside[0],leftside[1],leftside[2]);
        printf ("%f, %f, %f\n", rightside[0],rightside[1],rightside[2]);
        printf ("%s\n", datafilename);

*/
#endif

        for (int j=0; j<nrank; j++)
        {

	    start[j]  = (hsize_t)lbintersect[j];
	    stride[j] = 1;
	    count[j]  = (hsize_t)(rbintersect[j]-lbintersect[j]+1);
	    block[j]  = 1;


	    s_start[j]  = (hsize_t)lnintersect[j] ;
	    s_stride[j] = 1;
	    s_count[j]  = (hsize_t)(rnintersect[j]-lnintersect[j]+1);
	    s_block[j]  = 1;

#ifdef DEBUG   
            printf ("ORIGIN START %d, ORIGIN COUNT %d\n", s_start[j],s_count[j]);
            printf ("DEST  START %d, DEST   COUNT %d\n", start[j],count[j]);
#endif
        }

/*
        printf("%d, %d, %d\n", nsize[0],nsize[1],nsize[2]);
        printf("%d, %d, %d\n", start[0],start[1],start[2]);
        printf("%d, %d, %d\n", count[0],count[1],count[2]);


*/

        total_size_old = total_size;
        total_size += sourcesize;

        printf("-----> TOTAL SIZE for PE %d IS %ld\n",mype,total_size);
 

// allocate memory for loading data

	dataarray = new float[sourcesize];

/*
        scalar->resize(total_size);
	xpos->resize(total_size);
	ypos->resize(total_size);
	zpos->resize(total_size);
        smooth->resize(total_size);
*/
        points.resize(total_size);
	

        for(long jk=0; jk<sourcesize; jk++)dataarray[jk]=0.0;

// Set the coordinates

        long jaux = 0;
        for(long iaux=total_size_old; iaux<total_size; iaux++)
        {

               long ipp = jaux;
               long ic3d = (int)((float)ipp/(float)(s_count[1]*s_count[2]));
               long naux = ipp%(s_count[1]*s_count[2]);
               long jc3d = (int)((float)naux/((float)s_count[2]));
               long kc3d = naux - s_count[2]*jc3d;


               points[iaux].x   = lintersect[0] + dxxx[0] * (float)ic3d;
               points[iaux].y   = lintersect[1] + dxxx[1] * (float)jc3d;
               points[iaux].z   = lintersect[2] + dxxx[2] * (float)kc3d;
               points[iaux].r   = smooth_factor*dxxx[0];
               minradius = (minradius <= dxxx[0] ? minradius : dxxx[0]);
               maxradius = (maxradius >= dxxx[0] ? maxradius : dxxx[0]);

               jaux++;
#ifdef MESHMERGER
               colors[0][xcounter] = points[iaux].x;
               colors[1][xcounter] = points[iaux].y;
               colors[2][xcounter] = points[iaux].z;
               colors[3][xcounter] = smooth[i];
#endif
               xcounter++;
        }
	

// open source file 

        printf("READING DATA FROM %s\n", datafilename);
	source_id = H5Fopen(datafilename, H5F_ACC_RDONLY, H5P_DEFAULT);

// read dataset from SOURCE
// data are read in the following sequence:
// 0 = color 1 
// 1 = color 2 
// 2 = color 3
// 3 = intensity

        int kaux = 0;
#ifdef MESHMERGER
        number_of_fields2read = activefields;
#else
        number_of_fields2read = 4;
#endif
	for(int k=0; k<number_of_fields2read; k++)
	{

            jaux=0;
            kaux = sf[k];
#ifdef MESHMERGER
            ccounter=ccounter0;
            kaux = color[k];
#endif
            if(kaux >= 0)
            {
              completename = hgroup;
	      completename.append(fieldsnames[kaux]);

              printf("READING %s from Grid %d\n",completename.c_str(),i);
	      source_obj = H5Dopen(source_id,completename.c_str());
	      source_space = H5Dget_space(source_obj);

   	      H5Sget_simple_extent_dims(source_space, s_dims, s_maxdims);
#ifdef DEBUG   
	      printf("%d, %d, %d\n", s_dims[0], s_dims[1], s_dims[2]);
#endif

// create auxiliary memory space for reading, select region and read

              memoryspace = H5Screate_simple (nrank, s_count, s_count);
	    
	      H5Sselect_hyperslab(source_space, H5S_SELECT_SET, s_start, s_stride, s_count, s_block);

	      H5Dread(source_obj, H5T_NATIVE_FLOAT, memoryspace, source_space, H5P_DEFAULT, dataarray);
              H5Sclose (memoryspace);
            }

#ifdef MESHMERGER
// store data in arrays
            for (long ii=0; ii<sourcesize; ii++)
            {
                colors[k+4][ccounter] = dataarray[ii];
                ccounter++;
            }

#endif

#ifndef MESHMERGER
#define CASEMACRO__(num,str,noval,ss) \
      case num: \
        if (sf[num]>=0) \
          for(long iaux=total_size_old; iaux<total_size; iaux++) \
             {points[iaux].str = ss*dataarray[jaux]; jaux++;}\
        else \
          for(long iaux=total_size_old; iaux<total_size; iaux++) points[iaux].str = noval; \
        break;

            switch(k)
            {
               CASEMACRO__(0,e.r,1.0,1.0)
               CASEMACRO__(1,e.g,0,1.0)
               CASEMACRO__(2,e.b,0,1.0)
               CASEMACRO__(3,I,1.0,1.0)
               //CASEMACRO__(3,I,dxxx[0]*dxxx[0]*dxxx[0],1.0)
             }

#endif // not MESHMERGER
// end of loop over fields 
       }
#ifdef MESHMERGER
       ccounter0 = ccounter;
#endif
        
// free memory

        H5Fclose (source_id);
	delete [] dataarray;
   }

#ifdef MESHMERGER

// write data in file

   for(int ii=0; ii<10; ii++)
      cout << colors[0][ii] << " " << colors[1][ii] << " " << colors [2][ii] << " " << colors [3][ii] << endl;

   float * pointsaux = new float [pesize];


// write positions
   for (long jj=0; jj<pesize; jj++)pointsaux[jj]=colors[0][jj];
   H5Dwrite(outobj_id[0], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, pointsaux);
   //H5Dwrite(outobj_id[0], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, &colors[0][0]);
   for (long jj=0; jj<pesize; jj++)pointsaux[jj]=colors[1][jj];
   H5Dwrite(outobj_id[1], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, pointsaux);
   //H5Dwrite(outobj_id[0], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, &colors[1][0]);
   for (long jj=0; jj<pesize; jj++)pointsaux[jj]=colors[2][jj];
   H5Dwrite(outobj_id[2], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, pointsaux);
   //H5Dwrite(outobj_id[0], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, &colors[2][0]);
   for (long jj=0; jj<pesize; jj++)pointsaux[jj]=colors[3][jj];
   H5Dwrite(outobj_id[3], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, pointsaux);
   //H5Dwrite(outobj_id[0], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, &colors[2][0]);

// write fields
   for (int ii=0; ii<activefields; ii++)
     {
        for (long jj=0; jj<pesize; jj++)pointsaux[jj]=colors[ii+4][jj]; 
        //H5Dwrite(outobj_id[color[ii]], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, &colors[ii+3][0]);
        H5Dwrite(outobj_id[ii+4], H5T_NATIVE_FLOAT, memspace, outspace, fapl_id, pointsaux);
     }

   delete [] pointsaux;

// free HDF5 stuff

   H5Pclose(fapl_id);
   for (int ii=0; ii<activefields+4; ii++)H5Dclose(outobj_id[ii]);
   H5Sclose(memspace);
   H5Sclose(outspace);
   H5Fclose(output_file);
   
#endif

#ifdef DEBUG   
   //printf("TOTAL NUMBER OF CELLS AT LEVEL 0 = %ld\n", gcounter);
#endif

//   H5Fclose (file_id);


   fclose (pFile);

#ifdef MESHMERGER
   cout << "Data saved in " << outfile.c_str() << endl;
   exit (21);
#endif

   for(long ir=0; ir<total_size; ir++)points[ir].I = points[ir].I*(points[ir].r/minradius);
   //for(long ir=0; ir<total_size; ir++)points[ir].r = points[ir].r*sqrt(points[ir].r/minradius);

   //*maxr=maxradius;
   //*minr=minradius;
   cout << "RETURNING FROM ENZO" << endl;
#ifdef USEMPIOLD
   MPI_Allreduce(&maxradius, maxr, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&minradius, minr, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
#endif

#ifdef DEBUG
   printf("PE %d MANAGE %ld POINTS\n", mype, total_size);
   cout << "START " << startpesize << " END " << endpesize << " TOT " << pesize << endl;
#endif

   return total_size;

}

#endif  // HDF5
