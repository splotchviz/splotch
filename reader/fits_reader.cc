/*
 * Copyright (c) 2004-2014
 *              Marzia Rivi 
 *              University College London
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
#include <stdlib.h>
#include <fitsio.h>

#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "splotch/splotchutils.h"

using namespace std;


// read 3D fits cube
void fits_reader(paramfile &params, std::vector<particle_sim> &points)
{
    int      i, j, status=0, naxes;
    long     *dims, *fpixel;
    fitsfile *fitsptr; // FITS file pointer
    float rrr;
    int64 mybegin, npart, npart_total;
    
    if (mpiMgr.master())
        cout << "FITS DATA" << endl;
    
    // Open the file
    string datafile = params.find<string>("infile");
    fits_open_file(&fitsptr, datafile.c_str(), READONLY, &status);
    if ( status != 0 )
    {
        fits_report_error(stderr, status);
        exit(2);
    }
    
    // Get image dimensions
    fits_get_img_dim(fitsptr, &naxes, &status);
    
    // Get image axis lengths
    dims = (long*)calloc(naxes, sizeof(long));
    for (i=0; i<naxes; i++ ) dims[i] = 1;

    fits_get_img_size(fitsptr, naxes, dims, &status);
    
    npart_total = dims[0]*dims[1]*dims[2];
    cout << "x: " << dims[0] << "y: " << dims[1] << endl;
    rrr = params.find<float>("r",1.0);
    
    if (mpiMgr.master())
    {
        cout << "Input data file name: " << datafile << endl;
        cout << "Number of slices " << dims[2] << endl;
        cout << "Number of pixels " << npart_total << endl;
    }
    int64 myend;
    mpiMgr.calcShare (0, dims[2], mybegin, myend);
    npart = (myend-mybegin)*dims[0]*dims[1];
    points.resize(npart);
 
    
    mybegin = params.find<int>("cube_begin",mybegin);
    myend = params.find<int>("cube_end",myend);
    cout << "Selected slice range: " << mybegin << " - " << myend << endl;
    
    // Read in each slice in the FITS cube
    fpixel = (long*)calloc(naxes, sizeof(long));
    int imagesize = dims[0]*dims[1];
    float *image = (float*) calloc(imagesize,sizeof(float));
    
    int64 k=0;
    for (i=mybegin; i<myend; i++ )
    {
        // Read in slice
        fpixel[0] = 1;
        fpixel[1] = 1;
        fpixel[2] = i+1;
        
        fits_read_pix(fitsptr, TFLOAT, fpixel, dims[0]*dims[1], NULL, image, NULL, &status);
        if ( status != 0 )
        {
            printf(" FUNCS_FITS::READ_3D_FITS: Error reading pixel data from slice %i.\n",i+1);
            fits_report_error(stderr, status);
            exit(2);
        }
        for (j=0; j<imagesize; j++)
        {
            int i2 = j/dims[0];
            int i3 = j%dims[0];
            points[k].x = float(i3);
            points[k].y = float(i2);
            points[k].z = float(i);
            points[k].r = rrr;  //set smoothing length: assumed constant for all volume
            points[k].I = 0.5;
            if (isnan(image[j])) image[j] = 0;
            points[k].e.r = image[j];
            points[k].e.g = 0.0;
            points[k].e.b = 0.0;
            k++;
        }
            
    }
    // Close the image
    fits_close_file (fitsptr,&status);

    free(dims);
    free(image);
}


