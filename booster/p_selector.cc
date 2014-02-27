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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"
#include "booster/mesh_vis.h"

using namespace std;

void p_selector(vector<particle_sim> &points, Mesh_vis * Mesh, Mesh_dim MeshD, vector<particle_sim> &r_points)
{

   float fraction=1.0;
   long npart=points.size();
   srand ( time(NULL) );

   long i,j;
   long first;

   long reduced_npart=0;
   for (long k=0; k<MeshD.ncell; k++)reduced_npart += long(Mesh[k].weight*Mesh[k].num_particles);

   //vector<particle_sim> r_points;

   cout << "total number of cells = " << MeshD.ncell << endl;
   cout << "total number of particles = " << npart << endl;
   long count=0;
   for (j=0; j<MeshD.ncell; j++)
   {
      if(Mesh[j].active == true)
      {
      first = Mesh[j].offset;
      count++;
      for (i=0; i<long(Mesh[j].weight*Mesh[j].num_particles); i++)
      {

// build new array

        r_points.push_back(points[first]);
        first ++;
        
      }
      }
   }  
   cout << "processed cells = " << count << endl;
   cout << "processed particles = " << r_points.size() << endl;

/*
   FILE * pfile;
   pfile=fopen("pippo.dat", "w");
   for(long ii=0; ii<r_points.size(); ii++)
      fprintf(pfile, "%f %f %f\n", r_points[ii].x, r_points[ii].y, r_points[ii].z);
   fclose(pfile);
*/

}
