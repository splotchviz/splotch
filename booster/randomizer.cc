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

void randomizer(vector<particle_sim> &points, Mesh_vis * Mesh, Mesh_dim MeshD)
{

   //long npart=points.size();
   srand ( time(NULL) );

   long i,j;
   long first;
   particle_sim swap_part;

   for (j=0; j<MeshD.ncell; j++)
   {
      first = Mesh[j].offset;
      for (i=0; i<Mesh[j].num_particles; i++)
      {

// random number generation

        double random_double = double(rand())/double(RAND_MAX); 
        long random_long = long(random_double*(Mesh[j].num_particles-i));
        long index = first+random_long;

// swap particles

        swap_part = points[first];
        points[first] = points[index];
        points[index] = swap_part; 

        first++;
        

      }
   }  


}
