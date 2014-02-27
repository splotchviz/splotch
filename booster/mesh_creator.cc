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

#include "cxxsupport/arr.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"
#include "booster/mesh_vis.h"

using namespace std;

void mesh_creator(vector<particle_sim> &points, Mesh_vis ** Mesh, Mesh_dim * MeshD)
{

   float cellfactor=5.0;
   long maxcells=32;
   long npart=points.size();

// Step 1: calculate the grid length and the average distance between particles

   float L_max[3];
   float L_min[3];
   float L_size[3];
   float dx_max;
   
   for (int j=0; j<3; j++)L_max[j]=-1e30;
   for (int j=0; j<3; j++)L_min[j]=1e30;

   long i;
   for (i=0; i<npart; i++)
     {
	L_max[0] = max(L_max[0], points[i].x);
	L_max[1] = max(L_max[1], points[i].y);
	L_max[2] = max(L_max[2], points[i].z);
	L_min[0] = min(L_min[0], points[i].x);
	L_min[1] = min(L_min[1], points[i].y);
	L_min[2] = min(L_min[2], points[i].z);
     }

   for (int j=0; j<3; j++)L_max[j] += 0.01*L_max[j];
   for (int j=0; j<3; j++)L_min[j] -= 0.01*L_min[j];

   for (int j=0; j<3; j++)L_size[j]=abs(L_max[j]-L_min[j]);
   float vol=1.0;
   for (int j=0; j<3; j++)vol *= L_size[j];
   dx_max = vol/float(npart);
   dx_max = pow(dx_max, 0.333333f);

   float Dx[3];
   for (int j=0; j<3; j++)Dx[j]=cellfactor*dx_max;
   
// Step 2: calculate the number of cells in each direction

   long Num_cells[3];
   long tot_num_cells=1.0;

   for (int j=0; j<3; j++)
   {
      Num_cells[j]=ceil(L_size[j]/Dx[j]);
      if(Num_cells[j] > maxcells)
      {
         Num_cells[j]=maxcells;
         Dx[j]=L_size[j]/float(maxcells);
      }
      tot_num_cells *= Num_cells[j];
   } 

// Step 3: correct box size in terms of number of cells

   for (int j=0; j<3; j++)L_size[j]=float(Dx[j]*Num_cells[j]);

// Step 4: Define the mesh array

   *Mesh = new Mesh_vis [tot_num_cells];

// Step 5: set the Mesh variables independent of the particles

   //MeshD = new Mesh_dim;
   MeshD->Nx = Num_cells[0];
   MeshD->Ny = Num_cells[1];
   MeshD->Nz = Num_cells[2];
   MeshD->Lx = L_size[0];
   MeshD->Ly = L_size[1];
   MeshD->Lz = L_size[2];
   MeshD->Dx = Dx[0];
   MeshD->Dy = Dx[1];
   MeshD->Dz = Dx[2];
   MeshD->ncell = Num_cells[0]*Num_cells[1]*Num_cells[2]; 

   cout << 
   MeshD->Nx << " " <<
   MeshD->Ny << " " <<
   MeshD->Nz << " " <<
   MeshD->Lx << " " <<
   MeshD->Ly << " " <<
   MeshD->Lz << " " <<
   MeshD->ncell << endl;


   long index=0;
   for(long ii=0; ii<Num_cells[0]; ii++)
   {
   for(long jj=0; jj<Num_cells[1]; jj++)
   {
   for(long kk=0; kk<Num_cells[2]; kk++)
   {

      (*Mesh)[index].cell_index = index;

      (*Mesh)[index].Cx = Dx[0]*(float(ii)+0.5);
      (*Mesh)[index].Cy = Dx[1]*(float(jj)+0.5);
      (*Mesh)[index].Cz = Dx[2]*(float(kk)+0.5);
      (*Mesh)[index].posx = (*Mesh)[index].Cx+L_min[0];
      (*Mesh)[index].posy = (*Mesh)[index].Cy+L_min[1];
      (*Mesh)[index].posz = (*Mesh)[index].Cz+L_min[2];

      index++;

   }
   }
   }

   
// Step 6: Associate particle distribution to meshes

   long host;
   long host_r[3];
   long * num_particles;
   num_particles = new long [tot_num_cells];
   for (long j=0;j<tot_num_cells;j++)num_particles[j]=0;

   for (i=0; i<npart; i++)
   {
      host_r[0] = long((points[i].x-L_min[0])/Dx[0]);
      host_r[1] = long((points[i].y-L_min[1])/Dx[1]);
      host_r[2] = long((points[i].z-L_min[2])/Dx[2]);
      host = host_r[2] + Num_cells[2]*host_r[1] + Num_cells[2]*Num_cells[1]*host_r[0];
      num_particles[host]++;

   }

   for (long j=0;j<tot_num_cells;j++)(*Mesh)[j].num_particles=num_particles[j];

   long sum_offset=0;
   for (long j=0;j<tot_num_cells;j++)
   {
      (*Mesh)[j].offset = sum_offset;
      sum_offset += num_particles[j];

   }

// Set all the cells active by default
   for (long j=0;j<tot_num_cells;j++)(*Mesh)[j].active = true; 
   for (long j=0;j<tot_num_cells;j++)(*Mesh)[j].weight = 1.0; 

   delete [] num_particles;

// Step 7: create particles list... Expensive stuff!

   index=0;
   long * offset;
   long ncount=0;
   particle_sim point_aux;
   particle_sim point_aux1;
   offset = new long [tot_num_cells];
   for (long j=0;j<tot_num_cells;j++)offset[j]=0;

   point_aux = points[0];
   while (ncount<npart)
   {
      host_r[0] = long((point_aux.x-L_min[0])/Dx[0]);
      host_r[1] = long((point_aux.y-L_min[1])/Dx[1]);
      host_r[2] = long((point_aux.z-L_min[2])/Dx[2]);
      host = host_r[2] + Num_cells[2]*host_r[1] + Num_cells[2]*Num_cells[1]*host_r[0];
      index = (*Mesh)[host].offset+offset[host];
      offset[host]++;

      point_aux1 = points[index];
      points[index] = point_aux;
      point_aux=point_aux1; 
      ncount++;
   
   }
}
