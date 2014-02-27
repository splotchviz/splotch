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

void mesh_creator(vector<particle_sim> &points, Mesh_vis * Mesh, Mesh_dim * MeshD, long * Meshlist)
{

   float cellfactor=10.0;
   long maxcells=256;
   long npart=1;

// Step 1: calculate the grid length and the average distance between particles

   float L_max[3];
   float dx_max;
   
   for (int j=0; j<3; j++)L_max[j]=0.0;

   long i;
   for (i=0; i<npart; i++)
     {
	L_max[0] = max(L_max[0], points[i].x);
	L_max[1] = max(L_max[1], points[i].y);
	L_max[2] = max(L_max[2], points[i].z);
     }

   float vol=1.0;
   for (int j=0; j<3; j++)vol *= L_max[j];
   dx_max = vol/float(npart);
   dx_max = pow(dx_max, 0.333333f);

   float Dx[3];
   for (int j=0; j<3; j++)Dx[j]=cellfactor*dx_max;
   
// Step 2: calculate the number of cells in each direction

   long Num_cells[3];
   long tot_num_cells=1.0;

   for (int j=0; j<3; j++)
   {
      Num_cells[j]=ceil(L_max[j]/Dx[j]);
      if(Num_cells[j] > maxcells)
      {
         Num_cells[j]=maxcells;
         Dx[j]=L_max[j]/float(maxcells);
      }
      tot_num_cells *= Num_cells[j];
   } 

// Step 3: correct box size in terms of number of cells

   for (int j=0; j<3; j++)L_max[j]=float(Dx[j]*Num_cells[j]);

// Step 4: Define the mesh array

   Mesh = new Mesh_vis [tot_num_cells];
   Mesh_dim Meshaux;
   MeshD = &Meshaux;

// Step 5: set the Mesh variables independent of the particles

   MeshD->Nx = Num_cells[0];
   MeshD->Ny = Num_cells[1];
   MeshD->Nz = Num_cells[2];
   MeshD->Lx = L_max[0];
   MeshD->Ly = L_max[1];
   MeshD->Lz = L_max[2];
   MeshD->ncell = Num_cells[0]*Num_cells[1]*Num_cells[2]; 

   long index=0;
   for(long ii=0; ii<Num_cells[0]; ii++)
   {
   for(long jj=0; jj<Num_cells[1]; jj++)
   {
   for(long kk=0; kk<Num_cells[2]; kk++)
   {

      Mesh[index].cell_index = index;

      Mesh[index].Cx = Dx[0]*(float(ii)+0.5);
      Mesh[index].Cy = Dx[1]*(float(jj)+0.5);
      Mesh[index].Cz = Dx[2]*(float(kk)+0.5);

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

      host_r[0] = long(points[i].x/Dx[0]);
      host_r[1] = long(points[i].y/Dx[1]);
      host_r[2] = long(points[i].z/Dx[2]);
      host = host_r[2] + L_max[2]*host_r[1] + L_max[2]*L_max[1]*host_r[0];
      num_particles[host]++;

   }

   for (long j=0;j<tot_num_cells;j++)Mesh[i].num_particles=num_particles[j];

   long sum_offset=0;
   for (long j=0;j<tot_num_cells;j++)
   {
      Mesh[i].offset = sum_offset;
      sum_offset += num_particles[j];

   }

   delete [] num_particles;

// Step 7: create particles list... Expensive stuff!

   Meshlist = new long [npart];
   index=0;
   long * offset;
   offset = new long [tot_num_cells];
   for (long j=0;j<tot_num_cells;j++)offset[j]=0;

   for (i=0; i<npart; i++)
   {

      host_r[0] = long(points[i].x/Dx[0]);
      host_r[1] = long(points[i].y/Dx[1]);
      host_r[2] = long(points[i].z/Dx[2]);
      host = host_r[2] + L_max[2]*host_r[1] + L_max[2]*L_max[1]*host_r[0];
      index = Mesh[host].offset+offset[host];
      offset[host]++;

      Meshlist[index] = i;

   }

}
