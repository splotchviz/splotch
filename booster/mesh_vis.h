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


#ifndef TREEVIS
#define TREEVIS

#include "splotch/splotchutils.h"

struct Mesh_vis {

   float Cx,Cy,Cz;
   float posx,posy,posz;
   long cell_index;
   long num_particles;
   long offset;
   bool active;
   float weight;

};

struct Mesh_dim {

   float Lx,Ly,Lz;
   float Dx,Dy,Dz;
   long Nx,Ny,Nz;
   long ncell;

};


void mesh_creator(std::vector<particle_sim> &points, Mesh_vis ** Mesh, Mesh_dim * MeshD);

void p_selector(std::vector<particle_sim> &points, Mesh_vis * Mesh, Mesh_dim MeshD, std::vector<particle_sim> &r_points);

void randomizer(std::vector<particle_sim> &points, Mesh_vis * Mesh, Mesh_dim MeshD);

void m_rotation(paramfile &params, Mesh_vis ** p, Mesh_dim MeshD, const vec3 &campos, const vec3 &lookat, vec3 sky);

#endif
