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


#ifndef READER_H
#define READER_H

#include "splotch/splotchutils.h"

void gadget_reader(paramfile &params, int interpol_mode,
  std::vector<particle_sim> &p, std::vector<MyIDType> &id,
  std::vector<vec3f> &vel, int snr, double &time, double &boxsize);
void gadget_hdf5_reader(paramfile &params, int interpol_mode,
  std::vector<particle_sim> &p, std::vector<MyIDType> &id,
  std::vector<vec3f> &vel, int snr, double &time, double &redshift, double &boxsize);
void gadget_millenium_reader(paramfile &params, std::vector<particle_sim> &p, int snr, double *time);
void bin_reader_tab (paramfile &params, std::vector<particle_sim> &points);
void bin_reader_block (paramfile &params, std::vector<particle_sim> &points);

#ifdef SPLVISIVO
bool visivo_reader(paramfile &params, std::vector<particle_sim> &points,VisIVOServerOptions &opt);
#else
void visivo_reader();
#endif

long bin_reader_block_mpi (paramfile &params, std::vector<particle_sim> &points, float *maxr, float *minr, int mype, int npes);
void mesh_reader(paramfile &params, std::vector<particle_sim> &points);
void hdf5_reader(paramfile &params, std::vector<particle_sim> &points);
void tipsy_reader(paramfile &params, std::vector<particle_sim> &points);
void galaxy_reader(paramfile &params, std::vector<particle_sim> &points);
void h5part_reader(paramfile &params, std::vector<particle_sim> &points);
void ramses_reader(paramfile &params, std::vector<particle_sim> &points);
long enzo_reader (paramfile &params, std::vector<particle_sim> &points);
#endif
