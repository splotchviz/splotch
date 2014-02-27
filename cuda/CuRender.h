/*
 * Copyright (c) 2011-2014
 *              Marzia Rivi (1), Tim Dykes (2)
 *               (1) University of Oxford
 *               (2) University of Portsmouth
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

#ifndef CURENDER_H
#define CURENDER_H

#include "cuda/splotch_cuda2.h"
#include "cuda/splotch_cuda.h"


using namespace std;

int cu_draw_chunk(int mydevID, cu_particle_sim *d_particle_data, int nParticle, arr2<COLOUR> &Pic_host, cu_gpu_vars* gv, bool a_eq_e, float64 grayabsorb, int xres, int yres, bool doLogs);
int add_device_image(arr2<COLOUR> &Pic_host, cu_gpu_vars* gv, int xres, int yres);

#endif
