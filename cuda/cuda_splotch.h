/*
 * Copyright (c) 2010-2014
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

#ifndef CUDA_SPLOTCH_H
#define CUDA_SPLOTCH_H

#include "cxxsupport/string_utils.h"
#include "cxxsupport/walltimer.h"

#include "cuda/cuda_utils.h"
#include "cuda/cuda_render.h"
#include "splotch/splotchutils.h"

void cuda_rendercontext_init(paramfile& params, render_context& context);
void cuda_renderer_init(int& mydevID, int nTasksDev, arr2<COLOUR> &pic, vector<particle_sim> &particle, vector<COLOURMAP> &amap, paramfile &g_params, cu_cpu_vars& cv);
void cuda_rendering(int mydevID, int nTasksDev, arr2<COLOUR> &pic, std::vector<particle_sim> &particle, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, vec3 &sky, std::vector<COLOURMAP> &amap, float b_brightness, paramfile &g_params, cu_cpu_vars& cv);
void setup_colormap(int ptypes, std::vector<COLOURMAP> &amap, cu_gpu_vars* gv);

// NVIDIA device query functions defined in cuda_device_query.cu
int check_device(int rank);
void print_device_info(int rank, int dev);

#endif
