/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
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

#ifndef MIC_SPLOTCH_H
#define MIC_SPLOTCH_H


#include "cxxsupport/paramfile.h"
#include "cxxsupport/lsconstants.h"
#include "kernel/colour.h"
#include "kernel/transform.h"
#include "splotch/splotchutils.h"
#include "mic/mic_kernel.h"
#include "mic/mic_utils.h"
 
#include <vector>
#include <algorithm>
#include <stdio.h>
 
#pragma offload_attribute(push, target(mic))
//#ifdef __MIC__
//#include "papi/papi_wrapper.h"
//#endif
#pragma offload_attribute(pop)

void mic_init_offload();
void mic_allocate(mic_soa_particles&, std::vector<particle_sim>&, paramfile&);
void mic_rendering(paramfile&, std::vector<particle_sim>&, arr2<COLOUR>&, const vec3&, const vec3&,const vec3&, const vec3&, std::vector<COLOURMAP>&, float, mic_soa_particles&, bool);
const float* compute_transform(paramfile&, transform_data&, const vec3&, const vec3&, const vec3&, vec3); 
void compute_colormap(paramfile&, int, float*, bool*, float, std::vector<COLOURMAP>&, mic_color_map&);
void mic_free(mic_soa_particles&,int,int,paramfile&);
#endif