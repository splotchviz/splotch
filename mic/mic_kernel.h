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

#ifndef MIC_KERNEL_H
#define MIC_KERNEL_H

#include "mic/mic_utils.h"
#include <mathimf.h>

#pragma offload_attribute(push, target(mic))

void prepAllocators(int,int,int,int,int);
void prepDevicePics(int,int,int);
void rototranslate(int,float*, float*, float*, float*, float*, short*, bool*, transform_data&, const float*);
void colorize(int, float*,float*, float*, float*, short*, bool*, mic_color_map, float*, bool*);
void render(int, float*, float*, float*, float*, float*, float*, float*, bool*, int, int, mic_color*, render_data&);

#pragma offload_attribute(pop)

#endif
