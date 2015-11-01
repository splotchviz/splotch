/*
 * Copyright (c) 2004-2015
 *              Tim Dykes University of Portsmouth
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
 * Ascii file reader
 *
 * Currently only for Marzia/marc files
	Parameters:
	C1=0-7
	C2=-1-3
	C3=-1-3

	I=-1-3


	-1 = non existant (only for C2 & C3, if color is not vector)
	0 = gDES
	1 = rRES
	2 = iDES
	3 = zDES
	4 = yDES
	5 = gDES-rDES
	6 = gDES-iDES
	7 = gDES-zDES
 */




#include <cstdio>
#include <vector>
#include <string>

#include "reader.h"
#include "splotch/splotchutils.h"

//#define MARC_FORMAT2


 void ascii_reader(paramfile &params, std::vector<particle_sim> &points)
 {
 	// Get infile
 	std::string filename = params.find<std::string>("infile");

 	// Open file
 	FILE* infile = NULL;
 	infile = fopen(filename.c_str(), "r");
 	planck_assert(infile, "Failed to open file "+filename);

 	// How many values per line
 	#ifdef MARC_FORMAT2
 	int nqty = 8;
 	#else
 	int nqty = 7;//params.find<int>("nqty");
	#endif

 	// Which quanitites do we want
  	int qty_idx[4];

	qty_idx[0] = params.find<int>("C1");
	qty_idx[1] = params.find<int>("C2",-1);
	qty_idx[2] = params.find<int>("C3",-1);
	qty_idx[3] = params.find<int>("I",-1);

	//for(unsigned i = 1; i < 3; i++)
	//	planck_assert((qty_idx[i] > -2 && qty_idx[i] < 4), "Ascii reader: accepted value for C2&C3 -1-3");

	// Sampling
	// Sample factor is read in as percentage then converted to factor
	bool doSample = params.find<bool>("sampler",false);
	float sample_factor = params.find<float>("sample_factor",100);
	if(doSample)
	{
		if(sample_factor < 0 || sample_factor > 100)
		{
			if(mpiMgr.master())
				std::cout << "Invalid sample factor: " << sample_factor << "\n Use a percentage to sample, ie sample_factor=50\n";
			planck_fail("Ascii reader sampling fail");
		}
		else
		{
			if(mpiMgr.master())
			{
				std::cout << "Ascii reader sampling to " << sample_factor << std::endl;
			}
		}
	}

	// Set defaults of input p
	particle_sim p;
	p.r = params.find<int>("r",1);
	p.I = params.find<int>("I_force",1);
	p.type = 0;

	int count = 0;
	int np = 0;
	double xyz[3];
	double griz[8];

	while(1)
	{
		// Sample based on count
		if(count%100 > sample_factor)
		{
			continue;
		}

		#ifdef MARC_FORMAT2
		// id, x, y z, id, g, r, i, z, y
		int ret = fscanf(infile, "%*u %lf %lf %lf %*u %lf %lf %lf %lf %lf",
								 &xyz[0], &xyz[1], &xyz[2], &griz[0],&griz[1],&griz[2],&griz[3],&griz[4]);
		#else
		// Read particle
		//gDES,gDESerror,rDES,rDESerror,iDES,iDESerror,zDES,zDESerror,YDES,YDESerror,ebv_cos);
		int ret = fscanf(infile, "%*d %*u %*u %*f %*f %*f %*f %*f %*f %*f "
								 "%*f %*f %*f %*f %*f %lf %lf %lf %*f %*f "
								 "%*f %*f %*f %*f %*d %*d %*f %*f %lf %*f "
								 "%lf %*f %lf %*f %lf %*f %*f %*f %*f",
								 &xyz[0], &xyz[1], &xyz[2], &griz[0],&griz[1],&griz[2],&griz[3]);
		#endif

		if(feof(infile))
			break;
		if(ret != nqty)
		{
			std::cout << "nqty: " << nqty << " ret: " << ret << std::endl;
			planck_fail("Ascii reader: incorrect file format");
		}

		if(qty_idx[0] > 3)
		{
			griz[5] = griz[0] - griz[1];
			griz[6] = griz[0] - griz[2];
			griz[7] = griz[0] - griz[3];
		}

		p.x = (float)xyz[0];
		p.y = (float)xyz[1];
		p.z = (float)xyz[2];
		p.e.r = (float)griz[qty_idx[0]];
		if(qty_idx[1] > -1) p.e.g = (float)griz[qty_idx[1]];
		if(qty_idx[2] > -1) p.e.b = (float)griz[qty_idx[2]];

		if(qty_idx[3] > -1) p.I = (float)griz[qty_idx[3]];

		points.push_back(p);
		np++;

	}

	std::cout << "Ascii reader: read " << np << " points" << std::endl;

 	fclose(infile);
 }
