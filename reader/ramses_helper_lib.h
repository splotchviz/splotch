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


#ifndef RAMSES_HELPER_LIB_H
#define RAMSES_HELPER_LIB_H

#include <iostream>
#include <iterator>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include "F90_UnformattedIO.h"

//----------------------------------------------------------------------------
// Helper library for reading RAMSES data
// Tim Dykes
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// AMR file helper class
//----------------------------------------------------------------------------

class amr_file {
public:
	// Constructer calculates amr file name, opens and reads metadata
	amr_file(std::string repo, int fid)
	{
		GetFileName(repo, fid);
		file.Open(filename);

		ReadMetadata();
	}

	// Header struct
	// Not all metadata, only what is currently needed.
	struct amr_metadata {
		unsigned ncpu; // Number cpus/computational domains
		unsigned ndim; // Number spatial dimensions
		unsigned nx[3]; // Base mesh resolution (vec3)
		unsigned nlevelmax; // Max refinement levels
		unsigned ngridmax; // Max number of grids per domain
		unsigned nboundary; // Number of boundary cells (ghost cells)
		unsigned ngridcurrent; // Number of current active grids
		double boxlen; // Length of computational box
		unsigned twotondim; // two^ndim
		double xbound[3];
	};

	amr_metadata meta;
	F90_UnformattedIO file;

private:
	// Get file name
	void GetFileName(std::string repo, int fid)
	{
		// Work out filename according to ramses convention
		// Infile begins amr_
		filename = repo.substr(0,repo.size()-15)+"/amr_";

		// Append id with 5 digit precision
		std::stringstream ss;
		ss.width(5);
		ss.fill('0');
		ss << std::right << repo.substr(repo.size()-9,5); 

		// Append ext
		filename += (ss.str() + ".out"); 
		
		//Append id of current file to be read
		ss.str("");
		ss.width(5);
		ss.fill('0');
		ss << std::right << fid;
		filename += ss.str(); 
	}

	// Read metadata
	void ReadMetadata()
	{
		// Read metadata to meta
		file.ReadScalar(meta.ncpu);
		file.ReadScalar(meta.ndim);
		file.Read1DArray(meta.nx);
		file.ReadScalar(meta.nlevelmax);
		file.ReadScalar(meta.ngridmax);
		file.ReadScalar(meta.nboundary);
		file.ReadScalar(meta.ngridcurrent);
		file.ReadScalar(meta.boxlen);
		
		meta.twotondim = pow(2,meta.ndim);
		meta.xbound[0] = double(meta.nx[0])/2;
		meta.xbound[1] = double(meta.nx[1])/2;
		meta.xbound[2] = double(meta.nx[2])/2;

		// std::cout << "ncpu: " << meta.ncpu << std::endl;
		// std::cout << "ndim: " << meta.ndim << std::endl;
		// std::cout << "nx: " << meta.nx[0] << std::endl;
		// std::cout << "ny: " << meta.nx[1] << std::endl;
		// std::cout << "nz: " << meta.nx[2] << std::endl;
		// std::cout << "nlevelmax: " << meta.nlevelmax << std::endl;
		// std::cout << "ngridmax: " << meta.ngridmax << std::endl;
		// std::cout << "nboundary: " << meta.nboundary << std::endl;
		// std::cout << "ngridcurrent: " << meta.ngridcurrent << std::endl;
		// std::cout << "boxlen: " << meta.boxlen << std::endl;
	}

	// Fortran file
	std::string filename;

};

//----------------------------------------------------------------------------
// Hydro file helper class
//----------------------------------------------------------------------------
// Internal hydro variables + IDs for C1,C2,C3 parameters
// density     = 0;
// velocity_x  = 1;
// velocity_y  = 2;
// velocity_z  = 3;
// pressure    = 4;
// metallicity = 5;

class hydro_file {
public: 

	hydro_file(std::string repo, int fid)
	{
		GetFileName(repo, fid);
		file.Open(filename);
		ReadMetadata();
	}

	// Add more metadata when necessary...
	struct hydro_metadata {
		unsigned nvar;
	};
	
	hydro_metadata meta;
	F90_UnformattedIO file;

private:

	std::string filename;

	// Get file name
	void GetFileName(std::string repo, int fid)
	{
		// Work out filename according to ramses convention
		// Infile begins amr_
		filename = repo.substr(0,repo.size()-15)+"/hydro_";

		// Append id with 5 digit precision
		std::stringstream ss;
		ss.width(5);
		ss.fill('0');
		ss << std::right << repo.substr(repo.size()-9,5); 

		// Append ext
		filename += (ss.str() + ".out"); 
		
		//Append id of current file to be read
		ss.str("");
		ss.width(5);
		ss.fill('0');
		ss << std::right << fid;
		filename += ss.str(); 
	}

	void ReadMetadata() 
	{
		file.SkipRecord();
		file.ReadScalar(meta.nvar);
		file.SkipRecords(4);
	}
};

//-----------------------------------------------------------------------------------------------------------------------
// Particle file helper class
//-----------------------------------------------------------------------------------------------------------------------
// Data in particle file is in this order. Values are IDs used for red,green,blue parameters to read
// Note metallicity may or may not be in file.
// position
// vx = 0
// vy = 1
// vz = 2
// mass = 3
// identity 
// level 
// birth epoch 
// metallicity = 4 

class part_file {
public: 

	part_file(std::string repo, int fid)
	{
		GetFileName(repo, fid);
		file.Open(filename);
		ReadMetadata();
	}

	// Add more metadata when necessary...
	struct part_metadata {
			int 	ncpu;
			int 	ndim;
			int 	npart;
			// Other metadata unnecessary...
			// Real(16) localseed 
			// int nstar_tot;
			// double mstar_tot;
			// double mstar_lost;
			// int nsink;
	};
	
	part_metadata meta;
	F90_UnformattedIO file;

private:

	std::string filename;

	// Get file name
	void GetFileName(std::string repo, int fid)
	{
		// Work out filename according to ramses convention
		// Infile begins amr_
		filename = repo.substr(0,repo.size()-15)+"/part_";

		// Append id with 5 digit precision
		std::stringstream ss;
		ss.width(5);
		ss.fill('0');
		ss << std::right << repo.substr(repo.size()-9,5); 

		// Append ext
		filename += (ss.str() + ".out"); 
		
		//Append id of current file to be read
		ss.str("");
		ss.width(5);
		ss.fill('0');
		ss << std::right << fid;
		filename += ss.str(); 
	}

	void ReadMetadata() 
	{
		file.ReadScalar(meta.ncpu);
		file.ReadScalar(meta.ndim);
		file.ReadScalar(meta.npart);
		file.SkipRecords(5);
	}
};

//----------------------------------------------------------------------------
// Info file helper class
//----------------------------------------------------------------------------

class info_file {
public: 
	info_file() {}
	info_file(std::string repo)
	{
		Parse(repo);
	}

	// Parse info file. Arg0 is path of repository where info file is located
	void Parse(std::string repo)
	{
		getFileName(repo);
		std::ifstream infile(filename.c_str(), std::ios::in);
		if(!infile)
			std::cout << "Cannot open info file: " << filename  << std::endl;
		
		std::string scratch;
		std::getline(infile, scratch);	
		Read_rhs(scratch, ncpu);

		std::getline(infile, scratch);
		Read_rhs(scratch, ndim);

		std::getline(infile,scratch);
		std::getline(infile, scratch);
		Read_rhs(scratch, levelmin);

		for(int i = 0; i < 6; i++)
			std::getline(infile, scratch);

		Read_rhs(scratch, time);

		for(int i = 0; i < 11; i++)
			std::getline(infile, scratch);
		// Get ordering
		std::string str;
		Read_rhs(scratch, str);
		if(str.size() < 5)
		{
			std::cout << "Info file formatting incorrect, aborting." << std::endl;
			exit(0);
		}
		ordering = str.substr(5,str.size()-5);
		// If ordering is hilbert, read in domain boundaries
		if(ordering == "hilbert") {
			std::cout << "Hilbert ordering, reading domains..." << std::endl;
			std::getline(infile, scratch); // skip line
			// For each line read in domain index  minimum and maximum
			for(unsigned i = 0; i < ncpu; i++) {
				std::getline(infile, scratch);
				std::stringstream sss(scratch);
				int n;
				double min,max;
				sss >> n >> min >> max;
				ind_min.push_back(min);
				ind_max.push_back(max);
			}
			std::cout << ncpu << " domain boundaries read" << std::endl;
		}
		infile.close();
	}

	// Not everything, only what is currently needed
	unsigned ncpu;// Number of cpus used in simulation
	unsigned ndim;
	unsigned levelmin; // Minimum refinement level
	double time; // Snapshot timestamp
	std::string ordering;
	std::vector<double> ind_min; // Minimum domain boundaries for hilbert ordering as listed in info file
	std::vector<double> ind_max; // Maximum domain boundaries for hilbert ordering as listed in info file

	std::string filename;

private:

	void getFileName(std::string repo)
	{
		// Work out filename according to ramses convention

		// Infile begins amr_
		// filename = repo+"/info_";	

		// // Append id with 5 digit precision
		// std::stringstream ss;
		// ss.width(5);
		// ss.fill('0');
		// ss << std::right << repo.substr(repo.size()-9,5); 

		// // Append ext
		// filename += (ss.str() + ".txt"); 

		// We now use info file as the input so no need to work out filename
		filename = repo;
	}
	
	// Read the rightmost piece of data in a string to any type that can be converted via stringstream
	template<typename T>
	void Read_rhs(std::string line, T& data)
	{
		// Tokenise
		std::string buf;
		std::stringstream ss(line);
		std::vector<std::string> tokens;
		while(ss >> buf)
			tokens.push_back(buf);

		// Convert rhs to type and return 
		std::stringstream sss(tokens.back());
		sss >> data;
	}


};


#endif
