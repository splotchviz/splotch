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


#ifndef F90_UNFORMATTEDIO_H
#define F90_UNFORMATTEDIO_H
#include "cxxsupport/mpi_support.h"
//----------------------------------------------------------------------------
// Helper library for I/O with Fortran unformatted write files
// Tim Dykes
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Simple templates for 2/3D arrays used for reading file
//----------------------------------------------------------------------------
template<typename T>
class F90_Arr2D{
public: 
	F90_Arr2D() { arr = NULL;}
	void resize(unsigned x, unsigned y)
	{
		xdim = x;
		ydim = y;
		arr = new T[x*y];
	}

	void Delete(){ if(arr != NULL) delete[] arr;}

	T& operator ()(unsigned x, unsigned y) {return arr[x*ydim+y]; }

	unsigned xdim;
	unsigned ydim;

private: 
	T* arr;
};

template<typename T>
class F90_Arr3D{
public: 
	F90_Arr3D() { arr = NULL;}
	void resize(unsigned x, unsigned y, unsigned z)
	{
		xdim = x;
		ydim = y;
		zdim = z;
		arr = new T[x*y*z];
	}

	void Delete(){ if(arr != NULL) delete[] arr;}

	T& operator ()(unsigned x, unsigned y, unsigned z) {return arr[(x*ydim+y)*zdim+z]; }

	unsigned xdim;
	unsigned ydim;
	unsigned zdim;

private: 
	T* arr;
};


//----------------------------------------------------------------------------
// Unformatted file class, allows to read scalar and unknown size 1D arrays
// 2D and 3D arrays must be of a known size
// Allows skipping n records of any size (equivalent to READ(unit) ) 
//----------------------------------------------------------------------------

// Size of prefix and suffix record delimiters in bytes
// These delimiters indicate bytelength of record
#define PREPOST_DATA 4


class F90_UnformattedIO {
public:
	F90_UnformattedIO() {}

	~F90_UnformattedIO() {	file.close(); }

	// Open fortran unformatted file
	void Open(std::string filename) 
	{
		file.open(filename.c_str(), std::ios::in);
		if(!file.is_open())
			std::cout << "Failed to open fortran file: " << filename << std::endl;
	}

	template <typename T> 
	void ReadScalar(T& scalar)
	{
		// Read prefix, data and suffix. Check prefix and suffix = sizeof data expected
		unsigned pre, post;

		file.read((char*)&pre, PREPOST_DATA);
		file.read((char*)&scalar,sizeof(T));
		file.read((char*)&post,PREPOST_DATA);

		if((pre != sizeof(T)) || (pre != post)) {
			std::cout << "Failed scaler read from fortran file: pre != post" << std::endl;
		}
	}

	// Read 1d array of unknown size
	template <typename T>
	void Read1DArray(T* arr)
	{
		unsigned pre, post;
		file.read((char*)&pre, PREPOST_DATA);

		file.read((char*)&arr[0], pre);

		file.read((char*)&post, PREPOST_DATA);
		if(pre!=post)
		{
			if(mpiMgr.master())
				std::cout << "Failed read fortran 1d array: pre != post"<< std::endl;
		}
	}

	// Read subsection of 1D array
	// firstelement and numelements are in terms of array elements of size T
	template <typename T>
	void Read1DArray(T* arr, unsigned firstelement, unsigned numelements)
	{
		unsigned pre, post;
		file.read((char*)&pre, PREPOST_DATA);

		// Validate
		if((firstelement + numelements) > (pre/sizeof(T)))
		{
			std::cout << "Failed read fortran 1d array subsection: Requested more elements than array contains; from rank: " << mpiMgr.rank() << std::endl;
		}

		// Skip to start record
		if(firstelement>0)
			file.seekg(firstelement*sizeof(T),std::ios::cur);

		file.read((char*)&arr[0],numelements*sizeof(T));

		//Skip to end
		if(((firstelement+numelements)*sizeof(T))<pre)
			file.seekg((pre - ((firstelement+numelements)*sizeof(T))),std::ios::cur);

		file.read((char*)&post, PREPOST_DATA);
		if(pre!=post)
		{
			std::cout << "Failed read fortran 1d array subsection: pre != post; from rank: " << mpiMgr.rank() << std::endl;
		}
	}

	// Read 2d array of predefined size arr[xdim][ydim]
	// Note: F90_Arr2D/3D must be resized to the expected read size before reading.
	// This is due to no indication within record as to multidimensional array length
	template <typename T>
	void Read2DArray(F90_Arr2D<T> arr)
	{
		unsigned pre, post;
		file.read((char*)&pre, PREPOST_DATA);
		
		for(unsigned n = 0; n < arr.ydim; n++)
			for(unsigned m = 0; m < arr.xdim; m++) 
				file.read((char*)&arr(m,n), sizeof(T));

		file.read((char*)&post, PREPOST_DATA);
		if(pre!=post)
			std::cout << "Failed read fortran 2d array: pre != post" << std::endl;
	}

	// Read 3d array of predefined size arr[xdim][ydim][zdim]
	template <typename T>
	void Read3DArray(F90_Arr3D<T> arr)
	{
		unsigned pre, post;
		file.read((char*)&pre, PREPOST_DATA);

		for(unsigned l = 0; l < arr.zdim; l++)
			for(unsigned n = 0; n < arr.ydim; n++)
				for(unsigned m = 0; m < arr.xdim; m++) {
					file.read((char*)&arr(m,n,l), sizeof(T));
				}

		file.read((char*)&post, PREPOST_DATA);

		if(pre!=post)
			std::cout << "Failed read fortran 3d array: pre != post" << std::endl;
	}

	// Skip single record (scalar or array)
	void SkipRecord()
	{
		unsigned pre, post;
		try{
			file.read((char*)&pre, PREPOST_DATA);
			file.seekg(pre,std::ios::cur);
			file.read((char*)&post,PREPOST_DATA);
		}catch(...){
			throw std::runtime_error("Error seeking end of record");
		}

		if(pre != post)
			std::cout << "Failed record skip: pre != post" << std::endl;
	}

	// Skip N records (scalar or array)
	void SkipRecords(unsigned n)
	{
		for(unsigned i = 0; i < n; i++)
		{
			unsigned pre, post;
			try{
				file.read((char*)&pre, PREPOST_DATA);
				file.seekg(pre,std::ios::cur);
				file.read((char*)&post,PREPOST_DATA);
			}catch(...){
				throw std::runtime_error("Error seeking end of record");
			}

			if(pre != post)
				std::cout << "Failed record skip: pre != post" << std::endl;
		}
	}

	// Peek at next data item without moving on
	// Can be used to check delimiter contents
	template <typename T>
	void Peek(T& data)
	{
		file.read((char*)&data, sizeof(T));
		file.seekg(-sizeof(T),std::ios::cur);
	}

private:
	std::ifstream file;

};


#endif
