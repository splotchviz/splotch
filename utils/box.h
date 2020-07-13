#ifndef TESTBED_COMMON_BOX_H
#define TESTBED_COMMON_BOX_H


#ifdef USE_MPI
#include <mpi.h>
#endif

#include "debug.h"
#include <vector>

// This box holds 1 element T per field for maximum values, and 1 element T for minimum values
// Box size in bytes is 2*nFields*sizeof(T)
template<typename T, int nFields>
struct Box{
	T min[nFields];
	T max[nFields];
};

// get bounding values for all fields in T (must be able to access T fields with array subscript)
template<typename T, int nFields>
Box<T,nFields> getbox(const std::vector<T>& in)
{
	Box<T, nFields> bb;
	// Set initial box values to first item
	for(unsigned j = 0; j < nFields; j++)
		bb.min[j] = in[0];
	for(unsigned j = 0; j < nFields; j++)
		bb.max[j] = in[0];

	for(unsigned i = 0; i < in.size(); i++)
	{
		for(unsigned j = 0; j < nFields; j++)
			bb.min[j][j] = (in[i][j] < bb.min[j][j]) ? in[i][j] : bb.min[j][j];

		for(unsigned j = 0; j < nFields; j++)
			bb.max[j][j] = (in[i][j] > bb.max[j][j]) ? in[i][j] : bb.max[j][j];
	}
	return bb;
}

// Get bounding box
template<typename T, typename OP, int nFields>
void getbox(Box<T,nFields>& bb, const T* in, unsigned int size, OP* is_smaller)
{
	// Set initial box values to first item
	for(unsigned j = 0; j < nFields; j++)
		bb.min[j] = in[0];

	for(unsigned j = 0; j < nFields; j++)
		bb.max[j] = in[0];

	for(unsigned i = 0; i < size; i++)
	{
		// Call is smaller predicate for each field 
		for(unsigned j = 0; j < nFields; j++)
			bb.min[j] = (is_smaller[j](in[i], bb.min[j])) ? in[i] : bb.min[j];

		for(unsigned j = 0; j < nFields; j++)
			bb.max[j] = (is_smaller[j]( bb.max[j], in[i])) ? in[i] : bb.max[j];
	}
}


#ifdef USE_MPI

// As above but in parallel
template<typename T, typename OP, int nFields>
void getbox(Box<T,nFields>& bb, const T* in, unsigned int size, OP* is_smaller, int rank, int root, int comm_size)
{
	// Get local bounding box
	// Set initial box values to first item
	for(int j = 0; j < nFields; j++)
		bb.min[j] = in[0];

	for(int j = 0; j < nFields; j++)
		bb.max[j] = in[0];

	for(unsigned i = 0; i < size; i++)
	{
		// Call is smaller predicate for each field 
		for(int j = 0; j < nFields; j++)
			bb.min[j] = (is_smaller[j](in[i], bb.min[j])) ? in[i] : bb.min[j];

		for(int j = 0; j < nFields; j++)
			bb.max[j] = (is_smaller[j]( bb.max[j], in[i])) ? in[i] : bb.max[j];
	}

	// Get global bounding box+
	Box<T, nFields>* all_boxes = NULL;
	// If root, allocate some memory for storing boxes of these ranks
	if(rank==root)
		all_boxes = new Box<T, nFields>[comm_size];


	// Cast template box to char array and send to root
	int n_bytes = sizeof(Box<T,nFields>);
	int ret =  MPI_Gather(&bb, n_bytes, MPI_UNSIGNED_CHAR, all_boxes, n_bytes, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
	if(ret != MPI_SUCCESS)
	{
		ErrorMessage("getbox(): MPI_Gather() failed, error code %i\n", ret);
	}

	// Root work out global mins and maxes
	if(rank==root)
	{
		for(int i = 0; i < comm_size; i++)
		{
			for(int j = 0; j < nFields; j++)
				bb.min[j] = (is_smaller[j](all_boxes[i].min[j], bb.min[j])) ? all_boxes[i].min[j] : bb.min[j];

			for(int j = 0; j < nFields; j++)
				bb.max[j] = (is_smaller[j](bb.max[j], all_boxes[i].max[j])) ? all_boxes[i].max[j] : bb.max[j];				
		}
	}

	// Broadcast this back to everyone else
	ret = MPI_Bcast(&bb, n_bytes, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
	if(ret != MPI_SUCCESS)
	{
		ErrorMessage("getbox(): MPI_Bcast() failed, error code %i\n", ret);
	}
}


#endif


#endif