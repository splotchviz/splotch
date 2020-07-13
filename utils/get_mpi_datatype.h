#ifndef TESTBED_GET_MPI_DATATYPE_H
#define TESTBED_GET_MPI_DATATYPE_H

#include <mpi.h>
#include <stdio.h>
#include <assert.h>

// Catch other case
template<typename T>
inline MPI_Datatype get_MPI_Datatype() { printf("get_MPI_Datatype() caleld with unsupported type\n"); assert(0); }

template<>
inline MPI_Datatype get_MPI_Datatype<char>() { return MPI_CHAR; }

template<>
inline MPI_Datatype get_MPI_Datatype<unsigned char>() { return MPI_UNSIGNED_CHAR; }

// ??
//template<>
//inline MPI_Datatype get_MPI_Datatype<unsigned char>() { return MPI_BYTE; }

template<>
inline MPI_Datatype get_MPI_Datatype<short>() { return MPI_SHORT; }

template<>
inline MPI_Datatype get_MPI_Datatype<unsigned short>() { return MPI_UNSIGNED_SHORT; }

template<>
inline MPI_Datatype get_MPI_Datatype<unsigned>() { return MPI_UNSIGNED; }

template<>
inline MPI_Datatype get_MPI_Datatype<long>() { return MPI_LONG; }

template<>
inline MPI_Datatype get_MPI_Datatype<unsigned long>() { return MPI_UNSIGNED_LONG; }

template<>
inline MPI_Datatype get_MPI_Datatype<float>() { return MPI_FLOAT; }

template<>
inline MPI_Datatype get_MPI_Datatype<double>() { return MPI_DOUBLE; }

template<>
inline MPI_Datatype get_MPI_Datatype<long double>() { return MPI_LONG_DOUBLE; }

template<>
inline MPI_Datatype get_MPI_Datatype<long long int>() { return MPI_LONG_LONG_INT; }




#endif