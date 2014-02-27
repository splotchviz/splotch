# ============================================================================
# Name        : Makefile.am
# Author      : Gian Franco Marras
# Version     : 1.0
# Copyright   : free
# Description : A simple MPI Library I/O for access file in parallel system
# ============================================================================
m4dir = ${top_srcdir}/m4


libpartition_a_CPPFLAGS = $(enable_mpi) $(enable_debug) $(enable_io)
lib_LIBRARIES = libpartition.a
libpartition_a_SOURCES = partition.c
include_HEADERS = partition.h


if DEBUG
enable_debug=-DDEBUG
endif