#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# ============================================================================
# Name        : configure.ac
# Author      : Gian Franco Marras
# Version     : 1.0
# Copyright   : free
# Description : A simple MPI Library I/O for access file in parallel system
# ============================================================================

AC_PREREQ(2.59)
AC_INIT(mpiio, 1.0, [g.marras@cineca.it,m.rivi@cineca.it])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_CONFIG_SRCDIR([src/partition.c])
AC_CONFIG_HEADER([config.h])

# Checks for programs.
AC_PROG_CC

#enable debug and check mpi compiler
AC_ENABLE_DISABLE



# Checks for libraries.
AC_PROG_RANLIB

# FIXME: Replace `main' with a function in `-lm':
AC_CHECK_LIB([m], [main])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.
AC_FUNC_MALLOC

pwd=$(pwd)
AC_SUBST(pwd)
#AC_PREFIX_DEFAULT([.])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 man/Makefile])
AC_OUTPUT
