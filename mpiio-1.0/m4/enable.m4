# ============================================================================
# Name        : enable.m4
# Author      : Gian Franco Marras
# Version     : 1.0
# Copyright   : free
# Description : A simple MPI Library I/O for access file in parallel system
# ============================================================================

AC_DEFUN([AC_ENABLE_DISABLE],
[
#####################################################################################
#			  ENABLE    /   DISABLE
#####################################################################################
 
#####################################################################################
## ENABLE/DISABLE DEBUG
AC_ARG_ENABLE(debug,AC_HELP_STRING([--enable-debug=yes|no],[Enable support for debugging (default=no)]),
  [enable_debug=$enableval],
  [enable_debug=no]
)
if test "$enable_debug" = "yes"; then
AC_MSG_NOTICE([===============])
AC_MSG_NOTICE([  ENABLE DEBUG])
AC_MSG_NOTICE([===============])
CFLAGS=" -O0 -g -Wall"
CXXFLAGS=" -O0 -g -Wall"
FFLAGS=" -O0 -g"
FCFLAGS=" -O0 -g"
fi
AM_CONDITIONAL([DEBUG], [test $enable_debug = yes])

#####################################################################################
## CHECK MPI COMPILER
AC_MPI
CC=$MPICC

])