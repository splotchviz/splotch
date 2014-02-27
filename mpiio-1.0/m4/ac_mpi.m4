# ============================================================================
# Name        : partition.c
# Author      : Gian Franco Marras
# Version     : 1.0
# Copyright   : free
# Description : A simple MPI Library I/O for access file in parallel system
# ============================================================================

AC_DEFUN([AC_MPI],
[
# Check for MPI

AC_ARG_VAR(MPICC,[MPI C compiler command])
AC_CHECK_PROGS(MPICC, mpicc hcc mpxlc_r mpxlc mpcc cmpicc, 0)
if test "x0" = "x$MPICC"
        then
	 AC_MSG_NOTICE([=========================================================================])
         AC_MSG_NOTICE([ Cannot find MPICC compiler])
         AC_MSG_NOTICE([ Set MPICC variable or load the specific module])
         AC_MSG_NOTICE([ For more information, type the command: ./configure --help])
	 AC_MSG_NOTICE([=========================================================================])
         AC_MSG_ERROR([MPICHECK compiler failed])
fi
AC_SUBST(MPICC)


AC_ARG_VAR(MPICXX,[MPI C++ compiler command])
AC_CHECK_PROGS(MPICXX, mpic++ mpicxx mpiCC hcp mpxlC_r mpxlC mpCC cmpic++, 0)
if test "x0" = "x$MPICXX"
        then
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_NOTICE([ Cannot find MPICXX compiler])
         AC_MSG_NOTICE([ Set MPICXX variable or load the specific module])
         AC_MSG_NOTICE([ For more information, type the command: ./configure --help])
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_ERROR([MPICHECK compiler failed])
fi
AC_SUBST(MPICXX)


AC_ARG_VAR(MPIF77,[MPI F77 compiler command])
AC_CHECK_PROGS(MPIF77, mpif77 hf77 mpxlf_r mpxlf mpf77 cmpifc, 0)
if test "x0" = "x$MPIF77"
        then
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_NOTICE([ Cannot find MPIF77 compiler])
         AC_MSG_NOTICE([ Set MPIF77 variable or load the specific module])
         AC_MSG_NOTICE([ For more information, type the command: ./configure --help])
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_ERROR([MPICHECK compiler failed])
fi
AC_SUBST(MPIF77)


AC_ARG_VAR(MPIFC,[MPI Fortran compiler command])
AC_CHECK_PROGS(MPIFC, mpif90 mpxlf95_r mpxlf90_r mpxlf95 mpxlf90 mpf90 cmpif90c, $FC)
if test "x0" = "x$MPIFC"
        then
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_NOTICE([ Cannot find MPIFC compiler])
         AC_MSG_NOTICE([ Set MPIFC variable or load the specific module])
         AC_MSG_NOTICE([ For more information, type the command: ./configure --help])
         AC_MSG_NOTICE([=========================================================================])
         AC_MSG_ERROR([MPICHECK compiler failed])
fi
AC_SUBST(MPIFC)


])