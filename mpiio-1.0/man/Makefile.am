



.1in.1:
	@ echo Creating $@ man page...
	@ sed -e 's/#PACKAGE_NAME#/@PACKAGE_NAME@/g' \
	  -e 's/#PACKAGE_VERSION#/@PACKAGE_VERSION@/g' \
	  -e 's/#OMPI_DATE#/@OMPI_RELEASE_DATE@/g' \
	  > $@ < $<


mpi_man_pages = MPI_Ajo_read.1 MPI_Ajo_write.1 MPI_Ajo_partition.1

nodist_man_MANS = $(mpi_man_pages)
EXTRA_DIST = $(mpi_man_pages:.3=.3in)