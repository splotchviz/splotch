#######################################################################
#  Splotch V6                                                      #
#######################################################################

#--------------------------------------- Turn off Intensity  normalization
#OPT += -DNO_I_NORM

#--------------------------------------- Switch on DataSize
#OPT += -DLONGIDS

#--------------------------------------- Switch on HDF5
#OPT += -DHDF5
#OPT += -DH5_USE_16_API

#---------------------------------------- Enable FITSIO
#OPT += -DFITS

#--------------------------------------- Switch on MPI
#OPT += -DUSE_MPI
##OPT += -DUSE_MPIIO

#--------------------------------------- CUDA options
#OPT += -DCUDA
#OPT += -DHYPERQ

#--------------------------------------- Turn on VisIVO stuff
#OPT += -DSPLVISIVO

#--------------------------------------- Visual Studio Option
#OPT += -DVS

#--------------------------------------- MPI support for A!=E
#OPT += -DMPI_A_NEQ_E

#--------------------------------------- Independent absorption 
#OPT += -DINDEPENDENT_A

#--------------------------------------- Client server model
#OPT += -DCLIENT_SERVER
# Uncomment this to request a username on load
#OPT += -DSERVER_USERNAME_REQUEST

#--------------------------------------- Select target Computer
SYSTYPE="generic"
#SYSTYPE="mac"
#SYSTYPE="Linux-cluster"
#SYSTYPE="DAINT"
#SYSTYPE="GSTAR"
#SYSTYPE="SuperMuc"
#SYSTYPE="XC30-CCE"
#SYSTYPE="tiger"
### visualization cluster at the Garching computing center (RZG):
#SYSTYPE="RZG-SLES11-VIZ"
### generic SLES11 Linux machines at the Garching computing center (RZG):
#SYSTYPE="RZG-SLES11-generic"

# Set compiler executables to commonly used names, may be altered below!
ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
 CC       = mpic++
else
 CC       = g++
endif

# OpenMP compiler switch
#OMP      = -fopenmp

SUP_INCL = -I. -Icxxsupport -Ic_utils -Ivectorclass

# optimization and warning flags (g++)
OPTIMIZE = -Ofast -std=c++11 -pedantic -Wno-long-long -Wfatal-errors           \
           -Wextra -Wall -Wstrict-aliasing=2 -Wundef -Wshadow -Wwrite-strings  \
           -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align     \
           -Wpointer-arith -march=native 

# MPIIO library 
ifeq (USE_MPIIO,$(findstring USE_MPIIO,$(OPT)))
 SUP_INCL += -Impiio-1.0/include/
 LIB_MPIIO = -Lmpiio-1.0/lib -lpartition
endif

# Default paths for client-server dependencies 
ifeq (CLIENT_SERVER,$(findstring CLIENT_SERVER,$(OPT)))
    # LibWebsockets
    LWS_PATH = server/dep/LibWebsockets
    LIB_OPT +=  -L$(LWS_PATH)/lib -lwebsockets
    SUP_INCL += -I$(WSP_PATH)/include -I$(LWS_PATH)/include
    # LibTurboJPEG
    LIBTURBOJPEG_PATH = server/dep/libjpeg-turbo
    # RapidJSON
    RAPIDJSON_PATH 		= server/dep
    # WSRTI
    WSRTI_PATH = server/dep/WSRTI
  endif


#--------------------------------------- Specific configs per system type

ifeq ($(SYSTYPE),"generic")
  # OPTIMIZE += -O2 -g -D TWOGALAXIES
  OPTIMIZE += -O0 -g

  # Generic 64bit cuda setup
  ifeq (CUDA,$(findstring CUDA,$(OPT)))
  NVCC       =  nvcc
  NVCCARCH = -arch=sm_30
  NVCCFLAGS = -g  $(NVCCARCH) -dc -std=c++11
  CUDA_HOME  =  /opt/nvidia/cudatoolkit/default 
  LIB_OPT  += -L$(CUDA_HOME)/lib64 -lcudart
  SUP_INCL += -I$(CUDA_HOME)/include
  endif
endif


# NOTE: This is for osx >=10.9 i.e. clang not gcc
# Openmp isnt supported by default, so you must modify the paths to point to your
# build of clang with openmp support and an openmp runtime library as done below
# Dont forget to export DYLD_LIBRARY_PATH for the omp and cuda runtime libs
ifeq ($(SYSTYPE),"mac")
  OPT += -DSPLOTCHMAC
  ifeq (CUDA,$(findstring CUDA,$(OPT)))
	#CC = CC
	NVCC       = nvcc
	NVCCARCH   = -arch=sm_30
	CUDA_HOME  = /Developer/NVIDIA/CUDA-7.0/
	OPTIMIZE   = -Wall -stdlib=libstdc++ -Wno-unused-function -Wno-unused-variable -Wno-unused-const-variable
	LIB_OPT   += -L$(CUDA_HOME)/lib -lcudart
	NVCCFLAGS  = -g -ccbin /usr/bin/clang -dc -$(NVCCARCH)
	SUP_INCL  += -I$(CUDA_HOME)/include
	ifeq (-fopenmp,$(OMP))
		# Modify ccbin argument to point to your OpenMP enabled clang build (note clang++, this is important)
		NVCCFLAGS = -g -ccbin /Users/tims/Programs/clang+llvm-3.9.0-x86_64-apple-darwin/bin/clang++ -dc -$(NVCCARCH)
	endif
  endif
  ifeq (-fopenmp,$(OMP))
	# Change this as above
	CLANG_PATH = /Users/tims/Programs/clang+llvm-3.9.0-x86_64-apple-darwin/bin/
	OMP_RUNTIME_PATH = /Users/tims/Programs/Intel-OMP-RT/libomp_oss/exports/mac_32e/lib.thin/
	CC = $(CLANG_PATH)/clang++
	OPTIMIZE = -O3 -Wall -std=c++11 -D__apple_build_version__ 
	OMP = -fopenmp=libiomp5
	LIB_OPT += -L$(OMP_RUNTIME_PATH)
  endif
  ifeq (PREVIEWER,$(findstring PREVIEWER,$(OPT)))
	SUP_INCL += -I/opt/X11/include
  endif
  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
    	CC = mpic++
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
    HDF5_HOME = /usr/local
    LIB_HDF5  = -L$(HDF5_HOME)/lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
    HDF5_INCL = -I$(HDF5_HOME)/include
  endif
  ifeq (FITS,$(findstring FITS,$(OPT)))
  	LIB_OPT  +=  -L/Users/tims/Code/cfitsio/lib
  	SUP_INCL += -I/Users/tims/Code/cfitsio/include
  endif
endif


ifeq ($(SYSTYPE),"Linux-cluster")
  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC  =  mpiCC -g
  else
   CC  = g++
  endif
  OPTIMIZE += -O2 
  OMP = -fopenmp
  ifeq (CUDA,$(findstring CUDA,$(OPT)))
  CUDA_HOME = /usr/local/cuda/
  NVCC = nvcc
  NVCCARCH = -arch=sm_30
  NVCCFLAGS = -g  $(NVCCARCH) -dc -use_fast_math -std=c++11
  LIB_OPT  =  -L$(CUDA_HOME)/lib64 -lcudart
  SUP_INCL += -I$(CUDA_HOME)/include
  endif
endif

# Configuration for PIZ DAINT at CSCS
# Can be used for generic XC30 with GNU, note -march=broadwell in OPTIMIZE though
ifeq ($(SYSTYPE), "DAINT")
  CC = CC
  OPTIMIZE = -Ofast  -std=c++11 -pedantic -Wno-long-long -Wfatal-errors -Wextra -Wall -Wstrict-aliasing=2 -Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -march=native
  ifeq (CUDA, $(findstring CUDA, $(OPT)))
    #CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/7.028-1.0502.10742.5.1
    CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/default
    NVCC = nvcc
    NVCCARCH = -arch=sm_35
    NVCCFLAGS = -g  $(NVCCARCH) -dc -use_fast_math --std=c++11 -ccbin=CC
    LIB_OPT  += -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
    SUP_INCL += -I$(CUDATOOLKIT_HOME)/include
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
    HDF5_HOME = /opt/cray/hdf5-parallel/1.8.13/gnu/48/
    LIB_HDF5  = -L$(HDF5_HOME)lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
    HDF5_INCL = -I$(HDF5_HOME)include
  endif
endif

# XC40 with GPUs
ifeq ($(SYSTYPE), "tiger")
  CC = CC
  OPTIMIZE = -Ofast  -std=c++11 -pedantic -Wno-long-long -Wfatal-errors -Wextra -Wall -Wstrict-aliasing=2 -Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -march=native
  ifeq (CUDA, $(findstring CUDA, $(OPT)))
    #CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/7.028-1.0502.10742.5.1
    CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/default
    NVCC = nvcc
    NVCCARCH = -arch=sm_35
    NVCCFLAGS = -g  $(NVCCARCH) -dc -use_fast_math --std=c++11 -ccbin=CC
    LIB_OPT  += -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
    SUP_INCL += -I$(CUDATOOLKIT_HOME)/include
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
    HDF5_HOME = /opt/cray/hdf5-parallel/1.8.13/gnu/48/
    LIB_HDF5  = -L$(HDF5_HOME)lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
    HDF5_INCL = -I$(HDF5_HOME)include
  endif
endif

# Configuration for an XC30 with cray toolchain
ifeq ($(SYSTYPE), "XC30-CCE") 
  CC = CC
  OMP =
  OPTIMIZE =-h std=c++11 -hnomessage=12489 
  ifeq (CUDA, $(findstring CUDA, $(OPT)))
  CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/default
  NVCC = nvcc
  NVCCARCH = -arch=sm_30
  NVCCFLAGS = -g  $(NVCCARCH) -dc -use_fast_math -std=c++11 -ccbin=CC
  LIB_OPT  += -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
  SUP_INCL += -I$(CUDATOOLKIT_HOME)/include
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /opt/cray/hdf5-parallel/1.8.13/gnu/48/
  LIB_HDF5  = -L$(HDF5_HOME)lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
  HDF5_INCL = -I$(HDF5_HOME)include
  endif 
endif

ifeq ($(SYSTYPE),"GSTAR")
  ifeq (USE_MPI, $(findstring USE_MPI, $(OPT)))
   CC = mpic++ #-I/usr/local/x86_64/intel/openmpi-1.8.3/include -pthread
  else
   CC = g++
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /usr/local/x86_64/gnu/hdf5-1.8.15
  LIB_HDF5  = -L$(HDF5_HOME)/lib -lhdf5
  HDF5_INCL = -I$(HDF5_HOME)/include
  endif
  ifeq (CUDA,$(findstring CUDA,$(OPT)))
  NVCC       =  nvcc
  NVCCARCH = -arch=sm_20
  NVCCFLAGS = -g  $(NVCCARCH) -dc -std=c++11
  CUDA_HOME  =  /usr/local/cuda-7.5
  LIB_OPT  += -L$(CUDA_HOME)/lib64 -lcudart
  SUP_INCL += -I$(CUDA_HOME)/include
  endif
  OMP = -fopenmp
endif


ifeq ($(SYSTYPE),"SuperMuc")
  module unload mpi.ibm
  module load mpi.ibm/1.3_gcc
  module load gcc/5.1

  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC       = mpiCC
  else
   CC       = ifc
  endif
  OPTIMIZE = -O3 -g -std=c++11
  OMP =   -fopenmp
endif

# Configuration for the VIZ visualization cluster at the Garching computing centre (RZG):
# ->  gcc/OpenMPI_1.4.2
ifeq ($(SYSTYPE),"RZG-SLES11-VIZ")
  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC       = mpic++
  else
   CC       = g++
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /u/system/hdf5/1.8.7/serial
  LIB_HDF5  = -L$(HDF5_HOME)/lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
  HDF5_INCL = -I$(HDF5_HOME)/include
  endif
  OPTIMIZE += -O3 -march=native -mtune=native
  OMP       = -fopenmp
endif


# Configuration for SLES11 Linux clusters at the Garching computing centre (RZG):
# ->  gcc/IntelMPI_4.0.0, requires "module load impi"
ifeq ($(SYSTYPE),"RZG-SLES11-generic")
  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC       = mpigxx
  else
   CC       = g++
  endif
  ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /afs/ipp/home/k/khr/soft/amd64_sles11/opt/hdf5/1.8.7
  LIB_HDF5  = -L$(HDF5_HOME)/lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
  HDF5_INCL = -I$(HDF5_HOME)/include
  endif
  OPTIMIZE += -O3 -msse3
  OMP       = -fopenmp
endif

#--------------------------------------- Debug override 

# Override all optimization settings for debug build
ifeq ($(DEBUG), 1)
OPTIMIZE = -g -O0 -std=c++11
endif

#--------------------------------------- Build config

OPTIONS = $(OPTIMIZE) $(OPT)

EXEC = Splotch6-$(SYSTYPE)
EXEC1 = Galaxy

OBJS  =	kernel/transform.o cxxsupport/error_handling.o \
        reader/mesh_reader.o reader/visivo_reader.o \
	cxxsupport/mpi_support.o cxxsupport/paramfile.o cxxsupport/string_utils.o \
	cxxsupport/announce.o cxxsupport/ls_image.o reader/gadget_reader.o \
	reader/millenium_reader.o reader/bin_reader.o reader/bin_reader_mpi.o reader/tipsy_reader.o \
	splotch/splotchutils.o splotch/splotch.o \
	splotch/scenemaker.o splotch/splotch_host.o splotch/new_renderer.o cxxsupport/walltimer.o c_utils/walltime_c.o \
	booster/mesh_creator.o booster/randomizer.o booster/p_selector.o booster/m_rotation.o \
	reader/ramses_reader.o reader/enzo_reader.o reader/bonsai_reader.o reader/ascii_reader.o

OBJS1 = galaxy/Galaxy.o galaxy/GaussRFunc.o galaxy/Box_Muller.o galaxy/ReadBMP.o \
	galaxy/CalculateDensity.o galaxy/CalculateColours.o galaxy/GlobularCluster.o \
	galaxy/ReadImages.o galaxy/TirificWarp.o galaxy/Disk.o galaxy/RDiscFuncTirific.o

OBJSC = cxxsupport/paramfile.o cxxsupport/error_handling.o cxxsupport/mpi_support.o \
	c_utils/walltime_c.o cxxsupport/string_utils.o \
	cxxsupport/announce.o cxxsupport/ls_image.o cxxsupport/walltimer.o

ifeq (HDF5,$(findstring HDF5,$(OPT)))
  OBJS += reader/hdf5_reader.o
  OBJS += reader/gadget_hdf5_reader.o
  #OBJS += reader/galaxy_reader.o
  #OBJS += reader/h5part_reader.o
endif

ifeq (FITS,$(findstring FITS,$(OPT)))
  FITSIO_PATH = /Users/tims/Code/cfitsio
  SUP_INCL += -I$(FITSIO_PATH)/include
  LIB_OPT += -L$(FITSIO_PATH)/lib
  OBJS += reader/fits_reader.o
  LIB_FITSIO = -lcfitsio
endif

# CUDA config
ifeq (CUDA,$(findstring CUDA,$(OPT)))
OBJS += cuda/cuda_splotch.o cuda/cuda_policy.o cuda/cuda_utils.o cuda/cuda_device_query.o cuda/cuda_kernel.o cuda/cuda_render.o
CULINK = cuda/cuda_link.o
endif


# Debug object for utils, should be removed in favour of planck objects soon..
ifeq (MPI_A_NEQ_E, $(findstring MPI_A_NEQ_E,$(OPT)))
OBJS += utils/debug.o utils/composite.o
endif

# Client server objects, includes, and libraries
ifeq (CLIENT_SERVER, $(findstring CLIENT_SERVER,$(OPT)))

OBJS += server/server.o server/controller.o server/data.o                 \
        server/camera.o server/matrix4.o


LIB_OPT += -L$(LIBTURBOJPEG_PATH)/install/lib -lturbojpeg                \
            -L$(LWS_PATH)/lib -lwebsockets

SUP_INCL += -I$(LWS_PATH)/include -I$(LIBTURBOJPEG_PATH)                  \
            -I$(RAPIDJSON_PATH) -I$(WSRTI_PATH)/websocketplus/include     \
            -I$(WSRTI_PATH)/tjpp/include -I$(WSRTI_PATH)/syncqueue        \
            -I$(WSRTI_PATH)/serializer/include 
endif

INCL   = */*.h Makefile

CPPFLAGS = $(OPTIONS) $(SUP_INCL) $(HDF5_INCL) $(OMP) $(PREVIEWER_OPTS)

CUFLAGS = $(OPTIONS) $(SUP_INCL) $(OMP)

LIBS   = $(LIB_OPT) $(OMP)

.SUFFIXES: .o .cc .cxx .cpp .cu

.cc.o:
	$(CC) -c $(CPPFLAGS) -o "$@" "$<"

.cxx.o:
	$(CC) -c $(CPPFLAGS) -o "$@" "$<"

.cpp.o:
	$(CC) -c $(CPPFLAGS) -o "$@" "$<"

.cu.o:
	$(NVCC) $(NVCCFLAGS) -c --compiler-options "$(CUFLAGS)" -o "$@" "$<"

$(EXEC): $(OBJS) $(CULINK)
	$(CC) $(OPTIONS) $(OBJS) $(CULINK) $(LIBS) $(RLIBS) -o $(EXEC) $(LIB_MPIIO) $(LIB_HDF5) $(LIB_FITSIO)

$(EXEC1): $(OBJS1) $(OBJSC)
	$(CC) $(OPTIONS) $(OBJS1) $(OBJSC) $(LIBS) -o $(EXEC1) $(LIB_HDF5)

$(OBJS): $(INCL)

# In order to do "seperate compilation" for CUDA (CUDA >= 5) and also
# use the host linker for the final link step as opposed to nvcc we must
# add an intermediary step for nvcc (-arch=sm_30) dont forget -dc compile flag
$(CULINK):  $(OBJS) $(INCL)
	$(NVCC) $(NVCCARCH) -dlink $(OBJS) -o $(CULINK)

clean:
	rm -f $(OBJS) $(CULINK)

cleangalaxy:
	rm -f $(OBJS1)

realclean: clean
	rm -f $(EXEC)

