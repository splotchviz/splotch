# module for SuperMUC
#module unload mpi.ibm
#module load mpi.ibm/1.3_gcc
#module load gcc/5.1

#######################################################################
#  Splotch V6                                                      #
#######################################################################

#--------------------------------------- Switch on Previewer
#OPT += -DPREVIEWER

#--------------------------------------- Switch on DataSize
OPT += -DLONGIDS

#--------------------------------------- Switch on MPI
OPT += -DUSE_MPI
#OPT += -DUSE_MPIIO

#--------------------------------------- Switch on HDF5

# OPT += -DHDF5
#OPT += -DH5_USE_16_API

#--------------------------------------- Visual Studio Option
#OPT += -DVS

#--------------------------------------- CUDA options
#OPT += -DCUDA
#OPT += -DHYPERQ

#--------------------------------------- OpenCL options
#OPT += -DOPENCL
#OPT += -DNO_WIN_THREAD

#--------------------------------------- Turn on VisIVO stuff
#OPT += -DSPLVISIVO

#--------------------------------------- Turn off Intensity  normalization
#OPT += -DNO_I_NORM

#--------------------------------------- MIC options
# OPT += -DMIC

#--------------------------------------- Select target Computer
SYSTYPE="SuperMuc"
#SYSTYPE="generic"
#SYSTYPE="mac"
#SYSTYPE="SP6"
#SYSTYPE="GP"
#SYSTYPE="PLX"
#SYSTYPE="BGP"
#SYSTYPE="VIZ"
#SYSTYPE="EIGER"
#SYSTYPE="TODI"
#SYSTYPE="DAINT"

### Dommic cluster as CSCS:
#SYSTYPE="DMC-native"
#SYSTYPE="DMC-offload"

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
OMP      = -fopenmp


SUP_INCL = -I. -Icxxsupport -Ic_utils


# optimization and warning flags (g++)
OPTIMIZE =  -pedantic -Wno-long-long -Wfatal-errors -Wextra -Wall -Wstrict-aliasing=2 -Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -std=c++11
#-Wno-newline-eof -g
#-Wold-style-cast -std=c++11
ifeq ($(SYSTYPE),"generic")
# OPTIMIZE += -O2 -g -D TWOGALAXIES
 OPTIMIZE += -O2 -g

# Generic 64bit cuda setup
ifeq (CUDA,$(findstring CUDA,$(OPT)))
NVCC       =  nvcc
CUDA_HOME  =  /opt/nvidia/cudatoolkit/default
LIB_OPT  += -L$(CUDA_HOME)/lib64 -lcudart
SUP_INCL += -I$(CUDA_HOME)/include
NVCCFLAGS = -g -arch=sm_30 -dc
endif

endif

# OpenMP compiler switch
#OMP      = -fopenmp
SUP_INCL = -I. -Icxxsupport -Ic_utils

#CUDA_HOME = /usr/local/cuda/
ifeq (USE_MPIIO,$(findstring USE_MPIIO,$(OPT)))
 SUP_INCL += -Impiio-1.0/include/
endif

# NOTE: This is for osx >=10.9 i.e. clang not gcc
# Openmp isnt supported by default, so you must modify the paths to point to your
# build of clang with openmp support and an openmp runtime library as done below
# Dont forget to export DYLD_LIBRARY_PATH for the omp and cuda runtime libs
ifeq ($(SYSTYPE),"mac")
  OPT += -DSPLOTCHMAC
  ifeq (CUDA,$(findstring CUDA,$(OPT)))
	#CC = CC
	NVCC = nvcc
	CUDA_HOME = /Developer/NVIDIA/CUDA-7.0/
	OPTIMIZE = -Wall -stdlib=libstdc++ -Wno-unused-function -Wno-unused-variable -Wno-unused-const-variable
	LIB_OPT  += -L$(CUDA_HOME)/lib -lcudart
	NVCCFLAGS = -g -ccbin /usr/bin/clang -arch=sm_30 -dc
	SUP_INCL += -I$(CUDA_HOME)/include
	ifeq (-fopenmp,$(OMP))
		# Modify ccbin argument to point to your OpenMP enabled clang build (note clang++, this is important)
		NVCCFLAGS = -g -ccbin /Users/tims/Programs/Clang-OpenMP/build/Debug+Asserts/bin/clang++ -arch=sm_30 -dc
	endif
  endif
	ifeq (-fopenmp,$(OMP))
		# Change this as above
		CC = /Users/tims/Programs/Clang-OpenMP/build/Debug+Asserts/bin/clang++
		OPTIMIZE = -Wall -stdlib=libstdc++ -Wno-unused-function -Wno-unused-variable -Wno-unused-const-variable
		# These should point to your include and library folders for an openmp runtime library
		SUP_INCL += -I/Users/tims/Programs/Intel-OMP-RT/libomp_oss/exports/common/include/
		LIB_OPT += -L/Users/tims/Programs/Intel-OMP-RT/libomp_oss/exports/mac_32e/lib.thin/
	endif
	ifeq (PREVIEWER,$(findstring PREVIEWER,$(OPT)))
		SUP_INCL += -I/opt/X11/include
	endif
    ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
    	CC = mpic++
    endif
endif

ifeq ($(SYSTYPE),"SuperMuc")
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

# Configuration for DOMMIC at CSCS
ifeq ($(SYSTYPE),"DMC-native")
  CC = icpc -mmic -O2
#-vec-report6
  OPTIMIZE = -std=c++11 -pedantic -Wfatal-errors -Wextra -Wall -Wstrict-aliasing=2 -Wundef -Wshadow -Wwrite-strings -Woverloaded-virtual -Wcast-qual -Wpointer-arith
endif
ifeq ($(SYSTYPE),"DMC-offload")
  ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC = mpiicpc
  else
   CC = icpc
  endif
 OPTIMIZE = -Wall -O2
 #-opt-report-phase=offload
 #-vec-report2
 # -guide -parallel
endif

# configuration for TODI at CSCS
ifeq ($(SYSTYPE),"TODI")
 ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /opt/cray/hdf5/1.8.6/gnu/46/
  LIB_HDF5  = -L$(HDF5_HOME)/lib -lhdf5 -lz
  HDF5_INCL = -I$(HDF5_HOME)/include
 endif
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
   CC       = CC
 else
   CC       = g++ -DDEVS_PER_NODE=1
   #CC       = CC -DDEVS_PER_NODE=1 -DTODI
 endif
ifeq (CUDA,$(findstring CUDA,$(OPT)))
NVCC       =  nvcc -g -arch sm_35 -use_fast_math
LIB_OPT  += -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
SUP_INCL += -I$(CUDATOOLKIT_HOME)/include
endif
 OPTIMIZE = -O3
# OMP      = -fopenmp
endif

# Configuration for PIZ DAINT at CSCS
ifeq ($(SYSTYPE), "DAINT")

ifeq (CUDA, $(findstring CUDA, $(OPT)))
CUDATOOLKIT_HOME=/opt/nvidia/cudatoolkit/5.5.20-1.0501.7945.8.2/
NVCC = nvcc
NVCCARCH = -arch=sm_30
NVCCFLAGS = -g  $(NVCCARCH) -dc -use_fast_math
LIB_OPT  += -L$(CUDATOOLKIT_HOME)/lib64 -lcudart
SUP_INCL += -I$(CUDATOOLKIT_HOME)/include
OPTIMIZE = -O3
endif
 ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /opt/cray/hdf5-parallel/1.8.13/gnu/48/
  LIB_HDF5  = -L$(HDF5_HOME)lib -Wl,-rpath,$(HDF5_HOME)/lib -lhdf5 -lz
  HDF5_INCL = -I$(HDF5_HOME)include
 endif
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
	CC = cc
endif

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


ifeq ($(SYSTYPE),"SP6")
 ifeq (HDF5,$(findstring HDF5,$(OPT)))
  HDF5_HOME = /cineca/prodDF5_INCL = -I$(HDF5_HOME)/include/libraries/hdf5/1.8.4_ser/xl--10.1
  LIB_HDF5  = -L$(HDF5_HOME)/lib -lhdf5 -L/cineca/prod/libraries/zlib/1.2.3/xl--10.1/lib/ -lz -L/cineca/prod/libraries/szlib/2.1/xl--10.1/lib/ -lsz
 endif
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
  CC       =  mpCC_r
 else
  CC       =  xlc++
 endif
 OPTIMIZE =  -q64 -O3 -qarch=auto -qtune=auto -qinline
 LIB_OPT	 =  -bstackpsize:64k -bdatapsize:64k -btextpsize:64k
 OMP =
endif

ifeq ($(SYSTYPE),"EIGER")
ifeq (HDF5,$(findstring HDF5,$(OPT)))
HDF5_HOME = /scratch/eiger/cgheller/hdf5-1.8.7/install/
LIB_HDF5  = -L$(HDF5_HOME)/lib -lhdf5
HDF5_INCL = -I$(HDF5_HOME)/include
endif
ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
CC       =  mpic++
else
CC       =  c++
endif
#EIGER again
ifeq (CUDA,$(findstring CUDA,$(OPT)))
NVCC       =  nvcc -g
CUDA_HOME  =  /apps/eiger/Cuda-4.0/cuda
#LIB_OPT  += -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart
LIB_OPT  += -L$(CUDA_HOME)/lib64 -lcudart
SUP_INCL += -I/apps/eiger/NVIDIA_GPU_Computing_SDK/C/common/inc/ -I$(CUDA_HOME)/include#-I$(CUDAUTIL_INC) -I$(NVCC_HOME)/include
endif

OPTIMIZE =  -O3
OMP =
endif

ifeq ($(SYSTYPE),"GP")
 CC       =  nvcc -g
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
  CC       =  mpicxx -g -I$(CUDA_HOME)/sdk/common/inc -I$(CUDA_HOME)/sdk/C/common/inc -I$(CUDA_HOME)/include
 endif
 NVCC       =  nvcc -g
 OPTIMIZE = -O2
 LIB_OPT  =  -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart
 OMP =
 #-Xcompiler -openmp
 SUP_INCL += -I$(CUDA_HOME)/sdk/common/inc -I$(CUDA_HOME)/sdk/C/common/inc # -I$(CUDA_HOME)/include  -Icuda
endif

ifeq ($(SYSTYPE),"BGP")
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
  CC       = mpixlcxx_r
 else
  CC       = bgxlC_r
 endif
 OPTIMIZE = -O3 -qstrict -qarch=450d -qtune=450 # -qipa=inline
 LIB_OPT  =
 OMP =   -qsmp=omp -qthreaded
endif

ifeq ($(SYSTYPE),"PLX")
 ifeq (USE_MPI,$(findstring USE_MPI,$(OPT)))
  CC  =  mpiCC -g
 else
  CC  = g++
 endif
 OPTIMIZE = -O2 -DDEBUG
 OMP = #-fopenmp
 ifeq (CUDA,$(findstring CUDA,$(OPT)))
  NVCC = nvcc -arch sm_20 -use_fast_math
  LIB_OPT  =  -L$(CUDA_HOME)/lib64 -lcudart
  SUP_INCL += -I$(CUDA_HOME)/include -I$(CUDA_SDK)/CUDALibraries/common/inc
 endif
 ifeq (OPENCL,$(findstring OPENCL,$(OPT)))
  LIB_OPT  =  -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib -lOpenCL
  SUP_INCL += -I$(CUDA_HOME)/include
 endif
endif

#-L/home/pavel/NVIDIA_GPU_Computing_SDK/shared/lib
#
#--------------------------------------- Here we go

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

# OpenCL and CUDA config
ifeq (OPENCL,$(findstring OPENCL,$(OPT)))
 OBJS += opencl/splotch.o opencl/CuPolicy.o opencl/splotch_cuda2.o opencl/deviceQuery.o
else
ifeq (CUDA,$(findstring CUDA,$(OPT)))
  OBJS += cuda/cuda_splotch.o cuda/cuda_policy.o cuda/cuda_utils.o cuda/cuda_device_query.o cuda/cuda_kernel.o cuda/cuda_render.o
  CULINK = cuda/cuda_link.o
 endif
endif

# Intel MIC config
ifeq (MIC,$(findstring MIC,$(OPT)))
OBJS += mic/mic_splotch.o mic/mic_compute_params.o mic/mic_kernel.o mic/mic_allocator.o
OPTIONS += -offload-option,mic,compiler," -fopenmp -Wall -O3 -L. -z defs"
endif

ifeq (USE_MPIIO,$(findstring USE_MPIIO,$(OPT)))
 LIB_MPIIO = -Lmpiio-1.0/lib -lpartition
endif

##################################################
#        SPLOTCH PREVIEWER SECTION
##################################################

# Please choose rendering method. Choice will depend on your current drivers, OpenGL implementation and hardware capabilities
# PPFBO is recommended, but if your implementation does not support framebuffer objects try PPGEOM, if that is also not supported
# use FFSVBO, this uses the fixed function pipeline and should be available on most if not all hardware setups that support opengl.
#
# Uncomment below to use Fixed Function software rendering with Vertex Buffer Objects (faster method if no hardware acceleration)
#----------------------------------
#RENDER_METHOD = -DRENDER_FF_VBO
#----------------------------------
#
# Uncomment below to use Programmable Pipeline rendering using Vertex Buffer Objects and shaders + geometry shader
#----------------------------------
#RENDER_METHOD = -DRENDER_PP_GEOM
#----------------------------------
#
# Uncomment below to use Programmable Pipeline rendering using Vertex Buffer Objects and shaders + geometry shader + FBOs
#----------------------------------
RENDER_METHOD = -DRENDER_PP_FBO
#----------------------------------
#
# Uncomment below to use Programmable Pipeline rendering using Vertex Buffer Objects and shaders + geometry shader + FBOs + post processing filtering effects
#----------------------------------
#RENDER_METHOD = -DRENDER_PP_FBOF
#----------------------------------
#
# Uncomment for previewer DEBUG mode
#----------------------------------
 PREVIEWER_DEBUG = -DDEBUG_MODE=1
#----------------------------------

ifeq (PREVIEWER,$(findstring PREVIEWER,$(OPT)))
# Link libs

#ifeq ($SYSTYPE,"mac")
 LIB_OPT += -L/usr/X11/lib -lXext -lX11 -lGL
#else
# LIB_OPT += -lGL -lXext -lX11
#endif

# Current build specific settings
# Build specific objects are added to OBJS list, depending on renderer choice
# The current include for the auto-generated CurrentRenderer header file will be specified
# The render mode will be added to options to be passed on to the application
# To add a renderer, copy the if clause below and replace the object file and include file with your own
# The RENDER_MODE *must* be the exact, case dependant, name of your renderer class.
# Then add a render_method choice above
ifeq ($(RENDER_METHOD),-DRENDER_FF_VBO)
	OBJS_BUILD_SPECIFIC = previewer/libs/renderers/FF_VBO.o
endif

ifeq ($(RENDER_METHOD),-DRENDER_PP_GEOM)
	OBJS_BUILD_SPECIFIC = previewer/libs/renderers/PP_GEOM.o previewer/libs/materials/PP_ParticleMaterial.o previewer/libs/core/Shader.o
endif

ifeq ($(RENDER_METHOD),-DRENDER_PP_FBO)
	OBJS_BUILD_SPECIFIC = previewer/libs/renderers/PP_FBO.o previewer/libs/materials/PP_ParticleMaterial.o previewer/libs/core/Shader.o \
            previewer/libs/core/Fbo.o
endif

ifeq ($(RENDER_METHOD),-DRENDER_PP_FBOF)
	OBJS_BUILD_SPECIFIC = previewer/libs/renderers/PP_FBOF.o previewer/libs/materials/PP_ParticleMaterial.o previewer/libs/core/Shader.o \
            previewer/libs/core/Fbo.o
endif


OBJS +=   previewer/Previewer.o previewer/libs/core/Parameter.o previewer/libs/core/ParticleSimulation.o \
          previewer/libs/core/WindowManager.o previewer/libs/core/Camera.o previewer/libs/core/ParticleData.o \
          previewer/libs/core/MathLib.o previewer/libs/core/FileLib.o previewer/libs/events/OnQuitApplicationEvent.o \
          previewer/libs/events/OnKeyReleaseEvent.o previewer/libs/events/OnKeyPressEvent.o previewer/libs/events/OnExposedEvent.o \
          previewer/libs/events/OnButtonReleaseEvent.o previewer/libs/events/OnButtonPressEvent.o previewer/libs/events/OnMotionEvent.o \
          previewer/libs/core/Texture.o previewer/libs/animation/AnimationSimulation.o previewer/libs/core/Debug.o \
          previewer/libs/events/actions/CameraAction.o previewer/libs/materials/FF_ParticleMaterial.o \
          previewer/libs/animation/AnimationTypeLookUp.o  previewer/libs/core/Utils.o\
          previewer/libs/animation/AnimationData.o previewer/libs/animation/AnimationPath.o \
          previewer/simple_gui/GUIWindow.o previewer/simple_gui/SimpleGUI.o previewer/simple_gui/GUICommand.o

OBJS += $(OBJS_BUILD_SPECIFIC)

PREVIEWER_OPTS = $(RENDER_METHOD) $(PREVIEWER_DEBUG)

##################################################
#        END OF PREVIEWER SECTION
##################################################

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
	$(CC) $(OPTIONS) $(OBJS) $(CULINK) $(LIBS) $(RLIBS) -o $(EXEC) $(LIB_MPIIO) $(LIB_HDF5)

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

