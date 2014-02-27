#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#define DEBUG //only for debugging
#define __NO_STD_VECTOR
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cxxsupport/lsconstants.h"
#include "cxxsupport/string_utils.h"
#include "splotch/splotchutils.h"
#include "kernel/transform.h"

#include "opencl/splotch_cuda.h"
#include "opencl/Policy.h"

#include<CL/cl.h>

using namespace std;

template<typename T> T findParamWithoutChange(paramfile *param,
		std::string &key, T &deflt) {
	return param->param_present(key) ? param->find<T>(key) : deflt;
}

cl_context cxGPUContext; // OpenCL context
cl_command_queue cqCommandQueue; // OpenCL command que
cl_platform_id cpPlatform; // OpenCL platform
cl_device_id cdDevice; // OpenCL device
cl_program cpProgram; // OpenCL program
cl_kernel ckKernel; // OpenCL kernel

size_t szGlobalWorkSize; // 1D var for Total # of work items
size_t szLocalWorkSize; // 1D var for # of work items in the work group
size_t szParmDataBytes; // Byte size of context information
size_t szKernelLength; // Byte size of kernel code
cl_int ciErr1, ciErr2; // Error code var
bool first_kernel = true;
char* cPathAndName = NULL; // var for full paths to data, src, etc.
char* cSourceCL = NULL; // Buffer to hold source for compilation
const char* cExecutableName = NULL;

#ifdef MEM_OPTIMISE
const char* cSourceFile = "splotch_old.cl";
#endif
//--------------------------------------------------------------------
int pass = 0;

inline void checkErr(cl_int err, const char * name) 
{
	if (err != CL_SUCCESS) 
	{
		printf("%s\n", oclErrorString(err));
		//	sclPrintErrorFlags(err); Simple_CL deleted
		exit(EXIT_FAILURE);
	}
}

void getCuTransformParams(cu_param &para_trans, paramfile &params, vec3 &campos,
		vec3 &lookat, vec3 &sky) 
{
	int xres = params.find<int>("xres", 800), yres = params.find<int>("yres", xres);
	double fov = params.find<double>("fov", 45); //in degrees
	double fovfct = tan(fov * 0.5 * degr2rad);
	float64 xfac = 0.0, dist = 0.0;

	sky.Normalize();
	vec3 zaxis = (lookat - campos).Norm();
	vec3 xaxis = crossprod(sky, zaxis).Norm();
	vec3 yaxis = crossprod(zaxis, xaxis);
	TRANSFORM trans;
	trans.Make_General_Transform(TRANSMAT(xaxis.x, xaxis.y, xaxis.z, yaxis.x, yaxis.y, yaxis.z,
					zaxis.x, zaxis.y, zaxis.z, 0, 0, 0));
	trans.Invert();
	TRANSFORM trans2;
	trans2.Make_Translation_Transform(-campos);
	trans2.Add_Transform(trans);
	trans = trans2;
	bool projection = params.find<bool>("projection", true);

	if (!projection) 
	{
		float64 dist = (campos - lookat).Length();
		float64 xfac = 1. / (fovfct * dist);
		cout << " Field of fiew: " << 1. / xfac * 2. << endl;
	}

	float minrad_pix = params.find<float>("minrad_pix", 1.);

	//retrieve the parameters for transformation
	for (int i = 0; i < 12; i++)
		para_trans.p[i] = trans.Matrix().p[i];
	para_trans.projection = projection;
	para_trans.xres = xres;
	para_trans.yres = yres;
	para_trans.fovfct = fovfct;
	para_trans.dist = dist;
	para_trans.xfac = xfac;
	para_trans.minrad_pix = minrad_pix;
}

//cu_gpu_vars* copy;
void cu_init(int devID, int nP, cu_gpu_vars* pgv, paramfile &fparams,
		vec3 &campos, vec3 &lookat, vec3 &sky, int pict_size) 
{
	//Create the context
	cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);

#ifdef DEBUG
	printf("clCreateContext...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif

	// Create a command-queue
	cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);

#ifdef DEBUG
	printf("clCreateCommandQueue...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}

	printf("passed 1\n");
	pass++;
	fflush(stdout);
#endif

	size_t size = pgv->policy->GetFBufSize() << 20;
	pgv->d_fbuf = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL,
			&ciErr1);
	cout << " size:fbuffer " << (int) size << endl;

#ifdef DEBUG
	printf("clCreateBuffer...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif

	//allocate device memory for particle data
	size_t s = nP * sizeof(cu_particle_splotch);
	cout << " size - gou_oarticles: " << (int) (s + sizeof(cu_particle_splotch))
			<< endl;

	pgv->d_ps_render = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
			s + sizeof(cu_particle_splotch), NULL, &ciErr1);

#ifdef DEBUG
	printf("clCreateBuffer...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif

	//retrieve parameters
	cu_param tparams;
	getCuTransformParams(tparams, fparams, campos, lookat, sky);

	tparams.zmaxval = fparams.find<float>("zmax", 1.e23);
	tparams.zminval = fparams.find<float>("zmin", 0.0);
	tparams.ptypes = fparams.find<int>("ptypes", 1);

	for (int itype = 0; itype < tparams.ptypes; itype++) 
	{
		tparams.brightness[itype] = fparams.find<double>(
				"brightness" + dataToString(itype), 1.);
		tparams.col_vector[itype] = fparams.find<bool>(
				"color_is_vector" + dataToString(itype), false);
	}
	tparams.rfac = 1.5;

	pgv->par = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(cu_param),
			NULL, &ciErr1);

#ifdef DEBUG
	printf("clCreateBuffer...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif

	pgv->param_h = tparams;

#ifdef DEBUG
	printf("passed 2");
	pass++;
	fflush(stdout);
#endif

	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, pgv->par, CL_TRUE, 0,
			sizeof(cu_param), &tparams, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueWriteBuffer ()...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}
#endif

}

void transform(cu_particle_sim* p, unsigned int n, cu_gpu_vars* pgv,
		range_part* region) 
{
	cu_param dparams1 = pgv->param_h;
	//copy particle data to device
	size_t s = n * sizeof(cu_particle_splotch);

//#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		int m = i;
		float x, y, z;
		x = p[m].x * pgv->param_h.p[0] + p[m].y * pgv->param_h.p[1]
				+ p[m].z * pgv->param_h.p[2] + pgv->param_h.p[3];
		y = p[m].x * pgv->param_h.p[4] + p[m].y * pgv->param_h.p[5]
				+ p[m].z * pgv->param_h.p[6] + pgv->param_h.p[7];
		z = p[m].x * pgv->param_h.p[8] + p[m].y * pgv->param_h.p[9]
				+ p[m].z * pgv->param_h.p[10] + pgv->param_h.p[11];

		float xfac = pgv->param_h.xfac;
		const float res2 = 0.5 * pgv->param_h.xres;
		const float ycorr = .5f * (pgv->param_h.yres - pgv->param_h.xres);
		if (!pgv->param_h.projection) 
		{
			x = res2 * (x + pgv->param_h.fovfct * pgv->param_h.dist) * xfac;
			y = res2 * (y + pgv->param_h.fovfct * pgv->param_h.dist) * xfac
					+ ycorr;
		} else {
			xfac = 1. / (pgv->param_h.fovfct * z);
			x = res2 * (x + pgv->param_h.fovfct * z) * xfac;
			y = res2 * (y + pgv->param_h.fovfct * z) * xfac + ycorr;
		}

		float r = p[m].r;
		p[m].I /= r;
		r *= res2 * xfac;

		const float rfac = sqrt(r*r+ 0.25 * pgv->param_h.minrad_pix * pgv->param_h.minrad_pix)/r;
		r *= rfac;
		p[m].I = p[m].I / rfac;

		p[m].active = false;

		// compute region occupied by the partile
		const float rfacr = pgv->param_h.rfac * r;
		int minx = (int) (x - rfacr + 1);
		if (minx < pgv->param_h.xres) {
			minx = max(minx, 0);

			int maxx = (int) (x + rfacr + 1);
			if (maxx > 0) {
				maxx = min(maxx, pgv->param_h.xres);
				if (minx < maxx) {

					int miny = (int) (y - rfacr + 1);
					if (miny < pgv->param_h.yres) {
						miny = max(miny, 0);

						int maxy = (int) (y + rfacr + 1);
						if (maxy > 0) {
							maxy = min(maxy, pgv->param_h.yres);
							if (miny < maxy) 
							{
								region[m].minx = minx;
								region[m].miny = miny;
								region[m].maxx = maxx;
								region[m].maxy = maxy;

								p[m].active = true;
								p[m].x = x;
								p[m].y = y;
								p[m].r = r;
							}
						}
					}
				}
			}
		}
	}
#ifdef DEBUG
	printf("passed 3");
	pass++;
	fflush(stdout);
#endif

}

void load_ocl_program() {

#include "splotch_kernel.h"

// Create the program

	cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
			(const char **) &cSourceCL, NULL, &ciErr1);

#ifdef DEBUG
	printf("clCreateProgramWithSource...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);

	}
#endif

// Build the program with 'mad' Optimization option
#ifdef MAC
	char* flags = "-cl-fast-relaxed-math -DMAC";
#else
	char* flags = "-cl-fast-relaxed-math";
	//flags = "";
#endif
	ciErr1 = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);

	printf("clBuildProgram...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);

		char* build_log;
		size_t log_size;
		//cdDevice
		// First call to know the proper size
		clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0,
				NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		exit(1);
	}
//	printf("Build Complete\n");
}

void cu_init_colormap(cu_colormap_info h_info, cu_gpu_vars* pgv) 
{
	int tt = h_info.mapSize;
	//allocate memories for colormap and ptype_points and dump host data into it
	size_t size = sizeof(cu_color_map_entry) * tt;

	pgv->cu_col = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErr1);
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, pgv->cu_col, CL_TRUE, 0, size,
			h_info.map, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueWriteBuffer ()...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}
	printf("passed 6");
	fflush(stdout);
	pass++;
#endif

	size = sizeof(int) * h_info.ptypes;

	pgv->ptype_points = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, size, NULL, &ciErr1);
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, pgv->ptype_points, CL_TRUE, 0,
			size, h_info.ptype_points, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueWriteBuffer ()...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}
	printf("passed 7");
	fflush(stdout);
	pass++;
#endif

	//set fields of global variable pgv->d_colormap_info
	pgv->colormap_size = h_info.mapSize;
	pgv->colormap_ptypes = h_info.ptypes;
}

void cu_copy_particles_to_render(cu_particle_splotch *p, int n,
		cu_gpu_vars* pgv) 
{
	//copy filtered particles into device
	size_t size = n * sizeof(cu_particle_splotch);
	ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, pgv->d_ps_render, CL_TRUE, 0,
			size, p, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueWriteBuffer ()...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}
	printf("passed 8 n = %d\n", n);
	pass++;
	//printf("Size of particles(mb) %d\n", size/1024/1024);
	fflush(stdout);
#endif
}

void cu_render1(int nP, bool a_eq_e, float grayabsorb, cu_gpu_vars* pgv) 
{
	szGlobalWorkSize = nP;
	if (a_eq_e) 
        {
	  ckKernel = clCreateKernel(cpProgram, "k_render1", &ciErr1);

#ifdef DEBUG
	  printf("passed k_render - particles %d\n", nP);
	  printf("clCreateKernel (k_render1)...\n");
	  checkErr(ciErr1, "");

	  if (ciErr1 != CL_SUCCESS)
	  {
		printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	  }
#endif
        }
	else 
        {
          ckKernel = clCreateKernel(cpProgram, "k_render2", &ciErr1);
#ifdef DEBUG
	  printf("passed k_render - particles %d\n", nP);
	  printf("clCreateKernel (k_render2)...\n");
	  checkErr(ciErr1, "");

	  if (ciErr1 != CL_SUCCESS)
	  {
		printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	  }
#endif
        }
	ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem),
			(void*) &pgv->d_ps_render);

	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_int), (void*) &nP);

	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*) &pgv->d_fbuf);

	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_float),
			(void*) &grayabsorb);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(cl_int),
			(void*) &pgv->colormap_size);
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int),
			(void*) &pgv->colormap_ptypes);
	ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void*) &pgv->par);
	ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void*) &pgv->cu_col);
	ciErr1 |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem),
			(void*) &pgv->ptype_points);

#ifdef DEBUG
	printf("clSetKernelArg 0 - 3...\n\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");

	}
#endif

	ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL,
			&szGlobalWorkSize, NULL, 0, NULL, NULL); //local size

#ifdef DEBUG
	printf("clEnqueueNDRangeKernel (k_render)...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");
	}
#endif

	clReleaseKernel(ckKernel);
	clFinish(cqCommandQueue);
}

void cu_get_fbuf(cu_fragment_AeqE *h_fbuf, bool a_eq_e, unsigned long n,
		cu_gpu_vars* pgv) 
{
	clFinish(cqCommandQueue);

	size_t size;
	size = n * sizeof(cu_fragment_AeqE);
	ciErr1 = clEnqueueReadBuffer(cqCommandQueue, pgv->d_fbuf, CL_TRUE, 0, size,
			h_fbuf, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueReadBuffer (Dst size n - %d, size - %d)...\n\n", n,
			size / 1024 / 1024);
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");

	}
	printf("passed 9\n");
	pass++;
	fflush(stdout);
#endif

}

void cu_get_fbuf2(cu_fragment_AneqE *h_fbuf, bool a_eq_e, unsigned long n,
		cu_gpu_vars* pgv) 
{
	size_t size;
	size = n * sizeof(cu_fragment_AneqE);

	ciErr1 = clEnqueueReadBuffer(cqCommandQueue, pgv->d_fbuf, CL_TRUE, 0, size,
			h_fbuf, 0, NULL, NULL);

#ifdef DEBUG
	printf("clEnqueueReadBuffer (Dst size n - %d, size - %d)...\n\n", n,
			size / 1024 / 1024);
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n",
				__LINE__, __FILE__);
		checkErr(ciErr1, "");

	}

	printf("passed 9\n");
	pass++;
	fflush(stdout);
#endif

}

void cu_end(cu_gpu_vars* pgv) 
{
	if (pgv->d_ps_render)
		clReleaseMemObject(pgv->d_ps_render);

	if (pgv->d_fbuf)
		clReleaseMemObject(pgv->d_fbuf);

	delete pgv->policy;
}

int cu_get_chunk_particle_count(paramfile &params, CuPolicy* policy,
		size_t psize, float pfactor) 
{
	char cBuffer[512];
	ciErr1 = oclGetPlatformID(&cpPlatform); //selects OpenCL NVIDIA platform, if NVIDIA not present first found platform

	clGetPlatformInfo(cpPlatform, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer,
			NULL);
	printf(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
#ifdef DEBUG
	printf("clGetPlatformID...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif

	//Get the devices
	ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_ALL, 1, &cdDevice, NULL);

	size_t k;
	clGetDeviceInfo(cdDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &k,
			NULL);

#ifdef DEBUG
	printf("clGetDeviceIDs...\n");
	if (ciErr1 != CL_SUCCESS)
	{
		printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__,
				__FILE__);
		checkErr(ciErr1, "");
	}
#endif
	policy->setGMemSize(k);

	int gMemSize = policy->GetGMemSize();
	int fBufSize = policy->GetFBufSize();
	if (gMemSize <= fBufSize)
		return 0;

	int spareMem = 10;
	int arrayParticleSize = gMemSize - fBufSize - spareMem;

	return (int) (arrayParticleSize / psize / pfactor) * (1 << 20);
}

