/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
 *              Claudio Gheller ETH-CSCS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "mic_splotch.h"


// Size of chunk of data to process on MIC
#define CHUNK 100000000
// Number of signals being used
#define NSIGNALS 4


#define VERBOSE_MODE

// #define ROTOTRANS 0
// #define COLOR 1
// #define RENDER 2

// Device based color map
// At file scope for EVT_TARGET_MIC to apply
// In unnamed namespace to avoid multiple definition between mic_splotch and mic_kernel
namespace{
EVT_TARGET_MIC mic_color_map micmap;
}

void mic_init_offload()
{
  // Init mic omp threads with empty offload 
  // (if not already done by environment variable)
  // Do this asynchrnously
  #pragma offload_transfer target(mic:0)

  #pragma offload target(mic:0)
  {
    // Initialise openmp threads
    #pragma omp parallel
    {

    }
  }
}


void mic_allocate(mic_soa_particles& msp, std::vector<particle_sim>& p, paramfile& params)
{
	// Work out how many particles to process
	// Get CHUNK from param file?
	int chunksize;
	int nchunks;
	int psize = p.size();

	if(psize <= CHUNK)
	{
		chunksize = psize;
		nchunks = 1;
	}
	else
	{
		chunksize = CHUNK/2;
		nchunks = (psize/chunksize);
		if(psize%CHUNK) nchunks++;
	}
	if(mpiMgr.master())
	{
		printf("MPI ranks: %d\n", mpiMgr.num_ranks());
		printf("MIC rank 0 will process %d chunk(s) of %d particles\n",nchunks,chunksize);
		fflush(0);
	}
	// Check if we have already allocated memory
	// NOTE: If xres/yres change during an animation we have a problem
	if(msp.allocated)
	{
		if(chunksize > msp.chunksize)
		{
			// If so, free old data for reallocation
			mic_free(msp,nchunks,chunksize,params);
		}
		else
		{
			// Otherwise we dont need to allocate anything
			return;
		}
	}

	int xres = params.find<int>("xres",800);
    int yres = params.find<int>("yres",xres);
	int n_thread_groups = params.find<int>("n_thread_groups",1);
	int n_threads_per_group = params.find<int>("n_threads_per_group",-1);
	int tile_size = params.find<int>("tile_size",40);

	// Device image
	msp.devicePic = (mic_color*)_mm_malloc((xres*yres)*sizeof(mic_color), 64);

	// Allocate host memory buffer1
	msp.er0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.eg0 = (float*)_mm_malloc(chunksize*sizeof(float), 64);
	msp.eb0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.x0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.y0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.z0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.r0 = (float*)_mm_malloc(chunksize*sizeof(float), 64);
	msp.I0 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
	msp.type0 = (short*)_mm_malloc(chunksize*sizeof(short), 64); 
	msp.active0 = (bool*)_mm_malloc(chunksize*sizeof(bool), 64);

	// Offload clause cant offload a pointer stored as a struct element
	float* er00 = msp.er0;
	float* eg00 = msp.eg0;
	float* eb00 = msp.eb0;
	float* x00 = msp.x0;
	float* y00 = msp.y0;
	float* z00 = msp.z0;
	float* r00 = msp.r0;
	float* I00 = msp.I0;
	short* type00 = msp.type0;
	bool* active00 = msp.active0;
	mic_color* devicePic00 = msp.devicePic;

	int size = xres*yres;

	// Allocate first buffer worth of memory on mic along with space for device image
	#pragma offload_transfer target(mic:0) 				\
		nocopy(er00 : length(chunksize) ALLOC) 		\
		nocopy(eg00 : length(chunksize) ALLOC) 		\
		nocopy(eb00 : length(chunksize) ALLOC) 		\
		nocopy(x00 : length(chunksize) ALLOC) 		\
		nocopy(y00 : length(chunksize) ALLOC) 		\
		nocopy(z00 : length(chunksize) ALLOC) 		\
		nocopy(r00 : length(chunksize) ALLOC) 		\
		nocopy(I00 : length(chunksize) ALLOC) 		\
		nocopy(type00 : length(chunksize) ALLOC) 	\
		nocopy(active00 : length(chunksize) ALLOC) 	


	if(nchunks>1)
	{

		// Allocate host and device memory for buffer2
		msp.er1 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
		msp.eg1 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
		msp.eb1 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
		msp.x1 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
		msp.y1 = (float*)_mm_malloc(chunksize*sizeof(float), 64); 
		msp.z1 = (float*)_mm_malloc(chunksize*sizeof(float), 64);
		msp.r1 = (float*)_mm_malloc(chunksize*sizeof(float), 64);
		msp.I1 = (float*)_mm_malloc(chunksize*sizeof(float), 64);
		msp.type1 = (short*)_mm_malloc(chunksize*sizeof(short), 64); 
		msp.active1 = (bool*)_mm_malloc(chunksize*sizeof(bool), 64); 

		// Offload clause cant offload a pointer stored as a struct element
		float* er11 = msp.er1;
		float* eg11 = msp.eg1;
		float* eb11 = msp.eb1;
		float* x11 = msp.x1;
		float* y11 = msp.y1;
		float* z11 = msp.z1;
		float* r11 = msp.r1;
		float* I11 = msp.I1;
		short* type11 = msp.type1;
		bool* active11 = msp.active1;

		#pragma offload_transfer target(mic:0)			\
			nocopy(er11 : length(chunksize) ALLOC) 		\
			nocopy(eg11 : length(chunksize) ALLOC) 		\
			nocopy(eb11 : length(chunksize) ALLOC) 		\
			nocopy(x11 : length(chunksize) ALLOC) 		\
			nocopy(y11 : length(chunksize) ALLOC) 		\
			nocopy(z11 : length(chunksize) ALLOC) 		\
			nocopy(r11 : length(chunksize) ALLOC) 		\
			nocopy(I11 : length(chunksize) ALLOC) 		\
			nocopy(type11 : length(chunksize) ALLOC) 	\
			nocopy(active11 : length(chunksize) ALLOC)

		#ifdef VERBOSE_MODE
			printf("Allocated second buffer\n");
			fflush(0);
		#endif
	}

	#pragma offload_transfer target(mic:0) nocopy(devicePic00 : length(size) ALLOC)

	// Preallocate memory for pooled thread allocators
	#pragma offload target(mic:0)	in(xres,yres,tile_size,n_thread_groups,n_threads_per_group)	
	{
		// printf("offload block prepAllocators: xres %d yres %d tile_size %d n_threads_per_group %d tile_size %d \n",xres,yres,tile_size,n_thread_groups,n_threads_per_group);
		// fflush(0);
		prepAllocators(xres,yres,tile_size,n_thread_groups,n_threads_per_group);
	}		

	msp.allocated = true;
	msp.chunksize = chunksize;

	printf("Finished initial allocation\n");
	fflush(0);
}

void mic_rendering(paramfile& params, std::vector<particle_sim> &p, arr2<COLOUR> &pic, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, const vec3 &sky,  std::vector<COLOURMAP> &amap, float b_brightness, mic_soa_particles& msp, bool is_final_scene)
{

	tstack_push("Device allocation");
	if(mpiMgr.master())
		std::cout << "Allocating memory for MIC" << std::endl;
	mic_allocate(msp,p, params);
	tstack_pop("Device allocation");

	// Get CHUNK from param file?
	int chunksize;
	int nchunks;
	int psize = p.size();

	if(psize <= CHUNK)
	{
		chunksize = psize;
		nchunks = 1;
	}
	else
	{
		chunksize = CHUNK/2;
		nchunks = (psize/chunksize);
		if(psize%CHUNK) nchunks++;
	}

	int xres = params.find<int>("xres",800); 
    int yres = params.find<int>("yres",xres);

	//printf("Processing particles in chunks of %d from dataset of size %d\n", chunksize, psize);

	// Keep array pointers seperate for ease of offload
	// Get ptr values from mic aos memory object
	// NOTE: reformat this at some point
	float* er0 = msp.er0; float* eg0 = msp.eg0; float* eb0 = msp.eb0;
	float* x0 = msp.x0; float* y0 = msp.y0; float* z0 = msp.z0;
	float* r0 = msp.r0; float* I0 = msp.I0;

	float* er1 = msp.er1; float* eg1 = msp.eg1; float* eb1 = msp.eb1;
	float* x1 = msp.x1; float* y1 = msp.y1; float* z1 = msp.z1;
	float* r1 = msp.r1; float* I1 = msp.I1;

	short* type0 = msp.type0; short* type1 = msp.type1;
	bool* active0 = msp.active0; bool* active1 = msp.active1;
	mic_color* devicePic = msp.devicePic;

	// Signals
	int sig[NSIGNALS] = {0};

	tstack_push("AOS2SOA");

	// OMP parallel convert chunk1 AOS to SOA
	#pragma omp parallel for
	#pragma ivdep
	for(unsigned i = 0; i < chunksize; i++)
	{
		er0[i] = p[i].e.r;
		eg0[i] = p[i].e.g;
		eb0[i] = p[i].e.b;
		x0[i]  = p[i].x;
		y0[i]  = p[i].y;
		z0[i]  = p[i].z;
		r0[i]  = p[i].r;
		I0[i]  = p[i].I;
		type0[i] = p[i].type;
		active0[i] = p[i].active;
	}

	tstack_pop("AOS2SOA");

#ifdef VERBOSE_MODE
	printf("Converted to SOA done\n");
	fflush(0);
#endif

	tstack_push("Chunk0 transfer");
	tstack_push("Init transfer");
	// Send chunk0 to device buffer0 signal(sig0)
	#pragma offload_transfer target(mic:0)   		\
			in(er0 : length(chunksize) REUSE) 	 	\
			in(eg0 : length(chunksize) REUSE) 	 	\
			in(eb0 : length(chunksize) REUSE) 	 	\
			in(x0 : length(chunksize) REUSE) 	 	\
			in(y0 : length(chunksize) REUSE) 	 	\
			in(z0 : length(chunksize) REUSE) 	 	\
			in(r0 : length(chunksize) REUSE) 	 	\
			in(I0 : length(chunksize) REUSE) 	 	\
			in(type0 : length(chunksize) REUSE) 	\
			in(active0 : length(chunksize) REUSE) 	\
			signal(&sig[0])
	tstack_pop("Init transfer");

#ifdef VERBOSE_MODE
	printf("Data transfer initiated\n");
	fflush(0);
#endif

	tstack_push("Compute transform params");
	// Compute transform parameters
	transform_data* transdata = (transform_data*)_mm_malloc(sizeof(transform_data), 64);
	const float* transmatrix = compute_transform(params, *transdata, campos, centerpos, lookat, sky);
	tstack_pop("Compute transform params");

#ifdef VERBOSE_MODE
	printf("Transform params computed \n");
	fflush(0);
#endif
	
	tstack_push("Compute colormap");
	// Compute colormap
	int ptypes = params.find<int>("ptypes",1);
	float* brightness = (float*)_mm_malloc(ptypes*sizeof(float), 64);
	bool* col_is_vec = (bool*)_mm_malloc(ptypes*sizeof(bool), 64);
	compute_colormap(params, ptypes, brightness, col_is_vec, b_brightness, amap, micmap);
	tstack_pop("Compute colormap");	

#ifdef VERBOSE_MODE
	printf("Colormap computed\n");
	fflush(0);
#endif

	tstack_push("Wait for transfer");
	// Setup render parameters
	render_data* rendata = (render_data*)_mm_malloc(sizeof(render_data), 64);;
	rendata->n_thread_groups = params.find<int>("n_thread_groups",1);
	rendata->n_threads_per_group = params.find<int>("n_threads_per_group",-1);
	rendata->tile_size = params.find<int>("tile_size",40);
	//printf("N_THREAD_GROUPS: %d, N_THREADS_PER_GROUP: %d, TILE_SIZE: %d\n",rendata->n_thread_groups, rendata->n_threads_per_group, rendata->tile_size);

	// Alloc/init images + copy to device
	// Alloc index list for threadgroups

	// Send transform parameters to device
	// Also use this to pre-init sig[1] for chunkloop
	#pragma offload_transfer target(mic:0)			\
			in(transdata : length(1) ALLOC)			\
			in(transmatrix : length(12) ALLOC) 		\
			in(brightness : length(ptypes) ALLOC)	\
			in(col_is_vec : length(ptypes) ALLOC) 	\
			wait(&sig[0]) signal(&sig[1])
	tstack_pop("Wait for transfer");
	tstack_pop("Chunk0 transfer");

	// Send colourmap to device seperately to allow initialisation of both signals necessary for chunkloop
	int micmapsize = micmap.size;
	int* pcoloffset = micmap.ptype_color_offset;
	mic_color* mapc = micmap.mapcolor;
	float* mapv = micmap.mapvalue;
	int ntg = rendata->n_thread_groups;

	// Note rendata is copied to device here and *nocopied* for actual rende
	#pragma offload target(mic:0)					\
			in(micmapsize)							\
			in(pcoloffset : length(ptypes+1) ALLOC)	\
			in(mapc : length(micmapsize) ALLOC) 	\
			in(mapv : length(micmapsize) ALLOC)		\
			in(xres, yres)							\
			in(rendata : length(1) ALLOC)			\
			nocopy(micmap)
	{
		micmap.size = micmapsize;
		micmap.ptype_color_offset = pcoloffset;
		micmap.mapcolor = mapc;
		micmap.mapvalue = mapv;
		// Prealloc device images
		prepDevicePics(xres, yres, rendata->n_thread_groups);
	}

	// Presignal sig3
	#pragma offload target(mic:0) signal(&sig[3])
	{
	}

	int start = 0;
	int end = chunksize;

	tstack_push("Main Render");

	unsigned chunkid;

	//printf("Test devpic at render time\n");
	// Zero devicepic
	#pragma offload target(mic:0) in(xres)  	\
			in(yres)							\
			in(devicePic : length(0) REUSE)
			{
				#pragma omp parallel for
				for (int i = 0; i < xres*yres; ++i)
				{
					devicePic[i].r = 0;
					devicePic[i].g = 0;
					devicePic[i].b = 0;
				}
			}
#ifdef VERBOSE_MODE
	printf("Main Render\n");
	fflush(0);
#endif


	// Loop for all chunks of data, 2 at a time
	for(chunkid = 0; chunkid < nchunks; chunkid+=2)
	{

		int size = end-start;
		// Rototranslate + color + render data in device buffer0 wait(sig1) signal(sig2)
		#pragma offload target(mic:0) in(size)  	\
				in(er0 : length(size) REUSE) 	 	\
				in(eg0 : length(size) REUSE) 		\
				in(eb0 : length(size) REUSE) 		\
				in(x0 : length(size) REUSE) 		\
				in(y0 : length(size) REUSE) 		\
				in(z0 : length(size) REUSE) 		\
				in(r0 : length(size) REUSE) 	 	\
				in(I0 : length(size) REUSE) 	 	\
				in(type0 : length(size) REUSE) 	 	\
				in(active0 : length(size) REUSE) 	\
				nocopy(transdata, transmatrix)	 	\
				nocopy(brightness, col_is_vec)		\
				nocopy(micmap,devicePic, rendata)  	\
				wait(&sig[1]) signal(&sig[2])
		{

			// Performance/Timings
			#ifdef __MIC__
			// ------------- TIME -------------
			//  mic_timer mc;
			// // mc.start("Rototranslate");
			// mc.start("Rototranslate_color");
			// printf("rototrans\n");
			// fflush(0);

			// ------------- PAPI -------------
			//PapiWrapper pw;
			//pw.setDebug(true);
			//pw.setVerboseReport(true);
			//pw.init();
			//pw.startRecording(ROTOTRANS);

			#endif

			#ifdef VERBOSE_MODE
				printf("On MIC: chunk %d rototranslate\n", chunkid);
				fflush(0);
			#endif

			rototranslate(size, x0, y0, z0, r0, I0, type0, active0, *transdata, transmatrix);


			//#ifdef __MIC__
			// ------------- TIME -------------
			// mc.stop("Rototranslate");
			// //mc.start("Colorize");

			// ------------- PAPI -------------
			//pw.stopRecording();
			//pw.startRecording(COLOR);
			//#endif


			#ifdef VERBOSE_MODE
				printf("On MIC: chunk %d colorize\n", chunkid);
				fflush(0);
			#endif

			// Color
			colorize(size, er0, eg0, eb0, I0, type0, active0, micmap, brightness, col_is_vec);

			//#ifdef __MIC__
			// ------------- TIME -------------
			// mc.stop("Rototranslate_color");
			// //mc.stop("Colorize");
			// mc.start("Render");
			// ------------- PAPI -------------
			//pw.stopRecording();
			//pw.startRecording(RENDER);
			//pw.printAllRecords();
			//fflush(0);
			//#endif

			#ifdef VERBOSE_MODE
				printf("On MIC: chunk %d render\n", chunkid);
				fflush(0);
			#endif


			render(size, x0, y0, z0, er0, eg0, eb0, r0, active0, transdata->xres, transdata->yres, devicePic, *rendata);

			//#ifdef __MIC__
			//
			// ------------- TIME -------------
			// 	mc.stop("Render");
			// 	mc.print();
			// ------------- PAPI -------------
			// pw.stopRecording();
			// pw.printAllRecords();
			// fflush(0);
			//#endif __MIC__
		}


		// if(chunkid+1 < nchunks) Convert aos-soa + transfer chunkid+1 to buffer1 wait(sig3) signal (sig4)
		if(chunkid+1 < nchunks)
		{
			start = end;
			end = ((end + chunksize) > psize ) ? psize : (end + chunksize);
			// Convert host data from AOS to SOA in buffer1
			#pragma omp parallel for
			#pragma ivdep
			for(unsigned j = start; j < end ;j++)
			{
				int i = j - start;
				er1[i] = p[j].e.r;
				eg1[i] = p[j].e.g;
				eb1[i] = p[j].e.b;
				x1[i]  = p[j].x;
				y1[i]  = p[j].y;
				z1[i]  = p[j].z;
				r1[i]  = p[j].r;
				I1[i]  = p[j].I;
				type1[i] = p[j].type;
				active1[i] = p[j].active;
			}

			// Transfer to device buffer1 wait(sig3) signal(sig4)

			// !!!? WATCH FOR REUSE WHEN SIZE < CHUNKSIZE ?!!!
			#pragma offload_transfer target(mic:0)   \
					in(er1 : length(size) REUSE) 	 \
					in(eg1 : length(size) REUSE) 	 \
					in(eb1 : length(size) REUSE) 	 \
					in(x1 : length(size) REUSE) 	 \
					in(y1 : length(size) REUSE) 	 \
					in(z1 : length(size) REUSE) 	 \
					in(r1 : length(size) REUSE) 	 \
					in(I1 : length(size) REUSE) 	 \
					in(type1 : length(size) REUSE) 	 \
					in(active1 : length(size) REUSE) \
					wait(&sig[3]) signal(&sig[4])

			#ifdef VERBOSE_MODE
				printf("On MIC: chunk %d transfer\n", chunkid+1);
				fflush(0);
			#endif
		}


		// If available,transfer chunkid+2 to device buffer0 --> wait(sig2) signal(sig1)
		if(chunkid+2 < nchunks)
		{
			start = end;
			end = ((end + chunksize) > psize ) ? psize : (end + chunksize);

			// Convert data from AOS to SOA
			#pragma omp parallel for
			#pragma ivdep
			for(unsigned j = start; j < end ;j++)
			{
				int i = j - start;
				er0[i] = p[j].e.r;
				eg0[i] = p[j].e.g;
				eb0[i] = p[j].e.b;
				x0[i]  = p[j].x;
				y0[i]  = p[j].y;
				z0[i]  = p[j].z;
				r0[i]  = p[j].r;
				I0[i]  = p[j].I;
				type0[i] = p[j].type;
				active0[i] = p[j].active;
			}

			// Transfer to device buffer0 wait(sig2, sig4) signal(sig1)
			int size = end-start;

			#pragma offload_transfer target(mic:0)   	\
					in(er0 : length(size) REUSE) 	 	\
					in(eg0 : length(size) REUSE) 	 	\
					in(eb0 : length(size) REUSE) 	 	\
					in(x0 : length(size) REUSE) 	 	\
					in(y0 : length(size) REUSE) 	 	\
					in(z0 : length(size) REUSE) 	 	\
					in(r0 : length(size) REUSE) 	 	\
					in(I0 : length(size) REUSE) 	 	\
					in(type0 : length(size) REUSE) 	 	\
					in(active0 : length(size) REUSE) 	\
					wait(&sig[2], &sig[4]) signal(&sig[1])

			#ifdef VERBOSE_MODE
				printf("On MIC: chunk %d transfer\n", chunkid+2);
				fflush(0);
			#endif
		}

		if(chunkid+1 < nchunks)
		{
			// Rototranslate + color + render data in device buffer1 signal(sig3)
			#pragma offload target(mic:0) in(size)	 	\
					in(er1 : length(size) REUSE) 	 	\
					in(eg1 : length(size) REUSE) 	 	\
					in(eb1 : length(size) REUSE) 	 	\
					in(x1 : length(size) REUSE) 	 	\
					in(y1 : length(size) REUSE) 	 	\
					in(z1 : length(size) REUSE) 	 	\
					in(r1 : length(size) REUSE) 	 	\
					in(I1 : length(size) REUSE) 	 	\
					in(type1 : length(size) REUSE) 	 	\
					in(active1 : length(size) REUSE) 	\
					nocopy(transdata, transmatrix)	 	\
					nocopy(brightness, col_is_vec)	 	\
					nocopy(micmap,devicePic, rendata)	\
					signal(&sig[3])
			{

				#ifdef VERBOSE_MODE
					printf("On MIC: chunk %d rototranslate\n", chunkid+1);
					fflush(0);
				#endif

				// Rototranslate
				rototranslate(size, x1, y1, z1, r1, I1, type1, active1, *transdata, transmatrix);

				#ifdef VERBOSE_MODE
					printf("On MIC: chunk %d color\n", chunkid+1);
					fflush(0);
				#endif

				// Color
				colorize(size, er1, eg1, eb1, I1, type1, active1, micmap, brightness, col_is_vec);

				#ifdef VERBOSE_MODE
					printf("On MIC: chunk %d render\n", chunkid+1);
					fflush(0);
				#endif

				// Render
				render(size, x1, y1, z1, er1, eg1, eb1, r1, active1, transdata->xres, transdata->yres, devicePic, *rendata);
			}
		}
	}

	// Wait for signals, if even number of chunks we wait for sig3, if odd we wait for sig2
	#pragma offload_wait if(nchunks&1)  target(mic:0) wait(&sig[2])
	#pragma offload_wait if(!(nchunks&1)) target(mic:0) wait(&sig[3])

	#ifdef VERBOSE_MODE
		printf("Render complete\n");
		fflush(0);
	#endif

	tstack_pop("Main Render");
	tstack_push("Obtain image");

	// Get image back from mic
	tstack_push("Device2host copy");

	// REUSE so it doesnt free image data in case of another rendering
	#pragma offload_transfer target(mic:0) out(devicePic : length(xres*yres) REUSE)

	tstack_pop("Device2host copy");

	tstack_push("Hostpic2pic copy");

	#pragma omp parallel for
	for(unsigned i = 0; i < yres; i++)
	{
		for(unsigned j = 0; j < xres; j++)
		{
			pic[j][i].r = msp.devicePic[i*xres+j].r;
			pic[j][i].g = msp.devicePic[i*xres+j].g;
			pic[j][i].b = msp.devicePic[i*xres+j].b;
		}
	}

	tstack_pop("Hostpic2pic copy");
	tstack_pop("Obtain image");

	#ifdef VERBOSE_MODE
		printf("Image returned to host\n");
		fflush(0);
	#endif

	// printf("Free memory\n");
	// fflush(0);
	tstack_push("Free data");

	#pragma offload_transfer target(mic:0)			\
		nocopy(transdata : length(1) FREE)			\
		nocopy(transmatrix : length(12) FREE) 		\
		nocopy(brightness : length(ptypes) FREE)	\
		nocopy(col_is_vec : length(ptypes) FREE) 	

	#pragma offload_transfer target(mic:0)				\
			nocopy(pcoloffset : length(ptypes+1) FREE)	\
			nocopy(mapc : length(micmapsize) FREE) 		\
			nocopy(mapv : length(micmapsize) FREE)		\
			nocopy(rendata : length(1) FREE)		

	// printf("Freed MIC working memory\n");
	// fflush(0);
	// TODO: Check why transmatrix was const*
	_mm_free((float*)transmatrix);
	_mm_free(transdata);
	_mm_free(brightness);
	_mm_free(rendata);
	_mm_free(col_is_vec);
	_mm_free(micmap.ptype_color_offset);
 	_mm_free(micmap.mapcolor);
	_mm_free(micmap.mapvalue);

	// printf("Freed host working memory\n");
	// fflush(0);

	if(is_final_scene)
	{
		mic_free(msp, nchunks, chunksize, params);
	}
	#ifdef VERBOSE_MODE
		printf("Data freed\n");
		fflush(0);
	#endif

	tstack_pop("Free data");

}

//Free all particle related data on host and device
void mic_free(mic_soa_particles& msp, int nchunks, int chunksize, paramfile& params)
{

	float* er00 = msp.er0;
	float* eg00 = msp.eg0;
	float* eb00 = msp.eb0;
	float* x00 = msp.x0;
	float* y00 = msp.y0;
	float* z00 = msp.z0;
	float* r00 = msp.r0;
	float* I00 = msp.I0;
	short* type00 = msp.type0;
	bool* active00 = msp.active0;
	mic_color* devicePic00 = msp.devicePic;

	#pragma offload_transfer target(mic:0) 			\
		nocopy(er00 : length(chunksize) FREE) 		\
		nocopy(eg00 : length(chunksize) FREE) 		\
		nocopy(eb00 : length(chunksize) FREE) 		\
		nocopy(x00 : length(chunksize) FREE) 		\
		nocopy(y00 : length(chunksize) FREE) 		\
		nocopy(z00 : length(chunksize) FREE) 		\
		nocopy(r00 : length(chunksize) FREE) 		\
		nocopy(I00 : length(chunksize) FREE) 		\
		nocopy(type00 : length(chunksize) FREE) 	\
		nocopy(active00 : length(chunksize) FREE) 	
	
	int xres = params.find<int>("xres",800);
    int yres = params.find<int>("yres",xres);
	int size = xres*yres;
	
	#pragma offload_transfer target(mic:0) nocopy(devicePic00 : length(size) FREE)


	_mm_free(er00);
	_mm_free(eg00);
	_mm_free(eb00);
	_mm_free(x00);
	_mm_free(y00);
	_mm_free(z00);
	_mm_free(r00);
	_mm_free(I00);
	_mm_free(type00);
	_mm_free(active00);
	_mm_free(devicePic00);


	if(nchunks>1)
	{
		float* er11 = msp.er1;
		float* eg11 = msp.eg1;
		float* eb11 = msp.eb1;
		float* x11 = msp.x1;
		float* y11 = msp.y1;
		float* z11 = msp.z1;
		float* r11 = msp.r1;
		float* I11 = msp.I1;
		short* type11 = msp.type1;
		bool* active11 = msp.active1;

		#pragma offload_transfer target(mic:0)			\
			nocopy(er11 : length(chunksize) FREE) 		\
			nocopy(eg11 : length(chunksize) FREE) 		\
			nocopy(eb11 : length(chunksize) FREE) 		\
			nocopy(x11 : length(chunksize) FREE) 		\
			nocopy(y11 : length(chunksize) FREE) 		\
			nocopy(z11 : length(chunksize) FREE) 		\
			nocopy(r11 : length(chunksize) FREE) 		\
			nocopy(I11 : length(chunksize) FREE) 		\
			nocopy(type11 : length(chunksize) FREE) 	\
			nocopy(active11 : length(chunksize) FREE)

		_mm_free(er11);
		_mm_free(eg11);
		_mm_free(eb11);
		_mm_free(x11);
		_mm_free(y11);
		_mm_free(z11);
		_mm_free(r11);
		_mm_free(I11);
		_mm_free(type11);
		_mm_free(active11);
	}
}