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


#include "mic_kernel.h"
#ifdef __MIC__
#include "immintrin.h"
#endif

#define SPLOTCH_CLASSIC
EVT_TARGET_MIC const float pi = 3.14159265359;
EVT_TARGET_MIC const float h2sigma = 0.5*pow(pi,-1./6.);
EVT_TARGET_MIC const float sqrtpi = sqrt(pi);

#ifdef SPLOTCH_CLASSIC
EVT_TARGET_MIC const float bfak=0.5*pow(pi,-5./6.); // 0.19261
#endif

#ifdef SPLOTCH_CLASSIC
EVT_TARGET_MIC const float rfac=0.75;
#else
EVT_TARGET_MIC const float rfac=1.;
#endif


// 4D array of unsigned ints for particle indices
// Array of indices, for 2d array of tiles, for array of images (1 per threadgroup)
namespace {
	EVT_TARGET_MIC Array_T< Array_T< Array_T< Array_POD_T< unsigned > > > > grp_particle_lists;
	EVT_TARGET_MIC Array_T< Array_T< mic_color > > device_pics;
	EVT_TARGET_MIC Array_T<AlignedPoolAllocator> allocators;
	//EVT_TARGET_MIC Timer mic_timer;
}

// #define MEMALLOC 0
// #define PRERENDER 1
// #define FULLRENDER 2

void prepAllocators(int xres, int yres, int tile_size, int n_thread_groups, int n_threads_per_group)
{
	// printf("prepAllocators(): xres %d yres %d tile_size %d n_thread_groups %d n_threads_per_group %d\n",xres,yres,tile_size,n_thread_groups,n_threads_per_group);
	// fflush(0);
	// Get available threadcount
	int nt = 1;
	#pragma omp parallel 
	{
		#pragma omp single
		{
			#ifdef _OPENMP
			nt = omp_get_num_threads();
			#else
			nt = 1;
			#endif
		}
	}

	// Account for invalid user chosen thread grouping
	// TODO: not foolproof, could have a group of 1 thread vs groups of 30 etc, fix it.
	if(n_thread_groups < 2)
	{
		n_thread_groups = 1;
		n_threads_per_group = nt;
	}
	else if ((n_thread_groups*n_threads_per_group)>nt)
	{
		n_thread_groups = nt/n_threads_per_group;
		if(nt%n_threads_per_group)
			n_thread_groups++;
	}
	// Calculate number of tiles in X and Y dimensions
	int ntx=(xres+tile_size-1)/tile_size, nty=(yres+tile_size-1)/tile_size;

	// printf("prepAllocators(): n_thread_groups %d n_threads_per_group %d ntx %d nty %d \n", n_thread_groups,n_threads_per_group,ntx,nty);
	// fflush(0);

	// One allocator per thread
	allocators.resize(n_thread_groups*n_threads_per_group);

	// A set of threadgroups
	grp_particle_lists.resize(n_thread_groups);

	for(unsigned g_idx = 0; g_idx < n_thread_groups; g_idx++)
	{
		// Each group has a set of threads
		grp_particle_lists[g_idx].resize(n_threads_per_group);
		
		for(unsigned t_idx = 0; t_idx < n_threads_per_group; t_idx++)
		{
			// Each thread has an allocator that needs initialising and ntx*nty tiles
			// TODO: Allocate appropriate nbytes...
			allocators[g_idx*n_threads_per_group+t_idx].init(20000000,64);
			grp_particle_lists[g_idx][t_idx].resize(ntx*nty);

			// Give each list an allocator according to the thread that will be filling it
			for(unsigned nt_idx = 0; nt_idx < (ntx*nty); nt_idx++)
			{
				grp_particle_lists[g_idx][t_idx][nt_idx].giveAllocator(&allocators[g_idx*n_threads_per_group+t_idx]);
			}
		}
	}
	//mic_timer.reserve(10);
	// printf("Finished prepping allocators\n");
	// fflush(0);
}

void prepDevicePics(int xres, int yres, int n_thread_groups)
{
	// Alloc memory for images
	device_pics.resize(n_thread_groups); 
	mic_color fill;
	fill.r = 0;
	fill.g = 0;
	fill.b = 0;
	for(unsigned i = 0; i < n_thread_groups; i++)
	{
		device_pics[i].resize(xres*yres);
		device_pics[i].fill(fill);
	}
}

void rototranslate(int size,float* x, float* y, float* z, float* r, float* I, short* type, bool* active, transform_data& td, const float* trans)
{
	//mic_timer.start("rototrans");
	//printf("p[0] x: %.10f y: %.10f z: %.10f\n", x[0], y[0], z[0]);

	#pragma omp parallel
	{

		float xfac2 = td.xfac;
		float res2 = 0.5f*td.xres;
		long m;

		#pragma omp for
		#pragma ivdep
		for (m=0; m<size; ++m)
		{
			__assume_aligned(x, 64);
			__assume_aligned(y, 64);
			__assume_aligned(z, 64);
			__assume_aligned(r, 64);
			__assume_aligned(I, 64);
			__assume_aligned(active, 64);
			__assume_aligned(trans, 64);

			float x1,y1,z1,r1,i1;
			x1 = x[m]*trans[0] + y[m]*trans[1] + z[m]*trans[2] + trans[3];
			y1 = x[m]*trans[4] + y[m]*trans[5] + z[m]*trans[6] + trans[7];
			z1 = x[m]*trans[8] + y[m]*trans[9] + z[m]*trans[10] + trans[11];
			
			//Check for boundries
			active[m] = false;
			if(-z1 <= td.zminval) continue;
			if(-z1 >= td.zmaxval) continue;

			r1 = r[m];			
			i1 = I[m];
		   			
			if(!td.projection)
			{
    	  		x1 = res2 		  	 + x1*xfac2;
	    	  	y1 = res2 + td.ycorr + y1*xfac2;				
			}
			else
			{
				xfac2=-res2/(td.fovfct*z1);
				x1 = res2 + td.xshift            + x1*xfac2;
				y1 = res2 + td.yshift + td.ycorr + y1*xfac2;
			}

			#ifdef SPLOTCH_CLASSIC
				i1 *= 0.5f*td.bfak/r1;
				r1 *= 2;
			#else
				i1 *= 8.f/(pi*r1*r1*r1); // SPH kernel normalisation
				i1 *= (td.h2sigma*sqrtpi*r1); // integral through the center
			#endif

		    r1 *= xfac2;

		    float rcorr = sqrt(r1*r1 + td.minrad_pix*td.minrad_pix)/r1;
		    r1*=rcorr;
		    i1/=rcorr;

		    float posx=x1, posy=y1;
		    float rfacr=rfac*r1;

		    int minx=int(posx-rfacr+1);
		    if (minx>=td.xres) continue;
		    minx = (minx>0) ? minx : 0;
		    int maxx=int(posx+rfacr+1);
		    if (maxx<=0) continue;
		    maxx = (maxx<td.xres) ? maxx : td.xres;
		    if (minx>=maxx) continue;
		    int miny=int(posy-rfacr+1);
		    if (miny>=td.yres) continue;
		    miny = (miny>0) ? miny : 0;
		    int maxy=int(posy+rfacr+1);
		    if (maxy<=0) continue; 
		    maxy = (maxy < td.yres) ? maxy : td.yres;
		    if (miny>=maxy) continue;

		    active[m] = true;
		    x[m] = x1;
			y[m] = y1;
			z[m] = z1;
			r[m] = r1;
			I[m] = i1;
		}

	}
	//printf("p[0] x: %f y: %f z: %f\n", x[0], y[0], z[0]);
	//mic_timer.stop("rototrans");
}

void colorize(int size, float* er, float* eg, float* eb, float* I, short* type, bool* active, mic_color_map mmap, float* b, bool* colvec)
{
	//mic_timer.start("color");
	#pragma omp parallel
	{
		//---------- Normal version
	#ifndef __MIC__
		#pragma omp for
		#pragma ivdep
		#pragma vector aligned
		for (int m=0; m<size; ++m)
		{
			if(active[m])
			{
				short thistype = type[m];
				if(!colvec[thistype]) 
				{
					int index = mmap.ptype_color_offset[thistype];
					int end = mmap.ptype_color_offset[thistype+1];

					while(er[m] > mmap.mapvalue[index+1] && (index < end)) ++index;

					float fract = (er[m]-mmap.mapvalue[index])/(mmap.mapvalue[index+1]-mmap.mapvalue[index]);
					mic_color col1 = mmap.mapcolor[index];
					mic_color col2 = mmap.mapcolor[index+1];

					er[m] = col1.r + fract*(col2.r-col1.r);
					eg[m] = col1.g + fract*(col2.g-col1.g);
					eb[m] = col1.b + fract*(col2.b-col1.b);
				}

				float bmod = I[m] * b[thistype];
				er[m] *= bmod;
				eg[m] *= bmod;
				eb[m] *= bmod;
			}

		}
	#else
		// ---------- MIC Intrinsic based colorize ---------
		// Break loop into chunks of 16, each thread handles 16 particles at once
		// Work out the split - i.e remainder of particles to be processed in split loop
		// These are the <16 particles that begin on a 64byte boundary but do not fill 512K vector
		// And so are processed in the default manner
		int split = size%16; 
		#pragma omp for
		#pragma ivdep
		for (int m=0; m<(size-split); m+=16)	
		{
			// Create mask for active flag
			int activeMask = 0;
			#pragma vector aligned
			for(unsigned j = m, k = 0; j < (m+16); j++, k++)
			{
				activeMask |= (active[j] << (k));
			}
			__mmask16 _activeMask =  _mm512_int2mask(activeMask);
			
			// Obtain appropriate color & attribution value from colormap
			mm512f32 fract;
			mm512f32 col1r;
			mm512f32 col1g;
			mm512f32 col1b;
			mm512f32 col2r;
			mm512f32 col2b;
			mm512f32 col2g;
			mm512f32 _b;

			int colvecMask = 0;

			#pragma vector aligned
			for(unsigned j = m, k = 0; j < (m+16); j++, k++)
			{
				// Obtain mask for whether color is vector or not
				short thistype = type[j];
				colvecMask |= (!colvec[thistype] << k); 

				// Work out beginning and end indices for portion of colourmap relevant to this type
				int index = mmap.ptype_color_offset[thistype];
				int end = mmap.ptype_color_offset[thistype+1];

				// Find where current particles R value lies in the map
				while(er[j] > mmap.mapvalue[index+1] && (index < end)) ++index;

				// Work out what point between two colors of map the R value sits at, and then load 
				// the color and the color after
				fract.f32[k] = (er[j]-mmap.mapvalue[index])/(mmap.mapvalue[index+1]-mmap.mapvalue[index]);
				col1r.f32[k] = mmap.mapcolor[index].r;
				col1g.f32[k] = mmap.mapcolor[index].g;
				col1b.f32[k] = mmap.mapcolor[index].b;
				col2r.f32[k] = mmap.mapcolor[index+1].r;
				col2g.f32[k] = mmap.mapcolor[index+1].g;
				col2b.f32[k] = mmap.mapcolor[index+1].b;

				// Get brightness contributor for this type
				_b.f32[k] = b[thistype];
			}
			__mmask16 _colvecMask =  _mm512_int2mask(colvecMask);
			// Dual mask - to only affect particles where (colvec[thistype] == false && active==true)
			__mmask16 _dualMask = _mm512_kand(_colvecMask, _activeMask);

			// Load original colors to m512
			__m512 _er = _mm512_load_ps(&er[m]);
			__m512 _eg = _mm512_load_ps(&eg[m]);
			__m512 _eb = _mm512_load_ps(&eb[m]);
			__m512 _I = _mm512_load_ps(&I[m]);

			// Get diff betwen col2 and col1
			col2r.m512 = _mm512_sub_ps(col2r.m512, col1r.m512);
			col2g.m512 = _mm512_sub_ps(col2g.m512, col1g.m512);
			col2b.m512 = _mm512_sub_ps(col2b.m512, col1b.m512);

			// i.e. er[m] = col1.r + fract*(col2.r-col1.r))
			// Store col1 where appropriate via dualMask
			_er = _mm512_mask_mov_ps(_er, _dualMask, col1r.m512);
			_eg = _mm512_mask_mov_ps(_eg, _dualMask, col1g.m512);
			_eb = _mm512_mask_mov_ps(_eb, _dualMask, col1b.m512);

			// FMA fract and color to get attributino to pixel 
			_er = _mm512_mask3_fmadd_ps(fract.m512,col2r.m512,_er,_dualMask);
			_eg = _mm512_mask3_fmadd_ps(fract.m512,col2g.m512,_eg,_dualMask);
			_eb = _mm512_mask3_fmadd_ps(fract.m512,col2b.m512,_eb,_dualMask);

			// i.e float bmod = I[m] * b[thistype]; er[m] *= bmod;
			_I = _mm512_mul_ps(_I, _b.m512);
			_er = _mm512_mul_ps(_I,_er);
			_eg = _mm512_mul_ps(_I,_eg);
			_eb = _mm512_mul_ps(_I,_eb);	

			// Store modified colours
			_mm512_mask_store_ps(&er[m], _activeMask,_er);	
			_mm512_mask_store_ps(&eg[m], _activeMask,_eg);	
			_mm512_mask_store_ps(&eb[m], _activeMask,_eb);		

		}
		// Split loop, process remainder particles in default manner
		#pragma omp for
		for (int m=(size-split); m<size; ++m)
		{
			if(active[m])
			{
				short thistype = type[m];
				if(!colvec[thistype]) 
				{
					int index = mmap.ptype_color_offset[thistype];
					int end = mmap.ptype_color_offset[thistype+1];

					while(er[m] > mmap.mapvalue[index+1] && (index < end)) ++index;

					float fract = (er[m]-mmap.mapvalue[index])/(mmap.mapvalue[index+1]-mmap.mapvalue[index]);
					mic_color col1 = mmap.mapcolor[index];
					mic_color col2 = mmap.mapcolor[index+1];

					er[m] = col1.r + fract*(col2.r-col1.r);
					eg[m] = col1.g + fract*(col2.g-col1.g);
					eb[m] = col1.b + fract*(col2.b-col1.b);
				}

				float bmod = I[m] * b[thistype];
				er[m] *= bmod;
				eg[m] *= bmod;
				eb[m] *= bmod;
			}
		}
	#endif
	}
	//mic_timer.stop("color");
}

void render(int size, float* x, float* y, float* z, float* er, float* eg, float* eb, float* r, bool* active, int xres, int yres, mic_color* devicePic, render_data& rd)
{
	// #ifdef __MIC__
	// 	PapiWrapper pw;
	// 	pw.setDebug(true);
	// 	pw.setVerboseReport(true);
	// 	pw.init();
	// 	pw.startRecording(MEMALLOC);
	// #endif

	// printf("Ascertain thread grouping\n");
	// fflush(0);
	//mic_timer.start("prerender");
	int tile_size = rd.tile_size;
	int n_thread_groups = rd.n_thread_groups;
	int n_threads_per_group = rd.n_threads_per_group;

	int nt = 1;
	#pragma omp parallel 
	{
		#pragma omp single
		{
			#ifdef _OPENMP
			nt = omp_get_num_threads();
			#else
			nt = 1;
			#endif
		}
	}

	// Account for odd/invalid user chosen thread grouping
	if(n_thread_groups < 2)
	{
		n_thread_groups = 1;
		n_threads_per_group = nt;
	}
	else if ((n_thread_groups*n_threads_per_group)>nt)
	{
		n_thread_groups = nt/n_threads_per_group;
		if(nt%n_threads_per_group)
			n_thread_groups++;
	}

	// mic_timer mc;
	// printf("Ensure lists are empty\n");
	// fflush(0);

	// Number of subtiles in x and y dimensions 
	int ntx=(xres+tile_size-1)/tile_size, nty=(yres+tile_size-1)/tile_size;
	// Diagonal and horizontal radii of tile
	float rcell=sqrt(2.f)*(tile_size*0.5f-0.5f);
	float cmid0=0.5f*(tile_size-1);

	//mc.start("Prerender");
	//mc.start("memalloc");

	// Ensure lists are empty (in case of previous runs)
	for(unsigned g_idx = 0; g_idx < n_thread_groups; g_idx++)
	{
		for(unsigned t_idx = 0; t_idx < n_threads_per_group; t_idx++)
		{
			for(unsigned nt_idx = 0; nt_idx < (ntx*nty); nt_idx++)
			{
				grp_particle_lists[g_idx][t_idx][nt_idx].setCapacity(0);
			}
		}
	}	
	//mc.stop("memalloc");	

	// printf("Begin Prerender\n");
	// fflush(0);
	// #ifdef __MIC__
	// 	pw.stopRecording();
	// 	pw.startRecording(PRERENDER);
	// #endif

	// Pre render phase, create lists of particle indices to be rendered by tile-specific threads in render phase
	#pragma omp parallel 
	{
		// Assuming KMP_AFFINITY=granularity=fine,balanced 4threads per hw core
		// Get total number of threads, current thread ID and group ID it is assigned to
		// Number of threads in omp parallel sections should be static during runtime

		int nthreads, tid;

		#ifdef _OPENMP
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		#else
		nthreads = 1;
		tid = 1;
		#endif

		int gid = tid/n_threads_per_group; 
		int grouplocal_tid = tid - (n_threads_per_group*gid);

		//printf("ThreadID: %d, GroupID: %d", tid, gid);

		// Get boundaries of particle subsection for this thread
		int start = (size/nthreads) * tid;
		int end = (tid == nthreads-1) ? size : (size/nthreads) * (tid+1);

		// Debug bool for temp code a few lines below
		//bool mybool = false;

		// Assign particle index to per-tile list for each tile it affects
		for(unsigned i = start; i < end; i++)
		{
			// Radius of particle's influence 
			float rfacr = rfac*r[i];
			if(active[i])
			{
				// Min and max tiles potentially affected by particle
				int minx=(0 > int(x[i]-rfacr+1)/tile_size)  ? 0 : int(x[i]-rfacr+1)/tile_size;
				int maxx=(ntx-1 < int(x[i]+rfacr)/tile_size) ? ntx-1 : int(x[i]+rfacr)/tile_size;
				int miny=(0 > int(y[i]-rfacr+1)/tile_size)  ? 0 : int(y[i]-rfacr+1)/tile_size;
				int maxy=(nty-1 < int(y[i]+rfacr)/tile_size) ? nty-1 : int(y[i]+rfacr)/tile_size;
				// if(mybool)
				// {
				// 	printf("ThreadID: %d, GroupID: %d minx: %d, maxx: %d, miny: %d, maxy: %d, start: %d, end: %d\n", tid, gid, minx, maxx, miny, maxy, start, end);
				// 	fflush(0);
				// 	mybool = false;
				// }
				// 	Square of min distance between particle and cell before intersection for pythag
				float sumsq=(rcell+rfacr)*(rcell+rfacr);
				
				// Loop over all potentially affected tiles
				for (int ix=minx; ix<=maxx; ++ix)
				{
					// Center x point of current tile
					float cx=cmid0+ix*tile_size;
					for (int iy=miny; iy<=maxy; ++iy)
					{
						// Centre y point of current tile
						float cy=cmid0+iy*tile_size;
						// Pythagorus 
						float rtot2 = (x[i]-cx)*(x[i]-cx) + (y[i]-cy)*(y[i]-cy);
						// Add particle index to affected tile's list
						if (rtot2<sumsq)
						{
							//rtot2++;
							grp_particle_lists[gid][grouplocal_tid][(iy*ntx)+ix].push_back(i);
						}
							
					}
				}
			}
		}
	}

	//mic_timer.stop("prerender");
	//mc.stop("Prerender");
	//mc.start("Actual render");
	// #ifdef __MIC__
	// 	pw.stopRecording();
	// 	pw.startRecording(FULLRENDER);
	// #endif

	// Distribute work
	mic_exptable<float> xexp(-20.);

	mic_work_distributor wd (xres,yres,tile_size,tile_size);

	// printf("Begin full render\n");
	// fflush(0);
	//mic_timer.start("render");
	#pragma omp parallel
	{
		int nthreads, tid;
		
		#ifdef _OPENMP
		nthreads = omp_get_num_threads();
		tid = omp_get_thread_num();
		#else
		nthreads = 1;
		tid = 1;
		#endif

		int gid = tid/n_threads_per_group; 
		int grouplocal_tid = tid - (n_threads_per_group*gid);

		// Acertain number of tiles this thread should process and its starting tile
		int start = 0;
		int end = 0;
		int ntilesthisthread = (wd.ntiles()/n_threads_per_group);
		//printf("ThreadID: %d, GroupID: %d, ntilesthisthread %d\n", tid, gid, ntilesthisthread);

		// If we only have maximum one tile per thread
		if((ntilesthisthread == 0))
		{
			if(grouplocal_tid < wd.ntiles())
			{
				ntilesthisthread = 1;
				start = grouplocal_tid;
				end = start + 1;
			}
			else ntilesthisthread = 0;
		}
		// If multiple threads per tile then share equally
		else if(ntilesthisthread > 0)
		{
			int sparetiles = wd.ntiles() % n_threads_per_group;

			//printf("ThreadID: %d, GroupID: %d, sparetiles %d\n", tid, gid, sparetiles);
			if(sparetiles > 0 && sparetiles > grouplocal_tid)
			{
				ntilesthisthread++;
				start = grouplocal_tid * ntilesthisthread;
				end = start + ntilesthisthread;
			}
			else
			{
				start = (grouplocal_tid * ntilesthisthread) + sparetiles;
				end = start + ntilesthisthread;
			}
		}

		// Check for odd thread grouping - ie we have less threads than expected for full threadgroups
		// Final group accounts for extra tiles
		int missing_threads = (n_threads_per_group*n_thread_groups) - nthreads;
		 if(missing_threads)
		 {
		 	// How many spares are there
		 	int morespares = ((n_threads_per_group*n_thread_groups) - nthreads) * (wd.ntiles()/n_threads_per_group);
		  	if(gid == n_thread_groups-1)
		  	{
		  		// While we have spares left, distribute amongst group
		  		while(morespares>0)
		  		{
			 		if(morespares>grouplocal_tid)
			 		{
			 			start += grouplocal_tid;
			 			end += grouplocal_tid+1;
			 			ntilesthisthread++;
			 		}
			 		else
			 		{
			 			start += morespares;
			 			end += morespares;
			 		}
			 		morespares -= (n_threads_per_group-missing_threads);
		 		}
		  	}
		 }

		// Sanity check for odd thread grouping... 
		// TODO: Calculate share ratio to avoid this possibility
		if(start >= wd.ntiles())
		{
			start = 0;
			end = 0;
			ntilesthisthread = 0;
		}
		if(end > wd.ntiles())
			end = wd.ntiles();

		// FIX SHARE RATIO FOR WHEN FINAL GROUP HASNT GOT FULL NUMBER OF THREADS

		// At this point each thread knows the number of tiles it should process
		// and which ones they are, out of the set of tiles its threadgroup should
		// be processing
		//printf("ThreadID: %d, GroupID: %d, LocalID: %d,  ntiles: %d, start: %d, end: %d\n", tid, gid, grouplocal_tid, ntilesthisthread, start, end);
		 
		#ifdef __MIC__
		// Generate pixel mask for 5 pixels at a time
		int mask = 0b111111111111111;		
		__mmask16 _mask = _mm512_int2mask(mask);

		// Temp storage
		__m512 _att;
		__m512 _pixel;
		__m512 _pre2;
		 #endif

		// Storage for local copy of image and rendering data (+4 to avoid accidental overrun of data in intrinsics code)
		float* pre1 = (float*)_mm_malloc((tile_size+4)*sizeof(float), 64);
		mic_color* lpic = (mic_color*)_mm_malloc((tile_size*tile_size)*sizeof(mic_color), 64);

		for(unsigned tile = start; tile < end; tile++)
		{
			// Get tile info as it relates to full image
			// Allocate memory for subimage and fill black
			int x0, x1, y0, y1;
			wd.tile_info(tile,x0,x1,y0,y1);
			int x0s=x0, y0s=y0;
			x1-=x0; x0=0; y1-=y0; y0=0;

			mic_color mc;
			mc.r = 0; mc.g = 0; mc.b = 0;
			for(unsigned i = 0; i < (tile_size*tile_size); i++)
				lpic[i] = mc;

			// Get index of tile
			int tx, ty;
    		wd.tile_info_idx(tile,tx,ty);
    		//printf("ThreadID: %d, GroupID: %d, tidx: %d, tidy: %d, tile: %d\n", tid, gid, tx, ty, tile);

    		// Loop through all lists generated for this tile§
			for(unsigned i = 0; i < n_threads_per_group; i++)
			{
				Array_POD_T<unsigned>& idxList = grp_particle_lists[gid][i][(ty*ntx)+tx];

				// Loop through all particles in this index list 
				for(unsigned j = 0; j < idxList.size(); j++)
				{
					// Ascertain bounding box of pixels affected by particle
					unsigned p_idx = idxList[j];
					float rfacr = r[p_idx] * rfac;
			        float posx=x[p_idx];
			        float posy=y[p_idx];
			        posx-=x0s; posy-=y0s;
			        int minx=int(posx-rfacr+1);
			        minx = (minx > x0 ? minx : x0);
			        int maxx=int(posx+rfacr+1);
			        maxx = (maxx < x1 ? maxx : x1);
			        int miny=int(posy-rfacr+1);
			        miny = (miny > y0 ? miny : y0);
			        int maxy=int(posy+rfacr+1);
			        maxy = (maxy < y1 ? maxy : y1);

					float radsq = rfacr*rfacr;
					float sigma = h2sigma*r[p_idx];
					float stp = -1.f/(sigma*sigma);

					// Get negative particle color
					mic_color clr;
					clr.r = -er[p_idx];
					clr.g = -eg[p_idx];
					clr.b = -eb[p_idx];

					 #ifdef __MIC__
					// Load 5 copies of colour into vector 
					__m512 _clr = _mm512_setr_ps(clr.r, clr.g, clr.b,clr.r, clr.g, clr.b,clr.r, clr.g, clr.b,clr.r, clr.g, clr.b,clr.r, clr.g, clr.b,0);
					 #endif

					// This affects the contribution of the particle to a particular pixel
        			for (int pix_y=miny; pix_y<maxy; ++pix_y)
          				pre1[pix_y]=xexp(stp*(pix_y-posy)*(pix_y-posy));					

          			// For each pixel horizontal
          			for (int pix_x=minx; pix_x<maxx; ++pix_x)
					{
						// Work out actual bbox of pixels affected, modify contribution for each
						// This could be smaller than original bbox, but not larger
						float dxsq = (pix_x-posx)*(pix_x-posx);

						float dy= sqrt(radsq-dxsq);
						int miny2 = (miny > int(posy-dy+1) ? miny : int(posy-dy+1));
						int maxy2 = (maxy < int(posy+dy+1) ? maxy : int(posy+dy+1));
						float pre2 = xexp(stp*dxsq);
						#ifdef __MIC__
						_pre2 = _mm512_set1_ps(pre2);
						#endif
						// Add contribution to lpic pixel
						for (int pix_y=miny2; pix_y<maxy2; ++pix_y)
						{
							#ifdef __MIC__
							if((maxy2-pix_y) > 4){
								// Multiply pre1 by pre2
								//_att = _mm512_loadunpacklo_ps(_att,(void const*)&pre1[pix_y]);
								//_att = _mm512_loadunpackhi_ps(_att,((void const*)&pre1[pix_y])+64);
								_att =_mm512_setr_ps(pre1[pix_y],pre1[pix_y],pre1[pix_y],pre1[pix_y+1],pre1[pix_y+1],pre1[pix_y+1],pre1[pix_y+2],pre1[pix_y+2],pre1[pix_y+2],pre1[pix_y+3],pre1[pix_y+3],pre1[pix_y+3],pre1[pix_y+4],pre1[pix_y+4],pre1[pix_y+4],0);
								_att = _mm512_mul_ps(_att,_pre2);
								// Multiply att by color and add to previous pixel values
								int idx = (pix_x*tile_size)+pix_y;
								//_pixel = _mm512_mask_loadunpacklo_ps(_pixel, _mask,(void*)&lpic[idx]);
								//_pixel = _mm512_mask_loadunpackhi_ps(_pixel, _mask,((void*)&lpic[idx])+64);
								_pixel = _mm512_setr_ps(lpic[idx].r, lpic[idx].g, lpic[idx].b,lpic[idx+1].r, lpic[idx+1].g, lpic[idx+1].b,lpic[idx+2].r, lpic[idx+2].g, lpic[idx+2].b,lpic[idx+3].r, lpic[idx+3].g, lpic[idx+3].b,lpic[idx+4].r, lpic[idx+4].g, lpic[idx+4].b,0);
								_pixel = _mm512_fmadd_ps(_att,_clr,_pixel);
								// Unaligned store affected pixels back to image 
								_mm512_mask_packstorelo_ps((void*)&lpic[idx], _mask, _pixel);
								_mm512_mask_packstorehi_ps(((void*)&lpic[idx])+64, _mask, _pixel);
							 	 pix_y+=4;
							 }
							 else{
								float att = pre1[pix_y]*pre2;
								lpic[(pix_x*tile_size)+pix_y].r += att*clr.r;
								lpic[(pix_x*tile_size)+pix_y].g += att*clr.g;
								lpic[(pix_x*tile_size)+pix_y].b += att*clr.b;
							}
							#endif
						}
					}
				}
			} // For particles

			// Write lpic data for this chunk to full pic (swapping x and ys at same time)
      		for (int iy=0;iy<y1;iy++) 
				for (int ix=0;ix<x1;ix++)
					device_pics[gid][((iy+y0s)*xres)+ix+x0s]=lpic[(ix*tile_size)+iy];
		} // For chunks
	}

	//mc.stop("Actual render");

	//mc.start("Image Accumulate");
	// printf("Full render complete\n");
	// fflush(0);

	// // Accumulate images into one 
	for(unsigned i = 0; i < device_pics.size(); i++)
	{
		#pragma omp parallel for
		for(unsigned j = 0; j < yres; j++)
		{
			for(unsigned k = 0; k < xres; k++)
			{
				int idx1d = (j*xres)+k;
				devicePic[idx1d].r += device_pics[i][idx1d].r;
				devicePic[idx1d].g += device_pics[i][idx1d].g;
				devicePic[idx1d].b += device_pics[i][idx1d].b;
			}
		}
	}
	//printf("stage 3 complete\n");
	//fflush(0);
	//mc.stop("Image Accumulate");
	//mc.print();
	// #ifdef __MIC__
	// 	pw.stopRecording();
	// 	pw.printAllRecords();
	// 	fflush(0);
	// #endif
	//mic_timer.stop("render");
	//mic_timer.report();

}




