#ifndef NO_WIN_THREAD 
#include <pthread.h>
#endif
#include<stdio.h>

#include "opencl/splotch_cuda2.h"
#include "cxxsupport/string_utils.h"
#include "opencl/Policy.h"

using namespace std;

paramfile *g_params;
int ptypes = 0;
vector<particle_sim> particle_data; //raw data from file
vec3 campos, lookat, sky;
vector<COLOURMAP> amap;
wallTimerSet cuWallTimers;

void opencl_rendering(int mydevID, vector<particle_sim> &particle, int nDev, arr2<COLOUR> &pic)
//void opencl_rendering(int mydevID, int nDev, cu_color *pic, int xres, int yres) 
{
  particle_data = particle;
  long int nP = particle_data.size();
  	int xres = pic.size1();
  	int yres = pic.size2();
	pic.fill(COLOUR(0.0, 0.0, 0.0));
	//see if host must be a working thread
	bool bHostThread = g_params->find<bool>("use_host_as_thread", false);
	int nThread = bHostThread ? nDev + 1 : nDev;
	//init array info for threads control
	thread_info *tInfo = new thread_info[nThread];
	for (int i = 0; i < nDev; i++) 
	{
		tInfo[i].devID = mydevID + i;
		tInfo[i].pPic = new cu_color[xres*yres];
		tInfo[i].xres = xres;
		tInfo[i].yres = yres;
	}
	//make the last one work for host thread

	//decide how to divide particles range by another function
	DevideThreadsTasks(tInfo, nThread, bHostThread);

#ifndef NO_WIN_THREAD // create cuda threads on Windows using CreateThread function
	HANDLE *tHandle = new HANDLE[nThread];
	//issue the threads
	for (int i=0; i<nDev; i++)
	tHandle[i] = CreateThread( NULL, 0,
			(LPTHREAD_START_ROUTINE)cu_thread_func,&(tInfo[i]), 0, NULL );
	//issue the host thread too
	if (bHostThread)
	tHandle[nDev] = CreateThread( NULL, 0,
			(LPTHREAD_START_ROUTINE)host_thread_func,&(tInfo[nDev]), 0, NULL );
	WaitForMultipleObjects(nThread, tHandle, true, INFINITE);

#else // create cuda threads on Linux using pthread_create function
//  planck_assert(nDev <= 1, "can't have multiple cuda threads on Linux (yet), so 'gpu_number' must be 1");
	pthread_t *tHandle = new pthread_t[nThread];

	for (int i = 0; i < nDev; i++)
		pthread_create(&(tHandle[i]), NULL, cu_thread_func,
				(void *) &(tInfo[i]));

	void *status[nThread];
	for (int i = 0; i < nThread; ++i)
		pthread_join(tHandle[i], &status[i]);

#endif  //if not NO_WIN_THREAD

	// combine the results of multiple threads(devices + host) to pic
  	for (int i=0; i<nThread; i++)
      	   for (int x=0; x<xres; x++)
             for (int y=0; y<yres; y++)
             {
		int j = x*yres+y;
                pic[x][y].r += (tInfo[i].pPic[j]).r;
		pic[x][y].g += (tInfo[i].pPic[j]).g;
		pic[x][y].b += (tInfo[i].pPic[j]).b;
	     }
	if (g_params->getVerbosity())
		for (int i = 0; i < nThread; i++) {
			if (tInfo[i].devID != -1) {
				cout << endl << "Rank " << mpiMgr.rank() << ": Times of GPU"
						<< i << ":" << endl;
				GPUReport(tInfo[i].times);
				cout << endl;
			}
		}

	if (mpiMgr.master())
		cuWallTimers = tInfo[0].times;

 	 for (int i=1; i<nThread; i++) delete tInfo[i].pPic;
	 delete [] tInfo;
	 delete [] tHandle;
}

void DevideThreadsTasks(thread_info *tInfo, int nThread, bool bHostThread) {
	bool bTestLoadBalancing = g_params->find<bool>("test_load_balancing",
			false);
	unsigned int curStart = 0;
	int hostLoad = bHostThread ? g_params->find<int>("host_load", 0) : 0;
	int nDev = bHostThread ? nThread - 1 : nThread;
	int onePercent = particle_data.size() / 100;
	int averageDevLen = (nDev != 0) ? onePercent * (100 - hostLoad) / nDev : 0;

	for (int i = 0; i < nThread; i++) {
		tInfo[i].startP = curStart;
		if (tInfo[i].devID != -1) //not a host
				{
			if (bTestLoadBalancing) {
				int gpuLoad = g_params->find<int>("gpu_load" + dataToString(i),
						0);
				tInfo[i].endP = curStart + gpuLoad * onePercent - 1;
			} else
				tInfo[i].endP = curStart + averageDevLen - 1;
		} else //if this is a host
		{
			tInfo[i].endP = curStart + hostLoad * onePercent - 1;
		}
		curStart = tInfo[i].endP + 1;
	}

	tInfo[nThread - 1].endP = particle_data.size() - 1;
}

THREADFUNC cu_thread_func(void *pinfo) {
	//a new thread info object that will carry each chunk's drawing
	thread_info *pInfoOutput = (thread_info*) pinfo;

	// Initialize policy class
	CuPolicy *policy = new CuPolicy(*g_params);

	// num particles to manage at once
	float factor = g_params->find<float>("particle_mem_factor", 3);
	int len = cu_get_chunk_particle_count(*g_params, policy,
			sizeof(cu_particle_sim), factor);
	if (len == 0) {
		printf("\nGraphics memory setting error\n");
		mpiMgr.abort();
	}

	// Init
	cu_gpu_vars gv; //for each gpu/thread a variable pack is needed
	memset(&gv, 0, sizeof(cu_gpu_vars));
	gv.policy = policy;
	// enable device and allocate arrays
	cu_init(pInfoOutput->devID, len, &gv, *g_params, campos, lookat, sky,
			pInfoOutput->xres * pInfoOutput->yres);

	//CUDA Coloring
	setup_colormap(ptypes, &gv);

	int endP = pInfoOutput->endP;
	pInfoOutput->endP = pInfoOutput->startP;
	while (pInfoOutput->endP < endP) {
		pInfoOutput->endP = pInfoOutput->startP + len - 1; //set range
		if (pInfoOutput->endP > endP)
			pInfoOutput->endP = endP;
		cu_draw_chunk(pInfoOutput, &gv);
		pInfoOutput->startP = pInfoOutput->endP + 1;
	}
	cu_end(&gv);

}

void cu_draw_chunk(void *pinfo, cu_gpu_vars* gv) {

	//get the input info
	thread_info *tInfo = (thread_info*) pinfo;
	tInfo->times.start("gpu_thread");
	int nParticle = tInfo->endP - tInfo->startP + 1;
	// printf("Rank %d - GPU %d : Processing %d particles\n", mpiMgr.rank(), tInfo->devID, nParticle); fflush(stdout);

	paramfile &params(*g_params);

	//copy data address to local C-like array pointer d_particle_data
	cu_particle_sim *d_particle_data = &(particle_data[tInfo->startP]);

	//Transform on the host	
	tInfo->times.start("gtransform");
	range_part *minmax;
	minmax = new range_part[nParticle];
	transform(d_particle_data, nParticle, gv, minmax);
	tInfo->times.stop("gtransform");

	//Load CUDA kernels
	tInfo->times.start("ocl");
	load_ocl_program();
	tInfo->times.stop("ocl");

// ----------------------------------
// ----------- Rendering ------------
// ----------------------------------

	//get parameters for rendering
	int xres = params.find<int>("xres", 800), yres = params.find<int>("yres", xres);
	float64 grayabsorb = params.find<float>("gray_absorption", 0.2);
	bool a_eq_e = params.find<bool>("a_eq_e", true);

	//prepare fragment buffer memory space first
	cu_fragment_AeqE *fragBuf;
	cu_fragment_AneqE *fragBuf2;
	size_t nFBufInByte = gv->policy->GetFBufSize() << 20;
	int nFBufInCell;
	if (a_eq_e) 
	{
		nFBufInCell = nFBufInByte / sizeof(cu_fragment_AeqE);
		fragBuf = new cu_fragment_AeqE[nFBufInCell];
	} else {
		nFBufInCell = nFBufInByte / sizeof(cu_fragment_AneqE);
		fragBuf2 = new cu_fragment_AneqE[nFBufInCell];
	}

	int maxRegion = gv->policy->GetMaxRegion();
	int chunk_dim = nParticle;
	// new array of particles produced after filter and splitting
	cu_particle_splotch *cu_ps_filtered;
	cu_ps_filtered = new cu_particle_splotch[chunk_dim];

	range_part *range_filtered;
	range_filtered = new range_part[chunk_dim];

	int End_cu_ps = 0, Start_cu_ps = 0;
	while (End_cu_ps < nParticle) 
	{
		int nFragments2Render = 0;
		//filter and split particles to a cu_ps_filtered
		tInfo->times.start("gfilter");
		int pFiltered = filter_chunk(Start_cu_ps, chunk_dim, nParticle,
				maxRegion, nFBufInCell, d_particle_data, cu_ps_filtered,
				&End_cu_ps, &nFragments2Render, minmax, range_filtered);
		tInfo->times.stop("gfilter");

		tInfo->times.start("gcopy");
		cu_copy_particles_to_render(cu_ps_filtered, pFiltered, gv);
		tInfo->times.stop("gcopy");

		tInfo->times.start("grender");
		cu_render1(pFiltered, a_eq_e, (float) grayabsorb, gv);
		tInfo->times.stop("grender");
 
		//collect result
		tInfo->times.start("gcopy");
		if (a_eq_e)
			cu_get_fbuf(fragBuf, a_eq_e, nFragments2Render, gv);
		else
			cu_get_fbuf2(fragBuf2, a_eq_e, nFragments2Render, gv);
		tInfo->times.stop("gcopy");

		//combine chunks
		tInfo->times.start("gcombine");
 		if (a_eq_e)
			combine_chunk(0, pFiltered - 1, fragBuf,
					tInfo->pPic, xres, yres, range_filtered);

		else
			combine_chunk2(0, pFiltered - 1, fragBuf2,
					tInfo->pPic, xres, yres, range_filtered);
 

#ifdef MEM_OPTIMISE
		if (a_eq_e)
		combine_image(pFiltered-1,gv,yres,xres);
#endif

		tInfo->times.stop("gcombine");

		// printf("Rank %d - GPU %d : Rendered %d/%d particles \n",  mpiMgr.rank(), tInfo->devID, End_cu_ps, nParticle);

		Start_cu_ps = End_cu_ps;
	}

#ifdef MEM_OPTIMISE
	copy_pic(tInfo->pPic,gv);
#endif
	delete[] minmax;
	delete[] range_filtered;
	delete[] cu_ps_filtered;
	if (a_eq_e) delete[] fragBuf;
	else delete[] fragBuf2;
	tInfo->times.stop("gpu_thread");
}

//filter and split particles to a cu_ps_filtered of size nParticles
int filter_chunk(int StartP, int chunk_dim, int nParticle, int maxRegion,
		int nFBufInCell, cu_particle_sim *d_particle_data,
		cu_particle_splotch *cu_ps_filtered, int *End_cu_ps,
		int *nFragments2Render, range_part* minmax, range_part* minmax2) 
{
	cu_particle_splotch p, pNew;
	range_part min_max_change, min_max_v;
	int region, nsplit;
	bool finished = false;

	unsigned long posInFragBuf = 0;
	int pFiltered = 0;
	int i = StartP; // start chunk position in cu_ps

	// filter particles until cu_ps is finished or cu_ps_filtered array is full
	while (!finished && (i < nParticle)) 
	{
		//select valid ones
		p.I = d_particle_data[i].I;
		p.e.b = d_particle_data[i].e.b;
		p.e.r = d_particle_data[i].e.r;
		p.e.g = d_particle_data[i].e.g;

		p.r = d_particle_data[i].r;
		p.type = d_particle_data[i].type;
		p.x = d_particle_data[i].x;
		p.y = d_particle_data[i].y;

		min_max_v = minmax[i];
		if (d_particle_data[i].active) 
		{
			int h = min_max_v.maxy - min_max_v.miny;
			int w = min_max_v.maxx - min_max_v.minx;
			region = h * w;

			if (region <= maxRegion)
			{
				// set the start position of the particle in fragment buffer
				if ((pFiltered + 1 <= chunk_dim) && (posInFragBuf + region < nFBufInCell)) 
				{
					p.posInFragBuf = posInFragBuf;
					cu_ps_filtered[pFiltered] = p;
					cu_ps_filtered[pFiltered].maxx = min_max_v.maxx;
					cu_ps_filtered[pFiltered].minx = min_max_v.minx;

					minmax2[pFiltered] = min_max_v;
 
					pFiltered++;
					posInFragBuf += region;
				} 
				else finished = true;
			} 
			else 
			{ //particle too big -> split along y direction
				pNew = p;
				min_max_change = min_max_v;
				int w1 = (maxRegion % h == 0) ? (maxRegion / h) : (maxRegion / h + 1);
				nsplit = w / w1 + 1;
				if ((pFiltered + nsplit <= chunk_dim) && (posInFragBuf + region < nFBufInCell)) 
				{
				   for (int minx = min_max_v.minx; minx < min_max_v.maxx; minx += w1) 
				   {
					min_max_change.minx = minx; //minx,maxx of pNew need to be set
					min_max_change.maxx = (minx + w1 >= min_max_v.maxx) ? min_max_v.maxx : minx + w1;
					// set the start position of the particle in fragment buffer
					pNew.posInFragBuf = posInFragBuf;
					cu_ps_filtered[pFiltered] = pNew;
					minmax2[pFiltered] = min_max_change;
					cu_ps_filtered[pFiltered].maxx = min_max_change.maxx;
					cu_ps_filtered[pFiltered].minx = min_max_change.minx;
					pFiltered++;
					int newregion = (min_max_change.maxx - min_max_change.minx) 
							* (min_max_change.maxy - min_max_change.miny);
					posInFragBuf += newregion;
				   }
				} 
				else finished = true;
			}
		}
		i++;
	}

	*End_cu_ps = i;
	*nFragments2Render = posInFragBuf;

	return pFiltered; // return chunk position reached in cu_ps
}

void combine_chunk(int StartP, int EndP, cu_fragment_AeqE *fragBuf, cu_color *pPic, int xres, int yres,
		range_part* minmax) 
{
	for (int pPos = StartP, fPos = 0; pPos < EndP; pPos++) 
	{
		int x, y;
		for (x = minmax[pPos].minx; x < minmax[pPos].maxx; x++) 
		{
			for (y = minmax[pPos].miny; y < minmax[pPos].maxy; y++) 
			{
				int xy = x * yres + y;
				pPic[xy].r += fragBuf[fPos].aR;
				pPic[xy].g += fragBuf[fPos].aG;
				pPic[xy].b += fragBuf[fPos].aB;
				fPos++;
			}
		}
	}

}

void combine_chunk2(int StartP, int EndP,cu_fragment_AneqE *fragBuf, cu_color *pPic, 
		   int xres, int yres, range_part* minmax) 
{
	for (int pPos = StartP, fPos = 0; pPos < EndP; pPos++) 
	{
		for (int x = minmax[pPos].minx; x < minmax[pPos].maxx; x++) 
		{
			for (int y = minmax[pPos].miny; y < minmax[pPos].maxy; y++) 
			{
				int xy = x * yres + y;
				pPic[xy].r += fragBuf[fPos].aR * (pPic[xy].r - fragBuf[fPos].qR);
				pPic[xy].g += fragBuf[fPos].aG * (pPic[xy].g - fragBuf[fPos].qG);
				pPic[xy].b += fragBuf[fPos].aB * (pPic[xy].b - fragBuf[fPos].qB);
				fPos++;
			}
		}
	}

}

void setup_colormap(int ptypes, cu_gpu_vars* gv) 
{
	cu_color_map_entry *amapD; //amap for Device
	int *amapDTypeStartPos; //begin indexes of ptypes in the linear amapD[]
	amapDTypeStartPos = new int[ptypes];
	int curPtypeStartPos = 0;
	int size = 0;
	//first we need to count all the entries to get colormap size
	for (int i = 0; i < amap.size(); i++)
		size += amap[i].size();

	//then fill up the colormap amapD
	amapD = new cu_color_map_entry[size];
	int j, index = 0;
	for (int i = 0; i < amap.size(); i++) 
	{
		for (j = 0; j < amap[i].size(); j++) 
		{
			amapD[index].val = amap[i].getX(j);
			COLOUR c(amap[i].getY(j));
			amapD[index].color.r = c.r;
			amapD[index].color.g = c.g;
			amapD[index].color.b = c.b;
			index++;
		}
		amapDTypeStartPos[i] = curPtypeStartPos;
		curPtypeStartPos += j;
	}

	//now let cuda init colormap on device
	cu_colormap_info clmp_info;
	clmp_info.map = amapD;
	clmp_info.mapSize = size;
	clmp_info.ptype_points = amapDTypeStartPos;
	clmp_info.ptypes = ptypes;
	cu_init_colormap(clmp_info, gv);

	delete[] amapD;
	delete[] amapDTypeStartPos;
}

THREADFUNC host_thread_func(void *p) 
{
	thread_info *tInfo = (thread_info*) p;

	vector<particle_sim>::iterator i1, i2;
	i1 = particle_data.begin() + tInfo->startP;
	i2 = particle_data.begin() + tInfo->endP + 1;
	vector<particle_sim> particles(i1, i2);
//exit(0);
#ifndef OPENCL
	host_rendering(*g_params, particles, *(tInfo->pPic), campos, lookat, sky, amap);
#endif
}

void GPUReport(wallTimerSet &cuTimers) {
	cout << "Copy  (secs)               : " << cuTimers.acc("gcopy") << endl;
	cout << "Transforming Data (secs)   : " << cuTimers.acc("gtransform")
			<< endl;
	cout << "Load OpenCL kernel (secs)  : " << cuTimers.acc("ocl") << endl;
	cout << "Filter Sub-Data (secs)     : " << cuTimers.acc("gfilter") << endl;
	cout << "Rendering Sub-Data (secs)  : " << cuTimers.acc("grender") << endl;
	cout << "Combine Sub-image (secs)   : " << cuTimers.acc("gcombine") << endl;
	cout << "OpenCL thread (secs)       : " << cuTimers.acc("gpu_thread")
			<< endl;
}

void cuda_timeReport() {
	if (mpiMgr.master()) {
		cout << endl << "--------------------------------------------" << endl;
		cout << "Summary of timings" << endl;
		cout << "--------------------------------------------" << endl;
		cout << endl << "Times of GPU:" << endl;
		GPUReport(cuWallTimers);
		cout << "--------------------------------------------" << endl;
	}
}

