/*
 * Copyright (c) 2010-2014
 *              Marzia Rivi (1), Tim Dykes (2)
 *               (1) University of Oxford
 *               (2) University of Portsmouth
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
#include "cuda/cuda_splotch.h"

using namespace std;

void cuda_rendercontext_init(paramfile& params, render_context& context)
{
  int myID = mpiMgr.rank();
  int nDevNode = check_device(myID);     // number of GPUs available per node
  // We assume a geometry where processes use only one gpu if available
  //int nDevProc = 1;   // number of GPU required per process (unused)
  int nTasksNode = 1 ; //number of processes per node (=1 default configuration)
  context.cv.initialised = false;
  if (nDevNode > 0)
    {
#ifdef HYPERQ
    // all processes in the same node share one GPU
    context.mydevID = 0;
    nTasksNode = params.find<int>("tasks_per_node",1); //number of processes per node
    context.nTasksDev = nTasksNode;
    if (mpiMgr.master()) cout << "HyperQ enabled" << endl;
#else
    // only the first nDevNode processes of the node will use a GPU, each exclusively.
    nTasksNode = params.find<int>("tasks_per_node",1); //number of processes per node
    context.mydevID = myID%nTasksNode; //ID within the node
    context.nTasksDev = 1;
    if (mpiMgr.master())
    {
       cout << "Configuration supported is 1 gpu for each mpi process" << endl;
       cout << "Each node has " << nTasksNode << " ranks, and " << nDevNode << " gpus." << endl;
    }
    if (context.mydevID>=nDevNode)
      {
      cout << "There isn't a gpu available for process = " << myID << " computation will be performed on the host" << endl;
      context.mydevID = -1;
      }
#endif
    }
   else planck_fail("No GPUs are available");

  bool gpu_info = params.find<bool>("gpu_info",true);
  if (gpu_info)
    if (context.mydevID>=0 && mpiMgr.master()) print_device_info(myID, context.mydevID);
}

void cuda_renderer_init(int& mydevID, int nTasksDev, arr2<COLOUR> &pic, vector<particle_sim> &particle, vector<COLOURMAP> &amap, paramfile &g_params, cu_cpu_vars& cv)
{
  tstack_push("Device setup");
  cv.nP = particle.size();

  // Fill our output image with black
  pic.fill(COLOUR(0.0, 0.0, 0.0));
  cv.xres = pic.size1();
  cv.yres = pic.size2();

  // CUDA Init
  cu_Device(mydevID);
  
  // Initialize policy class
  // Is this deleted anywhere??
  cv.policy = new CuPolicy(cv.xres, cv.yres, g_params); 

#ifndef CUDA_FULL_ATOMICS
  int ntiles = cv.policy->GetNumTiles();
#else
  // For atomic implementation just a placeholder
  int ntiles = 0;
#endif

  // Initialise struct to hold gpu-destined variables
  memset(&cv.gv, 0, sizeof(cu_gpu_vars));
  cv.gv.policy = cv.policy;

  // Setup gpu colormap
  int ptypes = g_params.find<int>("ptypes",1);
  setup_colormap(ptypes, amap, &cv.gv);

  // Calculate how many particles to manage at once dependant on GPU memory size
  // Particle mem factor modifies how much memory we request
  float factor = g_params.find<float>("particle_mem_factor", 2);
  cv.len = cu_get_chunk_particle_count(&cv.gv, nTasksDev, sizeof(cu_particle_sim), ntiles, factor);
  if (cv.len <= 0)
  {
    cout << "Graphics memory setting error" << endl;
    mpiMgr.abort();
  }

  // Initialise device and allocate arrays
  int error =  cu_allocate(cv.len, ntiles, &cv.gv);
  if(error)
  {
    // Big problem!
  }

  cv.buf = new COLOUR[cv.xres*cv.yres];
  cv.initialised = true;
  tstack_pop("Device setup");
}


void cuda_rendering(int mydevID, int nTasksDev, arr2<COLOUR> &pic, vector<particle_sim> &particle, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, vec3 &sky, vector<COLOURMAP> &amap, float b_brightness, paramfile &g_params,cu_cpu_vars& cv)
{
  tstack_push("CUDA");
  pic.fill(COLOUR(0.0, 0.0, 0.0));
  if(!cv.initialised) cuda_renderer_init(mydevID, nTasksDev,pic,particle, amap, g_params, cv);

// This needs removing or cleaning somehow, e.g. put it in cv
#ifndef CUDA_FULL_ATOMICS
  // Create our host image buffer - to be incrementally filled by looped draw call
  arr2<COLOUR> Pic_host(xres,yres);
#else
  // For atomic implementation just a placeholder
  arr2<COLOUR> Pic_host;
#endif

    // Now we start
  float64 grayabsorb = g_params.find<float>("gray_absorption",0.2);
  bool a_eq_e = g_params.find<bool>("a_eq_e",true);

  bool doLogs;
  int error = cu_init_transform(g_params, campos, centerpos, lookat, sky, b_brightness, doLogs);

  int endP = 0;
  int startP = 0;
  int nPR = 0;
  int nchunk = cv.nP / cv.len;
  if(cv.nP%cv.len) nchunk++;
  //if(mpiMgr.master()) printf("NCHUNK: %i\n", nchunk);
  if(!error)
  {
    // Loop over chunks of particles as big as we can fit in dev mem
    while(endP < cv.nP)
    {
      cu_clear_device_img(cv.gv); 
      // Set range and draw first chunk
      endP = startP + cv.len;
      //if(mpiMgr.master()) printf("Render endP: %i\n", endP);
      if (endP > cv.nP) endP = cv.nP;
      nPR += cu_draw_chunk(mydevID, (cu_particle_sim *) &(particle[startP]), endP-startP, Pic_host, &cv.gv, a_eq_e, grayabsorb, cv.xres, cv.yres, doLogs);
      
  #ifndef CUDA_FULL_ATOMICS
      // Combine host render of large particles to final image
      // No need to do this for atomic implementation
      tstack_push("combine images");
      for (int x=0; x<xres; x++)
        for (int y=0; y<yres; y++)
          pic[x][y] += Pic_host[x][y];
      tstack_pop("combine images");
  #endif

      //cout << "Rank " << mpiMgr.rank() << ": Rendered " << nPR << "/" << cv.nP << " particles" << endl << endl;
      startP = endP;
      // Get device image and combine with final image
      if(nchunk>1)
      {
        tstack_push("add_device_image()");
        add_device_image(pic, &cv.gv, cv.xres, cv.yres, cv.buf);
        tstack_pop("add_device_image()");       
      }
    }    
    // If only one chunk we dont need to add_device_image each chunk, just memcpy it over when done
    if(nchunk==1) get_device_image(pic, &cv.gv, cv.xres, cv.yres);
  }

  tstack_pop("CUDA");
  bool final_frame=false;
  if(final_frame)
  {
      cu_end(&cv.gv);
      delete[] cv.buf;
  }
 }


void setup_colormap(int ptypes, vector<COLOURMAP> &amap, cu_gpu_vars* gv)
{
//init C style colormap
  cu_color_map_entry *amapD;//amap for Device
  int *amapDTypeStartPos; //begin indexes of ptypes in the linear amapD[]
  amapDTypeStartPos =new int[ptypes];
  int curPtypeStartPos =0;
  int size =0;
  //first we need to count all the entries to get colormap size
  for (int i=0; i<amap.size(); i++)
    size += amap[i].size();

  //then fill up the colormap amapD
  amapD =new cu_color_map_entry[size];
  int j,index =0;
  for(int i=0; i<amap.size(); i++)
    {
    for (j=0; j<amap[i].size(); j++)
      {
      amapD[index].val = amap[i].getX(j);
      COLOUR c (amap[i].getY(j));
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
  clmp_info.map =amapD;
  clmp_info.mapSize =size;
  clmp_info.ptype_points =amapDTypeStartPos;
  clmp_info.ptypes =ptypes;
  cu_init_colormap(clmp_info, gv);

  delete []amapD;
  delete []amapDTypeStartPos;
}
