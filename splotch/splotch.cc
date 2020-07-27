/*
 * Copyright (c) 2004-2014
 *              Martin Reinecke (1), Klaus Dolag (1)
 *               (1) Max-Planck-Institute for Astrophysics
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
#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>

#ifdef SPLVISIVO
#include "optionssetter.h"
#endif

#include "splotch/scenemaker.h"
#include "splotch/splotchutils.h"
#include "splotch/splotch_host.h"
#include "cxxsupport/walltimer.h"
#include "cxxsupport/ls_image.h"
#include "cxxsupport/announce.h"

#ifdef CUDA
#include "cuda/cuda_splotch.h"
#endif

#ifdef PREVIEWER
#include "previewer/simple_gui/SimpleGUI.h"
#include "previewer/libs/core/FileLib.h"
#include <string>
#endif

#ifdef MPI_A_NEQ_E
#include "utils/composite.h"
#endif

#ifdef CLIENT_SERVER
#include "server/server.h"
#endif


using namespace std;

#ifdef SPLVISIVO
int splotchMain (VisIVOServerOptions opt)
#else
int main (int argc, const char **argv)
#endif
  {

#ifdef PREVIEWER 
  bool pvMode=false;
  int paramFileArg=-1;
  for (int i=0; i<argc; i++)
    {
    // Look for command line switch (-pv launches previewer)
    if (string(argv[i]) == string("-pv"))
      pvMode = true;

    // Look for the parameter file
    if (previewer::FileLib::FileExists(argv[i]))
      paramFileArg = i;
    }
  // Preview mode is enabled
  if (pvMode)
    {
    // If param file exists launch app
    if (paramFileArg>0)
      {
      // Launch app with simple GUI
      previewer::simple_gui::SimpleGUI program;
      program.Load(string(argv[paramFileArg]));
      }
    else
      {
      // Output a message as input param file does not exist
      cout << "Invalid input parameter file." << endl;
      return 0;
      }
    }
#endif

  tstack_push("Splotch");
  tstack_push("Setup");
  bool master = mpiMgr.master();

  // Prevent the rendering to quit if the "stop" file
  // has been forgotten before Splotch was started
  if (master)
    if (file_present("stop")) remove("stop");

#ifdef SPLVISIVO
  // If Splotch is running within visivo, use parameters 
  // passed from visivo and initialize the paramfile 
  planck_assert(!opt.splotchpar.empty(),"usage: --splotch <parameter file>");
  paramfile params (opt.splotchpar.c_str(),true);
  splvisivo_init_params(params, opt);
#else
  // Print startup message and read parameter file
  module_startup ("splotch",argc,argv,2,"<parameter file>",master);
  paramfile params (argv[1],false);

  planck_assert(!params.param_present("geometry_file"),
    "Creating of animations has been changed, use \"scene_file\" instead of "
    "\"geometry_file\" ... ");
#endif

    // Raw particle data
  vector<particle_sim> particle_data;
  // Filtered data
  vector<particle_sim> r_points;
  // Create our rendering context
  render_context context;

// CUDA Initialization
#ifdef CUDA
  // Initialize our render context for CUDA rendering
  cuda_rendercontext_init(params, context);
#endif

// Colormap initilalization
#ifdef SPLVISIVO
  get_colourmaps(params,context.amap,opt);
#else
  get_colourmaps(params,context.amap);
#endif


  // Initialize the scene
  string outfile;
  sceneMaker sMaker(params); 
  update_res(params, context);

// Tree setup for MPI a!=e
#ifdef MPI_A_NEQ_E
  if(mpiMgr.num_ranks()>1)
  {
    sMaker.treeInit();
  }
#endif  

// Server initialization
#ifdef CLIENT_SERVER
  SplotchServer server;
  bool splotch_is_server = params.find<bool>("server", false);
  if(splotch_is_server) server.init(params,master);
  else if(master)  printf("Splotch server mode off\n");  
#endif

  tstack_pop("Setup");

// Server takes over execution if enabled
#ifdef CLIENT_SERVER
  if(splotch_is_server) 
  {
    server.run(sMaker, particle_data, r_points, context);
    server.finalize();
    tstack_pop("Splotch");
    return 0;
  }
  else if(master) 
    printf("Splotch server mode off, set <server=TRUE> in paramfile to turn it on\n");  
#endif


  // Render loop
  while(sMaker.getNextScene (particle_data, r_points, context.campos, context.centerpos, context.lookat, context.sky, outfile))
  {
    // Check for image resolution update
    update_res(params, context); 

    bool a_eq_e = params.find<bool>("a_eq_e",true);

#ifdef MPI_A_NEQ_E
    if(mpiMgr.num_ranks() > 1)
    {
      // Update opacity map res to match image
      sMaker.updateOpacityRes(context.pic, context.opacity_map);
      // Get bounding box from tree to give to renderer
      context.bbox = sMaker.treeBoundingBox(); 
    }
#endif

    // Apply a background, commented because doesnt work
    // as the bg is overwritten during OpenMP tiled rendering
    // do_background(params, pic, outfile, master);
    
#ifdef MPI_A_NEQ_E
    // For MPI a!=e we need to consider the additional ghost particles
    if(mpiMgr.num_ranks()>1)  
      context.npart = context.active_nodes[0]->local_size + context.active_nodes[0]->merged_ghosts;
    else                      
      context.npart = particle_data.size();
#else
    context.npart = particle_data.size();
#endif

    // Calculate boost factor for brightness
    bool boost = params.find<bool>("boost",false);
    context.b_brightness = boost ?
    float(context.npart)/float(r_points.size()) : 1.0;
    
    //server.ft.mark(5, "Setup: ");
    //server.ft.start(3);
    if(context.npart>0)
    {
       //("Npart>0: Rendering!\n");
#ifdef MPI_A_NEQ_E
        particle_sim* pData;

        if(mpiMgr.num_ranks()>1)
        {
            pData = active_nodes[0]->data;
            // Determine composite order before transformation
            particle_sim camera;
            camera.x = campos.x;
            camera.y = campos.y;
            camera.z = campos.z;
            sMaker.tree.depth_sort_leaves(camera,composite_order);           
        }
        else
        {
          pData = &particle_data[0];
        }
#else
        // Use correct vector dependant on boost usage
        std::vector<particle_sim>* pData;
        if (boost) pData = &r_points;
        else       pData = &particle_data;
#endif
       context.npart_all = context.npart;
       mpiMgr.allreduce (context.npart_all,MPI_Manager::Sum);

#ifdef CUDA       
      // CUDA or OPENCL rendering
      if (context.mydevID >= 0)
      {
        if (!a_eq_e) planck_fail("CUDA only supported for A==E so far");
        tstack_push("CUDA");
        cuda_rendering(context.mydevID, context.nTasksDev, context.pic, *pData, context.campos, context.centerpos, context.lookat, context.sky, context.amap, context.b_brightness, params, context.cv);
        tstack_pop("CUDA");
      }
      else
      { 
#endif
        // Default host rendering
        host_rendering(params, *pData, context);
#ifdef CUDA
      }
#endif
    }

    tstack_push("Post-processing");
    // Parallel image composition
    composite_images(context);
    
    // Master does final post processing
    if(master)
    {
      // For a_eq_q apply exponential
      exptable<float32> xexp(-20.0);
      if (mpiMgr.master() && a_eq_e)
#pragma omp parallel for
        for (int ix=0;ix<context.xres;ix++)
          for (int iy=0;iy<context.yres;iy++)
          { 
            context.pic[ix][iy].r=-xexp.expm1(context.pic[ix][iy].r);
            context.pic[ix][iy].g=-xexp.expm1(context.pic[ix][iy].g);
            context.pic[ix][iy].b=-xexp.expm1(context.pic[ix][iy].b);
          }

      // Gamma/contrast etc
      colour_adjust(params, context);
      // Colourbar
      if (params.find<bool>("colorbar",false))
        add_colorbar(params,context.pic,context.amap);
    }
    tstack_replace("Post-processing","Output");

    if(!params.find<bool>("AnalyzeSimulationOnly"))
      {
      if (master)
        {
        cout << endl << "saving file " << outfile << " ..." << endl;
        LS_Image img(context.pic.size1(),context.pic.size2());

#pragma omp parallel for
        for (tsize i=0; i<context.pic.size1(); ++i)
          for (tsize j=0; j<context.pic.size2(); ++j)
            img.put_pixel(i,j,Colour(context.pic[i][j].r,context.pic[i][j].g,context.pic[i][j].b));

        int pictype = params.find<int>("pictype",0);
        switch(pictype)
          {
          case 0:
            img.write_TGA(outfile+".tga");
            break;
          case 1:
            planck_fail("ASCII PPM no longer supported");
            break;
          case 2:
            img.write_PPM(outfile+".ppm");
            break;
          case 3:
            img.write_TGA_rle(outfile+".tga");
            break;
          default:
            planck_fail("No valid image file type given ...");
            break;
          }
        }
      }

    tstack_pop("Output");

  // Also meant to happen when using CUDA - unimplemented.
#if (defined(OPENCL))
        cuda_timeReport();
#endif
       timeReport(); 

    mpiMgr.barrier();
    // Abandon ship if a file named "stop" is found in the working directory.
    // ==>  Allows to stop rendering conveniently using a simple "touch stop".
    planck_assert (!file_present("stop"),"stop file found");
    }
#ifdef VS
  //Just to hold the screen to read the messages when debugging
  cout << endl << "Press any key to end..." ;
  getchar();
#endif
  }
