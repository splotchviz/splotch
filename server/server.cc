#include "server.h"
#include <cassert>
#include <cstring>
#include <sstream>
#include <ctype.h>
#include "splotch/splotch_host.h"
#include "json_maker.h"
#include "rapidjson/prettywriter.h"
#include "TJCompressor.h"
#include "JPEGImage.h"


#ifdef USE_MPI
#include "mpi.h"
#endif


//! Specialization for JPEGImage from tjpp library
// May not need WEB_BYTEARRAY, use ZRF_UNSIGNED_CHAR?
struct SerializeJPEGImage {
    using SS = srz::SerializePOD< size_t >;
    using WEB_BYTEARRAY = std::vector<unsigned char>;
    static srz::ByteArray Pack(const tjpp::JPEGImage& jpi,
                          srz::ByteArray buf = srz::ByteArray()) {
        buf = srz::PackArgs(jpi.Width(),jpi.Height(),jpi.PixelFormat(),jpi.ChrominanceSubSampling(),jpi.Quality(),jpi.CompressedSize());
        int sz = buf.size();
        buf.resize(sz+jpi.CompressedSize());
        memcpy(&buf[sz],jpi.DataPtr(),jpi.CompressedSize());
        return buf;
    }
    static WEB_BYTEARRAY PackDataOnlyWeb(const tjpp::JPEGImage& jpi,
                          WEB_BYTEARRAY buf = WEB_BYTEARRAY()) {
        int sz = 0;
        buf.resize(jpi.CompressedSize());
        memcpy(&buf[sz],jpi.DataPtr(),jpi.CompressedSize());
        return buf;
    }

    static srz::ConstByteIterator UnPack(srz::ConstByteIterator bi,
                                    tjpp::JPEGImage& jpi) {
      int w,h,pf,ss,q;
      size_t sz;
      bi = srz::SerializePOD<int>::UnPack(bi, w);
      bi = srz::SerializePOD<int>::UnPack(bi, h);
      bi = srz::SerializePOD<int>::UnPack(bi, pf);
      bi = srz::SerializePOD<int>::UnPack(bi, ss);
      bi = srz::SerializePOD<int>::UnPack(bi, q);
      bi = srz::SerializePOD<size_t>::UnPack(bi, sz);
      TJPF pf1 = (TJPF)pf;
      TJSAMP ss1 = (TJSAMP)ss;

      jpi.Reset(w,h,pf1,ss1,q);
      planck_assert(UncompressedSize(jpi) >= sz,"srz::UnPack: allocated buffer < recieved JPEGSize");
      memmove(jpi.DataPtr(),&(*bi),sz);
      jpi.SetCompressedSize(sz);
      return bi;
    }
};
 //! De-serialize data from byte array.
void UnPack(const srz::ByteArray& ba, tjpp::JPEGImage& d) {
    SerializeJPEGImage::UnPack(ba.begin(), d);
}

void SplotchServer::init(paramfile& p, bool master)
{
  // All ranks need params, only the master needs the rest of the setup
  params = &p;
  if(!master)
  {
    server_passive = true;
    return;
  }
  server_active = true;
  
  // Image buffer
  xres = params->find<int>("xres",800);
  yres = params->find<int>("yres",xres);
  image_size = xres*yres*3;
  image_buffer.resize(image_size);

  // Load initial waiting image
  set_waiting_image();
  image_modified = true;    
  waiting = true;    

  // Render parameters
  // Camera setup
  vec3f lookat(params->find<double>("lookat_x"),params->find<double>("lookat_y"),params->find<double>("lookat_z"));
  vec3f sky(params->find<double>("sky_x"),params->find<double>("sky_y"),params->find<double>("sky_z"));
  vec3f campos(params->find<double>("camera_x"),params->find<double>("camera_y"),params->find<double>("camera_z"));   
  float move_speed   = params->find<float>("move_speed",0.1);
  float rotate_speed = params->find<float>("rotate_speed",0.1);
  camera.init(campos,lookat,sky,move_speed,rotate_speed, ideal_mspf);

  // Observers setup
  // Could probably do this in constructors rather than manually?
  ims_events.mouse.attach(camera.mouse_observer);
  ims_events.keyboard.attach(camera.key_observer);
 

  // Communication setup
  wsocket_image_protocol = "splotch-image-stream-protocol";
  wsocket_control_protocol = "splotch-control-protocol";

  // Launch servers and services
  init_comms();
  launch_image_services();
  launch_event_services();

  // Initialize command system
  init_cmds();

  // Client stuff
  // This new is temporary until action queue is introduced
  clients = new ClientController(this);
  //ctl_events.client.attach(clients->ctl_observer);
  ims_events.client.attach(clients->observer);
  commands.client.attach(clients->observer);
  clients->load_default_client();

  // Start frame timer
  ft.start(1);
}

void SplotchServer::run(sceneMaker& sMaker, std::vector<particle_sim> &particle_data,
                        std::vector<particle_sim> &r_points, render_context& rc)
{
  // if !init fail
  bool running = true;
  exptable<float32> xexp(-20.0);
  bool a_eq_e = params->find<bool>("a_eq_e",true);
  context = &rc;
  while(running)
  {
    // Update
    update_state();
    handle_events();
    update_parameters(rc.campos, rc.centerpos, rc.lookat, rc.sky);
    update_scene(particle_data, r_points, rc); 
    if(scene_modified) try
    {
      // Render
      rc.npart = particle_data.size();
      // Calculate boost factor for brightness
      bool boost = params->find<bool>("boost",false);
      rc.b_brightness = boost ?
      float(rc.npart)/float(r_points.size()) : 1.0;

      if(rc.npart>0)
      {
        // Swap particle data with filtered data if boosting
        if (boost) particle_data.swap(r_points);

        // Get total particle count
        rc.npart_all = rc.npart;
        mpiMgr.allreduce (rc.npart_all,MPI_Manager::Sum);

  #ifdef CUDA
        if (mydevID >= 0)
        {
          planck_assert(a_eq_e, "CUDA only supported for A==E so far");
          tstack_push("CUDA");
          cuda_rendering(rc.mydevID, rc.nTasksDev, rc.pic, particle_data, rc.campos, rc.centerpos, rc.lookat, rc.sky, rc.amap, rc.b_brightness, *params, rc.cv);
          tstack_pop("CUDA");
        }
        else
        { 
  #endif
      host_rendering(*params, particle_data, rc);
  #ifdef CUDA
        }
  #endif

      }     
      // Postproc
      composite_images(rc);
      if(mpiMgr.master())
      {
        // For a_eq_q apply exponential
        if (mpiMgr.master() && a_eq_e)
  #pragma omp parallel for
          for (int ix=0;ix<rc.xres;ix++)
            for (int iy=0;iy<rc.yres;iy++)
            { 
              rc.pic[ix][iy].r=-xexp.expm1(rc.pic[ix][iy].r);
              rc.pic[ix][iy].g=-xexp.expm1(rc.pic[ix][iy].g);
              rc.pic[ix][iy].b=-xexp.expm1(rc.pic[ix][iy].b);
            }
        // Gamma/contrast etc
        colour_adjust(*params, rc);
        // Colourbar
        if (params->find<bool>("colorbar",false))
          add_colorbar(*params,rc.pic,rc.amap);
      }     
      image_modified = true;

    } catch(const PlanckError& e) {
     // spawn_print(e.what());
    }

    // Reformat the image array to a contiguous set of bytes in X order (x0y0,x1y0,x2y0)
    if(scene_modified)
    {
      unsigned char* buf = (unsigned char*)image_buffer.data();
      #pragma omp parallel for
      for (int j=rc.yres-1; j>=0; --j)
      {
        for (int i=0; i<rc.xres; ++i)
        {
          int yidx = rc.yres - j - 1;
          // For websockets we must also flip the image around horizontal axis
          // Logic here could be simplified
          buf[(yidx*rc.xres+i)*3  ]=(unsigned char)(rc.pic[i][yidx].r * 255);
          buf[(yidx*rc.xres+i)*3+1]=(unsigned char)(rc.pic[i][yidx].g * 255);
          buf[(yidx*rc.xres+i)*3+2]=(unsigned char)(rc.pic[i][yidx].b * 255);         
        }        
      }
    }

    send_image();      
  }
}

// Check for events and pass to appropriate handlers
void SplotchServer::handle_events()
{
  if(server_active)
  {
    // Enforce update rate limit
    camera.mspf = (mspf<max_mspf) ? max_mspf : mspf;
    // Events recieved from control stream server
    while(!cmd_queue.empty())
    { 
      commands.handler(cmd_queue.pop());
    }

    // Events recieved from image stream server
    while(!ims_ev_queue.empty())
    { 
       ims_events.handler(ims_ev_queue.pop());
    }   
  }
}

// Update internal state of server
void SplotchServer::update_state()
{
  // Update time: ms per frame
  #ifdef DEBUG
    // In debug mode output fps to console periodically
    if(server_active && !(frid%100))
    {
      mspf = ft.mark(1, "Splotch frame time: ");
      fflush(0);
    }   
    frid++;
    if(frid == std::numeric_limits<int>::max()) frid = 0; 
  #else 
   mspf = ft.mark(1); 
  #endif 

  // // If nothing is happening, mspf might be 0
  // // Should be at least 1 we will get a div/0 
  // // Probably better to force a wait if nothings happening...
  // if(mspf==0) mspf=32;

  // Reset the scene and image state
  image_modified = false;
  scene_modified = false;
}


// Update the current scene, unloading/loading data if necessary
void SplotchServer::update_scene( std::vector<particle_sim> &pdata,
                                  std::vector<particle_sim> &r_points, render_context& rc)
{
    // Check infile, if it has a new value we should load data
    std::string f = params->find<std::string>("infile","");


    // If we asked to reload the file, set current_file to "" which will autoreload infile
    // This is a poor mans way of changing file contents because file readers dont allow
    // Changing fields without simply reloading
    // if(reload_data)
    // {
    //   pdata.clear();
    //   r_points.clear();
    //   data.unload(); 
    //   data_loaded = false;  
    //   reload_data = false;
    //   current_file = ""; 
    // }

    // Master process handles waiting/loading
    // This wont work for MPI...
    if(server_active)
    {
      // Update resolution
      bool res_updated = update_res(xres, yres, rc);

      // Update data
      if(f != current_file) 
      {
        // If we already had data loaded, unload it
        if(data_loaded){
          unload_data(pdata, r_points);
          clear_interface();
          camera.active = false;
        }

        // If theres no new data to load
        if(f == "")
        {
          // Must redo waiting image if we werent already waiting
          // Or if the resolution changes
          if(!waiting || res_updated)
          {
            params->setParam<std::string>("infile",f);
            current_file = f;
            set_waiting_image();
            image_modified = true;
            waiting = true;   
          }
          return;
        }

        // Spawn loading thread 
        currently_loading = true;
        loading_thread = std::thread( [this]{ 
          set_loading_image();
          while(currently_loading){
            ims_send_queue.Push(image_buffer);
            std::this_thread::sleep_for(std::chrono::milliseconds(loadscreen_interval));
          }
        });
        try {    
          data.read(*params,pdata); 
          data_loaded = true;
          camera.active = true;
          new_scene = true;
          currently_loading = false;  
          waiting = false;    
        }
        catch(const PlanckError& e){
          //spawn_print(e.what());
          currently_loading = false;
          loading_thread.join();
          unload_data(pdata, r_points);
          params->setParam<std::string>("infile","");
          f = "";
          set_waiting_image();
          image_modified = true;
          waiting = true;  
          
        }
        // Set the new file as current
        current_file = f;
      }
      else if(data_loaded)
      {
        // File hasnt changed, if we have data, update the scene
        data.read(*params,pdata);       
      }
      else if(f=="")
      {
        return;
      }
    }
    // End loading if necessary
    // This must be done before any accesses to the image_buffer
    if(loading_thread.joinable())
      loading_thread.join();

    if(new_scene)
      init_scene(pdata, rc);

    // // If we are forcing a state update (e.g. from loading new parameter file)
    // if(send_state_update) {
    //   notify_state();
    //   send_state_update = false;
    // }
    // If the scene is modified, rendering will occur and the image will be too
    if(scene_modified) image_modified = true;

}

bool SplotchServer::update_res(int& xres, int& yres, render_context& rc)
{
    auto it = find_if(modified_params.begin(), modified_params.end(), [](const std::pair<std::string, std::string>& e){
      return(e.first == "xres" || e.first == "yres");
    });

    if(it != modified_params.end())
    {
      xres = params->find<int>("xres",xres);
      yres = params->find<int>("yres",yres);
      image_size = xres*yres*3;
      // Resolution changed
      if(image_size!=image_buffer.size())
      {
        // Update local image buffer
        image_buffer.resize(image_size);
        // Update rendering image buffer and context
        rc.pic.alloc(xres, yres);
        rc.xres = xres;
        rc.yres = yres;
        // Update clients
        //spawn_viewresize();
        return true;
      }      
    }
    return false;
}

void SplotchServer::init_scene(std::vector<particle_sim> &pdata, render_context& rc)
{
  // Fetch the values from the param object we store internally 
  // which may have been altered by the scene file or copied from the opt object.
  // Start with camera 
  rc.campos = vec3(params->find<double>("camera_x"), params->find<double>("camera_y"), params->find<double>("camera_z"));
  rc.lookat = vec3(params->find<double>("lookat_x"), params->find<double>("lookat_y"), params->find<double>("lookat_z"));
  rc.sky = vec3(params->find<double>("sky_x", 0), params->find<double>("sky_y", 0), params->find<double>("sky_z", 1));
  if (params->param_present("center_x"))
  rc.centerpos = vec3(params->find<double>("center_x"), params->find<double>("center_y"), params->find<double>("center_z"));
  else
  rc.centerpos = rc.campos;
  // Server camera is single precision
  camera.reposition(vec3f(rc.campos.x,rc.campos.y,rc.campos.z),
                    vec3f(rc.lookat.x,rc.lookat.y,rc.lookat.z),
                    vec3f(rc.sky.x,rc.sky.y,rc.sky.z));

  // *Local* number of particles 
  n_particles = pdata.size();

  // Colormap
  rc.amap.resize(0);
  get_colourmaps(*params, rc.amap);
  int ptypes = params->find("ptypes", 1);
  for (int itype=0;itype<ptypes;itype++)
  {
    if (params->find<bool>("color_is_vector"+dataToString(itype),false))
    {
      current_maps.push_back("");
    }
    else
    {
      current_maps.push_back(params->find<std::string>("palette"+dataToString(itype)));
    }
  }

  notify_interface(InterfaceType::ALL);

  // Add the normalization filter and refresh
  data.init_filters();
  data.refresh(*params, pdata);

  //std::vector<float> v = {0,100};
  //data.add_dynamic_filter(*params, "SDSS_g_Absolute", FilterTypeId::CLIP, v);
  // Build default names for saveable files
  image_name = params->find<std::string>("outfile","image");
  std::string default_outname = current_file.substr(current_file.find_last_of("/")+1);
  default_outname = default_outname.substr(0,default_outname.find_last_of('.'));
  current_ext = current_file.substr(current_file.find_last_of('.'));
  filtered_data_name = default_outname+"_filtered"+current_ext;

  scene_modified = true;  
  image_modified = true;
  new_scene = false;
}

void SplotchServer::clear_interface() 
{
  interface = interface_descriptor();
}

void  SplotchServer::unload_data(std::vector<particle_sim> &particle_data,
                                 std::vector<particle_sim> &r_points)
{
  current_maps.resize(0);
  std::vector<particle_sim>().swap(particle_data);
  std::vector<particle_sim>().swap(r_points);
  data.unload(); 
  data_loaded = false;     
}


// Update the parameters for Splotch rendering
// If running in parallel, we need to forward parameter updates to other tasks
void SplotchServer::update_parameters(vec3 &campos, vec3 &centerpos, vec3 &lookat, vec3 &sky)
{
  // Packet we send is structured: {header (byte), camera (9xfloat), parameters (map)}
  // header | 0x1 == true if camera data is included
  // header | 0x2 == true if parameter data is included
  enum HEADER: unsigned char {EMPTY = 0, CAMERA = 0x1, PARAMETERS = 0x2 };
  // Camera update needs to be done locally regardless of MPI usage
  #ifdef USE_MPI
  unsigned char header = 0;
  srz::ByteArray packet = srz::Pack(header);   
  #endif

  camera.move_speed = params->find<float>("move_speed", 0.1);
  camera.rotate_speed = params->find<float>("rotate_speed", 0.1);

  // Get camera data
  vec3f pos, target, up;
  if(server_active)
  {
    if(camera.modified)
    {
       // Get local camera info
      pos     = camera.eye();
      target  = camera.target();
      up      = camera.up();      
    }
    #ifdef USE_MPI
    // Pack camera data and flip header bit 
    if(camera.modified) 
    {
      srz::Pack(packet, pos, target, up);
      packet[0] |= HEADER::CAMERA;
    }
    // Send any other modified parameters
    if(modified_params.size())
    {
      srz::Pack(packet, modified_params);
      packet[0] |= HEADER::PARAMETERS;
    }
    #endif
  }

  #ifdef USE_MPI
  // Do the actual sending
  int to_send = packet.size();
  MPI_Bcast(&to_send, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(to_send>sizeof(header))
  {
    packet.resize(to_send);
    MPI_Bcast( &packet[0], packet.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  // Only passive server tasks need to actually unpack the data
  if(server_passive) 
  {
    srz::ConstByteIterator cbi = packet.begin();
    cbi = srz::UnPack(cbi, header);

    if(header & HEADER::CAMERA)
    {
      cbi = srz::UnPack(cbi, pos);
      cbi = srz::UnPack(cbi, target);
      cbi = srz::UnPack(cbi, up);
      camera.modified = true;
    }

    if(header & HEADER::PARAMETERS)
    {
      cbi = srz::UnPack(cbi, modified_params);
      // Then set our local param file with the new params
      for(const std::pair<std::string,std::string>& p : modified_params)
        params->setParam(p.first, p.second);  
    }
  }
  #endif

  // Finally update the camera
  if(camera.modified)
  {
    // Update splotch camera 
    campos = vec3(pos.x,pos.y,pos.z);
    centerpos = vec3(pos.x,pos.y,pos.z);
    lookat = vec3(target.x,target.y,target.z);
    sky = vec3(up.x,up.y,up.z);
    scene_modified = true;
    image_modified = true;
    camera.modified = false;       
  }

    // // Clear the modified parameter list
    // // Force scene update if we changed a parameter
    //if(modified_params.size()){
    //   // Check if we need to update colormap
    //   // int ptypes = params->find("ptypes", 1);
    //   // for(int t = 0; t < ptypes; t++){
    //   //   if(param_modified("palette"+dataToString(t))){
    //   //       try {   
    //   //         replace_colourmap(*params, rc.amap, t);
    //   //         scene_modified = true;
    //   //       } catch(const PlanckError& e) {
    //   //         printf("%s\n",e.what()); 
    //   //       }
    //   //   }
    //     // if(param_modified("quantity"+dataToString(t))){
    //     //     try {   
    //     //       switch(params->find<int>("simtype"))
    //     //       {
    //     //         case 2:
    //     //           params->setParam<std::string>("color_label"+dataToString(t), params->find<std::string>("quantity"+dataToString(t)));
    //     //           break;
    //     //         case 7:
    //     //           params->setParam<std::string>("C1", params->find<std::string>("quantity"+dataToString(t)));
    //     //           break;
    //     //         default:
    //     //         break;
    //     //       }        
    //     //       clear_scene();
    //     //       reload_data=true;      
    //     //       scene_modified = true;
    //     //     } catch(const PlanckError& e) {
    //     //       printf("%s\n",e.what()); 
    //     //     }
    //     // } 
    //   //}

       //scene_modified = true;
   //     modified_params.clear();
   // }
}

void SplotchServer::setup_interface(InterfaceType it)
{
  if(it == InterfaceType::ALL || it == InterfaceType::USER){
    // use this to decide whether to send a username or simply have a file loading window
   // interface.user.constrained = params->find<bool>("constrain_user", false);

  }

  if(it == InterfaceType::ALL || it == InterfaceType::VISUAL){
    // File
    std::string f = params->find<std::string>("infile", "");
    if(f.find_last_of("\\/") != std::string::npos)
      interface.file.name = f.substr(f.find_last_of("\\/"), f.size());
    else 
      interface.file.name = f;
    interface.file.type       = simtype2str((SimTypeId)params->find<int>("simtype", -1));
    interface.file.np         = n_particles;
    interface.file.boxsize[0] = data.particle_fields[FieldId::F_X].normalizer.minv;
    interface.file.boxsize[1] = data.particle_fields[FieldId::F_X].normalizer.maxv;
    interface.file.boxsize[2] = data.particle_fields[FieldId::F_Y].normalizer.minv;
    interface.file.boxsize[3] = data.particle_fields[FieldId::F_Y].normalizer.maxv;
    interface.file.boxsize[4] = data.particle_fields[FieldId::F_Z].normalizer.minv;
    interface.file.boxsize[5] = data.particle_fields[FieldId::F_Z].normalizer.maxv;
    interface.file.box_units  = params->find<std::string>("box_units", "Undefined");

    // Image
    interface.image.xres    = xres;
    interface.image.yres    = yres;
    interface.image.quality = quality;
    interface.image.name = image_name;

    // Interactivity
    interface.interaction.move_speed = camera.move_speed;
    interface.interaction.rotate_speed = camera.rotate_speed;

    // General render settings
    interface.render.colorbar = params->find<bool>("colorbar", false);

    // Type specific settings 
    int ptypes = params->find("ptypes", 1);
    if(interface.types.size()!=ptypes)
      interface.types.resize(ptypes);
    for(int t = 0; t < ptypes; t++)
    {
      interface.types[t].fd.clear();
      for(unsigned i = 0; i < data.particle_fields.size(); i++){
        field_descriptor new_fd;
        new_fd.id = static_cast<FieldId>(i);
        new_fd.name = data.particle_fields[i].name;
        new_fd.range_min = data.particle_fields[i].normalizer.minv;
        new_fd.range_max = data.particle_fields[i].normalizer.maxv;
        interface.types[t].fd.push_back(new_fd);
      }
      interface.types[t].smooth_factor = params->find<float>("smooth_factor"+dataToString(t), 1.0);
      interface.types[t].brightness = params->find<float>("brightness"+dataToString(t), 1.0);
      interface.types[t].field_opts.push_back("");
      interface.types[t].field_opts.insert( interface.types[t].field_opts.end(),
                                            data.available_fields.begin(),
                                            data.available_fields.end() );
    }
  }


  if(it == InterfaceType::ALL || it == InterfaceType::FILTER){
    // Filter menu
    // Add the available filter types to interface desc
    interface.filter_desc.type_opts = {"CLIP", "COMB_FIELD"};
    // Add available fields to interface desc
    interface.filter_desc.field_opts = data.available_fields;
    interface.filter_desc.filters = data.filters;
    interface.filter_desc.data_name = filtered_data_name;
    interface.filter_desc.npoints = data.filtered_size();
  }

}

// Add the current image to the buffer for sending
void SplotchServer::send_image()
{
  // Push back image buffer
  if(server_active && image_modified)
  {
    ims_send_queue.Push(image_buffer);
  }
}

// Close everything down
void SplotchServer::finalize()
{
  if(server_active) 
  {
    // Wait for service threads
    eventsend_thread.join();
    imagesend_thread.join();
    delete(clients);
  }
  // Deactivate server
  server_active = false;
  server_passive = false;
}

// This is analogous to mpiMgr.master()
// Should replace active() with master()
// And replace active_or_passive with active()
bool SplotchServer::active()
{
  return server_active;
}

// Is this task on the server
// This doesnt make much sense
bool SplotchServer::active_or_passive()
{
  return (server_active || server_passive);
}

bool SplotchServer::image_updated()
{
  return image_modified;
}

bool SplotchServer::scene_updated()
{
  return scene_modified;
}

int SplotchServer::get_xres() 
{
  return xres;
}
int SplotchServer::get_yres()
{
  return yres;
}

// Any setup thats needed for communication
void SplotchServer::init_comms()
{
  // Set websockets logging levels
  lws_set_log_level(LLL_ERR, nullptr);
  lws_set_log_level(LLL_WARN, nullptr);
  lws_set_log_level(LLL_NOTICE, nullptr);
  lws_set_log_level(LLL_INFO, nullptr);
  lws_set_log_level(LLL_DEBUG, nullptr);
  lws_set_log_level(LLL_PARSER, nullptr);
  lws_set_log_level(LLL_HEADER, nullptr);
  lws_set_log_level(LLL_EXT, nullptr);
  lws_set_log_level(LLL_CLIENT, nullptr);
  lws_set_log_level(LLL_LATENCY, nullptr);
  lws_set_log_level(LLL_ERR, 
                    [](int /*level*/, const char* msg) {
                        std::cerr << msg << std::endl;
                      });
  // Launch the websockets event and image servers
  ctl = new  WSocketMServer< CommandQueue& >(wsocket_control_protocol, //protocol name
                    200, //timeout: will spend this time to process
                          //websocket traffic, the higher the better
                    wsControlStreamPort, //port
                    cmd_queue, //callback
                    false, //recycle memory
                    0x1000, //input buffer size
                    0);  //min interval between sends in ms
 
  ims = new  WSocketMServer< EventQueue& >(wsocket_image_protocol, //protocol name
                    1000, //timeout: will spend this time to process
                          //websocket traffic, the higher the better
                    wsImageStreamPort, //port
                    ims_ev_queue, //callback
                    false, //recycle memory
                    0x1000, //input buffer size
                    0);  //min interval between sends in ms
}

// Launch the asynuc services for sending/receiving events
void SplotchServer::launch_event_services()
{
  // The event sending service
  auto eventsender = [this]() {
  while(server_active){
         //std::cout << "ctl connected clients: " <<ctl->ConnectedClients() << std::endl;
         if(ctl->ConnectedClients()>0){ 
            std::tuple<ClientId, WSMSGTYPE, std::vector<unsigned char> > tosend  = commands.sender.Pop();
            ctl->Push(std::get<2>(tosend), false, std::get<1>(tosend), std::get<0>(tosend));
         }
         else{
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
         } 
      }
    };
    eventsend_thread = std::thread(eventsender);
}

// Launch the async services for image sending
void SplotchServer::launch_image_services()
{
  quality = params->find<int>("tjpp_quality",95);


  // Create image sending service
  auto imagesender = [this]() {
    tjpp::JPEGImage jpegImage;
    tjpp::TJCompressor comp;
    jpegImage.Reset(xres,yres,TJPF_RGB,TJSAMP_420,quality);
    while(server_active) 
    {
      jpegImage = comp.Compress(std::move(jpegImage), (const unsigned char*) &(ims_send_queue.Pop())[0], xres, yres, TJPF_RGB, TJSAMP_420, quality);
        if(ims->ConnectedClients()>0) ims->Push(SerializeJPEGImage::PackDataOnlyWeb(jpegImage), false);
    }
  };
  imagesend_thread = std::thread(imagesender);
}

void SplotchServer::set_waiting_image()
{
  // Create annotated grey image
  blank_image = LS_Image(xres, yres);
  blank_image.fill(Colour(0,0,0));
  blank_image.annotate_centered(xres/2, yres/3, Colour(1,1,1), "Connected to Splotch Server");
  blank_image.annotate_centered(xres/2, 2*(yres/3), Colour(1,1,1), "Waiting for data input...");
  // Write to image buffer
  for(unsigned i = 0; i < yres; i++)
    for(unsigned j = 0; j < xres; j++)
        *(Colour8*)(&image_buffer[(i * xres + j)*3]) = blank_image.get_pixel(j, i);
}

void SplotchServer::set_loading_image()
{
  // Create annotated grey image
  blank_image = LS_Image(xres, yres);
  blank_image.fill(Colour(0,0,0));
  blank_image.annotate_centered(xres/2, yres/3, Colour(1,1,1), "Connected to Splotch Server");
  blank_image.annotate_centered(xres/2, 2*(yres/3), Colour(1,1,1), "Loading Data");
  // Write to image buffer
  for(unsigned i = 0; i < yres; i++)
    for(unsigned j = 0; j < xres; j++)
        *(Colour8*)(&image_buffer[(i * xres + j)*3]) = blank_image.get_pixel(j, i);
}

// Check if a parameter has been modified by searching the modified parameters vector
bool SplotchServer::param_modified(const std::string& p)
{
  return (std::find_if(modified_params.begin(), modified_params.end(), [&p](const std::pair<std::string, std::string> mp){
    return (p == mp.first);
  }) != modified_params.end());
}

rapidjson::Value SplotchServer::make_interface_descriptor(InterfaceType it, rapidjson::Document& dst)
{
  using namespace rapidjson;

  // Update the specific interface descriptor 
  setup_interface(it);

  // Create the descriptor
  Document::AllocatorType& a = dst.GetAllocator();
  Value desc(kObjectType);

  if(it == InterfaceType::USER || it == InterfaceType::ALL){

    // Username input tied to get_userinfo
    // User info (list of parameters tied to load params)
    Value user_group(kObjectType);
    user_group.AddMember("type", "Group", a);
    Value user_contents(kObjectType);


#ifdef SERVER_USERNAME_REQUEST
    std::vector<std::string> files = clients->retrieve(active_client).files;
    files.push_back("");
    user_contents.AddMember("Username", ui_text_input(a, active_client, "cmd_user"),a);
    user_contents.AddMember("Input File", ui_dropdown_text(a,current_parfile, files),a);  
    std::vector<std::string> args = { "User Settings/Input File" };
    user_contents.AddMember("Load", ui_button(a,"cmd_load", args),a); 
#else
    user_contents.AddMember("Input File", ui_text_input(a,current_parfile),a); 
    // Load expects name & string, we dont need name so just use string twice
    std::vector<std::string> args = { "User Settings/Input File" };
    user_contents.AddMember("Load", ui_button(a,"cmd_load", args),a); 
#endif

    user_group.AddMember("contents", user_contents, a);
    desc.AddMember("User Settings", user_group, a);
  }

  if(it == InterfaceType::VISUAL || it == InterfaceType::ALL){

      // First the file info
    Value file_obj(kObjectType);
    file_obj.AddMember("type", "Group", a);
    Value file_obj_contents(kObjectType);
    file_obj_contents.AddMember("Name", ui_text_display(a, interface.file.name),a);
    file_obj_contents.AddMember("Type", ui_text_display(a, interface.file.type),a);
    file_obj_contents.AddMember("Points", ui_number_display(a, interface.file.np),a);
    file_obj.AddMember("contents", file_obj_contents, a);
    desc.AddMember("File Info", file_obj, a);

    // Then image info
    Value image_obj(kObjectType);
    image_obj.AddMember("type", "Group", a);
    Value image_contents(kObjectType);
    image_contents.AddMember("Xres", ui_number_input(a, interface.image.xres),a);
    image_contents.AddMember("Yres", ui_number_input(a, interface.image.yres),a);
    image_contents.AddMember("Quality", ui_number_display(a, interface.image.quality),a);
    image_contents.AddMember("Filename", ui_text_input(a, interface.image.name),a);
    std::vector<std::string> args = { "Image Info/Filename" };
    image_contents.AddMember("Save", ui_button(a,"cmd_save_image", args),a);
    image_obj.AddMember("contents", image_contents, a);
    desc.AddMember("Image Info", image_obj, a);  

    // Now interactive menu
    Value interation_obj(kObjectType);
    interation_obj.AddMember("type", "Group", a);
    Value interaction_contents(kObjectType);
    interaction_contents.AddMember("move_speed", ui_number_input(a, interface.interaction.move_speed, "cmd_set"),a);
    interaction_contents.AddMember("rotate_speed", ui_number_input(a, interface.interaction.rotate_speed, "cmd_set"),a);
    // std::vector<std::string> args = { "Image Info/Filename" };
    // image_contents.AddMember("Save", ui_button(a,"cmd_save_image", args),a);
    interation_obj.AddMember("contents", interaction_contents, a);
    desc.AddMember("Interactivity", interation_obj, a);      

    // Global Render info
    Value render_obj(kObjectType);   
    render_obj.AddMember("type", "Group", a);
    Value render_contents(kObjectType);
    render_contents.AddMember("colorbar", ui_checkbox(a, interface.render.colorbar, "cmd_set"),a);
     
    // Type specific settings
    for(unsigned i = 0; i < interface.types.size(); i++){
      Value type_obj(kObjectType);
      type_obj.AddMember("type", "Group", a);
      Value type_contents(kObjectType);
      type_contents.AddMember("Type", ui_data(a,dataToString(i)),a);
      type_contents.AddMember(Value().SetString("smooth_factor"+dataToString(i),a), ui_number_input(a, interface.types[i].smooth_factor, "cmd_set"),a);
      type_contents.AddMember(Value().SetString("brightness"+dataToString(i),a), ui_number_input(a, interface.types[i].brightness, "cmd_set"),a);
      // Check current colormap name
      std::vector<std::string>& maps = clients->retrieve(active_client).colourmaps;
      std::string map = current_maps[i]; 
      // Add dropdown text & update button
      type_contents.AddMember("Colourmap", ui_dropdown_text(a, map, maps),a); 
      std::vector<std::string> args = { "Render Settings/Type "+dataToString(i)+"/Type",
                                        "Render Settings/Type "+dataToString(i)+"/Colourmap"};
      type_contents.AddMember("Update Colourmap", ui_button(a,"cmd_replace_colourmap", args),a);

      for(unsigned j = 0; j < interface.types[i].fd.size(); j++){

        Value field_obj(kObjectType);
        field_obj.AddMember("type","Group", a);
        Value field_contents(kObjectType);
        std::string field_id = FieldId2str(interface.types[i].fd[j].id);
        field_contents.AddMember("Field", ui_dropdown_text( a, interface.types[i].fd[j].name, 
                                                            interface.types[i].field_opts), a);
        field_contents.AddMember("Min", ui_number_display(a, interface.types[i].fd[j].range_min),a);
        field_contents.AddMember("Max", ui_number_display(a, interface.types[i].fd[j].range_max),a);
        field_contents.AddMember("ID", ui_data(a,field_id),a);
        std::vector<std::string> args = { "Render Settings/Type "+dataToString(i)+"/"+field_id+"/ID",
                                          "Render Settings/Type "+dataToString(i)+"/"+field_id+"/Field"};
        field_contents.AddMember("Update", ui_button(a,"cmd_reload_field", args),a); 
        field_obj.AddMember("contents", field_contents, a);
        type_contents.AddMember(Value().SetString(field_id, a), field_obj, a);
      }  
      type_obj.AddMember("contents", type_contents, a);
      render_contents.AddMember(Value().SetString(std::move("Type "+dataToString(i)),a), type_obj, a);  
    }
    render_obj.AddMember("contents", render_contents, a);
    desc.AddMember("Render Settings", render_obj, a);   
  }

  if(it == InterfaceType::FILTER || it == InterfaceType::ALL){
    // Info for filtering interface
    Value filter_obj(kObjectType);
    filter_obj.AddMember("type", "Group", a);
    Value filter_contents(kObjectType);
    // Add the filtered data submenu
    Value filter_data_obj(kObjectType);
    filter_data_obj.AddMember("type", "Group", a);
    Value filter_data_contents(kObjectType);
    //filter_data_contents.AddMember("Points", ui_number_display(a, interface.filter_desc.npoints),a); 
    filter_data_contents.AddMember("Name", ui_text_input(a, interface.filter_desc.data_name),a);
    std::vector<std::string> args = {"Filters/Filtered Data/Name"};
    filter_data_contents.AddMember("Save", ui_button(a,"cmd_save_filtered_data", args),a); 
    filter_data_obj.AddMember("contents", filter_data_contents, a);
    filter_contents.AddMember("Filtered Data", filter_data_obj, a);

    // Add the new filter submenu
    Value new_filter_obj(kObjectType);
    new_filter_obj.AddMember("type", "Group", a);
    Value new_filter_contents(kObjectType);
    new_filter_contents.AddMember("Type", ui_dropdown_text(a, interface.filter_desc.type_opts[0], interface.filter_desc.type_opts),a); 
    new_filter_contents.AddMember("Field", ui_dropdown_text(a, interface.filter_desc.field_opts[0], interface.filter_desc.field_opts),a);
    args = {"Filters/Create Filter/Type", "Filters/Create Filter/Field"};
    new_filter_contents.AddMember("Create", ui_button(a,"cmd_create_filter", args),a); 
    new_filter_obj.AddMember("contents", new_filter_contents, a);
    filter_contents.AddMember("Create Filter", new_filter_obj, a);

    // Then the current filters, skipping the last one because its a built in normalizer
    int n_filt = interface.filter_desc.filters.size()-1;
    for(int i = 0; i < n_filt; i++){
      filter_contents.AddMember(Value().SetString("Filter "+dataToString(interface.filter_desc.filters[i].id), a), ui_filter(a, interface.filter_desc.filters[i], interface.filter_desc.field_opts), a);
    }
    filter_obj.AddMember("contents", filter_contents, a);
    desc.AddMember("Filters", filter_obj, a);
  }
 
  return desc;
}

void SplotchServer::update_camera_parameters(){
  vec3f camPos = camera.eye();
  vec3f camTarget = camera.target();
  vec3f camUp = camera.up();
  params->setParam<double>("camera_x", camPos.x);
  params->setParam<double>("camera_y", camPos.y);
  params->setParam<double>("camera_z", camPos.z);
  params->setParam<double>("lookat_x",camTarget.x);
  params->setParam<double>("lookat_y",camTarget.y);
  params->setParam<double>("lookat_z",camTarget.z);
  params->setParam<double>("sky_x",camUp.x);
  params->setParam<double>("sky_y",camUp.y);
  params->setParam<double>("sky_z",camUp.z);    
}

std::string SplotchServer::metadata(){
  setup_interface(InterfaceType::ALL);
  update_camera_parameters();
  rapidjson::StringBuffer s;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> w(s);    
  w.StartObject();
  // Store the filename, type, original point count
  w.Key("Filename");
  w.String(current_file);
  w.Key("Type");
  w.String(interface.file.type);
  w.Key("Original Points");
  w.Uint(interface.file.np);
  //w.Key("Filtered Points");
  //w.Uint();
  w.Key("Filters");
  w.StartObject();
  int start_filter = 1;
  for(unsigned i = start_filter; i < interface.filter_desc.filters.size(); i++){
    const filter& f = interface.filter_desc.filters[i];
    w.Key("Filter "+dataToString(f.id));
    w.StartObject();
    w.Key("Type");
    w.String(FilterTypeId2str(f.type));
    w.Key("Field");
    w.String(f.name);
    switch(f.type){
      case FilterTypeId::CLIP: 
      w.Key("DefaultMin");
      w.String(f.defaults[0]);
      w.Key("DefaultMax");
      w.String(f.defaults[1]);
      w.Key("Min");
      w.String(f.args[0]);
      w.Key("Max");
      w.String(f.args[1]);
      break;
      default:
      break;
    }
    w.EndObject();
  }
  w.EndObject();
  w.Key("Parameters");
  w.StartObject();
  // Parameters
  using params_type = std::map<std::string,std::string>;
  const params_type& pars = params->getParams();
  for(auto it=pars.begin();it!=pars.end();it++){
    w.Key(it->first);
    w.String(it->second);
  }
  w.EndObject();
  w.EndObject();

  std::string result = s.GetString();

  return result;
}

// // Spawn a client print command
// void SplotchServer::spawn_print(std::string s) 
// {
//   std::vector<char> data;
//   ClientEventId event_type = ClientEventId::PRINT;
//   data = srz::Pack(event_type, s); 
//   printf("spawn_print: pushing to queue %lu bytes\n", data.size());
//   ctl_send_queue.Push(data);
// }

// // Spawn a client resize command
// void SplotchServer::spawn_viewresize() 
// {
//   std::vector<char> data;
//   ClientEventId event_type = ClientEventId::RESIZE;
//   data = srz::Pack(event_type, xres, yres); 
//   printf("spawn_resize: pushing to queue %lu bytes\n", data.size());
//   ctl_send_queue.Push(data);
// }

// // Spawn a client save command
// void SplotchServer::spawn_save() 
// {
//   // Update the parameter file with the current render parameters
//   vec3f camPos = camera.eye();
//   vec3f camTarget = camera.target();
//   vec3f camUp = camera.up();
//   params->setParam<double>("camera_x", camPos.x);
//   params->setParam<double>("camera_y", camPos.y);
//   params->setParam<double>("camera_z", camPos.z);
//   params->setParam<double>("lookat_x",camTarget.x);
//   params->setParam<double>("lookat_y",camTarget.y);
//   params->setParam<double>("lookat_z",camTarget.z);
//   params->setParam<double>("sky_x",camUp.x);
//   params->setParam<double>("sky_y",camUp.y);
//   params->setParam<double>("sky_z",camUp.z);     

//   // Serialize and send
//   // Simplest way to insert length of map is to compute after serializing
//   std::vector<char> data;
//   ClientEventId event_type = ClientEventId::FILE_DOWNLOAD;
//   int msg_size=0;
//   data = srz::Pack(event_type, msg_size, params->getParams());
//   msg_size = data.size() - 2*sizeof(int);
//   *((int*)&data[4]) = msg_size;
//   printf("spawn_save: pushing to queue %lu bytes\n", data.size());
//   ctl_send_queue.Push(data);
// }

// void SplotchServer::spawn_filelist(const std::vector<std::string>& vs) 
// {
//   std::vector<char> data;
//   ClientEventId event_type = ClientEventId::FILELIST;
//   data = srz::Pack(event_type, vs); 
//   printf("spawn_print: pushing to queue %lu bytes\n", data.size());
//   ctl_send_queue.Push(data);
// }


// void SplotchServer::notify_state()
// {
//   rapidjson::Value state(rapidjson::kObjectType);
//   // Update the parameter file with the current render parameters
//   vec3f camPos = camera.eye();
//   vec3f camTarget = camera.target();
//   vec3f camUp = camera.up();
//   params->setParam<double>("camera_x", camPos.x);
//   params->setParam<double>("camera_y", camPos.y);
//   params->setParam<double>("camera_z", camPos.z);
//   params->setParam<double>("lookat_x",camTarget.x);
//   params->setParam<double>("lookat_y",camTarget.y);
//   params->setParam<double>("lookat_z",camTarget.z);
//   params->setParam<double>("sky_x",camUp.x);
//   params->setParam<double>("sky_y",camUp.y);
//   params->setParam<double>("sky_z",camUp.z);     

//   // params_type is private member of paramfile...
//   using params_type = std::map<std::string,std::string>;
//   const params_type& pars = params->getParams();
//   for(auto it=pars.begin();it!=pars.end();it++)
//    state.AddMember(rapidjson::Value(it->first, rjdoc.GetAllocator()), rapidjson::Value(it->second, rjdoc.GetAllocator()), rjdoc.GetAllocator());

//   commands.send_client_message(BroadcastId(), commands.client_notify(rjdoc, "cmd_state", std::move(state)));
// }

void SplotchServer::notify_interface(InterfaceType it)
{
  using namespace rapidjson;
  // Create interface object
  Value v(kObjectType);
  v = make_interface_descriptor(it, rjdoc);
  // Send to all users
  commands.send_client_message(BroadcastId(), commands.client_notify(rjdoc, "cmd_interface_desc", std::move(v)));
}

void SplotchServer::notify_binary_download(ClientId clid, std::vector<unsigned char>& vec)
{
  commands.send_binary_message(clid, vec);
}

void SplotchServer::init_cmds()
{
  auto make_func = [this](rapidjson::Value (SplotchServer::*func)(ClientId, rapidjson::Document&, rapidjson::Document&)) {
    return [this, func] (ClientId clid, rapidjson::Document& src, rapidjson::Document& dst) {
      return (this->*func)(clid, src, dst);
    };
  };
 // commands.add_cmd(CommandId::NULL_CMD,   make_func(&SplotchServer::cmd_null));
  // commands.add_cmd(CommandId::HELP,      make_func(&SplotchServer::cmd_help));
  commands.add_cmd("cmd_set",                 make_func(&SplotchServer::cmd_set));
  //commands.add_cmd("cmd_notify",           make_func(&SplotchServer::cmd_notify));
  commands.add_cmd("cmd_get",                 make_func(&SplotchServer::cmd_get));
  commands.add_cmd("cmd_state",               make_func(&SplotchServer::cmd_state)); 
  commands.add_cmd("cmd_interface_desc",      make_func(&SplotchServer::cmd_interface_desc));
  commands.add_cmd("cmd_user",                make_func(&SplotchServer::cmd_user));
  commands.add_cmd("cmd_get_image",           make_func(&SplotchServer::cmd_get_image));
  commands.add_cmd("cmd_save_image",          make_func(&SplotchServer::cmd_save_image));
  commands.add_cmd("cmd_update_image_params", make_func(&SplotchServer::cmd_update_image_params));
  commands.add_cmd("cmd_replace_colourmap",   make_func(&SplotchServer::cmd_replace_colourmap));
  // commands.add_cmd(CommandId::ADD,       make_func(&SplotchServer::cmd_add));
  // commands.add_cmd(CommandId::SAVE,      make_func(&SplotchServer::cmd_save));
  // commands.add_cmd(CommandId::UPLOAD,    make_func(&SplotchServer::cmd_upload));
  // commands.add_cmd(CommandId::FILELIST,  make_func(&SplotchServer::cmd_filelist));
  commands.add_cmd("cmd_load",                make_func(&SplotchServer::cmd_load));
  commands.add_cmd("cmd_reload_field",        make_func(&SplotchServer::cmd_reload_field));
  // commands.add_cmd(CommandId::LOAD_ID,   make_func(&SplotchServer::cmd_load_id));
  // commands.add_cmd(CommandId::UNLOAD,    make_func(&SplotchServer::cmd_unload));
  // commands.add_cmd(CommandId::RELOAD,    make_func(&SplotchServer::cmd_reload));
  commands.add_cmd("cmd_create_filter",       make_func(&SplotchServer::cmd_create_filter));
  commands.add_cmd("cmd_update_filter",       make_func(&SplotchServer::cmd_update_filter));
  commands.add_cmd("cmd_remove_filter",       make_func(&SplotchServer::cmd_remove_filter));
  commands.add_cmd("cmd_save_filtered_data",  make_func(&SplotchServer::cmd_save_filtered_data));

}
 
// // Null command does nothing
// void SplotchServer::cmd_null(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   //printf("Server: null command\n");
// }

// // Get an existing parameter from the parameter file
// void SplotchServer::cmd_help(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(0, args_start, args_end)) return;
//   // Send help string
//   std::string s = 
//   R"( 
// -------------------
// --- Server info ---
// -------------------
// How to interact and control the server. 

// ------ Mouse ------
// Left click/Drag: Rotate around view target (default: global (0,0,0))
// Shift + Left click/drag: Rotate view target around camera origin

// ----- Keyboard ----
// w - Move camera forward
// a - Move camera left
// s - Move camera backward
// d - Move camera right
// q - Move camera up
// e - Move camera down
// shift+w - Move camera & view target forward
// shift+a - Move camera & view target left
// shift+s - Move camera & view target backward
// shift+d - Move camera & view target right
// shift+q - Move camera & view target up
// shift+e - Move camera & view target down

// ----- Commands ----
// These are typed into the command terminal, and sent by pressing return.
// Required arguments are denoted by e.g. X
// Optional arguments are denoted by e.g. [X]

// help
//   Print this Server Info dialog
// set X Y
//   Set existing key X to value Y in Splotch parameter file
// get X 
//   Print value Y for existing key X in Splotch parameter file
// add X Y
//   Add new key X with value Y 
// save [S]
//   Save server config file named S  
// upload
//   Upload a server config file
// load S
//   Load datafile S
// unload
//   Unload current datafile

//  )";
//  // Two more to add if/when filelist command is used
// // filelist
// //   List a set of files available to load ()
// // load_id I
// //   Load datafile by integer id listed in filelist command 
// spawn_print(s);
// }


// Set a parameter in the parameter file
rapidjson::Value SplotchServer::cmd_set(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::vector<std::string> keys, values;
  // If params are an array, it is assumed alternating key value strings
  // Hence require even number of strings >= 2
  if(src["params"].IsArray()) {
    if ((src["params"].Size() < 2) || (src["params"].Size() & 1) || !src["params"][0].IsString()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    for(unsigned i = 0; i < src["params"].Size(); i+=2) {
      keys.push_back(src["params"][i].GetString());
      values.push_back(src["params"][i+1].GetString());
    }
  } 
  // If they are an object, set key value pairs
  else if(src["params"].IsObject()) {
    for (auto itr = src["params"].MemberBegin(); itr != src["params"].MemberEnd(); ++itr) {
      keys.push_back(itr->name.GetString());
      values.push_back(itr->value.GetString());
    }
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }

  
  for(unsigned i = 0; i < keys.size(); i++){
   if(params->param_present(keys[i]))
    {
      params->setParam(keys[i], values[i]);
      // Check whether to update an existing modified parameter, or to add new one
      auto it = std::find_if(modified_params.begin(), modified_params.end(), [&keys, i](const std::pair<std::string, std::string>& e){
        return e.first == keys[i];
      });
      if(it!=modified_params.end())
      {
        it->first  = keys[i];
        it->second = values[i];
      }
      else modified_params.emplace_back(keys[i], values[i]);
       printf("Server: setting parameter %s to %s\n", keys[i].c_str(), values[i].c_str());
    }
    else
      return commands.error_value(dst, JRPCErr::SERVER_ERR, "cmd_set() Key "+keys[i]+" not found.");    
  }
 
  scene_modified = true;
  return commands.result_value(dst, rapidjson::Value(0));
}

// Get an existing parameter from the parameter file
rapidjson::Value SplotchServer::cmd_get(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  // Validate parameters
  // Only support by-position arguments
  if(!src.HasMember("params") || !src["params"].IsArray() || (src["params"].Size() != 1) || !src["params"][0].IsString()){
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }

  std::string key(src["params"][0].GetString());
  printf("Getting parameter %s\n", key.c_str());
  std::string val = params->find<std::string>(key, "parameter not found");
  printf("Parameter %s: %s\n", key.c_str(), val.c_str());

  return commands.result_value(dst, rapidjson::Value(val.c_str(), dst.GetAllocator()));
}

// // Add a new parameter to the Splotch parameter map
// void SplotchServer::cmd_add(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(2, args_start, args_end)) return;
//   srz::ConstByteIterator current = args_start;
//   std::string key, value;
//   current = srz::UnPack(current, key);
//   current = srz::UnPack(current, value);
//   params->setParam(key, value);
//   // Check whether to update an existing modified parameter, or to add new one
//   auto it = std::find_if(modified_params.begin(), modified_params.end(), [&key](const std::pair<std::string, std::string>& e){
//     return e.first == key;
//   });
//   if(it!=modified_params.end())
//   {
//     it->first  = key;
//     it->second = value;
//   }
//   else modified_params.emplace_back(key, value);
// }

// Get the current server state
// Returns a JS map equivalent to the current parameter file
rapidjson::Value SplotchServer::cmd_state(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  rapidjson::Value state(rapidjson::kObjectType);

  // Update the parameter file with the current render parameters
  vec3f camPos = camera.eye();
  vec3f camTarget = camera.target();
  vec3f camUp = camera.up();
  params->setParam<double>("camera_x", camPos.x);
  params->setParam<double>("camera_y", camPos.y);
  params->setParam<double>("camera_z", camPos.z);
  params->setParam<double>("lookat_x",camTarget.x);
  params->setParam<double>("lookat_y",camTarget.y);
  params->setParam<double>("lookat_z",camTarget.z);
  params->setParam<double>("sky_x",camUp.x);
  params->setParam<double>("sky_y",camUp.y);
  params->setParam<double>("sky_z",camUp.z);     

  // params_type is private member of paramfile...
  using params_type = std::map<std::string,std::string>;
  const params_type& pars = params->getParams();
  for(auto it=pars.begin();it!=pars.end();it++)
   state.AddMember(rapidjson::Value(it->first, dst.GetAllocator()), rapidjson::Value(it->second, dst.GetAllocator()), dst.GetAllocator());

  return commands.result_value(dst, std::move(state));
}



rapidjson::Value SplotchServer::cmd_interface_desc(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;
  Value v(kObjectType);
  v = make_interface_descriptor(InterfaceType::ALL, dst);
  return commands.result_value(dst, std::move(v));
}

// // Save comand downloads a splotch config file
// // This is a serialized map of Splotch parameters
// void SplotchServer::cmd_save(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(0, args_start, args_end)) return;
//   // Spawn event on control channel
//   printf("Sending savepoint\n");
//   spawn_save();
// }

// // Upload command recieves a file previously downloaded with the save command
// // File contains a serialized map of Splotch parameters which are then set
// void SplotchServer::cmd_upload(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(1, args_start, args_end)) return;
  
//   // Debug print message size
//   int bytes = args_end - args_start;
//   printf("Received upload file of %d bytes\n", bytes);
//   // Unpack map length and then map
//   int msg_size;
//   args_start = srz::UnPack(args_start, msg_size);  
//   std::map<std::string, std::string> input;
//   srz::UnPack(args_start, input);
//   // Set Splotch parameters according to map
//   std::map<std::string, std::string>::iterator it;
//   for(it=input.begin();it!=input.end();it++)
//   {
//     std::string key = it->first;
//     std::string value = it->second;
//     params->setParam(key, value);
//   }
// }

// // Print a list of available files
rapidjson::Value SplotchServer::cmd_user(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
      // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string name;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 2) || !src["params"][0].IsString()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    name = (src["params"][1].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    for (auto itr = src["params"].MemberBegin(); itr != src["params"].MemberEnd(); ++itr) {
      name = itr->value.GetString();
    }
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }


  if(clients->exists(name)){
    active_client = name;
    notify_interface(InterfaceType::USER);
    return commands.result_value(dst, rapidjson::Value(0));
  }

  return commands.error_value(dst, JRPCErr::SERVER_ERR, "Client "+name+" not found.");    
}

rapidjson::Value SplotchServer::cmd_get_image(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;
  // Create download ID
  int dlid =0;
      dlid = clients->retrieve(active_client).download_id++;

    tjpp::JPEGImage jpegImage;
    tjpp::TJCompressor comp;
    jpegImage.Reset(xres,yres,TJPF_RGB,TJSAMP_420,quality);
    jpegImage = comp.Compress(std::move(jpegImage), (const unsigned char*) &image_buffer[0], xres, yres, TJPF_RGB, TJSAMP_420, quality);

  // Trigger a binary download notification 
  notify_binary_download(clid, SerializeJPEGImage::PackDataOnlyWeb(jpegImage));

  // Return download ID
  Value v(kObjectType);
  v.AddMember("id", dlid, dst.GetAllocator());
  return commands.result_value(dst, std::move(v));  
}

rapidjson::Value SplotchServer::cmd_save_image(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst){
  using namespace rapidjson;
  // Get name
  // Validate parameters
  if(!src.HasMember("params")) 
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string name;
  // If it is an array, take the first
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 1) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    name = (src["params"][0]["Filename"].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()        && 
          src["params"].HasMember("name") && 
          src["params"]["name"].IsString()) {
    name = (src["params"]["name"].GetString());
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }

  // Get image 
  tjpp::JPEGImage jpegImage;
  tjpp::TJCompressor comp;
  jpegImage.Reset(xres,yres,TJPF_RGB,TJSAMP_420,quality);
  jpegImage = comp.Compress(std::move(jpegImage), (const unsigned char*) &image_buffer[0], xres, yres, TJPF_RGB, TJSAMP_420, quality);

  // Get metadata 
  std::string meta = metadata();

  // Get write location
  std::string path = params->find<std::string>("output_dir","./");
  std::string meta_file = path+name+".txt";
  std::string image_file = path+name+".jpg";

  // Save to write location
  std::ofstream outfile_meta(meta_file);
  if(outfile_meta.is_open()){
    outfile_meta << meta;
    outfile_meta.close();
  } else {
    return commands.error_value(dst, JRPCErr::INV_PARS, std::move(Value().SetString("Could not open file "+meta_file+" for writing", dst.GetAllocator())));
  }
  std::ofstream outfile_image(image_file);
  if(outfile_image.is_open()){
    std::vector<unsigned char> v = SerializeJPEGImage::PackDataOnlyWeb(jpegImage);
    std::copy((char*)&v[0], (char*)&v[v.size()], std::ostreambuf_iterator<char>(outfile_image));
    outfile_image.close();
  } else {
    return commands.error_value(dst, JRPCErr::INV_PARS, std::move(Value().SetString("Could not open file "+path+" for writing", dst.GetAllocator())));
  }

  return commands.result_value(dst, rapidjson::Value(0));
}


rapidjson::Value SplotchServer::cmd_update_image_params(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst){
  using namespace rapidjson;
  // Get name
  // Validate parameters
  if(!src.HasMember("params")) 
    return commands.error_value(dst, JRPCErr::INV_PARS);

  int x,y,q;
  // If it is an array, take the first
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 1) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    x = (src["params"][0]["Xres"].GetInt());
    y = (src["params"][0]["Yres"].GetInt());
    q = (src["params"][0]["Quality"].GetInt());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    x = (src["params"]["Xres"].GetInt());
    y = (src["params"]["Yres"].GetInt());
    q = (src["params"]["Quality"].GetInt());

  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }

  if(x != xres || y != yres)
  {
    printf("Update resolution");
  } 

  if(q!= quality){
    printf("Update quality");
  }

  return commands.result_value(dst, rapidjson::Value(0));
}


// Load file by name
rapidjson::Value SplotchServer::cmd_load(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string file;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 1) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    file = (src["params"][0]["Input File"].GetString());

  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    for (auto itr = src["params"].MemberBegin(); itr != src["params"].MemberEnd(); ++itr) {
      file = itr->value.GetString();
    }
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
  paramfile p;
  try{
    p = paramfile(file, false);
    *params = p;
    const std::map<std::string,std::string>&  m = params->getParams();
    for(auto e : m)
      modified_params.push_back(e);
    current_parfile = file;
  } catch(const PlanckError& e) {
    return commands.error_value(dst, JRPCErr::SERVER_ERR, std::move(Value().SetString(e.what(), dst.GetAllocator())));
  }

  return commands.result_value(dst, rapidjson::Value(0));
}

rapidjson::Value SplotchServer::cmd_reload_field(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string particle_fid, data_fid;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 2) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    particle_fid = (src["params"][0]["ID"].GetString());
    data_fid = (src["params"][1]["Field"].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    particle_fid = (src["params"]["ID"].GetString());
    data_fid = (src["params"]["Field"].GetString());    
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
    

  data.reload_field(*params, particle_fid, data_fid);
  notify_interface(InterfaceType::VISUAL);
  scene_modified = true;

  return commands.result_value(dst, rapidjson::Value(0));
}

// Add a filter
rapidjson::Value SplotchServer::cmd_create_filter(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string type, field;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 2) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    type  = (src["params"][0]["Type"].GetString());
    field = (src["params"][1]["Field"].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    type  = (src["params"]["Type"].GetString());
    field = (src["params"]["Field"].GetString());
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
  
  data.create_filter(*params, field, Str2FilterTypeId(type));
  notify_interface(InterfaceType::FILTER);

  return commands.result_value(dst, rapidjson::Value(0));
}

// Update a filter
rapidjson::Value SplotchServer::cmd_update_filter(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::vector<std::string> args;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if (!src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    for (auto& v : src["params"].GetArray())
      for (Value::ConstMemberIterator itr = v.MemberBegin(); itr != v.MemberEnd(); ++itr)
         args.push_back(itr->value.GetString());
  }
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    for (Value::ConstMemberIterator itr = src["params"].MemberBegin(); itr != src["params"].MemberEnd(); ++itr)
    {
      args.push_back(itr->value.GetString());
    }
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
  
  data.update_dynamic_filter(*params, args);
  scene_modified = true;

  return commands.result_value(dst, rapidjson::Value(0));
}


// Add a filter
rapidjson::Value SplotchServer::cmd_remove_filter(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  int id;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 1) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    stringToData(src["params"][0]["ID"].GetString(), id);
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    stringToData(src["params"]["ID"].GetString(), id);
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
  
  data.remove_dynamic_filter(id);
  notify_interface(InterfaceType::FILTER);

  return commands.result_value(dst, rapidjson::Value(0));
}

rapidjson::Value SplotchServer::cmd_save_filtered_data(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;
  // Get name
  // Validate parameters
  if(!src.HasMember("params")) 
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string name;
  // If it is an array, take the first
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 1) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    // Should also check on has member
    name = (src["params"][0]["Name"].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()        && 
          src["params"].HasMember("Name") && 
          src["params"]["Name"].IsString()) {
    name = (src["params"]["Name"].GetString());
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }

  // Get metadata 
  std::string meta = metadata();

  // Get write location
  std::string path = params->find<std::string>("output_dir","");
  std::string n = name;
  std::string meta_file = path+n.substr(0,n.find_last_of('.'))+".txt";
  std::string data_file = path+n;

  data.write_filtered(*params, data_file);

  // Save to write location
  std::ofstream outfile_meta(meta_file);
  if(outfile_meta.is_open()){
    outfile_meta << meta;
    outfile_meta.close();
  } else {
    return commands.error_value(dst, JRPCErr::INV_PARS, std::move(Value().SetString("Could not open file "+meta_file+" for writing", dst.GetAllocator())));
  }

  return commands.result_value(dst, rapidjson::Value(0));
}

// Add a filter
rapidjson::Value SplotchServer::cmd_replace_colourmap(ClientId clid, rapidjson::Document& src, rapidjson::Document& dst)
{
  using namespace rapidjson;

  // Validate parameters
  if(!src.HasMember("params"))
    return commands.error_value(dst, JRPCErr::INV_PARS);

  std::string type, name;
  // If it is an array, take the second element
  if(src["params"].IsArray()) {
    if ((src["params"].Size() != 2) || !src["params"][0].IsObject()) {
      return commands.error_value(dst, JRPCErr::INV_PARS);
    }
    type  = (src["params"][0]["Type"].GetString());
    name = (src["params"][1]["Colourmap"].GetString());
  } 
  // If it is an object, take the value of the last member 
  else if(src["params"].IsObject()) {
    type  = (src["params"]["Type"].GetString());
    name = (src["params"]["Colourmap"].GetString());
  } else {
    // Otherwise return err
    return commands.error_value(dst, JRPCErr::INV_PARS);
  }
  
  // Check type is valid
  int ptypes = params->find("ptypes", 1);
  int t = stringToData<int>(type);
  if(t < 0 || t >=ptypes) {
    return commands.error_value(dst, JRPCErr::INV_PARS, std::move(Value().SetString("Type out of range", dst.GetAllocator())));
  }

  // Replace the colourmap
  try{   
    replace_colourmap(*params, context->amap, t, name);
    current_maps[t] = name;
    scene_modified = true;
  } catch(const PlanckError& e) {
    return commands.error_value(dst, JRPCErr::SERVER_ERR, std::move(Value().SetString(e.what(), dst.GetAllocator())));
  }

  return commands.result_value(dst, rapidjson::Value(0));
}


    //   // int ptypes = params->find("ptypes", 1);
    //   // for(int t = 0; t < ptypes; t++){
    //   //   if(param_modified("palette"+dataToString(t))){
    //   //       try {   
    //   //         replace_colourmap(*params, rc.amap, t);
    //   //         scene_modified = true;
    //   //       } catch(const PlanckError& e) {
    //   //         printf("%s\n",e.what()); 
    //   //       }
    //   //   }


// // Load file by id from filelist
// void SplotchServer::cmd_load_id(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(1, args_start, args_end)) return;
//   srz::ConstByteIterator current = args_start;
//   std::string fileid;
//   current = srz::UnPack(args_start, fileid);
//   int id = std::stoi(fileid);
//   const std::vector<std::string> files = clients->get_filelist();
//   if(id < files.size() && id > -1)
//     params->setParam("infile", files[id]);
//   else
//     spawn_print("Filelist option "+fileid+" not found in filelist\n");
// }

// // Unload current data
// void SplotchServer::cmd_unload(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(0, args_start, args_end)) return;
//   params->setParam("infile", std::string(""));
// }

// // Reload current data
// void SplotchServer::cmd_reload(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end)
// {
//   if(!commands.validate_argc(0, args_start, args_end)) return;
//   reload_data = true;
// }




