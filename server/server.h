#ifndef SPLOTCH_SERVER_H
#define SPLOTCH_SERVER_H

#include <signal.h>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>
#include <functional>
#include <utility>

#include "cxxsupport/paramfile.h"
#include "cxxsupport/arr.h"
#include "cxxsupport/ls_image.h"
#include "kernel/colour.h"
#include "splotch/scenemaker.h"
#include "splotch/splotchutils.h"

#include "event.h"
#include "controller.h"
#include "command.h"
#include "interface.h"

#include "WSocketMServer.h"

#include "SyncQueue.h"
#define ZRF_int32_size
#include "Serialize.h"

#include "fast_timer.h"
#include "data.h"

class SplotchServer
{
public:
	SplotchServer() {}
	void  init(paramfile& p, bool master);
  void  run(sceneMaker& sMaker, std::vector<particle_sim> &particle_data,
            std::vector<particle_sim> &r_points, render_context& rc);
	void  handle_events();
  void  update_state();
  void  update_scene( std::vector<particle_sim> &particle_data, 
                      std::vector<particle_sim> &r_points, render_context& rc);
  void  update_parameters(vec3 &campos, vec3 &centerpos, vec3 &lookat, vec3 &sky);
  void  update_camera_parameters();

  bool  update_res(int& x, int& y, render_context& rc);
  void  init_scene(std::vector<particle_sim> &pdata, render_context& rc);

  rapidjson::Value make_interface_descriptor(InterfaceType it, rapidjson::Document& dst);
  void  setup_interface(InterfaceType it);
  void  clear_interface();
  void  send_image(); 
  void  unload_data(std::vector<particle_sim> &particle_data,
          std::vector<particle_sim> &r_points);
	void  finalize();
	bool  active();
  bool  active_or_passive();
  bool  image_updated();
  bool  scene_updated();
  int   get_xres();
  int   get_yres();
  std::string metadata();

  fast_timer ft;
  std::vector<char> image_buffer;

  WSocketMServer< EventQueue& >* ims;
  WSocketMServer< CommandQueue& >* ctl;
  std::string wsocket_image_protocol;
  std::string wsocket_control_protocol;
  const int wsControlStreamPort = 8882;
  const int wsImageStreamPort = 8881;


  interface_descriptor interface;
  long n_particles;

  // // Make these public until internal action queue is implemented
  // void spawn_print(std::string s);
  // void spawn_viewresize();
  bool image_modified         = false;

  // Notifications
  rapidjson::Document rjdoc;
  //void notify_state();
  void notify_interface(InterfaceType it);
  void notify_binary_download(ClientId clid, std::vector<unsigned char>& vec);
  void notify_binary_download(ClientId clid, std::vector<unsigned char>&& vec){
    notify_binary_download(clid,vec);
  };
private:
  EventQueue ims_ev_queue;
  CommandQueue cmd_queue;
  Data data;
  render_context* context = NULL;
  // Spawning events to send
  //void spawn_print(std::string s);
  // //void spawn_viewresize();
  // void spawn_save();
  // void spawn_filelist(const std::vector<std::string>& vs);

  // Communication
  void init_comms();
  void launch_event_services();
  void launch_image_services();

  // Waiting/loading
  void set_waiting_image();
  void set_loading_image();

  // Scene handling
  bool param_modified(const std::string& p);

  bool server_active    = false;
  bool server_passive   = false;
  paramfile* params;

  // Scene
  std::string current_file    = "";
  std::string current_parfile = "";
  std::string current_ext     = "";
  bool scene_modified         = false;
 // bool image_modified         = false;
  bool waiting                = false;
  bool new_scene              = false;
  bool send_state_update      = false; 

  // Colormaps
  std::vector<std::string> current_maps;
  std::vector<std::string> available_maps;

  // Data loading
  bool currently_loading          = false;
  bool data_loaded                = false;
  const int loadscreen_interval   = 250; 
  bool reload_data                = false;

  // Image
  int xres                    = 0;
  int yres                    = 0;
  int image_size              = -1;
  int quality                 = 0;
  std::string image_name      = "";
  LS_Image blank_image;

  // Filters
  std::string filtered_data_name = "";

  // Framerate
  int   frid                  = 0;
  float mspf                  = 0;
  float ideal_mspf            = 32;
  float max_mspf              = 32;

  // Controllers 
  EventController     ims_events;
  CameraController    camera;
  ClientController*    clients;
  bool valid_client = false;
  std::string active_client = "";

  // Threading
  std::thread imagesend_thread;
  std::thread eventreceive_thread;
  std::thread eventsend_thread;
  std::thread loading_thread;

  // Parallel parameter forwarding
  std::vector<std::pair<std::string, std::string> > modified_params;

  // Events
  SyncQueue< std::vector<char> > ims_send_queue;

  // Commands
  CommandController commands;
  void init_cmds();
  // void cmd_null(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end);
  // void cmd_help(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end);
  rapidjson::Value cmd_set(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_notify(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_get(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_state(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_interface_desc(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  //void cmd_add(ClientId& clid, rapidjson::Document& d);
  // void cmd_save(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end);
  // void cmd_upload(srz::ConstByteIterator args_start, srz::ConstByteIterator args_end);
  rapidjson::Value cmd_user(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_get_image(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_save_image(ClientId , rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_update_image_params(ClientId , rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_replace_colourmap(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_load(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_reload_field(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  // void cmd_load_id(ClientId& clid, rapidjson::Document& d);
  // void cmd_unload(ClientId& clid, rapidjson::Document& d);
  // void cmd_reload(ClientId& clid, rapidjson::Document& d);
  rapidjson::Value cmd_create_filter(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_update_filter(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_remove_filter(ClientId, rapidjson::Document& src, rapidjson::Document& dst);
  rapidjson::Value cmd_save_filtered_data(ClientId, rapidjson::Document& src, rapidjson::Document& dst);



};

#endif
