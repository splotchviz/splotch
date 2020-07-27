#include "controller.h"
#include "server.h"

  ClientController::ClientController(SplotchServer* server){
    observer.bind_callback(
      [this](const ClientEvent& c) {
          this->handler(c);
      });
    if(server) s = server;
    else planck_fail("Client controller recieved NULL server pointer\n");
  }

  void ClientController::handler(const ClientEvent& c)
  {
    switch(c.eid)
    {
      // When a client connects to the server
      // We  send a resize event automatically, followed by the most recent image
      case EventId::IMG_CONNECT:
       printf("Image server client connected\n");
       //s->spawn_viewresize();
       s->image_modified = true;
       break;
      case EventId::IMG_DISCONNECT:
        printf("Image server client disconnected\n");
      break;
      case EventId::CMD_CONNECT:
       // When a client connects to the server
      // We  send a resize event automatically, followed by the most recent image
       printf("Control server client connected\n");
       s->notify_interface(InterfaceType::USER);
       s->image_modified = true;
       break;
      case EventId::CMD_DISCONNECT:
        printf("Control server client disconnected\n");
      break;
      default:
      break;   
    }  
  }

  // Client support is experimental, default client hardcoded here. 
  // Full client support reads this client data from an input file.
  void ClientController::load_default_client()
  {
    clients[""] = ClientData();
    clients[""].files = { 
                          "/Users/tims/Splotch/web/splotch-server/splotch/default.par",
                         // "/Users/tims/Splotch/web/splotch-server/splotch/snap092.par", 
                          "/Users/tims/Splotch/web/splotch-server/splotch/hdf5.par",
                          "/Users/tims/Splotch/web/splotch-server/splotch/millenium.par",
                          "/Users/tims/Splotch/web/splotch-server/splotch/mill-cut.par"
                        };

    clients[""].download_id = 0;

    clients[""].colourmaps = {
                                "palettes/Blue.pal",
                                "palettes/BlueSimple.pal",
                                "palettes/CyanPurple.pal",
                                "palettes/Fancy.pal",
                                "palettes/Klaus.pal",
                                "palettes/LongFancy.pal",
                                "palettes/M51.pal",
                                "palettes/M51_stars.pal",
                                "palettes/NewSplotch.pal",
                                "palettes/OldSplotch.pal",
                                "palettes/Orion.pal",
                                "palettes/OrionNew1.pal",
                                "palettes/RedBlue.pal",
                                "palettes/RedSimple.pal",
                                "palettes/Stars.pal",
                                "palettes/Tipsy.pal",
                                "palettes/TronInv.pal",
                                "palettes/Yellow.pal",
                            };
  }

  ClientController::ClientData& ClientController::client_data(std::string s)
  {
    if(clients.find(s) == clients.end())
      return clients[""];
    else return clients[s];
  }
