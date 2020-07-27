#ifndef SPLOTCH_CONTROLLER_H
#define SPLOTCH_CONTROLLER_H

#include "camera.h"
#include "event.h"
class SplotchServer;

class CameraController{
public:
  CameraController(){
    mouse_observer.bind_callback(
      [this](const MouseEvent& m) {
          this->mouse_handler(m);
      });
    key_observer.bind_callback(
      [this](const KeyEvent& k) {
          this->key_handler(k);
      });
  }

  void init(vec3f& pos, vec3f& at, vec3f& up, float m, float r, float imspf){
    camera.Create(pos, at, up);
    move_speed = m;
    rotate_speed = r;
    ideal_mspf = imspf;
  }

  void reposition(vec3f&& pos, vec3f&& at, vec3f&& up)
  {
    camera.Create(pos, at, up);
  }

  void mouse_handler(const MouseEvent& m){
    if(active)
    {  
      float scale = (ideal_mspf/mspf); 
      float local_rot = rotate_speed*scale;
      switch(m.eid)
      {
        case EventId::MOUSE_DOWN:
        case EventId::MOUSE_UP:
          mouse_x = m.x;
          mouse_y = m.y;
        break;
        case EventId::MOUSE_DRAG:
        {
          if(m.button & int(MouseButtonId::LEFT))
          {
            int motion_x = m.x - mouse_x;
            int motion_y = m.y - mouse_y;
            float x_rot = 0, y_rot = 0;
            if(motion_x || motion_y)
            {
              x_rot = -motion_x*local_rot; 
              y_rot = -motion_y*local_rot;
              if(m.button>>8 & int(KeyModifierId::SHIFT))
                camera.RotateTargetAround(x_rot, y_rot, 0.f);
              else
                camera.RotateAroundTarget(x_rot, y_rot, 0.f);

              mouse_x = m.x;
              mouse_y = m.y;
              modified = true;
            }         
          }
        }
        break;

        default:
        break;
      }
    }
  }

  void key_handler(const KeyEvent& k){
    // Scale for smooth movement with varying fps
    if(active)
    {
      float scale = (ideal_mspf/mspf); 
      float distance = move_speed*scale;

      if(k.eid == EventId::KEY_DOWN)
      {
        if(k.mod & int(KeyModifierId::SHIFT))
        {

          if     (k.key == 'w')  camera.MoveCamAndTargetForward(distance);
          else if(k.key == 'd')  camera.MoveCamAndTargetRight(distance);
          else if(k.key == 'q')  camera.MoveCamAndTargetUpward(distance);
          else if(k.key == 's')  camera.MoveCamAndTargetForward(-distance);
          else if(k.key == 'a')  camera.MoveCamAndTargetRight(-distance);
          else if(k.key == 'e')  camera.MoveCamAndTargetUpward(-distance);     
        }
        else
        {
          if     (k.key == 'w')  camera.MoveForward(distance);
          else if(k.key == 'd')  camera.MoveRight(distance);
          else if(k.key == 'q')  camera.MoveUpward(distance);
          else if(k.key == 's')  camera.MoveForward(-distance);
          else if(k.key == 'a')  camera.MoveRight(-distance);
          else if(k.key == 'e')  camera.MoveUpward(-distance);  
        }
        modified = true;
      }
    }
  }

  // Passthroughs for camera info
  // Not necessarily normalized
  vec3f eye(){return camera.GetCameraPosition();}
  vec3f target(){return camera.GetTarget();}
  vec3f up(){return camera.GetUpVector();}


  Observer<MouseEvent>  mouse_observer;
  Observer<KeyEvent>    key_observer;
  float mspf          = 100;
  bool modified       = false;
  float move_speed    = 0.1;
  float rotate_speed  = 0.1;
  bool active         = false;
private: 
  // Camera
  Camera camera;
  float ideal_mspf  = 32; 
  float mouse_x     = 0;
  float mouse_y     = 0;
};

//
// Handles client interaction
// Temporarily holds a pointer to the server
// Will be replaced by placing actions in internal action queue
//
class ClientController
{
public: 
struct ClientData {
    int download_id;
    std::vector<std::string> files;
    std::vector<std::string> colourmaps;
  };
  ClientController(SplotchServer* server);
  void handler(const ClientEvent& c);
  void load_default_client();
  ClientData& client_data(std::string s);
  bool exists(std::string s) {
    return clients.count(s);
  } 
  ClientData& retrieve(std::string name){
    return clients[name];
  }

  Observer<ClientEvent>  observer;

private: 
  SplotchServer* s;
  std::map<std::string, ClientData> clients;
};

#endif