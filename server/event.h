#ifndef SPLOTCH_UTILS_EVENT_H
#define SPLOTCH_UTILS_EVENT_H

// External dependency
#include "SyncQueue.h"
#include "observe.h"
#define ZRF_int32_size
#include "Serialize.h"
#include "WSocketMServer.h"

// // Server commands (ids of command events)
// enum class CommandId: int { NULL_CMD = 0, HELP = 1, SET = 2, GET = 3, ADD = 4, SAVE = 5, 
//                             UPLOAD = 6, FILELIST = 7, LOAD = 8, LOAD_ID = 9, UNLOAD = 10,
//                             RELOAD = 11 };
// static  CommandId StrToCommandId(std::string id) {
//       static std::map< std::string, CommandId > s2c = {
//           {"null",     CommandId::NULL_CMD},
//           {"help",     CommandId::HELP},
//           {"set",      CommandId::SET},
//           {"get",      CommandId::GET},
//           {"add",      CommandId::ADD},
//           {"save",     CommandId::SAVE},
//           {"upload",   CommandId::UPLOAD},
//           {"filelist", CommandId::FILELIST},
//           {"load",     CommandId::LOAD},
//           {"load_id",  CommandId::LOAD_ID},
//           {"unload",   CommandId::UNLOAD},
//           {"reload",   CommandId::RELOAD},
//       };
//       auto it = s2c.find(id);
//       if(it==s2c.end()){
//          printf("Server: StrToCommandId() invalid command string %s\n", id.c_str());      
//          return CommandId::NULL_CMD;  
//       }
//       return it->second;
//   }

// Client events (i.e. those sent to client)
enum class ClientEventId: int { NULL_CMD = 0, PRINT = 1, RESIZE = 2, FILE_DOWNLOAD = 3, HELP = 4, FILELIST = 5};


// Server events
enum class EventId: int { NULL_EVENT = 0, MOUSE_DOWN = 1, MOUSE_UP = 2, MOUSE_DRAG = 3, KEY_DOWN = 4, KEY_UP = 5,
                          MOUSE_WHEEL = 6, RESIZE = 7, CMD_STRING = 8, CMD_ID = 9, IMG_CONNECT = 10, IMG_DISCONNECT = 11, 
                          CMD_CONNECT = 12, CMD_DISCONNECT = 13, RPC = 14 };



static  std::string EventToStr(EventId id) {
      static std::map< EventId, std::string > e2s = {
          {EventId::NULL_EVENT,  "null event"},
          {EventId::MOUSE_DOWN,  "mouse down"},
          {EventId::MOUSE_UP,    "mouse up"},
          {EventId::MOUSE_DRAG,  "mouse drag"},
          {EventId::KEY_DOWN,    "key down"},
          {EventId::KEY_UP,      "key up"},
          {EventId::MOUSE_WHEEL, "mouse wheel"},
          {EventId::RESIZE,      "resize window"},
          {EventId::CMD_STRING,  "command string"}, 
          {EventId::CMD_ID,      "command id"},
          {EventId::IMG_CONNECT,     "image connect"}, 
          {EventId::IMG_DISCONNECT,  "image disconnect"},          
          {EventId::CMD_CONNECT,     "cmd connect"}, 
          {EventId::CMD_DISCONNECT,  "cmd disconnect"},
          {EventId::RPC,  "rpc"}
      };
      if(id < EventId::NULL_EVENT || id > EventId::RPC){
         printf("Server: EventToStr() invalid event id %d\n", id);
         return e2s[EventId::NULL_EVENT];
       }
      return e2s[id];
  }

// -1 is variable
static  int EventDataLength(EventId id) {
      static std::map< EventId, int > e2i = {
          {EventId::NULL_EVENT,  0},
          {EventId::MOUSE_DOWN,  12},
          {EventId::MOUSE_UP,    12},
          {EventId::MOUSE_DRAG,  12},
          {EventId::KEY_DOWN,    12},
          {EventId::KEY_UP,      12},
          {EventId::MOUSE_WHEEL, 12},
          {EventId::RESIZE,      8},
          {EventId::CMD_STRING,  260}, 
          {EventId::CMD_ID,      -1},
          {EventId::IMG_CONNECT,     0},
          {EventId::IMG_DISCONNECT,  0}
      };
      if(id < EventId::NULL_EVENT || id > EventId::IMG_DISCONNECT){
         printf("Server: EventDataLength() invalid event id %d\n", id);
         return e2i[EventId::NULL_EVENT];        
      }
      return e2i[id];
  }

// Base serialized event structure
struct event{
  int cid = -1;
  EventId eid = EventId::NULL_EVENT;
  std::vector<char> data;
};


enum class MouseButtonId: int { NONE = 0x0, LEFT = 0x1, RIGHT = 0x2, MIDDLE = 0x4, FOUR = 0x8, FIVE = 0x16};
// Real event structures
struct MouseEvent{
  EventId eid;
  int x;
  int y;
  int button;
};
static MouseEvent DeserializeMouseEvent(EventId& eid, std::vector<char>& data){
  MouseEvent m;
  m.eid = eid;
  srz::ConstByteIterator cbi = data.begin();
  cbi = srz::SerializePOD<int>::UnPack(cbi, m.x);
  cbi = srz::SerializePOD<int>::UnPack(cbi, m.y);
  cbi = srz::SerializePOD<int>::UnPack(cbi, m.button);
  return m;
}


enum class KeyModifierId: int { NONE = 0x0, ALT = 0x1, CTRL = 0x2, META = 0x4, SHIFT = 0x8};
struct KeyEvent{
  EventId eid;
  int key;
  int mod;
};
static KeyEvent DeserializeKeyEvent(EventId& eid, std::vector<char>& data){
  KeyEvent k;
  k.eid = eid;
  srz::ConstByteIterator cbi = data.begin();
  cbi = srz::SerializePOD<int>::UnPack(cbi, k.key);
  cbi = srz::SerializePOD<int>::UnPack(cbi, k.mod);
  k.key = std::tolower(k.key);
  return k;
}

struct ClientEvent{
  EventId eid;
};
static ClientEvent DeserializeClientEvent(EventId& eid, std::vector<char>& data){
  ClientEvent c;
  c.eid = eid;
  return c;
}

// Server-side event controller
class EventController
{
public:
  void handler(event&& e){
      switch(e.eid)
      {
        case EventId::MOUSE_DOWN:
        case EventId::MOUSE_UP:
        case EventId::MOUSE_DRAG:
        case EventId::MOUSE_WHEEL:
          mouse.notify( DeserializeMouseEvent(e.eid, e.data) );
        break;
        case EventId::KEY_DOWN:
        case EventId::KEY_UP:
          keyboard.notify( DeserializeKeyEvent(e.eid, e.data) );
        break;     
        case EventId::IMG_CONNECT:
        case EventId::IMG_DISCONNECT:
          client.notify( DeserializeClientEvent(e.eid, e.data) );
        break;       
        default:
          // This should probably be an error as we shouldnt arrive here at all
          printf("Control server: Warning, event not recognised.\n");
        break;
      }
  }

  // Event observables
  Observable<MouseEvent>    mouse;
  Observable<KeyEvent>      keyboard;
  //Observable<CommandEvent>  command;
  Observable<ClientEvent>   client;

private:

};

// Server-side event queue
class EventQueue
{
public:
  // To add events arriving via websocket
   void operator()(WSSTATE s, ClientId cid, const char* in, int len, bool, bool)
   {
      // Check if we are simply connecting/disconnecting
      // Replace input data with spawned event if so
      EventId connectEvent;
      if(s == WSSTATE::CONNECT) {
          connectEvent = EventId::IMG_CONNECT;
          in = (const char*)&connectEvent;
          len = 4;
      } else if(s == WSSTATE::DISCONNECT) {
          connectEvent = EventId::IMG_DISCONNECT;
          in = (const char*)&connectEvent;
          len = 4;
      }
      int temp_cid = 0;
      add_event(in, len, temp_cid);
   }

  // To add events arriving from anywhere else
  void operator()(const char* e, int len, int cid)
  {
  	add_event(e, len, cid);
  } 

  bool empty()
  {
    return q.Empty();
  }

  event pop()
  {
      return q.Pop();
  }

private:
  void add_event(const char* in, int len, int cid)
  {
    // Read event ID from first four bytes 
    int evid = *((int*)in);
  	EventId eid = static_cast<EventId>(evid);
  	in += sizeof(int);

  	// Validate event id, could just be a warning and return, to ignore garbled messages
  	if(eid < EventId::MOUSE_DOWN || eid > EventId::IMG_DISCONNECT)
  	{
  		planck_fail("parse_event() invalid event id\n");
  	}

    //Sprintf("Recieved event with id %d: %s\n", evid, EventToStr(eid).c_str());
    
    // Create a new event to store
    event out;
    out.eid = eid;
    out.cid = cid;

    // Subtract event id from size of message to get data length
    len -= sizeof(int);

    // Validate event data
    // For fixed length events lookup event ID, compare to expected event length 
    // If expected length is -1 then its variable (command event)
    int expected = EventDataLength(eid);
  	if(expected != -1 && len != expected) 
  	{
  		printf("parse_event(): warning, ignoring event id: %d, data len %d < expected data len: %d", eid, len, expected);
  		return;
  	}

    if(len>0) 
    {
      out.data.resize(len);
      std::copy(in, in+len, &(out.data[0]));
    }

  	q.Push(std::move(out));
  }

private:
  // Synchronous queue to hold events
  SyncQueue< event > q;
};



#endif