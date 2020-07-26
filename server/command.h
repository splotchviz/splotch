#ifndef SPLOTCH_SERVER_COMMAND_H
#define SPLOTCH_SERVER_COMMAND_H

#include "SyncQueue.h"
#include "event.h"
#include "json_maker.h"

enum JRPCErr: int { SUCCESS = 0, PARSE_ERR = -32700, INV_REQ = -32600, METH_NOT_FOUND = -32601, 
                    INV_PARS = -32602, INTERNAL_ERR = -32603, SERVER_ERR = -32000};

static std::map< JRPCErr, std::string > JRPCErrors = {
    {SUCCESS,         "No Error"},
    {PARSE_ERR,       "Parse error"},
    {INV_REQ,         "Invalid Request"},
    {METH_NOT_FOUND,  "Method not found"},
    {INV_PARS,        "Invalid params"},
    {INTERNAL_ERR,    "Internal error"},
    {SERVER_ERR,      "Server error"}, // available to -32099
};

struct JsonRPC{
  EventId     eid;
  ClientId    clid;
  std::string rpc;
};

// Command receiving queue
class CommandQueue
{
public:
  // To add events arriving via websocket
   void operator()(WSSTATE s, ClientId clid, const char* in, int len, bool, bool){
      // Check if we are simply connecting/disconnecting
      // Replace input data with spawned event if so
      EventId eid;
      if(s == WSSTATE::CONNECT || s == WSSTATE::DISCONNECT) {
        eid = (s==WSSTATE::CONNECT) ? EventId::CMD_CONNECT : EventId::CMD_DISCONNECT;
      }
      else{
        eid = EventId::RPC;
      }
      add_cmd(in, len, clid, eid);
   }

  bool empty(){
    return q.Empty();
  }

  JsonRPC pop(){
      return q.Pop();
  }

private:
  void add_cmd(const char* in, int len, ClientId clid, EventId eid){
    // Commands arrive in JSON-RPC 2.0 format
    // but we dont check anything until the command is handled
    // To avoid unnecessary multiple parses/copies of the dom
    JsonRPC out;
    out.eid = eid;
    out.clid = clid;
    out.rpc = std::string(in, len);
    q.Push(std::move(out));
  }

  // Synchronous queue to hold events
  SyncQueue< JsonRPC > q;
};

class CommandController{
  using Command   = std::function<rapidjson::Value(ClientId, rapidjson::Document&, rapidjson::Document&)>;
  using CommandMap = std::map<std::string, Command>;
  public:

  void add_cmd(std::string cmid, Command&& cmd){
    cmdmap[cmid] = cmd;
  }

  void handler(JsonRPC&& jrpc){

    if(jrpc.eid == EventId::CMD_CONNECT || jrpc.eid == EventId::CMD_DISCONNECT){
             // Create client event
        ClientEvent c; 
        c.eid = jrpc.eid;
        // Send it to the handler
        client.notify(c);
        return;     
    }

    // Parse in situ to document, checking for parse error
    rapidjson::Document d;
    rapidjson::Value id;
    if(d.ParseInsitu(&jrpc.rpc[0]).HasParseError()){
      // printf("\nError(offset %u): %s\n", 
      //   (unsigned)d.GetErrorOffset(),
      //   rapidjson::GetParseError_En(d.GetParseError()));
      send_client_message(jrpc.clid, client_response(d, id, error_value(d, JRPCErr::PARSE_ERR)));
      return;
    }

    // Check for batch (IsArray)

    // Check for id
    if(d.HasMember("id")){
      if(d["id"].IsNumber() || d["id"].IsString() || d["id"].IsNull()){
        id = d["id"];
      } else {
        printf("Err: %s\n", JRPCErrors[JRPCErr::INV_REQ].c_str());  
        send_client_message(jrpc.clid, client_response(d, id, error_value(d, JRPCErr::INV_REQ)));
        return;  
      }
    }
    // Check for jsonrpc=="2.0" 
    if(!d.HasMember("jsonrpc") || !(d["jsonrpc"] == "2.0")){
      printf("Err: %s\n", JRPCErrors[JRPCErr::INV_REQ].c_str());
      send_client_message(jrpc.clid, client_response(d, id, error_value(d, JRPCErr::INV_REQ)));
      return;
    }
    // Check for method name
    std::string method;
    if(!d.HasMember("method") || !d["method"].IsString()){
      printf("Err: %s\n", JRPCErrors[JRPCErr::INV_REQ].c_str());
      send_client_message(jrpc.clid, client_response(d, id, error_value(d, JRPCErr::INV_REQ)));
      return;
    } else {
      method = d["method"].GetString();
    }
    // Check method exists
    d["method"].GetString();
    if(!(cmdmap.count(method))){
      printf("Err: %s\n", JRPCErrors[JRPCErr::METH_NOT_FOUND].c_str());
      send_client_message(jrpc.clid, client_response(d, id, error_value(d, JRPCErr::METH_NOT_FOUND)));
      return;
    }

    // Give the document and request value
    // If batch these would be different..
    // Get back response value
    rapidjson::Value v = cmdmap.at(method)(jrpc.clid, d, d);

    // If ID != NULL we must reply
    if(!id.IsNull()){
      send_client_message(jrpc.clid, client_response(d, id, std::move(v)));
    }

  }

  void send_client_message(ClientId clid, rapidjson::Value&& message){
      // Stringify
      rapidjson::StringBuffer buffer;
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      message.Accept(writer);
      printf("JSON: %s\n", buffer.GetString());
      // Copy to byte buffer and send
      std::vector<unsigned char> serial(buffer.GetSize());
      memcpy(serial.data(),buffer.GetString(), buffer.GetSize());
      sender.Push(std::make_tuple(clid, WSMSGTYPE::TEXT, std::move(serial)));    
  }

  void send_binary_message(ClientId clid, std::vector<unsigned char>& message){
    sender.Push(std::make_tuple(clid, WSMSGTYPE::BINARY, std::move(message)));  
  }

  rapidjson::Value client_notify(rapidjson::Document& d, std::string method, rapidjson::Value&& params){
    rapidjson::Value notification(rapidjson::kObjectType);
    rapidjson::Document::AllocatorType& a = d.GetAllocator();
    notification.AddMember("jsonrpc","2.0", a);
    notification.AddMember("method", rapidjson::Value(method, a).Move(), a);
    if(!params.IsNull())
      notification.AddMember("params", params, a);
    return notification;
  }

  rapidjson::Value client_response(rapidjson::Document& d, rapidjson::Value& id, rapidjson::Value&& result) {
    // Build json response
    rapidjson::Value response(rapidjson::kObjectType);
    rapidjson::Document::AllocatorType& a = d.GetAllocator();
    response.AddMember("jsonrpc","2.0", a);
    response.AddMember("id", id, a);
    if(result.HasMember("result") && !result["result"].IsNull())
      response.AddMember("result", result["result"], a);
    else if(result.HasMember("error"))
      response.AddMember("error", result["error"], a);
    return response;
  }

  rapidjson::Value error_value(rapidjson::Document& d, const JRPCErr& err, std::string msg){
    return error_value(d, err, rapidjson::Value(msg, d.GetAllocator()));
  }

  rapidjson::Value error_value(rapidjson::Document& d, const JRPCErr& err, rapidjson::Value data = rapidjson::Value()){
    rapidjson::Value error(rapidjson::kObjectType);
    rapidjson::Document::AllocatorType& a = d.GetAllocator();
    error.AddMember("error", rapidjson::kObjectType, a);
    error["error"].AddMember("code",(int)err, a);
    error["error"].AddMember("message", rapidjson::Value(JRPCErrors[err], a).Move(), a);
    if(!data.IsNull())
      error["error"].AddMember("data", data, a);
    return error;
  }

  rapidjson::Value result_value(rapidjson::Document& d, rapidjson::Value value = rapidjson::Value()){
    rapidjson::Value result(rapidjson::kObjectType);
    rapidjson::Document::AllocatorType& a = d.GetAllocator();
    result.AddMember("result",value, a);
    return result;
  }

  SyncQueue< std::tuple<ClientId, WSMSGTYPE, std::vector<unsigned char> > >    sender;
  CommandQueue              reciever;
  Observable<ClientEvent>   client;
private: 
  CommandMap      cmdmap;
};

#endif