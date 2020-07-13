#ifndef SPLOTCH_JSON_H
#define SPLOTCH_JSON_H

#include "cxxsupport/error_handling.h"
#include "server/filter.h"
#define RAPIDJSON_HAS_STDSTRING 1
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

static rapidjson::Value ui_data(rapidjson::Document::AllocatorType& a, std::string value)
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "Data", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetString(std::move(value), a), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_text_display(rapidjson::Document::AllocatorType& a, std::string value)
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "TextDisplay", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetString(std::move(value), a), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_number_display(rapidjson::Document::AllocatorType& a, long value)
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "NumberDisplay", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetInt64(value), a);
  meta.AddMember("step", Value().SetInt64(1), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_number_display(rapidjson::Document::AllocatorType& a, int value)
{
  return std::move(ui_number_display(a, (long)value));
}

static rapidjson::Value ui_number_display(rapidjson::Document::AllocatorType& a, double value)
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "NumberDisplay", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetDouble(value), a);
  meta.AddMember("step", Value().SetDouble(0.01), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_text_input(rapidjson::Document::AllocatorType& a, std::string value, std::string rpc = "")
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "TextInput", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetString(std::move(value), a), a);
  if(rpc!="")
    meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  element.AddMember("meta", meta, a);
  return element;
}



static rapidjson::Value ui_number_input(rapidjson::Document::AllocatorType& a, long value, std::string rpc = "")
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "NumberInput", a); 
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetInt64(value), a);
  if(rpc!="")
    meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_number_input(rapidjson::Document::AllocatorType& a, int value, std::string rpc = "")
{
  return std::move(ui_number_input(a, (long)value, rpc));
}


static rapidjson::Value ui_number_input(rapidjson::Document::AllocatorType& a, double value, std::string rpc = "")
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "NumberInput", a); 
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetDouble(value), a);
  if(rpc!="")
    meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_number_input(rapidjson::Document::AllocatorType& a, float value, std::string rpc = "")
{
  return std::move(ui_number_input(a, (double)value, rpc));
}

static rapidjson::Value ui_checkbox(rapidjson::Document::AllocatorType& a, bool value, std::string rpc = "")
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "Checkbox", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetBool(value), a);
  if(rpc!="")
    meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_dropdown_text( rapidjson::Document::AllocatorType& a, 
                                          std::string value, 
                                          const std::vector<std::string>& options, 
                                          std::string rpc = "")
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "DropdownText", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetString(std::move(value), a), a);

  Value opt(kArrayType);
  bool valid = false;
  for(auto i : options) {
    opt.PushBack(Value().SetString(std::move(i), a), a);
    if(i == value)
      valid = true;
  }
  if(!valid)
    planck_fail("ui_dropdown_text value must exist in options");
  meta.AddMember("options",opt, a);

  if(rpc!="")
    meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  element.AddMember("meta", meta, a);
  return element;
}


static rapidjson::Value ui_button(rapidjson::Document::AllocatorType& a, 
                                  std::string rpc,
                                  const std::vector<std::string>& rpc_args)
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "Button", a);
  Value meta(kObjectType);
 // meta.AddMember("value", Value().SetString(std::move(name), a), a);
  meta.AddMember("rpc", Value().SetString(std::move(rpc), a), a);
  Value args(kObjectType);
  for(unsigned i = 0; i < rpc_args.size(); i++)
    args.AddMember(Value().SetString("Arg"+dataToString(i), a), Value().SetString(rpc_args[i], a), a);
  meta.AddMember("args", args, a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_range(rapidjson::Document::AllocatorType& a, 
                                 double min,
                                 double max,
                                 double value )
{
  using namespace rapidjson;
  Value element(kObjectType);
  element.AddMember("type", "Range", a);
  Value meta(kObjectType);
  meta.AddMember("value", Value().SetDouble(value), a);
  meta.AddMember("min", Value().SetDouble(min), a);
  meta.AddMember("max", Value().SetDouble(max), a);
  element.AddMember("meta", meta, a);
  return element;
}

static rapidjson::Value ui_filter(rapidjson::Document::AllocatorType& a, filter& f, std::vector<std::string> fields)
{
    using namespace rapidjson;
    // Info for filtering interface
    Value filter_obj(kObjectType);
    filter_obj.AddMember("type", "Group",a);
    Value filter_content(kObjectType);
    filter_content.AddMember("ID", ui_data(a, dataToString(f.id)),a);
    filter_content.AddMember("Type", ui_text_display(a, FilterTypeId2str(f.type)),a);
    filter_content.AddMember("Field", ui_text_display(a, f.name),a);
    switch(f.type){
      case FilterTypeId::CLIP:
      {
        float def0, def1, arg0, arg1;
        stringToData(f.defaults[0], def0);
        stringToData(f.defaults[1], def1);
        stringToData(f.args[0], arg0);
        stringToData(f.args[1], arg1);
        filter_content.AddMember("Min", ui_range(a, def0, def1, arg0),a);
        filter_content.AddMember("Max", ui_range(a, def0, def1, arg1),a);
        std::vector<std::string> update_args = {  "Filters/Filter "+dataToString(f.id)+"/ID", 
                                                  "Filters/Filter "+dataToString(f.id)+"/Min", 
                                                  "Filters/Filter "+dataToString(f.id)+"/Max"};
        filter_content.AddMember("Update", ui_button(a, "cmd_update_filter", update_args),a);

        //std::vector<std::string> remove_args = {  "Filters/Filter "+dataToString(f.id)+"/ID" };
        //filter_contents.PushBack(ui_button(a,"Remove", "cmd_remove_filter", remove_args),a);         
      }
      break;
      case FilterTypeId::COMB_FIELD:
      {
        filter_content.AddMember("Operation",  ui_dropdown_text(a,f.args[0], OpId::ops),a);
        filter_content.AddMember("RHS", ui_dropdown_text(a,f.args[1], fields),a);
        std::vector<std::string> update_args = {  "Filters/Filter "+dataToString(f.id)+"/ID",
                                                  "Filters/Filter "+dataToString(f.id)+"/Operation", 
                                                  "Filters/Filter "+dataToString(f.id)+"/RHS"};
        filter_content.AddMember("Update", ui_button(a, "cmd_update_filter", update_args),a);        
      }
      break;
      default:
      break;
    }
    filter_obj.AddMember("contents",filter_content, a);
    return  filter_obj;
}

#endif