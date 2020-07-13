#ifndef SPLOTCH_INTERFACE_H
#define SPLOTCH_INTERFACE_H

#include <string>
#include <vector>
#include "filter.h"

enum class InterfaceType: int { ALL, USER, VISUAL, FILTER };

// Describes the grouping mechanisms for user interface generation
struct user_descriptor{
  bool                      constrained;
  std::string               name;
  std::vector<std::string>  input_file;
  std::vector<std::string>  input_pars;
};

struct datafile_descriptor{
  std::string name;
  std::string type;
  long        np;
  float       boxsize[6];
  std::string box_units;   
};

struct interaction_descriptor{
  float move_speed;
  float rotate_speed;
};

struct image_descriptor{
  int xres;
  int yres;
  int quality;
  std::string name;
};

struct render_descriptor{
  bool colorbar;
};

struct field_descriptor{
  FieldId id;
  std::string name;
  double range_min;
  double range_max;
};

struct type_descriptor{
  std::vector<field_descriptor> fd;
  std::vector<std::string>      field_opts;
  float smooth_factor;
  float brightness;
};

struct filter_descriptor{
  std::vector<std::string>  type_opts;
  std::vector<std::string>  field_opts;
  std::vector<filter>       filters;
  std::string               data_name;
  unsigned long             npoints;
};

struct interface_descriptor{
  user_descriptor               user;
  datafile_descriptor           file;
  image_descriptor              image;
  interaction_descriptor        interaction;
  render_descriptor             render;
  std::vector<type_descriptor>  types;
  filter_descriptor             filter_desc;
};



#endif