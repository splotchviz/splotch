#ifndef SPLOTCH_HOST_H
#define SPLOTCH_HOST_H

#include "splotch/splotchutils.h"

namespace host_funct {

void particle_project(paramfile &params, std::vector<particle_sim> &p,
  const vec3 &campos, const vec3 &lookat, vec3 sky);
void particle_colorize(paramfile &params, std::vector<particle_sim> &p,
  std::vector<COLOURMAP> &amap, float b_brightness);
void particle_sort(std::vector<particle_sim> &p, int sort_type, bool verbose);
//common interface for CPU and GPU version
void render_new (particle_sim *p, int npart, arr2<COLOUR> &pic,
  bool a_eq_e, float32 grayabsorb);
}

void host_rendering (paramfile &params, std::vector<particle_sim> &particle_data,
  arr2<COLOUR> &pic, const vec3 &campos, const vec3 &lookat, const vec3 &sky,
  std::vector<COLOURMAP> &amap, float b_brightness, tsize npart_all);

#endif
