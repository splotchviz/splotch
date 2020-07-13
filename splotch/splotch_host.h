#ifndef SPLOTCH_HOST_H
#define SPLOTCH_HOST_H

#include "splotch/splotchutils.h"

#ifdef MPI_A_NEQ_E
#include "utils/box.h"
namespace host_funct {

void particle_project(paramfile &params, particle_sim* p, tsize np,
  const vec3 &campos, const vec3 &lookat, vec3 sky, const vec3 &centerpos);
void particle_colorize(paramfile &params, particle_sim* p, tsize np,
  std::vector<COLOURMAP> &amap, float b_brightness);
void particle_sort(std::vector<particle_sim> &p, int sort_type, bool verbose);
//common interface for CPU and GPU version
void render_new (particle_sim *p, int npart, arr2<COLOUR> &pic,
  bool a_eq_e, float32 grayabsorb, int tile_size, Box<float,3>& bbox, vec3* plane_normals, arr2<COLOUR>& opacity_map);
}

// void host_rendering (paramfile &params, particle_sim* p,
//   arr2<COLOUR> &pic, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, const vec3 &sky,
//   std::vector<COLOURMAP> &amap, float b_brightness, tsize npart, tsize npart_all, Box<float, 3> bbox, arr2<COLOUR>& opacity_map);

#else

namespace host_funct {

void particle_project(paramfile &params, std::vector<particle_sim> &p,
  const vec3 &campos, const vec3 &lookat, vec3 sky, const vec3 &centerpos);
void particle_colorize(paramfile &params, std::vector<particle_sim> &p,
  std::vector<COLOURMAP> &amap, float b_brightness);
void particle_sort(std::vector<particle_sim> &p, int sort_type, bool verbose);
//common interface for CPU and GPU version
void render_new (particle_sim *p, int npart, arr2<COLOUR> &pic,
  bool a_eq_e, float32 grayabsorb);
}

//void host_rendering (paramfile &params, std::vector<particle_sim> &particle_data,
//  arr2<COLOUR> &pic, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, const vec3 &sky,
//  std::vector<COLOURMAP> &amap, float b_brightness, tsize npart_all);
#endif

void host_rendering (paramfile &params, std::vector<particle_sim> &particle_data, render_context &rc);

#endif
