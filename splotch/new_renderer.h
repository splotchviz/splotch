#ifndef NEW_RENDERER_H
#define NEW_RENDERER_H
#ifndef MPI_A_NEQ_E

#include "splotch/splotchutils.h"

void host_rendering_new (paramfile &params, std::vector<particle_sim> &particle_data,
  arr2<COLOUR> &pic, const vec3 &campos, const vec3 &centerpos, const vec3 &lookat, const vec3 &sky,
  std::vector<COLOURMAP> &amap, float b_brightness, tsize npart_all);

#endif
#endif
