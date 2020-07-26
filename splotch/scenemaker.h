#ifndef SPLOTCH_SCENE_MAKER_H
#define SPLOTCH_SCENE_MAKER_H

#include <vector>
#include <string>
#ifdef WINDOWSCINDER
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "cxxsupport/paramfile.h"
#include "splotch/splotchutils.h"
#include "reader/IFile.h"

#ifdef MPI_A_NEQ_E
#ifndef USE_MPI
#error MPI_A_NEQ_E only available when USE_MPI is set
#endif
#include "utils/kdtree.h"
#endif

class sceneMaker
  {
  private:
    struct scene
      {
      paramfile sceneParameters;

      std::string outname;
      bool keep_particles, reuse_particles;

      scene (const paramfile &scnParms,
          const std::string &oname, bool keep, bool reuse)
        : sceneParameters(scnParms), outname(oname),
          keep_particles(keep), reuse_particles(reuse) {}
      scene (const std::string &oname, bool keep, bool reuse)
        : outname(oname), keep_particles(keep), reuse_particles(reuse)
        { sceneParameters.setVerbosity(false); }
      };

    std::vector<scene> scenes;
    int cur_scene;
    paramfile &params;
    int interpol_mode;
    double boxsize;

    // only used if interpol_mode>0
    std::vector<particle_sim> p1, p2;
    std::vector<MyIDType> id1, id2;
    std::vector<MyIDType> idx1, idx2;

    int snr1_now, snr2_now;

    // buffers to hold the times relevant to the *currently loaded snapshots*
    double time1, time2;
    double redshift1, redshift2;

    // only used if interpol_mode>1
    std::vector<vec3f> vel1, vel2;

    // only used if the same particles are used for more than one scene
    std::vector<particle_sim> p_orig;

    // only used if interpol_mode>0
    void particle_interpolate(std::vector<particle_sim> &p, double frac) const;

    void particle_normalize(std::vector<particle_sim> &p, bool verbose) const;

    void fetchFiles(std::vector<particle_sim> &particle_data, double fidx);

    // --- routines and variables for the MPI parallelization of the particle
    //     interpolation ---
    // exchange particle IDs
    void MpiFetchRemoteParticles();

  public:
  sceneMaker (paramfile &par);
  bool getNextScene (std::vector<particle_sim> &particle_data,
      std::vector<particle_sim> &r_points, vec3 &campos, vec3 &centerpos,
      vec3 &lookat, vec3 &sky, std::string &outfile);

  void updateCurrentScene (std::vector<particle_sim> &particle_data, bool new_data = false);

  #ifdef CLIENT_SERVER
  void unloadData(bool force_dealloc = true);
  #endif
  IFile* file = NULL;

#ifdef MPI_A_NEQ_E
  KDtree<particle_sim, 3> tree;
  KDtree<particle_sim, 3> tree_orig;
  void treeInit();
  Box<float,3> treeBoundingBox(render_context& context);
  void treeUpdateOpacityRes(arr2<COLOUR>& pic, arr2<COLOUR>& opac);
#endif
  bool is_final_scene();
  };

#endif
