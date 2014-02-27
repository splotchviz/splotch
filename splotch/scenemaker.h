#ifndef SPLOTCH_SCENE_MAKER_H
#define SPLOTCH_SCENE_MAKER_H

#include <vector>
#include <string>

#include "cxxsupport/paramfile.h"
#include "splotch/splotchutils.h"

#define NEW_MPISTUFF

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
        : outname(oname), keep_particles(keep), reuse_particles(reuse) {}
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
#ifndef NEW_MPISTUFF
    // reset the data structures after rendering
    void MpiStripRemoteParticles();
    // save the number of particles which are initially in p2
    MyIDType numberOfLocalParticles;
    // vectors to hold a backup of the data at time2
    std::vector<particle_sim> p2Backup;
    std::vector<MyIDType> id2Backup;
    std::vector<MyIDType> idx2Backup;
    std::vector<vec3f> vel2Backup;
    // --- routines and variables necessary for the MPI parallelization of the
    //     interpolation ---
#endif

  public:
  sceneMaker (paramfile &par);
  bool getNextScene (std::vector<particle_sim> &particle_data,
      std::vector<particle_sim> &r_points, vec3 &campos,
      vec3 &lookat, vec3 &sky, std::string &outfile);
  };

#endif
