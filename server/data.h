#include <string>
#include <vector>
#include <thread>
#include <future>
#include <chrono>
#include "reader/IFile.h"
#include "reader/hdf5_reader.h"
#include "splotch/splotchutils.h"
#include "cxxsupport/paramfile.h"
#include "server/filter.h"


class Data{
public:
  struct field_info{
    FieldId             id;
    std::string        name;
    Normalizer<float>  normalizer;
  };
  typedef std::pair<field_info, std::vector<float>> field_cache;

  Data() : n_fields(8) { 
    particle_fields.resize(n_fields); 
    operations =  {
                    [](float a, float b) -> float { return a;},
                    [](float a, float b) -> float { return a + b; },
                    [](float a, float b) -> float { return a - b; },
                    [](float a, float b) -> float { return a * b; },
                    [](float a, float b) -> float { return a / b; },
                  };
  }
  void unload();
  void read(paramfile &params, std::vector<particle_sim> &points, bool cache = true);
  void refresh(paramfile &params, std::vector<particle_sim> &points, bool cache = true);
  void range(std::vector<particle_sim> &points, FieldId fid = FieldId::NONE);
  void range(field_info& f, std::vector<float>& v);

  void reload_field(paramfile& params, std::string fid, std::string field_name);

  unsigned long filtered_size();

  // Filtering
  //void add_dynamic_filter(paramfile &params, std::string fieldname, FilterTypeId ftid, std::vector<float> args);
  void init_filters();
  void create_filter(paramfile &params, std::string fieldname, FilterTypeId ftid);
  void update_dynamic_filter(paramfile &params, const std::vector<std::string>& args);
  void apply_particle_filter(paramfile& params, field_info* particle_field, filter& flt, const std::vector<std::string>& args);
  void apply_data_filter(paramfile& params, field_cache* data_field, filter& flt, const std::vector<std::string>& args);
  void apply_dynamic_filters(std::vector<particle_sim> &points);
  void remove_dynamic_filter(int id);
  bool write_filtered(paramfile& params, std::string filename);
  void unfilter(unsigned marker);


  template<typename T> void filter_if(float* field, T& f, unsigned marker){
    #pragma omp parallel for
    for(unsigned i = 0; i < pcache.size(); i++){
      filter_id[i] &= ~marker;
      if(f(field[i])) filter_id[i] |= marker;
    }
  }

  template<typename T> void apply_filter(float* field, T& f){
    #pragma omp parallel for
    for(unsigned i = 0; i < pcache.size(); i++){
      f(field[i], i);
    }
  }

  template<typename T> void apply_filter(std::vector<particle_sim>& p, FieldId fid, T& f){
    switch(fid)
    {
      case FieldId::F_X:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].x, i);
      break;
      case FieldId::F_Y:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].y, i);
      break;
      case FieldId::F_Z:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].z, i);
      break;
      case FieldId::F_CR:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].e.r, i);
      break;
      case FieldId::F_CG:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].e.g, i);
      break;
      case FieldId::F_CB:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].e.b, i);
      break;
      case FieldId::F_R:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].r, i);
      break;
      case FieldId::F_I:
        #pragma omp parallel for
        for(unsigned i = 0; i < p.size(); i++) f(p[i].I, i);
      break;
      default:
      break;
    }
  }

  std::vector<field_info>  particle_fields;
  std::vector<std::string> available_fields;
  std::vector<filter> filters;

private:
  void create_file(paramfile &params);
 

  IFile* file = NULL; 

  const int n_fields;
  std::vector<particle_sim> pcache;
  bool cached = false;

  // Filtering
  // Unique ID for each filter
  // 0 is reserved for normalization filter which is always applied
  int filter_identifier = 0;

  std::vector<unsigned> filter_id;
  std::vector<field_cache> cached_fields;
  unsigned filter_marker = 1;
  unsigned long npoints_filtered;
  std::future<bool> filter_io_future;
  std::mutex filter_prep_mutex;
  std::mutex filter_io_mutex;
  std::vector<float (*)(float, float)> operations;

   bool is_marking_filter(FilterTypeId ftid);
  filter make_filter_type(field_info* p_field, field_cache* d_field, int id, std::string fname, FilterTypeId ftid);

};