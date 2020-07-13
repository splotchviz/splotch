
#ifndef READER_IFILE_H
#define READER_IFILE_H
#include <vector>
#include <string>
#include "cxxsupport/paramfile.h"
#include "splotch/splotchutils.h"

// File interface for server readers
class IFile { 
public:
  virtual void open(paramfile& pars) = 0;  
  virtual bool is_open() { return file_open; } 
  virtual void close(paramfile& params) = 0;

  virtual void read_info(paramfile& pars) = 0;

  // Fill point array with splotch formatted particles
  virtual void read(paramfile &params, std::vector<particle_sim> &points) = 0;

  virtual void read_field(paramfile &params, std::string name, std::vector<float>& storage) = 0;
  virtual void write_filtered(paramfile &params, std::string& filename, std::vector<unsigned long long>& idxs) = 0;

  virtual void reset(paramfile& params) = 0;
  virtual void particle_fields(std::vector<std::string>&) = 0;
  virtual void available_fields(std::vector<std::string>&) = 0;

  virtual ~IFile() = default;

protected:
  std::string name;
  bool file_open            = false;
  bool info_read            = false;
};


#endif