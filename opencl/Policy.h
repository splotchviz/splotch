/*
CuPolicy is the class that knows the overall state of  application.
All 'magic numbers' are out of this class.
*/
#ifndef CUPOLICY_H
#define CUPOLICY_H

#include "cxxsupport/paramfile.h"
#include "opencl/splotch_cuda.h"


class CuPolicy
  {
  private:
    int m_blockSize, maxregion, fbsize;
    size_t gmsize;
  public:
    CuPolicy(paramfile &Param);

    int GetMaxRegion();
    int GetFBufSize();
    int GetGMemSize();
    void setGMemSize(size_t s);

  };

#endif //CUPOLICY_H
