#ifndef SPLOTCH_UTILS_COMPOSITE_H
#define SPLOTCH_UTILS_COMPOSITE_H

#include "cxxsupport/arr.h"
#include "kernel/colour.h"
#include <vector>

void composite(arr2<COLOUR>& pic, arr2<COLOUR>& opac, std::vector<int>& order, int rank, int nranks);

#endif