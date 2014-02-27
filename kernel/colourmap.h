/*
 *  Ray++ - Object-oriented ray tracing library
 *  Copyright (C) 1998-2001 Martin Reinecke and others.
 *  See the AUTHORS file for more information.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Library General Public
 *  License as published by the Free Software Foundation; either
 *  version 2 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Library General Public License for more details.
 *
 *  You should have received a copy of the GNU Library General Public
 *  License along with this library; if not, write to the Free
 *  Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 *  See the README file for more information.
 */

#ifndef RAYPP_COLOURMAP_H
#define RAYPP_COLOURMAP_H

#include <vector>
#include "kernel/colour.h"
#include "cxxsupport/sort_utils.h"
#include "cxxsupport/math_utils.h"

template<typename T> class anythingMap
  {
  private:
    bool sorted;
    std::vector<double> x;
    std::vector<T> y;

  public:
    void addVal (double x_, const T &val)
      {
      sorted=false;
      x.push_back(x_);
      y.push_back(val);
      }

    void sortMap()
      {
      std::vector<size_t> idx;
      buildIndex(x.begin(),x.end(),idx);
      sortByIndex(x.begin(),x.end(),idx);
      sortByIndex(y.begin(),y.end(),idx);
      sorted = true;
      }

    T getVal (double x_)
      {
      planck_assert(x.size()>0,"trying to access an empty map");
      if (x.size()==1) return y[0];
      if (!sorted) sortMap();
      tsize index;
      double frac;
      interpol_helper (x.begin(), x.end(), x_, index, frac);
      return (1.-frac)*y[index]+frac*y[index+1];
      }
    T getVal_const (double x_) const
      {
      planck_assert(x.size()>0,"trying to access an empty map");
      planck_assert(sorted,"map must be sorted");
      if (x.size()==1) return y[0];
      tsize index;
      double frac;
      interpol_helper (x.begin(), x.end(), x_, index, frac);
      return (1.-frac)*y[index]+frac*y[index+1];
      }

    size_t size() const { return x.size(); }
    double getX (size_t idx) { if (!sorted) sortMap(); return x[idx]; }
    T getY (size_t idx) { if (!sorted) sortMap(); return y[idx]; }
  };

typedef anythingMap<COLOUR> COLOURMAP;

#endif
