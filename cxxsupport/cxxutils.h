/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*! \file cxxutils.h
 *  Various convenience functions used by the Planck LevelS package.
 *
 *  Copyright (C) 2002-2011 Max-Planck-Society
 *  \author Martin Reinecke \author Reinhard Hell
 */

#ifndef PLANCK_CXXUTILS_H
#define PLANCK_CXXUTILS_H

#include "error_handling.h"
#include "datatypes.h"

#include "string_utils.h"
#include "math_utils.h"
#include "share_utils.h"
#include "sort_utils.h"
#include "announce.h"

/*! Resizes \a container to zero and releases its memory. Typically used for
    std::vector.
    Taken from http://www.gotw.ca/gotw/054.htm */
template<typename T> inline void releaseMemory (T &container)
  { T().swap(container); }

/*! Releases all unused memory that \a container might have. Typically used for
    std::vector.
    Taken from http://www.gotw.ca/gotw/054.htm */
template<typename T> inline void shrinkToFit (T &container)
  { T(container).swap(container); }

#endif
