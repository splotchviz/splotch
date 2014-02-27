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

#include "kernel/transform.h"
#include "cxxsupport/lsconstants.h"

using namespace std;

TRANSMAT:: TRANSMAT (float32 a00, float32 a01, float32 a02,
                     float32 a10, float32 a11, float32 a12,
                     float32 a20, float32 a21, float32 a22,
                     float32 a30, float32 a31, float32 a32)
  {
  entry[0][0] = a00;
  entry[1][0] = a01;
  entry[2][0] = a02;

  entry[0][1] = a10;
  entry[1][1] = a11;
  entry[2][1] = a12;

  entry[0][2] = a20;
  entry[1][2] = a21;
  entry[2][2] = a22;

  entry[0][3] = a30;
  entry[1][3] = a31;
  entry[2][3] = a32;
  }

TRANSMAT &TRANSMAT::operator*= (const TRANSMAT &b)
  {
  TRANSMAT a(*this);
  for (int i=0 ; i<4 ; ++i)
    for (int j=0 ; j<3 ; ++j)
      entry[j][i] = a.entry[0][i] * b.entry[j][0]
                  + a.entry[1][i] * b.entry[j][1]
                  + a.entry[2][i] * b.entry[j][2];
  entry[0][3] += b.entry[0][3];
  entry[1][3] += b.entry[1][3];
  entry[2][3] += b.entry[2][3];
  return *this;
  }

TRANSMAT TRANSMAT::Inverse () const
  {
  TRANSMAT tmp;

  tmp.entry[0][0] = entry[1][1]*entry[2][2] - entry[2][1]*entry[1][2];
  tmp.entry[0][1] = entry[0][1]*entry[2][2] - entry[2][1]*entry[0][2];
  tmp.entry[0][2] = entry[0][1]*entry[1][2] - entry[1][1]*entry[0][2];

  tmp.entry[1][0] = entry[1][0]*entry[2][2] - entry[2][0]*entry[1][2];
  tmp.entry[1][1] = entry[0][0]*entry[2][2] - entry[2][0]*entry[0][2];
  tmp.entry[1][2] = entry[0][0]*entry[1][2] - entry[1][0]*entry[0][2];

  tmp.entry[2][0] = entry[1][0]*entry[2][1] - entry[2][0]*entry[1][1];
  tmp.entry[2][1] = entry[0][0]*entry[2][1] - entry[2][0]*entry[0][1];
  tmp.entry[2][2] = entry[0][0]*entry[1][1] - entry[1][0]*entry[0][1];

  float64 d = (entry[0][0]*tmp.entry[0][0] -
               entry[1][0]*tmp.entry[0][1] +
               entry[2][0]*tmp.entry[0][2]);

  planck_assert(abs(d)>1e-10,"degenerate matrix in TRANSMAT::Inverse()");

  d = 1./d;

  tmp.entry[0][0] *= d;
  tmp.entry[2][0] *= d;
  tmp.entry[1][1] *= d;
  tmp.entry[0][2] *= d;
  tmp.entry[2][2] *= d;

  d = -d;

  tmp.entry[1][0] *= d;
  tmp.entry[0][1] *= d;
  tmp.entry[2][1] *= d;
  tmp.entry[1][2] *= d;

  tmp.entry[0][3] = - (tmp.entry[0][0]*entry[0][3] +
                       tmp.entry[0][1]*entry[1][3] +
                       tmp.entry[0][2]*entry[2][3]);
  tmp.entry[1][3] = - (tmp.entry[1][0]*entry[0][3] +
                       tmp.entry[1][1]*entry[1][3] +
                       tmp.entry[1][2]*entry[2][3]);
  tmp.entry[2][3] = - (tmp.entry[2][0]*entry[0][3] +
                       tmp.entry[2][1]*entry[1][3] +
                       tmp.entry[2][2]*entry[2][3]);

  return tmp;
  }

void TRANSMAT::SetToIdentity ()
  {
  for (int m=0; m<12; ++m) p[m]=0;
  entry[0][0]=entry[1][1]=entry[2][2]=1;
  }

void TRANSMAT::SetToZero ()
  { for (int m=0; m<12; ++m) p[m]=0; }

void TRANSMAT::Transpose ()
  {
  swap(entry[0][1], entry[1][0]);
  swap(entry[0][2], entry[2][0]);
  swap(entry[1][2], entry[2][1]);
  entry[0][3] = entry [1][3] = entry[2][3] = 0.0;
  }

ostream &operator<< (ostream &os, const TRANSMAT &mat)
  {
  for (int i=0;i<4;++i)
    os << mat.entry[0][i] << ' ' << mat.entry[1][i]
       << ' ' << mat.entry[2][i] << endl;
  return os;
  }

void TRANSFORM::Make_Scaling_Transform (const vec3 &vec)
  {
  planck_assert((vec.x>0) && (vec.y>0) && (vec.z>0),
    "invalid scaling transformation");

  matrix.SetToIdentity();
  matrix.entry[0][0]=vec.x;
  matrix.entry[1][1]=vec.y;
  matrix.entry[2][2]=vec.z;

  inverse.SetToIdentity();
  inverse.entry[0][0]=1.0/vec.x;
  inverse.entry[1][1]=1.0/vec.y;
  inverse.entry[2][2]=1.0/vec.z;
  }

void TRANSFORM::Make_Translation_Transform (const vec3 &vec)
  {
  matrix.SetToIdentity();
  matrix.entry[0][3]=vec.x;
  matrix.entry[1][3]=vec.y;
  matrix.entry[2][3]=vec.z;

  inverse.SetToIdentity();
  inverse.entry[0][3]=-vec.x;
  inverse.entry[1][3]=-vec.y;
  inverse.entry[2][3]=-vec.z;
  }

void TRANSFORM::Make_Axis_Rotation_Transform
  (const vec3 &axis, float64 angle)
  {
  vec3 V = axis.Norm();
  angle *= degr2rad;
  float64 cosx = cos (angle), sinx = sin (angle);

  matrix.SetToZero();
  matrix.entry[0][0] = V.x * V.x + cosx * (1.0 - V.x * V.x);
  matrix.entry[1][0] = V.x * V.y * (1.0 - cosx) + V.z * sinx;
  matrix.entry[2][0] = V.x * V.z * (1.0 - cosx) - V.y * sinx;
  matrix.entry[0][1] = V.x * V.y * (1.0 - cosx) - V.z * sinx;
  matrix.entry[1][1] = V.y * V.y + cosx * (1.0 - V.y * V.y);
  matrix.entry[2][1] = V.y * V.z * (1.0 - cosx) + V.x * sinx;
  matrix.entry[0][2] = V.x * V.z * (1.0 - cosx) + V.y * sinx;
  matrix.entry[1][2] = V.y * V.z * (1.0 - cosx) - V.x * sinx;
  matrix.entry[2][2] = V.z * V.z + cosx * (1.0 - V.z * V.z);
  inverse = matrix;
  inverse.Transpose();
  }

void TRANSFORM::Make_Shearing_Transform
  (float32 xy, float32 xz, float32 yx, float32 yz, float32 zx, float32 zy)
  {
  matrix.SetToIdentity();

  matrix.entry[1][0] = xy;
  matrix.entry[2][0] = xz;
  matrix.entry[0][1] = yx;
  matrix.entry[2][1] = yz;
  matrix.entry[0][2] = zx;
  matrix.entry[1][2] = zy;

  inverse = matrix.Inverse();
  }

void TRANSFORM::Add_Transform (const TRANSFORM &trans)
  {
  matrix *= trans.matrix;
  TRANSMAT tmp=trans.inverse;
  tmp*=inverse;
  inverse = tmp;
  }

vec3 TRANSFORM::TransPoint (const vec3 &vec) const
  {
  const float32 *p = matrix.p;
  return vec3
    (vec.x*p[0] + vec.y*p[1] + vec.z*p[2] + p[3],
     vec.x*p[4] + vec.y*p[5] + vec.z*p[6] + p[7],
     vec.x*p[8] + vec.y*p[9] + vec.z*p[10]+ p[11]);
   }

vec3 TRANSFORM::InvTransPoint (const vec3 &vec) const
  {
  const float32 *p = inverse.p;
  return vec3
    (vec.x*p[0] + vec.y*p[1] + vec.z*p[2] + p[3],
     vec.x*p[4] + vec.y*p[5] + vec.z*p[6] + p[7],
     vec.x*p[8] + vec.y*p[9] + vec.z*p[10]+ p[11]);
  }

vec3 TRANSFORM::TransDirection (const vec3 &vec) const
  {
  const float32 *p = matrix.p;
  return vec3
    (vec.x*p[0] + vec.y*p[1] + vec.z*p[2],
     vec.x*p[4] + vec.y*p[5] + vec.z*p[6],
     vec.x*p[8] + vec.y*p[9] + vec.z*p[10]);
  }

vec3 TRANSFORM::InvTransDirection (const vec3 &vec) const
  {
  const float32 *p = inverse.p;
  return vec3
    (vec.x*p[0] + vec.y*p[1] + vec.z*p[2],
     vec.x*p[4] + vec.y*p[5] + vec.z*p[6],
     vec.x*p[8] + vec.y*p[9] + vec.z*p[10]);
  }

vec3 TRANSFORM::TransNormal (const vec3 &vec) const
  {
  const float32 *p = inverse.p;
  return vec3
    (vec.x*p[0] + vec.y*p[4] + vec.z*p[8],
     vec.x*p[1] + vec.y*p[5] + vec.z*p[9],
     vec.x*p[2] + vec.y*p[6] + vec.z*p[10]);
  }

vec3 TRANSFORM::InvTransNormal (const vec3 &vec) const
  {
  const float32 *p = matrix.p;
  return vec3
    (vec.x*p[0] + vec.y*p[4] + vec.z*p[8],
     vec.x*p[1] + vec.y*p[5] + vec.z*p[9],
     vec.x*p[2] + vec.y*p[6] + vec.z*p[10]);
  }

TRANSFORM Scaling_Transform (const vec3 &vec)
  {
  TRANSFORM trans;
  trans.Make_Scaling_Transform (vec);
  return trans;
  }

TRANSFORM Translation_Transform (const vec3 &vec)
  {
  TRANSFORM trans;
  trans.Make_Translation_Transform (vec);
  return trans;
  }

TRANSFORM Axis_Rotation_Transform (const vec3 &axis, float64 angle)
  {
  TRANSFORM trans;
  trans.Make_Axis_Rotation_Transform (axis, angle);
  return trans;
  }

TRANSFORM Shearing_Transform
  (float32 xy, float32 xz, float32 yx, float32 yz, float32 zx, float32 zy)
  {
  TRANSFORM trans;
  trans.Make_Shearing_Transform (xy, xz, yx, yz, zx, zy);
  return trans;
  }

ostream &operator<< (ostream &os, const TRANSFORM &t)
  {
  os << "Transform\n{\n" << t.matrix << t.inverse << "}" << endl;
  return os;
  }
