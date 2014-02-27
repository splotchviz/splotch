//-----------------------------------------------------------------------------
// Copyright (c) 2005-2006 dhpoware. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
//-----------------------------------------------------------------------------
//
// This is a stripped down version of the dhpoware 3D Math Library. To download
// the full version visit: http://www.dhpoware.com/source/mathlib.html
//
//  Edits made by Tim Dykes - integration of matrix functions and splotch vec3 
//  class
//
//-----------------------------------------------------------------------------

#include "MathLib.h"

const float Math::PI = 3.1415926f;
const float Math::EPSILON = 1e-6f;

const Matrix4 Matrix4::IDENTITY(1.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 1.0f);

void Matrix4::rotate(const vec3f &axis, float degrees)
{
    // Creates a rotation matrix about the specified axis.
    // The axis must be a unit vector. The angle must be in degrees.
    //
    // Let u = axis of rotation = (x, y, z)
    //
    //             | x^2(1 - c) + c  xy(1 - c) + zs  xz(1 - c) - ys   0 |
    // Ru(angle) = | yx(1 - c) - zs  y^2(1 - c) + c  yz(1 - c) + xs   0 |
    //             | zx(1 - c) - ys  zy(1 - c) - xs  z^2(1 - c) + c   0 |
    //             |      0              0                0           1 |
    //
    // where,
    //  c = cos(angle)
    //  s = sin(angle)

    degrees = Math::degreesToRadians(degrees);

    float x = axis.x;
    float y = axis.y;
    float z = axis.z;
    float c = cosf(degrees);
    float s = sinf(degrees);

    mtx[0][0] = (x * x) * (1.0f - c) + c;
    mtx[0][1] = (x * y) * (1.0f - c) + (z * s);
    mtx[0][2] = (x * z) * (1.0f - c) - (y * s);
    mtx[0][3] = 0.0f;

    mtx[1][0] = (y * x) * (1.0f - c) - (z * s);
    mtx[1][1] = (y * y) * (1.0f - c) + c;
    mtx[1][2] = (y * z) * (1.0f - c) + (x * s);
    mtx[1][3] = 0.0f;

    mtx[2][0] = (z * x) * (1.0f - c) + (y * s);
    mtx[2][1] = (z * y) * (1.0f - c) - (x * s);
    mtx[2][2] = (z * z) * (1.0f - c) + c;
    mtx[2][3] = 0.0f;

    mtx[3][0] = 0.0f;
    mtx[3][1] = 0.0f;
    mtx[3][2] = 0.0f;
    mtx[3][3] = 1.0f;
}

Matrix4 perspective( float fovx, float ar, float near, float far )
{
    float range = tan( Math::degreesToRadians(fovx) / 2.0f ) * near;
    float left = -range;
    float right = range;
    float bottom = -range / ar;
    float top = range / ar;

    Matrix4 result;
    result[0][0] = (2.0f * near) / (right - left);
    result[0][1] = 0;
    result[0][2] = 0;
    result[0][3] = 0;
    result[1][0] = 0;
    result[1][1] = (2.0f * near) / (top - bottom);
    result[1][2] = 0;
    result[1][3] = 0;
    result[2][0] = 0;
    result[2][1] = 0;
    result[2][2] = - (far + near) / (far - near);
    result[2][3] = - 1.0f;
    result[3][0] = 0;
    result[3][1] = 0;
    result[3][2] = - 2.0f * ( far * near) / (far - near);
    result[3][3] = 0;
    return result;
}

//orthographic (same as glOrtho implementation)
Matrix4 orthographic(float left, float right, float bottom, float top, float near, float far)
{
    float rl = right - left;
    float tb = top - bottom;
    float fn = far - near;
    float tx = - (right + left) / (right - left);
    float ty = - (top + bottom) / (top - bottom);
    float tz = - (far + near) / (far - near);

    Matrix4 orthoMatrix;
    
    orthoMatrix[0][0] = 2.0f / rl;
    orthoMatrix[0][1] = 0.0f;
    orthoMatrix[0][2] = 0.0f;
    orthoMatrix[0][3] = tx;
    orthoMatrix[1][0] = 0.0f;
    orthoMatrix[1][1] = 2.0f / tb;
    orthoMatrix[1][2] = 0.0f;
    orthoMatrix[1][3] = ty;
    orthoMatrix[2][0] = 0.0f;
    orthoMatrix[2][1] = 0.0f;
    orthoMatrix[2][2] = 2.0f / fn;
    orthoMatrix[2][3] = tz;
    orthoMatrix[3][0] = 0.0f;
    orthoMatrix[3][1] = 0.0f;
    orthoMatrix[3][2] = 0.0f;
    orthoMatrix[3][3] = 1.0f;

    return orthoMatrix;
}

