/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
 *              Claudio Gheller ETH-CSCS
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "mic_splotch.h"

#define SPLOTCH_CLASSIC
const float32 h2sigma = 0.5*pow(pi,-1./6.);
const float32 sqrtpi = sqrt(pi);

#ifdef SPLOTCH_CLASSIC
const float32 bfak=0.5*pow(pi,-5./6.); // 0.19261
#endif


// Compute the parameters to use for rototranslation on the device
const float* compute_transform(paramfile& params, transform_data& td, const vec3& campos,const vec3& centerpos, const vec3& lookat, vec3 sky)
{

  int xres = params.find<int>("xres",800),
      yres = params.find<int>("yres",xres);
  float32 zmaxval = params.find<float32>("zmax",1.e23),
          zminval = params.find<float32>("zmin",0.0);
  zminval = std::max(0.f,zminval);

  float32 ycorr = .5f*(yres-xres);
  float32 res2 = 0.5f*xres;
  // Fov is in degrees
  float32 fov = params.find<float32>("fov",45);
  float32 fovfct = tan(fov*0.5f*degr2rad);

  sky.Normalize();
  vec3 zaxis = (centerpos-lookat).Norm();
  vec3 xaxis = crossprod (sky,zaxis).Norm();
  vec3 yaxis = crossprod (zaxis,xaxis);
  TRANSFORM trans;
  trans.Make_General_Transform
    (TRANSMAT(xaxis.x,xaxis.y,xaxis.z,
              yaxis.x,yaxis.y,yaxis.z,
              zaxis.x,zaxis.y,zaxis.z,
              0,0,0));
  trans.Invert();
  TRANSFORM trans2;
  trans2.Make_Translation_Transform(-campos);
  trans2.Add_Transform(trans);
  trans=trans2;

  vec3 tcpos=trans.TransPoint(centerpos);
  bool projection = params.find<bool>("projection",true);

  float32 dist = (centerpos-lookat).Length();
  float32 xfac = res2/(fovfct*dist);
  double xshift=xfac*tcpos.x,
         yshift=xfac*tcpos.y;
  if (!projection)
    std::cout << " Horizontal field of fiew: " << xres/xfac << std::endl;

  float32 minrad_pix = params.find<float32>("minrad_pix",1.);

  td.xfac = xfac;
  td.xres = xres;
  td.yres = yres;
  td.zminval = zminval;
  td.zmaxval = zmaxval;
  td.ycorr = ycorr;
  td.dist = dist;
  td.fovfct = fovfct;
  td.h2sigma = h2sigma;
  td.sqrtpi = sqrtpi;
  td.projection = projection;
  td.minrad_pix = minrad_pix;
  td.xshift = xshift;
  td.yshift = yshift;

  #ifdef SPLOTCH_CLASSIC
  td.bfak = bfak;
  #else
  td.bfak = 1;
  #endif 

  // Get transformation matrix
  float* transmatrix = (float*)_mm_malloc(12*sizeof(float), 64);
  for(unsigned i = 0; i < 12; i++)
  {
  	transmatrix[i] = trans.Matrix().p[i];
  	//std::cout << "Translation matrix [" << i << "]: " << transmatrix[i] << std::endl;
  }
  return transmatrix;
}

// Compute a c style colourmap to use on the device. Similar to method used in CUDA version.
void compute_colormap(paramfile& params, int ptypes, float* brightness, bool* col_is_vec, float b_brightness, std::vector<COLOURMAP> &amap, mic_color_map& mmap)
{
	// Get brightness + col_is_vec from param file
	// Brightness is multiplied by b_brightness modifier for boost ( = 1 if not using boost )
	for(int i = 0; i < ptypes; i++)
	{
		brightness[i] = params.find<float32>("brightness"+dataToString(i),1.f);
		brightness[i] *= b_brightness;
		col_is_vec[i] = params.find<bool>("color_is_vector"+dataToString(i),false);
	}

	// Compute colormap for device
	mmap.ptype_color_offset = (int*)_mm_malloc((ptypes+1)*sizeof(int),64); 
	 
	 // First we need to count all the entries to get colormap size
	mmap.size = 0;
  	for (int i=0; i<amap.size(); i++)
    	mmap.size += amap[i].size();

  mmap.mapcolor = (mic_color*)_mm_malloc(mmap.size*sizeof(mic_color),64);
  mmap.mapvalue = (float*)_mm_malloc(mmap.size*sizeof(float),64);

	// Then fill up the colormap 
	int j,index =0;
	int thisTypeOffset = 0;
	for(int i=0; i<amap.size(); i++)
	{
		for (j=0; j<amap[i].size(); j++)
		{
			mmap.mapvalue[index] = amap[i].getX(j);
			COLOUR c (amap[i].getY(j));
			mmap.mapcolor[index].r = c.r;
			mmap.mapcolor[index].g = c.g;
			mmap.mapcolor[index].b = c.b;
			index++;
		}
		mmap.ptype_color_offset[i] = thisTypeOffset;
		thisTypeOffset += j;
	}
	// End point
	mmap.ptype_color_offset[ptypes] = mmap.size;

  // printf("Colormap info: Index size: %d\n");
  // for(unsigned i = 0; i < index; i++)
  // {
  //  printf("Index: %d \t Mapvalue: %f \t Mapcolor R: %f G %f B %f\n",i,  mmap.mapvalue[i], mmap.mapcolor[i].r ,mmap.mapcolor[i].g ,mmap.mapcolor[i].b );  
  // }

}


