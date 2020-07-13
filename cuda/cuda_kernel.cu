/*
 * Copyright (c) 2010-2014
 *              Marzia Rivi (1), Tim Dykes (2)
 *               (1) University of Oxford
 *               (2) University of Portsmouth
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

#ifndef __KERNEL__
#define __KERNEL__

 #include "cuda_kernel.cuh"
//help functions

__constant__ cu_param dparams;
__constant__ cu_color_map_entry dmap[MAXSIZE];
__constant__ int ptype_points[10];

__device__ __forceinline__ void clamp (float minv, float maxv, float &val)
{
  val = min(maxv, max(minv, val));
}

__device__ __forceinline__   double my_asinh (double val)
{ return log(val+sqrt(1.+val*val)); }

//fetch a color from color table on device
__device__ __forceinline__ cu_color get_color(int ptype, float val, int map_size, int map_ptypes)
{
  //first find the right entry for this ptype
  int     start, end;
  start = ptype_points[ptype];
  if ( ptype == map_ptypes-1)//the last type
    end = map_size-1;
  else
    end = ptype_points[ptype+1]-1;

  //search the section of this type to find the val
  int i=start;
  while ((val>dmap[i+1].val) && (i<end)) ++i;

  const float fract = (val-dmap[i].val)/(dmap[i+1].val-dmap[i].val);
  cu_color clr1=dmap[i].color, clr2=dmap[i+1].color;
  cu_color        clr;
  clr.r =clr1.r + fract*(clr2.r-clr1.r);
  clr.g =clr1.g + fract*(clr2.g-clr1.g);
  clr.b =clr1.b + fract*(clr2.b-clr1.b);

  return clr;
}

// Transform+coloring by kernel
#if !defined(CUDA_FULL_ATOMICS)
__global__ void k_process(cu_particle_sim *p, int *p_active, int n, int mapSize, int types, int tile_sidex, int tile_sidey, int width, int nxtiles, int nytiles)
#else
__global__ void k_process(cu_particle_sim *p, int n, int mapSize, int types)
#endif
{
  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >=n) return;

  int ptype = p[m].type;
  float r = p[m].r;
  float er = p[m].e.r;
  float eg = p[m].e.g;
  float eb = p[m].e.b;
  float I = p[m].I;

    // Normalization and clamping 

#ifndef NO_I_NORM
  // Norm and clamp I
    if (dparams.inorm_maxs[ptype]==dparams.inorm_mins[ptype])
      I = 1;
    else
      I = (max(dparams.inorm_mins[ptype],min(dparams.inorm_maxs[ptype],I))-dparams.inorm_mins[ptype])/(dparams.inorm_maxs[ptype]-dparams.inorm_mins[ptype]);
#endif

  // Norm and clamp er
    if (dparams.cnorm_maxs[ptype]==dparams.cnorm_mins[ptype])
      er = 1;
    else
      er = (max(dparams.cnorm_mins[ptype],min(dparams.cnorm_maxs[ptype],er))-dparams.cnorm_mins[ptype])/(dparams.cnorm_maxs[ptype]-dparams.cnorm_mins[ptype]);
  
  // If col_vector[t]
  // norm and clamp eg and eb
    if(dparams.col_vector[ptype])
    {
      if (dparams.cnorm_maxs[ptype]==dparams.cnorm_mins[ptype])
        eg = 1;
      else
        eg = (max(dparams.cnorm_mins[ptype],min(dparams.cnorm_maxs[ptype],er))-dparams.cnorm_mins[ptype])/(dparams.cnorm_maxs[ptype]-dparams.cnorm_mins[ptype]);

      if (dparams.cnorm_maxs[ptype]==dparams.cnorm_mins[ptype])
        eb = 1;
      else
        eb = (max(dparams.cnorm_mins[ptype],min(dparams.cnorm_maxs[ptype],er))-dparams.cnorm_mins[ptype])/(dparams.cnorm_maxs[ptype]-dparams.cnorm_mins[ptype]);
    }

  //now do x,y,z
 // float zminval = 0.0;
 // float zmaxval = 1e23;
  float x,y,z;
  x =p[m].x*dparams.p[0] + p[m].y*dparams.p[1] + p[m].z*dparams.p[2] + dparams.p[3];
  y =p[m].x*dparams.p[4] + p[m].y*dparams.p[5] + p[m].z*dparams.p[6] + dparams.p[7];
  z =p[m].x*dparams.p[8] + p[m].y*dparams.p[9] + p[m].z*dparams.p[10]+ dparams.p[11];

#ifdef CUDA_FULL_ATOMICS
  if(-z <= 0.0f){p[m].active = false;return;};
  if(-z >= 1e23){p[m].active = false;return;};
#else
  if(-z <= 0.0f){p[m].active = false; p_active[m]=-1;return;};
  if(-z >= 1e23){p[m].active = false; p_active[m]=-1;return;};
#endif
  //do r
  float xfac2 = dparams.xfac;
  //const float   res2 = 0.5f*dparams.xres;
  //const float   ycorr = 0.5f*(dparams.yres-dparams.xres);
  if (!dparams.projection)
    {
      x = 0.5f*dparams.xres + x*xfac2;
      y = 0.5f*dparams.xres + y*xfac2 + 0.5f*(dparams.yres-dparams.xres);
    }
  else
    {
      xfac2= -(0.5f*dparams.xres)/(dparams.fovfct*z);
      x = 0.5f*dparams.xres + dparams.xshift + x*xfac2;
      y = 0.5f*dparams.xres + dparams.yshift + y*xfac2 + 0.5f*(dparams.yres-dparams.xres);
    }

#ifdef SPLOTCH_CLASSIC
  I *= 0.5f*dparams.bfak/r;
  r*= 2;  
#else
  //I *= 8.f/(Pi*r*r*r);  //SPH kernel normalization
  //I *= dparams.h2sigma*sqrtf(Pi)*r;  //integral through the center
  I *= 8.f*dparams.h2sigma/(sqrtf(Pi)*r*r);
#endif

  r *= xfac2 * dparams.smooth_fac[ptype];
  const float rcorr= sqrtf(r*r + dparams.minrad_pix*dparams.minrad_pix)/r;
  r *= rcorr;
#ifdef SPLOTCH_CLASSIC
  I /= rcorr;
#else
  I /= rcorr*rcorr;
#endif
  I *= dparams.brightness[ptype];

  p[m].active = false;

#if !defined(CUDA_FULL_ATOMICS) 
  // Tiled implementation has seperate active array for filtering
  p_active[m] = -1; 
#endif

  // compute region occupied by the partile
  //float raux=dparams.rfac;
  const float rfacr=dparams.rfac*r;
  int minx=int(x-rfacr+1.f);
  if (minx>=dparams.xres) return;
  minx=max(minx,0);

  int maxx=int(x+rfacr+1.f);
  if (maxx<=0) return;
  maxx=min(maxx,dparams.xres);
  if (minx>=maxx) return;

  int miny=int(y-rfacr+1.f);
  if (miny>=dparams.yres) return;
  miny=max(miny,0);

  int maxy=int(y+rfacr+1.f);
  if (maxy<=0) return;
  maxy=min(maxy,dparams.yres);
  if (miny>=maxy) return;
  p[m].active = true;
  
  p[m].x = x;
  p[m].y = y;
  p[m].r = r;
  p[m].I = I;

//coloring
// get color, associated from physical quantity contained in e.r, from lookup table
  cu_color e;
  e.r=er;
  e.g=eg;
  e.b=eb;

  if (!dparams.col_vector[ptype])
     e = get_color(ptype, e.r, mapSize, types);

  p[m].e.r = e.r*I;
  p[m].e.g = e.g*I;
  p[m].e.b = e.b*I; 
  
#if !defined(CUDA_FULL_ATOMICS) 
  // Tiled implementation
  // Manage particles outside the image but that influence it
  if(x < 0.0 || x >= (float)dparams.xres){p_active[m] = -2; return;};
  if(y < 0.0 || y >= (float)dparams.yres){p_active[m] = -2; return;};
  // Active particle = tile_id to which it belongs to
  /////////////CLAAAAA p_active[m] = int(y)/tile_sidey + int(x)/tile_sidex*nytiles; 
  p_active[m] = int(y/float(tile_sidey)) + int(x/float(tile_sidex))*nytiles; 
 // if (p_active[m] < 0 || p_active[m] > nxtiles*nytiles) {printf("x=%f, y=%f, flag=%d\n",x,y,p_active[m]);}
  if ((maxx-minx)*(maxy-miny) <= 1) p_active[m] = nxtiles*nytiles; // point-like particles 
  if (int(rfacr)>=width) 
  {
      p_active[m] = -2; // particle to be removed and copied back to the host 
      //printf("x=%f, y=%f, rfacr=%d, WIDTH=%d \n",p[m].r,raux,int(rfacr),width);
  }
#endif

}
 
// Calculates logs, asinh is commented out because if it is used
// it is done on host
__global__ void k_range(int nP, cu_particle_sim *p)
{

  //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >=nP) return;

  // Get current particle type
  int ptype = p[m].type;

  // Check if we need to log10 intensity
  if (dparams.log_int[ptype])
  { 
    if(p[m].I > 0)
        p[m].I = log10(p[m].I);
    else
        p[m].I = -38;
  }

  if (dparams.log_col[ptype])
  {
    if(p[m].e.r > 0)
      {
      p[m].e.r = log10(p[m].e.r);
      }
    else
      p[m].e.r =-38;
  }

  if (dparams.col_vector[ptype])
  {
    if (dparams.log_col[ptype])
    {
      p[m].e.g = log10(p[m].e.g);
      p[m].e.b = log10(p[m].e.b);
    }
  }
}


#if !defined(CUDA_FULL_ATOMICS) 
// Tiled implementation 

// change of linear coordinate: from tile to global image
// lpix -> (x,y) -> (X,Y) -> gpix 
__device__ int pixelLocalToGlobal(int lpix, int xo, int yo, int width, int tile_sidey)
{
  // global 2D coordinates
  int x = xo + lpix/(tile_sidey+2*width);
  int y = yo + lpix%(tile_sidey+2*width);

  return x*dparams.yres+y;
}

#define NPSIZE 16

//device render function k_render1
// a_eq_e = false is not supported
__global__ void k_renderC2
__launch_bounds__(256, 8) 
(int nP, cu_particle_sim *part, int *tileId, int *tilepart, cu_color *pic, cu_color *pic1, cu_color *pic2, cu_color *pic3, int tile_sidex, int tile_sidey, int width, int nytiles)
{
   extern __shared__ cu_color Btile[];
   __shared__ int local_chunk_length, start;
   __shared__ cu_color e[NPSIZE];
   __shared__ float radsq[NPSIZE], stp[NPSIZE];
   __shared__ float posx[NPSIZE], posy[NPSIZE];
   __shared__ int minx[NPSIZE], maxx[NPSIZE], miny[NPSIZE], maxy[NPSIZE];
   
   // 1D length of Btile (tile plus boundarys) = 2x1D length of normal tile
   int tileBsize = (tile_sidex+2*width)*(tile_sidey+2*width);
   // Each block assigned a tile
   int tile = tileId[blockIdx.x]; // tile number 

   if (threadIdx.x == 0)
   {

      if (blockIdx.x == 0) start = 0;
      else start = tilepart[blockIdx.x-1];
      local_chunk_length = tilepart[blockIdx.x] - start;
   }
   __syncthreads();

   int xo = (tile/nytiles)*tile_sidex - width;  // Btile origin x
   int yo = (tile%nytiles)*tile_sidey - width;  // Btile origin y

  // Inizialise Btile
   // Block Dim is width of Btile, each thread inits the y column of pixels corresponding to its index on the x
  for (int i=threadIdx.x; i<tileBsize; i=i+blockDim.x) 
  {
     Btile[i].r = 0.0f;  Btile[i].g = 0.0f;   Btile[i].b = 0.0f;
  }
   __syncthreads();

  int x,y,k;
  int j = 0;
  // Check we dont go over width of tile(shouldnt be hardcoded to 16...)
  int last = min(NPSIZE, blockDim.x);
  // Now do the rendering: each thread processes a pixel of particle i
  // We do 16 particles per loop of this while block
  while (j < local_chunk_length) 
  {
      // The first 16 threads get the particles color/position/render params
      k = threadIdx.x; 
      if(k < last)
      {
        cu_particle_sim p = part[start+k+j];
        e[k] = p.e;
        posx[k] = p.x; posy[k] = p.y;
        float rfacr = dparams.rfac*p.r;
        radsq[k] = rfacr*rfacr;
        stp[k] = -1.f/(dparams.h2sigma*dparams.h2sigma*p.r*p.r);

        // Get min and max pixels affected, clamp to edges of image
        minx[k]=int(p.x-rfacr+1.f);
        minx[k]=max(minx[k],0);
        maxx[k]=int(p.x+rfacr+1.f);
        maxx[k]=min(maxx[k],dparams.xres); 
        miny[k]=int(p.y-rfacr+1.f);
        miny[k]=max(miny[k],0);
        maxy[k]=int(p.y+rfacr+1.f);
        maxy[k]=min(maxy[k],dparams.yres);
      }
      __syncthreads(); 
      j += last; //blockDim.x;
      if (j > local_chunk_length) last = local_chunk_length%last; //blockDim.x;
      // Loop over 16 particles
      for (int i=0; i<last; i++)
      {
         // Number of pixels affected by this particle
         int reg = (maxx[i]-minx[i])*(maxy[i]-miny[i]);
         // Render pixel threadIdx.x of particle i
         if (threadIdx.x < reg)
         {
           // global pixel coordinates
           x = threadIdx.x/(maxy[i]-miny[i]) + minx[i];
           y = threadIdx.x%(maxy[i]-miny[i]) + miny[i];
           // Get local pixal coords within btile
           // global pixel index = x*dparams.yres+y
           // localx = x-xo,   localy = y-yo 
           int lp = (x-xo)*(tile_sidey+2*width) + y-yo;  //local pixel index
         //  if (lp >= tileBsize) printf("lp = %d, tile=%d, x=%d, y=%d xo =%f yo=%f tile_sidey=%d width=%d\n",lp,tile,x,y,xo,yo,tile_sidey,width);
          // How far are we from the pixel centre
           float dsq = (y-posy[i])*(y-posy[i]) + (x-posx[i])*(x-posx[i]);
           if (dsq<radsq[i])
           {
            // Attribution is exp of stp * distance from centre
             float att = __expf(stp[i]*dsq);
             Btile[lp].r += -att*e[i].r;
             Btile[lp].g += -att*e[i].g;
             Btile[lp].b += -att*e[i].b;
           }
          }
      } 
      __syncthreads();  
  }

#ifdef CUDA_ATOMIC_TILE_UPDATE

    // Atomic update of image
   for (int i=threadIdx.x; i<9*tile_sidex*tile_sidey; i=i+blockDim.x) 
   {
      int kk = pixelLocalToGlobal(i,xo,yo,width,tile_sidey);
      atomicAdd(&(pic[kk].r),Btile[i].r);
      atomicAdd(&(pic[kk].g),Btile[i].g);
      atomicAdd(&(pic[kk].b),Btile[i].b);
   }

#else

// Update inner tile in the global image
// First work out the starting point of the inner tile within the btile
// (i.e. we skip the top boundary row, and the first pixel row of the left boundary column)
  int k0 = width*(tile_sidey+2*width) + width; 
  for (int i=threadIdx.x; i<tile_sidex*tile_sidey; i=i+blockDim.x) 
  {
     // Add correction to skip boundary pixels
     j = k0 + i + (i/tile_sidey)*2*width; 
     // Convert to global coords
     k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
     pic[k].r += Btile[j].r;
     pic[k].g += Btile[j].g;
     pic[k].b += Btile[j].b;
  }
  __syncthreads();

// update boundary in 3 steps: 
// 1. columns
  int ymax = yo + tile_sidey+2*width;
  int xmax = xo + tile_sidex+2*width;
  // Step is used for to split threads between left column/right column
  // or top row/bottom row, or corners
  int step = blockDim.x/2;

  // i.e. left column
  if ((threadIdx.x < step)  && (yo >= 0))
  {
    // Get to start pixel of left column (on furthest left pixel of btile)
    k0 = width*(tile_sidey+2*width);
    for (int i = threadIdx.x; i<tile_sidex*width; i=i+step) 
    {
      j = k0 + i + (i/width)*(tile_sidey+width); //add correction due to the boundary
      k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
      pic1[k].r += Btile[j].r;
      pic1[k].g += Btile[j].g;
      pic1[k].b += Btile[j].b;
    }
  }
  // i.e. right column
  else if ((threadIdx.x >= step)  && (ymax <= dparams.yres))
  {
    // Furthest left pixel of righthand column
    k0 = width*(tile_sidey+2*width) + width + tile_sidey; 
    for (int i = threadIdx.x - step; i<tile_sidex*width; i=i+step) 
    {
      j = k0 + i + (i/width)*(tile_sidey+width); //add correction due to the boundary
      k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
      pic1[k].r += Btile[j].r;
      pic1[k].g += Btile[j].g;
      pic1[k].b += Btile[j].b;
    }
  }
  __syncthreads();

// 2. rows
  if ((threadIdx.x < step) && (xo >= 0))
  {
    k0 = width; 
    for (int i=threadIdx.x; i<tile_sidey*width; i=i+step) 
    {
      j = k0 + i + (i/tile_sidey)*2*width; //add correction due to the boundary
      k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
      pic2[k].r += Btile[j].r;
      pic2[k].g += Btile[j].g;
      pic2[k].b += Btile[j].b;
    }
  }
  else if ((threadIdx.x >= step)  && (xmax <= dparams.xres))
  {
    k0 = width + (width+tile_sidex)*(tile_sidey+2*width); // starting point
    for (int i=threadIdx.x - step; i<tile_sidey*width; i=i+step) 
    {
      j = k0 + i + (i/tile_sidey)*2*width; //add correction due to the boundary
      k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
      pic2[k].r += Btile[j].r;
      pic2[k].g += Btile[j].g;
      pic2[k].b += Btile[j].b;
    }
  }
  __syncthreads();

// 3. corners
// dimension corners = 1/4 dimension blocks
  int i;
  if ((threadIdx.x < blockDim.x/4) && (xo >= 0) && (yo >= 0))
  {
     j = threadIdx.x + (threadIdx.x/width)*(tile_sidey+width);
     k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
     pic3[k].r += Btile[j].r;
     pic3[k].g += Btile[j].g;
     pic3[k].b += Btile[j].b;
  }
  else if ((threadIdx.x >= blockDim.x/4 && threadIdx.x < blockDim.x/2) && (xo >= 0) && (ymax <= dparams.yres))
  {
     k0 = width + tile_sidey; 
     i = threadIdx.x - blockDim.x/4; 
     j = k0 + i + (i/width)*(tile_sidey+width);
     k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
     pic3[k].r += Btile[j].r;
     pic3[k].g += Btile[j].g;
     pic3[k].b += Btile[j].b;
  }
  else if ((threadIdx.x >= blockDim.x/2 && threadIdx.x < 3*blockDim.x/4) && (xmax <= dparams.xres) && (yo >= 0))
  {
     k0 = (width + tile_sidex)*(tile_sidey+2*width);
     i = threadIdx.x - blockDim.x/2; 
     j = k0 + i + (i/width)*(tile_sidey+width);
     k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
     pic3[k].r += Btile[j].r;
     pic3[k].g += Btile[j].g;
     pic3[k].b += Btile[j].b;
  }
  else if ((threadIdx.x >= 3*blockDim.x/4) && (xmax <= dparams.xres) && (ymax <= dparams.yres))
  {
     k0 = (width + tile_sidex)*(tile_sidey+2*width) + width + tile_sidey;
     i = threadIdx.x - 3*blockDim.x/4; 
     j = k0 + i + (i/width)*(tile_sidey+width);
     k = pixelLocalToGlobal(j,xo,yo,width,tile_sidey);
     pic3[k].r += Btile[j].r;
     pic3[k].g += Btile[j].g;
     pic3[k].b += Btile[j].b;
  }
#endif

}


// Index C3 particles
// Work out th eindex of the pixel each particle affects
__global__ void k_indexC3(int n, cu_particle_sim *part, int *index)
{
   //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >= n) return;
 
  cu_particle_sim p = part[m];
  float rfacr = dparams.rfac*p.r;
  int x = int(p.x-rfacr+1.f);
  x = max(x,0);
  int y = int(p.y-rfacr+1.f);
  y = max(y,0);
  index[m] = x*dparams.yres+y; 
  //pixel = -p.e
}

__global__ void k_renderC3(int nC3, int *index, cu_particle_sim *part, cu_color *pic)
{
   //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >= nC3) return;

  pic[index[m]].r += - part[m].e.r;
  pic[index[m]].g += - part[m].e.g;
  pic[index[m]].b += - part[m].e.b;
  
}

#ifndef CUDA_ATOMIC_TILE_UPDATE
__global__ void k_add_images(int n, cu_color *pic, cu_color *pic1, cu_color *pic2, cu_color *pic3)
{
   //first get the index m of this thread
  int m=blockIdx.x *blockDim.x + threadIdx.x;
  if (m >n) return;

   pic[m].r += pic1[m].r + pic2[m].r + pic3[m].r;
   pic[m].g += pic1[m].g + pic2[m].g + pic3[m].g;
   pic[m].b += pic1[m].b + pic2[m].b + pic3[m].b;
}
#endif

// End tiled implementation
#else

// --------------------------------------
// Render for full atomic implementation
// --------------------------------------
__global__ void k_render(int nP, cu_particle_sim *part, cu_color *pic)
{
  // Get index, double check its not out of bounds 
  // (launch parameters mean it shouldnt be...)
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >=nP) return;

  cu_particle_sim p = part[idx];

  if(p.active)
  {
    // Work out radial factor
    float rfacr = dparams.rfac*p.r;
    float radsq = rfacr*rfacr;
    float stp = -1.f/(dparams.h2sigma*dparams.h2sigma*p.r*p.r);

    // Get min and max pixels affected, clamp to image boundary
    int minx=int(p.x-rfacr+1.f);
    minx=max(minx,0);
    int maxx=int(p.x+rfacr+1.f);
    maxx=min(maxx,dparams.xres); 
    int miny=int(p.y-rfacr+1.f);
    miny=max(miny,0);
    int maxy=int(p.y+rfacr+1.f);
    maxy=min(maxy,dparams.yres);


    // For each pixel on x
    for(int x = minx; x < maxx; ++x)
    {
      
      // Work out x dist from centre and new yminmax 
      float dxsq = (x - p.x)*(x-p.x);
      float dy = sqrt(radsq-dxsq);
      int miny2=max(miny,int(p.y-dy+1)),
          maxy2=min(maxy,int(p.y+dy+1));
      float pre2 = __expf(stp*dxsq);
      // For each pixel on y
      for(int y = miny2; y < maxy2; ++y)
      {
          // Work out y dist from centre  
          float dysq = (y - p.y) * (y - p.y);
          float att = __expf(stp*dysq);
          // Update global image
          atomicAdd(&(pic[x*dparams.yres+y].r),-att*p.e.r*pre2);
          atomicAdd(&(pic[x*dparams.yres+y].g),-att*p.e.g*pre2);
          atomicAdd(&(pic[x*dparams.yres+y].b),-att*p.e.b*pre2);      

      }
    
    }
  }
  //__syncthreads();
  //if(idx == 3) printf("px3 r: %f g: %f b: %f\n", pic[3].r,pic[3].g,pic[3].b);
  
}
// End atomic implementation
#endif


#endif

