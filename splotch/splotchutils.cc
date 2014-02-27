/*
 * Copyright (c) 2004-2013
 *              Martin Reinecke (1), Klaus Dolag (1)
 *               (1) Max-Planck-Institute for Astrophysics
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
#include <fstream>
#include "splotch/splotchutils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/string_utils.h"

using namespace std;

double my_asinh (double val)
  { return log(val+sqrt(1.+val*val)); }


void add_colorbar(paramfile &params, arr2<COLOUR> &pic, vector<COLOURMAP> &amap)
  {
  int xres = pic.size1(), yres=pic.size2();
  int offset=0;
  int ptypes = params.find<int>("ptypes",1);

  for(int itype=0;itype<ptypes;itype++)
    {
    if (params.find<bool>("color_is_vector"+dataToString(itype),false))
      cout << " adding no color bar for type " << itype
           << " as it is color vector ..." << endl;
    else
      {
      cout << " adding color bar for type " << itype << " ..." << endl;
      for (int x=0; x<xres; x++)
        {
        COLOUR e=amap[itype].getVal(x/float64(xres));
        for (int y=0; y<10; y++)
          pic[x][yres-offset-1-y] = e;
        }
      offset += 10;
      }
    }
  }

#ifdef SPLVISIVO
void get_colourmaps (paramfile &params, vector<COLOURMAP> &amap, VisIVOServerOptions &opt)
#else
void get_colourmaps (paramfile &params, vector<COLOURMAP> &amap)
#endif
  {
  int ptypes = params.find<int>("ptypes",1);

  bool master = mpiMgr.master();
  amap.resize(ptypes);

  if (master)
    cout << "building color maps (" << ptypes << ")..." << endl;
  for (int itype=0;itype<ptypes;itype++)
    {
    if (params.find<bool>("color_is_vector"+dataToString(itype),false))
      {
      if (master)
        cout << " color of ptype " << itype << " is vector, so no colormap to load ..." << endl;
      }
    else
      {
      bool VisIVOPalette=false;
#ifdef SPLVISIVO
//reading colortable from visivo
      string paletteFile=params.find<string>("palette"+dataToString(itype),"none"); //itype is 0 in VisIVO
      if(paletteFile=="none")  //VisIVO color table
        {
        int nVVColours=0;
        SelectLookTable(&opt);  //the Table is loaded only one time
        nVVColours=opt.extPalR.size();
        double step = 1./(nVVColours-1);
        for (int i=0; i<opt.extPalR.size(); i++)
          {
          float rrr,ggg,bbb;
          rrr=(float)opt.extPalR[i]; //these vale are already normalized to 255
          ggg=(float)opt.extPalG[i];
          bbb=(float)opt.extPalB[i];
          amap[itype].addVal(i*step,COLOUR(rrr,ggg,bbb));
          }
        VisIVOPalette=true;
        }
#endif
      if (!VisIVOPalette)
        {
        ifstream infile (params.find<string>("palette"+dataToString(itype)).c_str());
        planck_assert (infile,"could not open palette file  <" +
          params.find<string>("palette"+dataToString(itype)) + ">");
        string dummy;
        int nColours;
        infile >> dummy >> dummy >> nColours;
        if (master)
          cout << " loading " << nColours << " entries of color table of ptype " << itype << endl;
        double step = 1./(nColours-1);
        for (int i=0; i<nColours; i++)
          {
          float rrr,ggg,bbb;
          infile >> rrr >> ggg >> bbb;
          amap[itype].addVal(i*step,COLOUR(rrr/255,ggg/255,bbb/255));
          }
        } //if(!VisIVOPalette)

      }
    amap[itype].sortMap();
    }
  }

void timeReport()
  {
  if (mpiMgr.master())
    tstack_report("Splotch");
  }

void hostTimeReport(wallTimerSet &Timers)
  {
  cout << "Ranging Data (secs)        : " << Timers.acc("range") << endl;
  cout << "Build Index List (secs)    : " << Timers.acc("buildindex") << endl;
  cout << "Interpolating Data (secs)  : " << Timers.acc("interpolate") << endl;
  cout << "Transforming Data (secs)   : " << Timers.acc("transform") << endl;
  cout << "Sorting Data (secs)        : " << Timers.acc("sort") << endl;
  cout << "Coloring Sub-Data (secs)   : " << Timers.acc("coloring") << endl;
  cout << "Rendering Sub-Data (secs)  : " << Timers.acc("render") << endl;
  }

bool file_present(const string &name)
  {
  ifstream test(name.c_str());
  return test;
  }
