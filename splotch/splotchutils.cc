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

#include <regex>
#include "splotch/splotchutils.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/string_utils.h"
#include "cxxsupport/lsconstants.h"
#include "cxxsupport/ls_image.h"

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
      //cout << " adding color bar for type " << itype << " ..." << endl;
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



// Clear a single palette from the colourmap array and replace it with a different one
void replace_colourmap(paramfile &params, vector<COLOURMAP> &amap, int itype, std::string palette="")
  {
  int ptypes = params.find<int>("ptypes",1);
  planck_assert((amap.size() == ptypes), "Cannot replace colormap: number of entries in map < ptypes");
  planck_assert((itype < ptypes), "Invalid type ID for colourmap replacement, id >= ptypes");
  bool master = mpiMgr.master();
    if (params.find<bool>("color_is_vector"+dataToString(itype),false))
      {
      if (master)
        cout << " color of ptype " << itype << " is vector, so no colormap to load ..." << endl;
      }
    else
      {
        // Load and check the palette file
        std::string path;
        if(palette != "") 
          path = palette;
        else
          path = params.find<string>("palette"+dataToString(itype));
        ifstream infile (path);
        planck_assert (infile,"could not open palette file  <" + path + ">");
        check_palette_format(infile);

        // Update palette for this type
        amap[itype].clear();
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
        amap[itype].sortMap();
        params.setParam<string>("palette"+dataToString(itype), path);
      }    
  }

// Check the format of a palette file when loaded
// We expect it to have format:
// Name
// 0100
// n_colours
// r g b 
// (repeat one per line for n_colours lines)

void check_palette_format(ifstream& infile)
{
  std::string line;
  std::vector<std::string> lines;

    // Format of file 
  // Match anything for the name
  std::vector<std::regex> exprs = {std::regex(".*"),std::regex("0100"),std::regex(R"(\d+)"),std::regex(R"(\s*\d+\s*\d+\s*\d+\s*)")};

  // Check first three lines
  for(unsigned i = 0; i < 3; i++)
  {
    if(getline(infile, line))
      if(std::regex_match(line, exprs[i]))
        continue;
    planck_assert(0,"Palette file format incorrect\n");
  }

  // Read number of entries from previous line
  // and check each line against the regex
  int ncols = stringToData<int>(line);
  int cols_found = 0;
  while(getline(infile, line)){
    if(!std::regex_match(line, exprs[3]))
      planck_assert(0,"Palette file format incorrect, colour record mismatched regex \\s*\\d+\\s*\\d+\\s*\\d+\n");
    cols_found++;
  }
  if(cols_found != ncols){
    planck_assert(0,"Palette file format incorrect, "+dataToString(cols_found)+" colours found of "+dataToString(ncols)+" expected\n");
  }
  infile.clear();
  infile.seekg(0, ios::beg);
}

// Parallel image compositing
// For MPI_A_NEQ_E case we do a depth ordered composition
// otherwise a simple MPI Sum
void composite_images(render_context& c)
{
  tstack_push("Composite");
#ifdef MPI_A_NEQ_E
    if(mpiMgr.num_ranks()>1) 
      composite(pic,opacity_map,composite_order, mpiMgr.rank(), mpiMgr.num_ranks());
#else
    mpiMgr.allreduceRaw
      (reinterpret_cast<float *>(&c.pic[0][0]),3*c.xres*c.yres,MPI_Manager::Sum);
#endif
  tstack_pop("Composite");
}

void colour_adjust(paramfile &params, render_context& c)
{
  // Modify gamma, brightness and contrast settings for image
  double gamma=params.find<double>("pic_gamma",1.0);
  double helligkeit=params.find<double>("pic_brightness",0.0);
  double kontrast=params.find<double>("pic_contrast",1.0);

  if (gamma != 1.0 || helligkeit != 0.0 || kontrast != 1.0)
  {
    cout << endl << "image enhancement (gamma,brightness,contrast) = ("
        << gamma << "," << helligkeit << "," << kontrast << ")" << endl;
#pragma omp parallel for
    for (tsize i=0; i<c.pic.size1(); ++i)
      for (tsize j=0; j<c.pic.size2(); ++j)
      {
        c.pic[i][j].r = kontrast * pow((double)c.pic[i][j].r,gamma) + helligkeit;
        c.pic[i][j].g = kontrast * pow((double)c.pic[i][j].g,gamma) + helligkeit;
        c.pic[i][j].b = kontrast * pow((double)c.pic[i][j].b,gamma) + helligkeit;
      }
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
  return bool(test);
  }

#ifdef MPI_A_NEQ_E
// Particle tree accessors
bool xcomp_smaller(const particle_sim &p1, const particle_sim &p2) {return (p1.x < p2.x) ? true : false;}
bool ycomp_smaller(const particle_sim &p1, const particle_sim &p2) {return (p1.y < p2.y) ? true : false;}
bool zcomp_smaller(const particle_sim &p1, const particle_sim &p2) {return (p1.z < p2.z) ? true : false;}
float xcomp_diff(const particle_sim &p1, const particle_sim &p2) {return p1.x - p2.x;}
float ycomp_diff(const particle_sim &p1, const particle_sim &p2) {return p1.y - p2.y;}
float zcomp_diff(const particle_sim &p1, const particle_sim &p2) {return p1.z - p2.z;}
float x_accessor(const particle_sim &p1) {return p1.x;}
float y_accessor(const particle_sim &p1) {return p1.y;}
float z_accessor(const particle_sim &p1) {return p1.z;}
float r_accessor(const particle_sim &p1) {return p1.r;}
float ghost_accessor(const particle_sim &p1) {return p1.ghost;}
void ghost_setter(particle_sim &p1) {p1.ghost = 1;}
#endif

void update_res(paramfile& params, render_context& c)
{
    int new_xres = params.find<int>("xres",800);
    int new_yres = params.find<int>("yres",new_xres);
    if(new_xres != c.xres || new_yres != c.yres)
    {
      c.xres = new_xres;
      c.yres = new_yres;
      c.pic.alloc(c.xres, c.yres);
    }
}

void do_background(paramfile& params, arr2<COLOUR>& pic, const std::string& outfile)
{
  bool background = params.find<bool>("background",false);
  if(background)
    {
    if (mpiMgr.master())
      {
        cout << endl << "reading file Background/" << outfile << " ..." << endl;
        LS_Image img;
        img.read_TGA("Background/"+outfile+".tga");

      #pragma omp parallel for
        for (tsize i=0; i<pic.size1(); ++i)
          for (tsize j=0; j<pic.size2(); ++j)
          {
            Colour8 c=img.get_pixel(i,j);
            pic[i][j].r=c.r/256.0;
            pic[i][j].g=c.g/256.0;
            pic[i][j].b=c.b/256.0;
        }
      }
    mpiMgr.bcastRaw(&pic[0][0].r,3*pic.size1()*pic.size2());
    }
}

void checkbbox(std::vector<particle_sim>& p, paramfile& params)
{ 
      // this is to check the bounding box
  float minx=1e15;
  float maxx=-1e15;
  float miny=1e15;
  float maxy=-1e15;
  float minz=1e15;
  float maxz=-1e15;
  for (unsigned long m1=0; m1<p.size(); ++m1)
  {
      minx = std::min(minx,p[m1].x);
      maxx = std::max(maxx,p[m1].x);
      miny = std::min(miny,p[m1].y);
      maxy = std::max(maxy,p[m1].y);
      minz = std::min(minz,p[m1].z);
      maxz = std::max(maxz,p[m1].z);
  }

  if (mpiMgr.master())
  {
    std::cout << "minx, maxx: " << minx << " " << maxx << std::endl;
    std::cout << "miny, maxy: " << miny << " " << maxy << std::endl;
    std::cout << "minz, maxz: " << minz << " " << maxz << std::endl;
  }    
}

void checkrth(float r_th, int npart, std::vector<particle_sim>& p, paramfile &/*params*/)
{
    // this is to check the .r distribution:
  long countsmall=0;
  long countinter1=0;
  long countinter2=0;
  long countinter3=0;
  long countinter4=0;
  long countlarge=0;
  long countactive=0;

  if(r_th != 0.0)
    {
    for (long m=0; m<npart; ++m)
      {
        if(p[m].active == true)
        {
    if(p[m].r < 1.0)countsmall++;
          if(p[m].r >= 1.0 && p[m].r < r_th)countinter1++;
          if(p[m].r >= 1.0 && p[m].r < 2*r_th)countinter2++;
          if(p[m].r >= 1.0 && p[m].r < 4*r_th)countinter3++;
          if(p[m].r >= 1.0 && p[m].r < 8*r_th)countinter4++;
          if(p[m].r >= 8*r_th)countlarge++;
          countactive++;
        }
      }
    if (mpiMgr.master())
      {
  std::cout << "NUMBER OF ACTIVE PARTICLES = " << countactive << std::endl;
  std::cout << "PARTICLES WITH r < 1 = " << countsmall << std::endl;
  std::cout << "PARTICLES WITH 1 <= r < " << r_th << " = " << countinter1 << std::endl;
  std::cout << "PARTICLES WITH 1 <= r < " << 2*r_th << " = " << countinter2 << std::endl;
  std::cout << "PARTICLES WITH 1 <= r < " << 4*r_th << " = " << countinter3 << std::endl;
  std::cout << "PARTICLES WITH 1 <= r < " << 8*r_th << " = " << countinter4 << std::endl;
  std::cout << "PARTICLES WITH r >= " << 8*r_th << " = " << countlarge << std::endl;
      }
    }
}

static std::map< SimTypeId, std::string > st2s = {
          {SimTypeId::NONE,             "None"},
          {SimTypeId::BIN_TABLE,        "Binary Table"},
          {SimTypeId::BIN_BLOCK,        "Binary Block"},
          {SimTypeId::GADGET,           "Gadget"},
          {SimTypeId::ENZO,             "Enzo"},
          {SimTypeId::GADGET_MILLENIUM, "Gadget Millenuim"},
          {SimTypeId::BIN_BLOCK_MPI,    "Binary Block MPI"},
          {SimTypeId::MESH,             "Mesh"},
          {SimTypeId::REGULAR_HDF5,     "HDF5"},
          {SimTypeId::GADGET_HDF5,      "Gadget HDF5"}, 
          {SimTypeId::VISIVO,           "Visivo"}, 
          {SimTypeId::TIPSY,            "Tipsy"},
          {SimTypeId::H5PART,           "H5Part"},
          {SimTypeId::RAMSES,           "Ramses"},
          {SimTypeId::BONSAI,           "Bonsai"},
          {SimTypeId::ASCII,            "Ascii"},
          {SimTypeId::FITS,             "FITS"},
};

 std::string simtype2str(SimTypeId simtype) {
      auto it = st2s.find(simtype);
      planck_assert(it!=st2s.end(), "simtype2str() simtype doesnt exist");
      return it->second;
}

static std::map< FieldId, std::string > fid2s = {
          {FieldId::NONE,   "None"},
          {FieldId::F_X,    "X"},
          {FieldId::F_Y,    "Y"},
          {FieldId::F_Z,    "Z"},
          {FieldId::F_CR,   "CR"},
          {FieldId::F_CG,   "CG"},
          {FieldId::F_CB,   "CB"},
          {FieldId::F_R,    "R"},
          {FieldId::F_I,    "I"}
};

 std::string FieldId2str(FieldId fid) {
      auto it = fid2s.find(fid);
      planck_assert(it!=fid2s.end(), "FieldId2str() field id doesnt exist");
      return it->second;
}

FieldId str2FieldId(std::string str) {
    auto it = fid2s.begin();
    while(it != fid2s.end() && it->second != str) it++;
    planck_assert(it!=fid2s.end(), "str2FieldId() string doesnt exist");
    return it->first;
}


#ifdef SPLVISIVO
void splvisivo_init_params(paramfile& params, VisIVOServerOptions& opt)
{
  params.setParam("camera_x",opt.spPosition[0]);
  params.setParam("camera_y",opt.spPosition[1]);
  params.setParam("camera_z",opt.spPosition[2]);
  params.setParam("lookat_x",opt.spLookat[0]);
  params.setParam("lookat_y",opt.spLookat[1]);
  params.setParam("lookat_z",opt.spLookat[2]);
  params.setParam("outfile",visivoOpt->imageName);
  params.setParam("interpolation_mode",0);
  params.setParam("simtype",10);
  params.setParam("fov",opt.spFov);
  params.setParam("ptypes",1);  
}
#endif
