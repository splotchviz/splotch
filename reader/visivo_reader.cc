/*
 * Copyright (c) 2004-2014
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


#include <iostream>
#include <vector>
#include <string>

#include "cxxsupport/arr.h"
#include "cxxsupport/sort_utils.h"
#include "cxxsupport/paramfile.h"
#include "cxxsupport/mpi_support.h"
#include "cxxsupport/bstream.h"
#include "splotch/splotchutils.h"

#ifdef SPLVISIVO
#include "visivoutils.h"
#endif
#ifndef SPLVISIVO
void visivo_reader ()
{
  return;
}
#else

using namespace std;

namespace {


void visivo_reader_finish (vector<particle_sim> &points)
  {
  float minr=1e30;
  float maxr=-1e30;
  for (tsize i=0; i<points.size(); ++i)
    {
    points[i].type=0;
    minr = min(minr,points[i].r);
    maxr = max(maxr,points[i].r);
    }
  mpiMgr.allreduce(maxr,MPI_Manager::Max);
  mpiMgr.allreduce(minr,MPI_Manager::Min);
  }

} // unnamed namespace

/* In this case we expect the file to be written as a binary table
   xyzrabcdexyzrabcdexyzrabcde...xyzrabcde */

/* In this case we expect the file to be written as
   xxxxxxxxxxxxxxxxxxxxyyyyyyyyyyyyyyyyyyyy...TTTTTTTTTTTTTTTTTTTT */
bool visivo_reader (paramfile &params, vector<particle_sim> &points, VisIVOServerOptions &opt)
  {
//    bifstream inp;
  int nfields;
  int64 mybegin, npart, npart_total;
  arr<int> qty_idx;
  if (mpiMgr.master())
    cout << "VISIVO BINARY FILE" << endl;
  npart=opt.nRows;
  npart_total=opt.nRows;
  points.resize(npart);
  int itype=0; // itype =0 until ptype=1
  float fix_size = params.find<float>("size_fix"+dataToString(itype),1.0);
  float size_fac = params.find<float>("size_fac"+dataToString(itype),1.0);
  
  if(opt.goodAllocation) //data is already in memory
  {
//CLAUDIO      	points.x =(float)opt.tableData[opt.columns.find(opt.xField)->second][i];
	for(int i=0;i<opt.nRows;i++)
	{

        if(opt.needSwap)
	{
     		points[i].x =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.xField)->second][i]));
     		points[i].y =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.yField)->second][i]));
     		points[i].z =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.zField)->second][i]));
		if(opt.columns.find(opt.sprField)!=opt.columns.end())
		  points[i].r =(fix_size==0.0) ? 
		    size_fac*(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.sprField)->second][i])) :fix_size ;
		
		if(opt.columns.find(opt.spIField)!=opt.columns.end())
		  points[i].I =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.spIField)->second][i]));

		if(opt.columns.find(opt.spC1Field)!=opt.columns.end())
		  points[i].e.r =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.spC1Field)->second][i]));

		if(opt.columns.find(opt.spC2Field)!=opt.columns.end())
		  points[i].e.g =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.spC2Field)->second][i]));
		else
		  points[i].e.g=0.0;
		
		if(opt.columns.find(opt.spC3Field)!=opt.columns.end())
		  points[i].e.b =(float)floatSwap((char *)(&opt.tableData[opt.columns.find(opt.spC3Field)->second][i]));
		else
		  points[i].e.b=0.0;

	  
	}  else {
/*	        clog<<opt.xField<<" "<<opt.columns.find(opt.xField)->second<<std::endl;
		int firstIndex=opt.columns.find(opt.xField)->second;
		 clog<<firstIndex<<" "<<i<<" "<<opt.tableData[firstIndex][i]<<std::endl;*/
		 
     		points[i].x = opt.tableData[opt.columns.find(opt.xField)->second][i];

		points[i].y = opt.tableData[opt.columns.find(opt.yField)->second][i];
      		points[i].z = opt.tableData[opt.columns.find(opt.zField)->second][i];
		if(opt.columns.find(opt.sprField)!=opt.columns.end())
		        points[i].r = (fix_size==0.0) ? 
		          size_fac*opt.tableData[opt.columns.find(opt.sprField)->second][i] : fix_size;
		else
			points[i].r=size_fac*1.0;
		if(opt.columns.find(opt.spIField)!=opt.columns.end())
		        points[i].I =opt.tableData[opt.columns.find(opt.spIField)->second][i];
		else
			points[i].I=0.5;
		if(opt.columns.find(opt.spC1Field)!=opt.columns.end())
		        points[i].e.r =opt.tableData[opt.columns.find(opt.spC1Field)->second][i];
		else
			points[i].e.r=1.0;
		if(opt.columns.find(opt.spC2Field)!=opt.columns.end())
		        points[i].e.g =opt.tableData[opt.columns.find(opt.spC2Field)->second][i];
		else
		        points[i].e.g=0.0;
		if(opt.columns.find(opt.spC3Field)!=opt.columns.end())
		        points[i].e.b =opt.tableData[opt.columns.find(opt.spC3Field)->second][i];
		else
		        points[i].e.b=0.0;

	}
	} // for
  } else {  //read from file   
       ifstream inFile;
	float readValue;
       inFile.open(opt.path.c_str(),ios::binary); 
       if (!inFile)
	{ 
		std::cerr << "Could not open input file "<<opt.path<< std::endl; 
		return false; // to be exit
	}
        inFile.seekg((opt.x*opt.nRows)* sizeof(float), ios::beg);
	for(int i=0;i<opt.nRows;i++)
	{
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].x =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].x =(float)readValue;
	}
  	inFile.seekg((opt.y*opt.nRows)* sizeof(float), ios::beg);
	for(int i=0;i<opt.nRows;i++)
	{
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].y =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].y =(float)readValue;
	}
  	inFile.seekg((opt.z*opt.nRows)* sizeof(float), ios::beg);
	for(int i=0;i<opt.nRows;i++)
	{
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].z =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].z =(float)readValue;
	}
	
	if(opt.columns.find(opt.sprField)!=opt.columns.end())
	{
	  inFile.seekg((opt.spr*opt.nRows)* sizeof(float), ios::beg);
	  for(int i=0;i<opt.nRows;i++)
	  {
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].r = (fix_size==0.0) ? size_fac*(float)floatSwap((char *)(&readValue)):fix_size;
    		else
      			points[i].r = (fix_size==0.0) ?  size_fac*(float)readValue : fix_size;
	  }
	} else 
	  for(int i=0;i<opt.nRows;i++)
	    points[i].r=size_fac*1.0;
	
	if(opt.columns.find(opt.spIField)!=opt.columns.end())
	{
	  inFile.seekg((opt.spI*opt.nRows)* sizeof(float), ios::beg);
	  for(int i=0;i<opt.nRows;i++)
	  {
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].I =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].I =(float)readValue;
	  }
	} else
	  for(int i=0;i<opt.nRows;i++)
	     points[i].I =0.5;
	
	if(opt.columns.find(opt.spC1Field)!=opt.columns.end())
	{
	  inFile.seekg((opt.spC1*opt.nRows)* sizeof(float), ios::beg);
	  for(int i=0;i<opt.nRows;i++)
	  {
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].e.r =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].e.r =(float)readValue;
	  }
	} else 
	  for(int i=0;i<opt.nRows;i++)
	    points[i].e.r=1.0;
	
	if(opt.columns.find(opt.spC2Field)!=opt.columns.end())
	{
	  inFile.seekg((opt.spC2*opt.nRows)* sizeof(float), ios::beg);
	  for(int i=0;i<opt.nRows;i++)
	  {
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].e.g =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].e.g =(float)readValue;
	  }
	} else
	  for(int i=0;i<opt.nRows;i++)
	     points[i].e.g=0.0; 

	if(opt.columns.find(opt.spC3Field)!=opt.columns.end())
	{
	  inFile.seekg((opt.spC3*opt.nRows)* sizeof(float), ios::beg);
	  for(int i=0;i<opt.nRows;i++)
	  {
 		inFile.read((char *)( &readValue),sizeof(float));
   		if(opt.needSwap)
     			points[i].e.b =(float)floatSwap((char *)(&readValue));
    		else
      			points[i].e.b =(float)readValue;
	  }
	} else
	  for(int i=0;i<opt.nRows;i++)
	     points[i].e.b=0.0; 

	  
	}// if opt.goodAllocation

  visivo_reader_finish (points);
  return true;
}
#endif
