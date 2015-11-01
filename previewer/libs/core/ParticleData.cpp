/*
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
 * 		Tim Dykes and Ian Cant
 * 		University of Portsmouth
 *
 */

#include "ParticleData.h"
// Need previewer include, in cpp to avoid circular deps
#include "previewer/Previewer.h"

namespace previewer
{

	void ParticleData::Load()
	{
		
		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();
		sceneMaker sMaker(*splotchParams);

		vec3 dummyCam = vec3(0,0,0);
		vec3 dummyAt = vec3(0,0,0);
		vec3 dummySky = vec3(0,0,0);
		vec3 dummyCent = vec3(0,0,0);
		std::string dummyOutfile = " ";
	
		// Splotch param find
		bool boost = splotchParams->find<bool>("boost",false);

		// Previewer param find
		//bool boost = param.GetParameter<bool>("boost");

		if(boost)
		{
			//if boost particledata is second argument(boosted data)
		  bool ret = sMaker.getNextScene(dummySplotchParticleData, particleList, dummyCam, dummyCent, dummyAt, dummySky, dummyOutfile);
			if(!ret) std::cout << "sMaker.getNextScene failed." << std::endl;
		}
		else 
		{
			//else use full data set (first arg)
		  bool ret = sMaker.getNextScene(particleList, dummySplotchParticleData, dummyCam, dummyCent, dummyAt, dummySky, dummyOutfile);
			if(!ret) std::cout << "sMaker.getNextScene failed." << std::endl;		
		}

		// Get colourmap
		int ret = FileLib::LoadColourPalette(Previewer::parameterInfo, colourMaps);
		if(!ret)
			std::cout << "Colour palette in param file is invalid" << std::endl;

		// Get per ptype parameters
		numTypes = splotchParams->find<int>("ptypes",1);

		for(unsigned i = 0; i < numTypes; i++)
		{
			brightness.push_back(splotchParams->find<float>("brightness"+dataToString(i),1.f) * splotchParams->find<float>("pv_brightness_mod"+dataToString(i),1.f));
			colour_is_vec.push_back(splotchParams->find<bool>("color_is_vector"+dataToString(i),0));
			smoothing_length.push_back(splotchParams->find<float>("size_fix"+dataToString(i),0) );
		}
		
		// Get radial mod
		radial_mod = splotchParams->find<double>("pv_radial_mod",1.f);



		// Copy data into our own structure
		OriginalRGBData.resize(particleList.size());

		for(unsigned i = 0; i < particleList.size(); i++)
		{
		
			/*particleList[i].r *= splotchParams->find<double>("pv_radial_mod",1.f);*/

			// Store orginal colour, to use in regeneration of particle colour with new colourmap
			OriginalRGBData[i].x = particleList[i].e.r;
			OriginalRGBData[i].y = particleList[i].e.g;
			OriginalRGBData[i].z = particleList[i].e.b;


			// Generate colour in same way splotch does (Add brightness here):
			if(!colour_is_vec[particleList[i].type])
				particleList[i].e = colourMaps[particleList[i].type].getVal_const(particleList[i].e.r) * particleList[i].I/* * brightness[particleList[i].type]*/;
			else
				particleList[i].e *= particleList[i].I /** brightness[particleList[i].type]*/;

		}

		// Compute and store bounding box of data
		BBox.Compute(particleList);

		DebugPrint("Particles loaded\nbbox data:\n");
		DebugPrint("minX: %f\n", BBox.minX);
		DebugPrint("maxX: %f\n", BBox.maxX);
		DebugPrint("miny: %f\n", BBox.minY);
		DebugPrint("maxY: %f\n", BBox.maxY);
		DebugPrint("minZ: %f\n", BBox.minZ);
		DebugPrint("maxZ: %f\n", BBox.maxZ);

		DebugPrint("Number of particles to be rendered: %i\n", (int)particleList.size());

	}	

	void ParticleData::ReloadColorData()
	{
		DebugPrint("Reloading color data\n");

		// Get new colourmaps
		int ret = FileLib::LoadColourPalette(Previewer::parameterInfo, colourMaps);
		if(!ret)
		{
			std::cout << "Color palette is invalid" << std::endl;
			return;
		}
DebugPrint("Reloading color data\n");
		//Reload colour data with new palette
		for(unsigned i = 0; i < particleList.size(); i++)
		{
			if(!colour_is_vec[particleList[i].type])
				// Use original RGB data as particleList's colour data will have been overwritten
				particleList[i].e = colourMaps[particleList[i].type].getVal_const(OriginalRGBData[i].x)  ;
		}
DebugPrint("Reloading color data\n");
		for(unsigned i = 0; i < numTypes; i++)
		{
			if(!colour_is_vec[i])
				DebugPrint("Loaded new colours for type: %i\n", i);
			else
				DebugPrint("Colour is vector, so no new palette to load for type: %i\n", i);
		}

	}

	void ParticleData::SetPalette(std::string paletteFilename, int particleType)
	{
		DebugPrint("Setting new palette\n");

		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();

		// Check type is valid
		if(particleType >= splotchParams->find<int>("ptypes"))
		{
			ErrorMessage("No particles of type %i\n", particleType);
		}

		// Check if particle type has vector colour
		if (splotchParams->find<bool>("color_is_vector"+dataToString(particleType),false))
		{
			ErrorMessage("Color of ptype %i is vector, so cannot set new colourMap\n", particleType);
		}
 
		// Return if there is no palette parameter available for specified particle type
		// E.G. particle type is invalid
		if(!splotchParams->param_present("palette"+dataToString(particleType)))
		{
			ErrorMessage("Cannot set new palette for particle type %i \n", particleType);
		}

		DebugPrint("Checked for present parameter");

		// Write new palette filename to parameter file
		splotchParams->setParam<std::string>("palette"+dataToString(particleType),paletteFilename);

		DebugPrint("Set new parameter\n");
	}

	std::string ParticleData::GetPalette(int particleType)
	{
		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();

		if(!splotchParams->param_present("palette"+dataToString(particleType)))
		{
			std::cout << "No palette in parameter file for particle type: " << particleType << std::endl;
			std::string ret = "NULL";
			return ret;
		}
		else
		{
			return splotchParams->find<std::string>("palette"+dataToString(particleType),"NULL");
		}

	}

	BoundingBox ParticleData::GetBoundingBox() const
	{
		return BBox;
	}

	ParticleList ParticleData::GetParticleList() const
	{
		return particleList;
	}

	std::vector<float> ParticleData::GetParameterBrightness() const
	{
		return brightness;
	}

	float ParticleData::GetRadialMod() const
	{
		return radial_mod;
	}

	std::vector<float> ParticleData::GetParameterSmoothingLength() const
	{
		return smoothing_length;
	}

}
