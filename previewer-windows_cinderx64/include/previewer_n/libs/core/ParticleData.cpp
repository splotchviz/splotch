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
#include "../../application.h"

namespace previewer
{

	void ParticleData::Load()
	{
		
		paramfile* splotchParams = Previewer::parameterInfo.GetParamFileReference();
		sceneMaker sMaker(*splotchParams);

		vec3 dummyCam = vec3(0, 0, 0);
		vec3 dummyAt = vec3(0, 0, 0);
		vec3 dummySky = vec3(0, 0, 0);
		vec3 dummyCent = vec3(0, 0, 0);
		std::string dummyOutfile = " ";

		// Splotch param find
		bool boost = splotchParams->find<bool>("boost", false);

		// Previewer param find
		//bool boost = param.GetParameter<bool>("boost");

		if (boost)
		{
			//if boost particledata is second argument(boosted data)
			bool ret = sMaker.getNextScene(dummySplotchParticleData, particleList, dummyCam, dummyCent, dummyAt, dummySky, dummyOutfile);
			if (!ret) std::cout << "sMaker.getNextScene failed." << std::endl;
		}
		else
		{
			//else use full data set (first arg)
			bool ret = sMaker.getNextScene(particleList, dummySplotchParticleData, dummyCam, dummyCent, dummyAt, dummySky, dummyOutfile);
			if (!ret) std::cout << "sMaker.getNextScene failed." << std::endl;
		}

		// Get colourmap
		int ret = FileLib::LoadColourPalette(Previewer::parameterInfo, colourMaps);
		if (!ret)
			std::cout << "Colour palette in param file is invalid" << std::endl;

		// Get per ptype parameters
		numTypes = splotchParams->find<int>("ptypes", 1);

		for (unsigned i = 0; i < numTypes; i++)
		{
			//brightness.push_back(splotchParams->find<float>("brightness" + dataToString(i), 1.f) * splotchParams->find<float>("pv_brightness_mod" + dataToString(i), 1.f));
			colour_is_vec.push_back(splotchParams->find<bool>("color_is_vector" + dataToString(i), 0));
			//smoothing_length.push_back(splotchParams->find<float>("size_fix" + dataToString(i), 0));
		}

		// Get radial mod
		radial_mod = splotchParams->find<double>("pv_radial_mod", 1.f);



		// Copy data into our own structure
		OriginalRGBData.resize(particleList.size());

		for (unsigned i = 0; i < particleList.size(); i++)
		{

			/*particleList[i].r *= splotchParams->find<double>("pv_radial_mod",1.f);*/
			/*
			// Store orginal colour, to use in regeneration of particle colour with new colourmap
			OriginalRGBData[i].x = particleList[i].e.r;
			OriginalRGBData[i].y = particleList[i].e.g;
			OriginalRGBData[i].z = particleList[i].e.b;
			*/

			// Generate colour in same way splotch does (Add brightness here):
			if (!colour_is_vec[particleList[i].type])
				particleList[i].e = colourMaps[particleList[i].type].getVal_const(particleList[i].e.r) * particleList[i].I/* * brightness[particleList[i].type]*/;
			else
				particleList[i].e *= particleList[i].I /** brightness[particleList[i].type]*/;

		}
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