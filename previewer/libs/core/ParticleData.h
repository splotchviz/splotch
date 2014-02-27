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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLEDATA
#define SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLEDATA

//Debug include
#include "previewer/libs/core/Debug.h"

#include <vector>
#include <string>
#include "Particle.h"
#include "Parameter.h"
#include "BoundingBox.h"
#include "FileLib.h"

#include "splotch/scenemaker.h"
#include "splotch/splotchutils.h"

namespace previewer
{
	// Type definition for a list of particles
	typedef std::vector<particle_sim> ParticleList;

	// Stores and loads particle information from a specific particle
	// simulation
	class ParticleData
	{
	public:
		void Load();

		void ReloadColourData();

		void SetPalette(std::string paletteFilename, int particleType);
		std::string GetPalette(int particleType);

		BoundingBox GetBoundingBox() const;

		ParticleList GetParticleList() const;

		std::vector<float> GetParameterBrightness() const;
		float GetRadialMod() const;
		std::vector<float> GetParameterSmoothingLength() const;

	private:
		ParticleList particleList;
		BoundingBox  BBox;

		std::vector<COLOURMAP> colourMaps;

		ParticleList dummySplotchParticleData;
		std::vector<vec3> OriginalRGBData;

		unsigned numTypes;
		std::vector<float> brightness;
		std::vector<bool> colour_is_vec;
		std::vector<float> smoothing_length;
		float radial_mod;
	};
}

#endif