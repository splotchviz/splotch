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

#ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONPATH
#define SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONPATH

#include <string>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cmath>

#include "previewer/libs/animation/DataPoint.h"
#include "previewer/libs//animation/interpolations/InterpolationMath.h"
#include "previewer/libs/animation/AnimationTypeLookUp.h"
#include "previewer/libs/core/Utils.h"

namespace previewer
{
		typedef std::vector<DataPoint> DataPointList;
		typedef std::vector<DataPointF> DataPointListF;

	class AnimationPath
	{
	public:
		AnimationPath(std::string, std::string);

		void AddPoint(const DataPoint&);
		void RemovePoint();
		void AddInterpolatedPoint(const DataPoint& dp);
		void InterpolatePoints();

		void SetComponent(const std::string&);
		std::string GetComponent();

		void SetParameter(const std::string&);
		std::string GetParameter();

		void SetTension(const float&);
		float GetTension();

		void SetBias(const float&);
		float GetBias();

		void SetContinuity(const float&);
		float GetContinuity();

		DataPointList GetDataPointList();
		DataPointList GetInterpolatedDataPointList();

		int Size();

	private:
		DataPointList DPList;
		DataPointList InterpolatedDPList;

		std::string component;
		std::string parameter;

		int totalFrames;

		void InterpolateF(DataPointList&);
		void LinearInterpolatePath(float&, int&, DataPointListF&, DataPointList&);
		void CubicInterpolatePath(float&, int&, DataPointListF&, DataPointList&);
		void CHKBInterpolatePath(float&, int&, DataPointListF&, DataPointList&);


		float tension;
		float bias;
		float continuity;

	};

}

#endif
