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

#include "previewer/libs/animation/AnimationPath.h"

#include "previewer/libs/animation/AnimationData.h"
#include "previewer/libs/animation/AnimationSimulation.h"

namespace previewer
{
	AnimationPath::AnimationPath(std::string _comp, std::string _param) : component(_comp), parameter(_param)
	{
	}

	void AnimationPath::AddPoint(const DataPoint& dp)
	{
		DPList.push_back(dp);
	}

	void AnimationPath::RemovePoint()
	{
		if(!DPList.empty())
			DPList.pop_back();
	}

	void AnimationPath::AddInterpolatedPoint(const DataPoint& dp)
	{
		InterpolatedDPList.push_back(dp);
	}

	void AnimationPath::InterpolatePoints()
	{
		DataPointList DPL;

		DebugPrint("Created new DPL");

		// Check what type of interpolation we plan to do
		// Interpolating other types is not yet supported...
		switch(AnimationTypeLookUp::Query(component, parameter))
		{
			case FLOAT:
				DebugPrint("Calling interpolate");
				InterpolateF(DPL);
				break;

			case DOUBLE:
				// InterpolateD();
				break;

			case VEC3FLOAT:
				// InterpolateV3F();
				break;

			case VEC3DOUBLE:
				// InterpolateV3D();
				break;

			default:

				break;
		}

		DebugPrint("Storing");
		InterpolatedDPList = DPL;
	}

	int AnimationPath::Size()
	{
		return DPList.size();
	}

	void AnimationPath::SetComponent(const std::string& _comp)
	{
		component = _comp;
	}

	std::string AnimationPath::GetComponent()
	{
		return component;
	}

	void AnimationPath::SetParameter(const std::string& _param)
	{
		parameter = _param;
	}

	std::string AnimationPath::GetParameter()
	{
		return parameter;
	}

	void AnimationPath::SetTension(const float& _tension)
	{
		tension = _tension;
	}

	float AnimationPath::GetTension()
	{
		return tension;
	}

	void AnimationPath::SetBias(const float& _bias)
	{
		bias = _bias;
	}

	float AnimationPath::GetBias()
	{
		return bias;
	}

	void AnimationPath::SetContinuity(const float& _cont)
	{
		continuity = _cont;
	}	

	float AnimationPath::GetContinuity()
	{
		return continuity;
	}

	DataPointList AnimationPath::GetDataPointList()
	{
		return DPList;
	}

	DataPointList AnimationPath::GetInterpolatedDataPointList()
	{
		return InterpolatedDPList;
	}


	void AnimationPath::InterpolateF(DataPointList& NewDPL)
	{
		DebugPrint("Max animation time:  ", AnimationData::GetMaxAnimationTime());

		DebugPrint("Converting GDPL to FDPL");
		// Convert generic data point list to float data point list
		DataPointListF DPLF;
		DPLF.resize(DPList.size());
		for(unsigned i = 0; i < DPLF.size(); i++)
		{
			DPLF[i].interpType = DPList[i].interpType;
			DPLF[i].time = DPList[i].time;
			DPLF[i].value = std::atof(DPList[i].value.c_str());
		}

		DebugPrint("Adding start point if necessary");
		// If path starts partway through the animation, repeat first element at beginning of animation
		if(DPLF.front().time > 0)
		{
			std::vector<DataPointF>::iterator it = DPLF.begin();
			DataPointF dp = DPLF.front();
			dp.time = 0;
			DPLF.insert(it, dp);
			DebugPrint("It was necessary");
		}

		DebugPrint("Adding final point if necessary");
		// If path ends before end of animation, repeat final point at end time
		if(DPLF.back().time < AnimationData::GetMaxAnimationTime())
		{
			DataPointF dp = DPLF.back();
			dp.time = AnimationData::GetMaxAnimationTime();
			DPLF.push_back(dp);
			DebugPrint("It was necessary");
		}

		DebugPrint("Getting total frames");
		// Get total number of frames for animation 
		totalFrames = AnimationData::GetMaxAnimationTime() * AnimationSimulation::GetFPS();

		DebugPrint("Total Frames = ", totalFrames);

		//DebugPrint("DPLF size: ", DPLF.size());

		// Do interpolation for point(n) and point (n+1), for all points up to size() - 1
		for(int i = 0; i < (int)(DPLF.size() - 1); i++)
		{
			DebugPrint("Working out segment length");
			// Get length of segment (time between point[n] and point[n+1]) as percentage of total time
			float segmentLength = abs(DPLF[i].time - DPLF[i+1].time);

			float percentageOfTotal = 0;
			if(AnimationData::GetMaxAnimationTime() > 0 && segmentLength > 0)
				percentageOfTotal = segmentLength / AnimationData::GetMaxAnimationTime();

			DebugPrint("Working out fpseg");
			// Work out number of frames allocated to segment
			float framesPerSegment = 1;
			if(percentageOfTotal > 0)
				framesPerSegment = totalFrames * percentageOfTotal;

			DebugPrint("Fpseg = ", framesPerSegment);

			DebugPrint("Working out increment");
			// Compute increment value for mu (interpolation interval) 
			// Ceiling fpseg for evenly spaced frames per segment at the expensive of having exact number of frames requested
			float increment = 1/ceil(framesPerSegment);

			DebugPrint("Increment = ", increment);

			DebugPrint("Interpolating!");
			// Interpolate
			switch(DPLF[i].interpType)
			{
				case LINEAR:
					DebugPrint("Interpolation type - LINEAR");
					LinearInterpolatePath(increment, i, DPLF, NewDPL);
					break;

				case CUBIC:
					DebugPrint("Interpolation type - CUBIC");
					CubicInterpolatePath(increment, i, DPLF, NewDPL);
					break;

				case CUBIC_HERMITE_KB:
					DebugPrint("Interpolation type - CUBIC HERMITE");
					CHKBInterpolatePath(increment, i, DPLF, NewDPL);
					break;

				default:
					std::cout << "invalid interpolation type given" << std::endl;
					break;
			}
			DebugPrint("Finished Interpolating! I = ", i);
		}
		DebugPrint("DPLF size: ", (float)(DPLF.size()));
	}

	void AnimationPath::LinearInterpolatePath(float& increment, int& iter, DataPointListF& DPLF, DataPointList& NewDPL)
	{
		float mu = 0;

		DebugPrint("Increment = ", increment);

		for(; mu < 1; mu += increment)
		{
			DebugPrint("Computing interpolation for mu = ", mu);
			float newValue = LinearInterpolate(mu, DPLF[iter].value, DPLF[iter+1].value);
			DebugPrint("Value is: ", newValue);


			DebugPrint("Creating new data point");
			DataPoint dp;
			dp.value = Utils::ToString(newValue);
			dp.interpType = DPLF[iter].interpType;
			dp.time = DPLF[iter].time;

			DebugPrint("Pushing to new dpl");
			NewDPL.push_back(dp);		
		}
	}

	void AnimationPath::CubicInterpolatePath(float& increment, int& iter, DataPointListF& DPLF, DataPointList& NewDPL)
	{
		float mu = 0;

		for(; mu < 1; mu += increment)
		{
			float newValue;

			//repeat element 0 as element -1 for first element
			if(iter == 0)
				newValue = CubicInterpolate(mu, DPLF[iter].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+2].value);
			//repeat element n+1 as element n+2 for second to last element
			else if(iter == (int)(DPLF.size()- 2))
				newValue = CubicInterpolate(mu, DPLF[iter-1].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+1].value);
			//standard interpolation
			else
				newValue = CubicInterpolate(mu, DPLF[iter-1].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+2].value);
			
			DataPoint dp;
			dp.value = Utils::ToString(newValue);
			dp.interpType = DPLF[iter].interpType;
			dp.time = DPLF[iter].time;
			
			NewDPL.push_back(dp);
		}	
	}

	void AnimationPath::CHKBInterpolatePath(float& increment, int& iter, DataPointListF& DPLF, DataPointList& NewDPL)
	{
		float mu = 0;

		for(; mu < 1; mu += increment)
		{
			float newValue;

			//repeat element 0 as element -1 for first element
			if(iter == 0)
				newValue = CHKBInterpolate(mu, DPLF[iter].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+2].value, tension, bias, continuity);
			//repeat element n+1 as element n+2 for second to last element
			else if(iter == (int)(DPLF.size()- 2))
				newValue = CHKBInterpolate(mu, DPLF[iter-1].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+1].value, tension, bias, continuity);
			//standard interpolation
			else
				newValue = CHKBInterpolate(mu, DPLF[iter-1].value, DPLF[iter].value, DPLF[iter+1].value, DPLF[iter+2].value, tension, bias, continuity);
			
			DataPoint dp;
			dp.value = Utils::ToString(newValue);
			dp.interpType = DPLF[iter].interpType;
			dp.time = DPLF[iter].time;
			
			NewDPL.push_back(dp);
		}	
	}
}
