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

#include "WindowManager.h"

namespace previewer
{
	int WindowManager::GetSimulationWidth()
	{
		return simulationWidth;
	}
	int WindowManager::GetSimulationHeight()
	{
		return simulationHeight;
	}
	int WindowManager::GetSimulationMinX()
	{
		return simulationMinX;
	}
	int WindowManager::GetSimulationMinY()
	{
		return simulationMinY;
	}

	void WindowManager::SetSimulationWidth(int newWidth)
	{
		simulationWidth = newWidth;
	}
	void WindowManager::SetSimulationHeight(int newHeight)
	{
		simulationHeight = newHeight;
	}
	void WindowManager::UpdateFPS()
	{
		struct timeval t;
	  	gettimeofday(&t, NULL);

	  	previousFrameTime = currentTime;
	 	currentTime = (t.tv_sec + (t.tv_usec*0.000001));

	 	//Check for 0
	 	if(!previousFrameTime)
	 		previousFrameTime=currentTime;

	 	realtElapsed = currentTime - previousFrameTime;
	 	tElapsed = currentTime - lastRecordedTime;

	 	frameCtr++;
	 	// Check for 0
	  	if(!lastRecordedTime)
	  		lastRecordedTime = currentTime;
	 	// Update every half second	  	
	  	if(tElapsed >= 0.1)
	 	{
	 		spf = (currentTime - lastRecordedTime)/frameCtr;
	 		fps = 1/spf;
	 		lastRecordedTime = currentTime;
	 		frameCtr = 0;
	 	}

	 	//std::cout << "SPF: " << spf << std::endl;
	}

	double WindowManager::GetFPS()
	{
		return fps;
	}

	double WindowManager::GetSPF()
	{
		return spf;
	}

	double WindowManager::GetElapsedTime()
	{
		return realtElapsed;
	}

	// Static definitions - screen information
	int WindowManager::simulationWidth;
	int WindowManager::simulationHeight;
	int WindowManager::simulationMinX = 0;
	int WindowManager::simulationMinY = 0;


	// Static definitions - FPS variables
	double WindowManager::fps = 0;
	double WindowManager::currentTime = 0;
	double WindowManager::lastRecordedTime = 0;
	double WindowManager::tElapsed = 0;
	int WindowManager::frameCtr = 0;
	double WindowManager::spf = 0;

	double WindowManager::previousFrameTime = 0;
	double WindowManager::realtElapsed = 0;

	void WindowManager::SetSimulationSize(int x, int y)
	{
		simulationWidth = x;
		simulationHeight = y;
	}
	void WindowManager::SetSimulationPosition(int x, int y)
	{
		simulationMinX = x;
		simulationMinY = y;
	}
}