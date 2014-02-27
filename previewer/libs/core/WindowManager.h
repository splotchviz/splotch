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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_WINDOWMANAGER
#define SPLOTCH_PREVIEWER_LIBS_CORE_WINDOWMANAGER

//Debug include
#include "previewer/libs/core/Debug.h"

// General includes
#include <sys/time.h>

namespace previewer
{
	// Class for the core window functionality which can be inherited to
	// gain access to the window iteself
	class WindowManager
	{
	public:
		static int GetSimulationWidth();
		static int GetSimulationHeight();
		static int GetSimulationMinX();
		static int GetSimulationMinY();
		static void SetSimulationSize(int x, int y);
		static void SetSimulationPosition(int x, int y);
		static void SetSimulationWidth(int);
		static void SetSimulationHeight(int);

		//FPS functions
		static void UpdateFPS();
		static double GetFPS();
		static double GetSPF();
		static double GetElapsedTime();

	private:
		// Screen information
		static int simulationWidth;
		static int simulationHeight;
		static int simulationMinX;
		static int simulationMinY;

		// FPS variables
		static double fps; //Frames per second
		static double currentTime;
		static double lastRecordedTime;
		static double tElapsed;
		static double frames;
		static double spf; //Seconds per frame
		static int frameCtr;
		static double previousFrameTime;
		static double realtElapsed;

	};
}

#endif