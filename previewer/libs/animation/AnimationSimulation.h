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

#ifndef SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONSIMULATION
#define SPLOTCH_PREVIEWER_LIBS_ANIMATION_ANIMATIONSIMULATION

//Debug include
#include "previewer/libs/core/Debug.h"

#include "previewer/libs/core/ParticleSimulation.h"
#include "previewer/libs/animation/AnimationData.h"
#include "previewer/libs/events/actions/CameraAction.h"
#include "previewer/libs/events/OnKeyPressEvent.h"
#include "previewer/libs/core/Utils.h"

#include <math.h>
#include <string>
#include <map>

namespace previewer
{
	// Provides the animation specific sections to the application. Requires a
	// particle simulation to be supplied, otherwise no particle data is rendered
	// only the path
	class AnimationSimulation
	{
	public:
		// Load function
		void Load(std::string);
		void Run();
		void Unload();

		// Event response
		//void OnKeyPress(Event);
		
		// Set animation point
		void SetAnimationPoint(std::string, std::string, std::string, int);

		// Write animation file
		void WriteAnimFile();

		// Load animation file
		void LoadAnimFile();

		// Write animation list to run splotch with
		void WriteSplotchAnimFile(std::string);

		// Set/Get filename for current animation
		void SetFilename(std::string);
		std::string GetFilename();

		// Set animation fps
		static void SetFPS(float);
		static float GetFPS();

		static std::string GetExePath();

		// Set current time for animation point
		void SetTime(int);
		float GetTime();

		// Set animation point at the current time
		void SetPoint();

		// Remove last set point
		void RemovePoint();

		// Interpolate created points
		void Interpolate();

		// Preview the current animation
		void Preview();

		// Allow you to create new animation having already loaded/created one
		void UnloadCurrentAnim();

		// Set camera and lookat interpolation types
		void SetCameraInterpolation(int);
		void SetLookatInterpolation(int);

	private:
		// Particle simulation and renderer
		ParticleSimulation particleSim;

		// Animation data (loaded from file?)
		AnimationData animation;

		// Path list to store interpolated data
		AnimationPathList animationPaths;

		// Animation file handling
		// File info
		std::string currentFile;

		// Animation information
		int currentFrame;
		int totalFrames;
		static int timeValue;
		static float fps;
		static std::string exepath;
		bool isActive;
		bool demoPlaying;
		bool fileReady;

		void PlayDemo();

		int cameraInterpType;
		int lookatInterpType;
	};
}

#endif
