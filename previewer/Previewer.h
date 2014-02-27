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

#ifndef SPLOTCH_PREVIEWER_PREVIEWER
#define SPLOTCH_PREVIEWER_PREVIEWER

// Event includes
#include "libs/events/OnButtonPressEvent.h"
#include "libs/events/OnButtonReleaseEvent.h"
#include "libs/events/OnExposedEvent.h"
#include "libs/events/OnKeyPressEvent.h"
#include "libs/events/OnKeyReleaseEvent.h"
#include "libs/events/OnMotionEvent.h"
#include "libs/events/OnQuitApplicationEvent.h"
#include "libs/core/Event.h"

// Member includes
#include "libs/core/Parameter.h"
#include "libs/core/ParticleSimulation.h"
#include "libs/animation/AnimationSimulation.h"
#include "cxxsupport/vec3.h"

// Usage includes
#include "libs/core/WindowManager.h"
#include "libs/core/Camera.h"
#include "cxxsupport/paramfile.h"

#include <string>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>

#ifdef SPLOTCHMAC
#include <libproc.h>
#endif
 
namespace previewer
{
	// Provides the main entry point to the application from splotch. Also acts as
	// the main running code that is controlling the flow of the application
	class Previewer : public OnQuitApplicationEvent
	{
	public:
		void Load(std::string);

		void Run();

		void Unload();

		void DetectSystemCapabilities();

		void OnQuitApplication(Event);

		void TriggerOnButtonPressEvent(std::string buttonPressed, float xPosition, float yPosition);
		void TriggerOnButtonReleaseEvent(std::string buttonReleased, float xPosition, float yPosition);
		void TriggerOnExposedEvent();
		void TriggerOnKeyPressEvent(std::string keyPressed);
		void TriggerOnKeyReleaseEvent(std::string keyReleased);
		void TriggerOnMotionEvent(float xPosition, float yPosition);
		void TriggerOnQuitApplicationEvent();

		void SetRenderWidth(int width);
		void SetRenderHeight(int height);
		void SetRenderSize(int width, int height);
		void SetRenderPositionX(int x);
		void SetRenderPositionY(int y);
		void SetRenderPosition(int x, int y);

		double GetFPS();
		void SetAnimationPoint(int time);
		void RemoveAnimationPoint();
		void SetMovementSpeed(int speed);
		void SetRotationSpeed(int speed);
		void PreviewAnimation();
		void SaveAnimationFile(std::string filename);
		void LoadAnimationFile(std::string filename);
		void UnloadAnimationFile();
		void SetXRes(int xRes);
		void SetYRes(int yRes);
		void SetFOV(int fov);
		void SetCameraInterpolation(int);
		void SetLookatInterpolation(int);


		void WriteParameterFile(std::string outpath);
		void WriteSplotchAnimationFile(std::string outpath);
		
		void ReloadColourData();
		void SetPalette(std::string paletteFilename, int particleType);
		std::string GetPalette(int particleType);
		void Interpolate();

		// Viewing splotch images
		void ViewImage(std::string);
		void StopViewingImage();

		// Modify previewing brightness
		void SetRenderBrightness(int, float);
		float GetRenderBrightness(int);

		void SetSmoothingLength(int, float);
		float GetSmoothingLength(int);

		void SetParameter(std::string, std::string);
		// Note this getParameter function is only useful for displaying parameters. 
		// They are not converted to their original format, they are kept as string
		std::string GetParameter(std::string);

		void ResetCamera();

		static Parameter parameterInfo;

	private:
		ParticleSimulation particleSim;
		AnimationSimulation animationSim;

		bool isApplicationTerminating;
		vec3_t<int> screenSize;
		vec3_t<int> screenPosition;
	};
}

#endif