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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLESIMULATION
#define SPLOTCH_PREVIEWER_LIBS_CORE_PARTICLESIMULATION

// Debug include
#include "previewer/libs/core/Debug.h"

// Member includes
#include "../renderers/IRenderer.h"
#include "../core/ParticleData.h"

// Automatically generated header to include the current renderer
// Created by makefile on build
#include "../renderers/CurrentRenderer.h"

// Usage includes
#include "Parameter.h"
#include "previewer/libs/core/WindowManager.h"
#include "GL/gl.h"

// Event includes
#include "Event.h"
#include "../events/OnButtonPressEvent.h"
#include "../events/OnButtonReleaseEvent.h"

namespace previewer
{
	// Contains the main functionality for simulating a universe including the
	// renderer to show the previewer on screen and the data representing the
	// simulation itself
	class ParticleSimulation
	{
	public:
		void Load(std::string); // With data

		void Run();

		void Unload();

		static int GetRenderWidth();
		static int GetRenderHeight();

		static int GetRenderXMin();
		static int GetRenderYMin();
		static int GetRenderXMax();
		static int GetRenderYMax();

		static int GetSimWindowWidth();
		static int GetSimWindowHeight();
		static float GetAspectRatio();

		static int GetFOV();
		static void SetFOV(int);

		static void SetMoveSpeed(int);
		static int GetMoveSpeed();
		static void SetRotationSpeed(int);
		static int GetRotationSpeed();

		void ReloadColourData();

		void SetPalette(std::string paletteFilename, int particleType);
		std::string GetPalette(int particleType);

		static void SetXRes(int,bool);
		static void SetYRes(int,bool);
		static int GetXRes();
		static int GetYRes();

		Camera& GetCameraReference();

		// View tga image
		void ViewImage(std::string);
		void StopViewingImage();

		// Modify previewing brightness
		void SetRenderBrightness(int, float);
		float GetRenderBrightness(int);

		void SetSmoothingLength(int, float);
		float GetSmoothingLength(int);

		void ResetCamera();
		static std::string GetExePath();
		
	private:

		static void Update(bool);
		static void UpdateResolution(bool);

		static IRenderer* renderer;
		static ParticleData particles;

		// Handling drawing correct size/shape image in window 
		static int pSimWindowWidth;
		static int pSimWindowHeight;
		static int renderWidth;
		static int renderHeight;
		static int renderXMin;
		static int renderXMax;
		static int renderYMin;
		static int renderYMax;
		static float aspectRatio;
		static int fieldOfView;

		// Is it the first run (do not update renderer screen if so)
		static bool firstRun;

		// Actual resolution for parameter file
		static int xres;
		static int yres;
		
		// Used to force renderer update
		static bool rendererUpdated;

		// Interaction with simulation render
		static int moveSpeed;
		static int rotationSpeed;

		static std::string exepath;

		// For image viewing mode
		bool viewingImage;
	};
}

#endif