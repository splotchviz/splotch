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

#ifndef SPLOTCH_PREVIEWER_LIBS_CORE_CAMERA
#define SPLOTCH_PREVIEWER_LIBS_CORE_CAMERA

//Debug include
#include "previewer/libs/core/Debug.h"

// Member includes
#include "../../../cxxsupport/vec3.h"
#include "MathLib.h"

// Usage includes
#include "BoundingBox.h"

// Event includes
#include "Event.h"
#include "../events/OnButtonPressEvent.h"
#include "../events/OnButtonReleaseEvent.h"
#include "../events/OnKeyPressEvent.h"
#include "../events/OnKeyReleaseEvent.h"

// Action includes
#include "../events/actions/CameraAction.h"

namespace previewer
{
	// Provides an enumeration for the box faces
	enum
	{
		FRONT,
		BACK,
		LEFT,
		RIGHT,
		TOP,
		BOTTOM
	};

	// Provides the functionality within the renderers for a movable
	// and configurable camera through the use of event subscriptions.
	class Camera : public CameraAction
	{
	public:
		// Getters
		vec3f GetCameraPosition();
		vec3f GetLookAt();
		vec3f GetUpVector();

		// Setters
		void SetLookAt(vec3f);
		void SetFieldOfView(float);

		// Creators
		void Create(BoundingBox);
		void Create(vec3f, vec3f);
		void Create(vec3f, vec3f, vec3f);

		// Makes the camera look at a bounding box
		void LookAtBox(BoundingBox);
		void LookAtBox(BoundingBox, int);

		// Rotation functionality - all do the same thing or as they say
		void Rotate(vec3f);
		void Rotate(float, float, float);
		void RotateYaw(float);
		void RotatePitch(float);
		void RotateRoll(float);
		void RotateAroundTarget(vec3f, float, float, float);

		// Move functions relative to world axes 
		void Move(vec3f);
		void Move(float, float, float);

		// Move functions relative to local axes
		void MoveForward(float);
		void MoveUpward(float);
		void MoveRight(float);

		// Projections
		void SetPerspectiveProjection(float, float, float, float);
		void SetOrthographicProjection(float, float, float, float, float, float);

		// Give camera focus
		void SetFocusState(bool);

		//Matrix access
		Matrix4 GetMVPMatrix();
		Matrix4 GetViewMatrix();
		Matrix4 GetProjectionMatrix();

		// Actions
		void ActionMoveCameraPosition(vec3f, int);
		void ActionMoveCameraLookAt(vec3f);
		//void ActionMoveCameraUpVector(vec3f);
		void ActionGetCameraPos(vec3f&);
		void ActionGetCameraLookAt(vec3f&);
		void ActionGetCameraUp(vec3f&);
		void SetMainCameraStatus(bool);

	private:
		// Camera positions and targets including orientation
		vec3f cameraPosition;
		vec3f cameraLookAt; // Relative to cameraPosition (normalised directional vector)
		vec3f cameraUpVector;
		vec3f cameraRightVector;

		// Camera projection settings
		float cameraFieldOfView;
		float cameraAspectRatio;
		float cameraNearClip;
		float cameraFarClip;

		// Cameras view and projection matrices
		Matrix4 cameraViewMatrix;
		Matrix4 cameraProjectionMatrix;

		// Internal functions for creating view matrix
		void ConstructViewMatrix(bool);

		bool hasFocus;

	};

}

#endif