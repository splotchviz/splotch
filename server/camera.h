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


#include "../cxxsupport/vec3.h"
#include "matrix4.h"

// // Event includes
// #include "Event.h"
// #include "../events/OnButtonPressEvent.h"
// #include "../events/OnButtonReleaseEvent.h"
// #include "../events/OnKeyPressEvent.h"
// #include "../events/OnKeyReleaseEvent.h"

// Provides the functionality within the renderers for a movable
// and configurable camera through the use of event subscriptions.
class Camera 
{
public:
	// Getters
	vec3f GetCameraPosition();
	vec3f GetLookAt();
	vec3f GetUpVector();
	vec3f GetTarget();

	// Setters
	void SetLookAt(vec3f);
	void SetFieldOfView(float);
	void SetMaxAcceleration(vec3f);
	void SetMaxSpeed(vec3f);

	// Creators
	void Create(vec3f, vec3f);
	void Create(vec3f, vec3f, vec3f);

	// Makes the camera look at a bounding box
	void SetTarget(vec3f);

	// Rotation functionality - all do the same thing or as they say
	void Rotate(vec3f);
	void Rotate(float, float, float);
	void RotateYaw(float);
	void RotatePitch(float);
	void RotateRoll(float);
	void RotateAroundTarget(float, float, float);
	void RotateTargetAround(float, float, float);
	// Move functions relative to world axes 
	void Move(vec3f);
	void Move(float, float, float);

	// Move functions relative to local axes
	// Camera only
	void MoveForward(float);
	void MoveUpward(float);
	void MoveRight(float);
	// Camera and target
	void MoveCamAndTargetForward(float);
	void MoveCamAndTargetUpward(float);
	void MoveCamAndTargetRight(float);
	// Target only
	void MoveTargetForward(float);
	void MoveTargetUpward(float);
	void MoveTargetRight(float);

	// Physical movement
	void AccelForward(float amount);
	void AccelUpward(float amount);
	void AccelRight(float amount);
	void DecelForward(float amount);
	void DecelUpward(float amount);
	void DecelRight(float amount);
	void UpdateSpeed();
	void UpdateVelocity();
	void UpdatePosition();


	// Projections
	void SetPerspectiveProjection(float, float, float, float);
	void SetOrthographicProjection(float, float, float, float, float, float);

	// Give camera focus
	void SetFocusState(bool);

	//Matrix access
	Matrix4 GetMVPMatrix();
	Matrix4 GetViewMatrix();
	Matrix4 GetProjectionMatrix();

private:
	// Camera positions and targets including orientation
	vec3f cameraPosition;
	vec3f cameraLookAt; // Relative to cameraPosition (normalised directional vector)
	vec3f cameraTarget; // Setable target point to lookat/rotate around
	vec3f cameraUpVector;
	vec3f cameraRightVector;

	// Proper movement for smooth camera transitions
	// Acceleration and speed are treated seperately for x y z, 
	// corresponding to right, up, forward respectively
	vec3f acceleration;
	vec3f maxAcceleration;
	vec3f velocity;
	vec3f speed;
	vec3f maxSpeed;

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


#endif