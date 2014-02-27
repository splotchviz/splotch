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

#include "Camera.h"

namespace previewer
{
	vec3f Camera::GetCameraPosition()
	{
		return cameraPosition;
	}
	vec3f Camera::GetLookAt()
	{
		return cameraLookAt;
	}
	vec3f Camera::GetUpVector()
	{
		return cameraUpVector;
	}

	void Camera::SetLookAt(vec3f target)
	{
		// Calculate relative direction from the current position
		cameraLookAt = cameraPosition - target;

		// Normalize the right and up vectors
		cameraRightVector = crossprod(cameraUpVector, cameraLookAt);
		cameraRightVector.Normalize();

		cameraUpVector = crossprod(cameraLookAt, cameraRightVector);
		cameraUpVector.Normalize();

		// Construct a view matrix (special case)
		ConstructViewMatrix(false);
		return;
	}

	void Camera::SetFieldOfView(float fov)
	{
		cameraFieldOfView = fov;
		return;
	}

	void Camera::Create(BoundingBox box)
	{
		// Look at the bounding box
		LookAtBox(box, FRONT);
		return;
	}

	void Camera::Create(vec3f position, vec3f lookAt)
	{
		// Call the more useful overloaded function
		Create(position, lookAt, vec3f(0, 0, 1));
		return;
	}

	void Camera::Create(vec3f position, vec3f lookAt, vec3f up)
	{
		// Set cameras position, up vector and fov (no modifications needed here)
		cameraPosition = position;
		cameraUpVector = up;

		// Set the look at point for the camera (the cameras right vector is calculated here!)
		SetLookAt(lookAt);
		return;
	}

	void Camera::LookAtBox(BoundingBox box)
	{
		// Call the more useful overloaded function
		LookAtBox(box, FRONT);
		return;
	}

	void Camera::LookAtBox(BoundingBox box, int face)
	{
		// Find the correct camera position to look at the box given the prescribed
		// field of view
		switch(face)
		{
			case FRONT:
			{
				float distanceWidth = (((box.maxX - box.minX)/2)) / (tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxY;
				float distanceHeight = (((box.maxZ - box.minZ)/2)) / (tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxY;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = (box.maxX + box.minX)/2;
				cameraPosition.y = distanceFromBox;
				cameraPosition.z = (box.maxZ + box.minZ)/2;

				cameraUpVector = vec3f(0, 0, 1);
				break;
			}

			case LEFT:
			{
				float distanceWidth = (((box.maxY - box.minY)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxX;
				float distanceHeight = (((box.maxZ - box.minZ)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxX;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = distanceFromBox;
				cameraPosition.y = (box.maxY + box.minY)/2;
				cameraPosition.z = (box.maxZ + box.minZ)/2;

				cameraUpVector = vec3f(0, 0, 1);
				break;
			}

			case RIGHT:
			{
				float distanceWidth = (((box.maxY - box.minY)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxX;
				float distanceHeight = (((box.maxZ - box.minZ)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxX;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = -distanceFromBox;
				cameraPosition.y = (box.maxY + box.minY)/2;
				cameraPosition.z = (box.maxZ + box.minZ)/2;

				cameraUpVector = vec3f(0, 0, 1);
				break;
			}

			case BACK:
			{
				float distanceWidth = (((box.maxX - box.minX)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxY;
				float distanceHeight = (((box.maxZ - box.minZ)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxY;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = (box.maxX + box.minX)/2;
				cameraPosition.y = -distanceFromBox;
				cameraPosition.z = (box.maxZ + box.minZ)/2;

				cameraUpVector = vec3f(0, 0, 1);
				break;
			}

			case TOP:
			{
				float distanceWidth = (((box.maxX - box.minX)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxZ;
				float distanceHeight = (((box.maxY - box.minY)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxZ;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = (box.maxX + box.minX)/2;
				cameraPosition.y = (box.maxY + box.minY)/2;
				cameraPosition.z = distanceFromBox;

				cameraUpVector = vec3f(0, 1, 0);
				break;
			}

			case BOTTOM:
			{
				float distanceWidth = (((box.maxX - box.minX)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxZ;
				float distanceHeight = (((box.maxY - box.minY)/2) / tan(Math::degreesToRadians(cameraFieldOfView/2))) + box.maxZ;

				float distanceFromBox = (distanceWidth > distanceHeight) ? distanceWidth : distanceHeight;

				cameraPosition.x = (box.maxX + box.minX)/2;
				cameraPosition.y = (box.maxY + box.minY)/2;
				cameraPosition.z = -distanceFromBox;

				cameraUpVector = vec3f(0, 1, 0);
				break;
			}
		}

		// Set the look at point for the camera (the cameras right vector is calculated here)
		SetLookAt(box.centerPoint);
		return;
	}

	void Camera::Rotate(vec3f rotation)
	{
		// Call overloaded member
		Rotate(rotation.x, rotation.y, rotation.z);
		return;
	}
	void Camera::Rotate(float yaw, float pitch, float roll)
	{
		// Create a rotation matrix
		Matrix4 rotationMatrix;

		if(yaw != 0.0f)
		{
			// Create the yaw component in the roation matrix
			rotationMatrix.rotate(cameraUpVector, yaw);

			// Transform the right and look at vector values
			cameraRightVector = cameraRightVector * rotationMatrix;
			cameraLookAt = cameraLookAt * rotationMatrix;
		}

		if(pitch != 0.0f)
		{
			// Create the pitch component in the rotation matrix
			rotationMatrix.rotate(cameraRightVector, pitch);

			// Transform the up and look at vector values
			cameraUpVector = cameraUpVector * rotationMatrix;
			cameraLookAt = cameraLookAt * rotationMatrix;
		}

		if(roll != 0.0f)
		{
			// Create the roll component in the rotation matrix
			rotationMatrix.rotate(cameraLookAt, roll);

			// Transform the right and up vector values
			cameraUpVector = cameraUpVector * rotationMatrix;
			cameraRightVector = cameraRightVector * rotationMatrix;
		}

		return;
	}
	void Camera::RotateYaw(float yaw)
	{
		// Create a rotation matrix
		Matrix4 rotationMatrix;

		if(yaw != 0.0f)
		{
			// Create the yaw component in the roation matrix
			rotationMatrix.rotate(cameraUpVector, yaw);

			// Transform the right and look at vector values
			cameraRightVector = cameraRightVector * rotationMatrix;
			cameraLookAt = cameraLookAt * rotationMatrix;
		}

		return;
	}
	void Camera::RotatePitch(float pitch)
	{
		// Create a rotation matrix
		Matrix4 rotationMatrix;

		if(pitch != 0.0f)
		{
			// Create the pitch component in the rotation matrix
			rotationMatrix.rotate(cameraRightVector, pitch);

			// Transform the up and look at vector values
			cameraUpVector = cameraUpVector * rotationMatrix;
			cameraLookAt = cameraLookAt * rotationMatrix;
		}

		return;
	}
	void Camera::RotateRoll(float roll)
	{
		// Create a rotation matrix
		Matrix4 rotationMatrix;

		if(roll != 0.0f)
		{
			// Create the roll component in the rotation matrix
			rotationMatrix.rotate(cameraLookAt, roll);

			// Transform the right and up vector values
			cameraUpVector = cameraUpVector * rotationMatrix;
			cameraRightVector = cameraRightVector * rotationMatrix;
		}

		return;
	}

	void Camera::RotateAroundTarget(vec3f target, float yaw, float pitch, float roll)
	{
		Matrix4 rotationMtxH;
		Matrix4 rotationMtxP;

		vec3f focusVector = cameraPosition - target;

		if(yaw != 0.0f)
		{
			rotationMtxH.rotate(cameraUpVector, yaw);
			focusVector = focusVector * rotationMtxH;
		}

		if(pitch != 0.0f)
		{
			rotationMtxP.rotate(cameraRightVector, pitch);
			focusVector = focusVector * rotationMtxP;
		}

		if(roll != 0.0f)
		{
			// Ignore roll for now
		}

		cameraPosition = focusVector + target;

		SetLookAt(target);
	}

	void Camera::Move(vec3f movement)
	{
		// Move the camera relative to world axes
		cameraPosition += movement;
		return;
	}
	void Camera::Move(float x, float y, float z)
	{
		// Move the camera relative to world axes
		cameraPosition.x += x;
		cameraPosition.y += y;
		cameraPosition.z += z;
		return;
	}
	void Camera::MoveForward(float distance)
	{
		// Transform movement relative to local axes 
		cameraPosition -= cameraLookAt * distance;
		return;
	}
	void Camera::MoveUpward(float distance)
	{
		// Transform movement relative to local axes 
		cameraPosition += cameraUpVector * distance;
		return;
	}
	void Camera::MoveRight(float distance)
	{
		// Transform movement relative to local axes 
		cameraPosition += cameraRightVector * distance;
		return;
	}

	void Camera::SetFocusState(bool newFocus)
	{
		hasFocus = newFocus;
	}

	void Camera::SetPerspectiveProjection(float fov, float aspectRatio, float nearClip, float farClip)
	{
		SetFieldOfView(fov);
		// Create and set camera's projection matrix (perspective) 
		cameraProjectionMatrix = perspective(fov, aspectRatio, nearClip, farClip);
	}

	void Camera::SetOrthographicProjection(float left, float right, float bottom, float top, float near, float far)
	{
		// Create and set camera's projection matrix (orthgraphic) 
		cameraProjectionMatrix = orthographic(left, right, bottom, top, near, far);
	}

	Matrix4 Camera::GetMVPMatrix()
	{
		
		// Rebuild the view matrix
		ConstructViewMatrix(true);

		// Create and return ModelViewProjection Matrix
		return (cameraViewMatrix*cameraProjectionMatrix);
	}

	Matrix4 Camera::GetViewMatrix()
	{
		// Rebuild and return the view matrix
		ConstructViewMatrix(true);

		return cameraViewMatrix;
	}

	Matrix4 Camera::GetProjectionMatrix()
	{
		return cameraProjectionMatrix;
	}


	void Camera::ConstructViewMatrix(bool orthoganalizeAxes)
	{
		// Orthogonalize axes if necessary (ensure up and right vectors are perpendicular to lookat and each other)
		if(orthoganalizeAxes)
		{
			// Look at
			cameraLookAt.Normalize();

			// Up vector
			cameraUpVector = crossprod(cameraLookAt, cameraRightVector);
			cameraUpVector.Normalize();

			// Right vector
			cameraRightVector = crossprod(cameraUpVector, cameraLookAt);
			cameraRightVector.Normalize();
		}

		// Build the view matrix itself
		cameraViewMatrix[0][0] = cameraRightVector.x;
		cameraViewMatrix[1][0] = cameraRightVector.y;
		cameraViewMatrix[2][0] = cameraRightVector.z;
		cameraViewMatrix[3][0] = -dotprod(cameraRightVector, cameraPosition);

		cameraViewMatrix[0][1] =  cameraUpVector.x;
		cameraViewMatrix[1][1] =  cameraUpVector.y;
		cameraViewMatrix[2][1] =  cameraUpVector.z;
		cameraViewMatrix[3][1] =  -dotprod(cameraUpVector, cameraPosition);

		cameraViewMatrix[0][2] = cameraLookAt.x;
		cameraViewMatrix[1][2] = cameraLookAt.y;
		cameraViewMatrix[2][2] = cameraLookAt.z;
		cameraViewMatrix[3][2] = -dotprod(cameraLookAt, cameraPosition);

		cameraViewMatrix[0][3] = 0;
		cameraViewMatrix[1][3] = 0;
		cameraViewMatrix[2][3] = 0;
		cameraViewMatrix[3][3] = 1;

		return;
	}

	void Camera::ActionMoveCameraPosition(vec3f newPosition, int type)
	{
		switch(type)
		{
			case XCOORD:
				cameraPosition.x = newPosition.x;
				break;

			case YCOORD:
				cameraPosition.y = newPosition.y;
				break;

			case ZCOORD:
				cameraPosition.z = newPosition.z;
				break;
		}
	}

	void Camera::ActionMoveCameraLookAt(vec3f newLookAt)
	{
		SetLookAt(cameraPosition - newLookAt);
	}

	void Camera::ActionGetCameraPos(vec3f& position)
	{
		position = cameraPosition;
	}

	void Camera::ActionGetCameraLookAt(vec3f& lookAt)
	{
		lookAt = cameraLookAt;
	}

	void Camera::ActionGetCameraUp(vec3f& upVec)
	{
		upVec = cameraUpVector;
	}


	void Camera::SetMainCameraStatus(bool newStatus)
	{
		if(newStatus)
		{
			CameraActionSubscribe();
		}
		else
		{
			CameraActionUnsubscribe();
		}
	}


}