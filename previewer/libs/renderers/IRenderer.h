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

#ifndef SPLOTCH_PREVIEWER_LIBS_RENDERERS_IRENDERER
#define SPLOTCH_PREVIEWER_LIBS_RENDERERS_IRENDERER

//Debug include
#include "previewer/libs/core/Debug.h"

#include "previewer/libs/core/Camera.h"
#include "previewer/libs/core/ParticleData.h"
#include "previewer/libs/core/BoundingBox.h"
#include "previewer/libs/materials/IMaterial.h"
#include "previewer/libs/materials/FF_ParticleMaterial.h"
#include "previewer/libs/core/WindowManager.h"

#include "previewer/libs/core/Event.h"
#include "previewer/libs/events/OnKeyPressEvent.h"
#include "previewer/libs/events/OnMotionEvent.h"

#include "GL/gl.h"


namespace previewer
{
	// Provides the interface for access to renderers. Note all renderers
	// must implement the IRenderer interface for them to be accessible
	// to the particle simulation
	class IRenderer :  	public OnKeyPressEvent,
						public OnMotionEvent
	{
	public:
		
		virtual ~IRenderer() {};

		virtual void Load(const ParticleData&) = 0;
		virtual void Draw() = 0;
		virtual void Unload() = 0;
		virtual void Update() = 0;

		virtual void OnKeyPress(Event) = 0;
		virtual void OnMotion(Event) = 0;

		void Clear(float r, float g, float b, float a)
		{	
			glClearColor(r, g, b, a);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		void LoadImage(std::string file)
		{
			imageViewingMaterial = new FF_ParticleMaterial();
			imageViewingMaterial->Load();
			imageViewingMaterial->SetTexture(true);
			imageViewingMaterial->LoadTexture(file, GL_TEXTURE_2D);

			imageWidth = imageViewingMaterial->GetTexWidth();
			imageHeight = imageViewingMaterial->GetTexHeight();
		}

		void DrawImage(int xMin, int yMin, int width, int height)
		{

			// get actual width/height!
			glViewport(xMin, yMin, width, height);

			glClearColor(0.1, 0.1, 0.1, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			imageViewingMaterial->Bind();

			glBegin(GL_QUADS);
		    glTexCoord2f(0,0);  glVertex3f(-1, -1, 0);
		    glTexCoord2f(0,1);  glVertex3f(-1, 1, 0);
		    glTexCoord2f(1,1);  glVertex3f(1, 1, 0);
		    glTexCoord2f(1,0);  glVertex3f(1, -1, 0);
			glEnd();

			imageViewingMaterial->Unbind();

			//PrintOpenGLError();
		}

		// Get a camera reference
		Camera& GetCameraReference()
		{
			return camera;
		}

		int GetImageWidth()
		{
			return imageWidth;
		}

		int GetImageHeight()
		{
			return imageHeight;
		}

		void SetRenderBrightness(unsigned type, float b)
		{
			// Check if brightness container has element for this type already
			if(type > (brightness.size()-1))
				brightness.resize(type+1, 1);

			// Set the brightness
			brightness[type] = b;
		}

		float GetRenderBrightness(int type)
		{
			return brightness[type];
		}

		void SetSmoothingLength(unsigned type, float sl)
		{
			// Check if brightness container has element for this type already
			if(type > (smoothingLength.size()-1))
				smoothingLength.resize(type+1, 1);

			// Set the brightness
			smoothingLength[type] = sl;
		}

		float GetSmoothingLength(int type)
		{
			return smoothingLength[type];
		}

		void SetRadialMod(float rm)
		{
			radial_mod = rm;
		}

		float GetRadialMod()
		{
			return radial_mod;
		}

		void ResetCamera()
		{
			camera.Create(dataBBox);
		}

	protected:
		Camera camera;
		ParticleList particleList;
		IMaterial* material;
		BoundingBox dataBBox;
		std::vector<float> brightness;
		std::vector<float> smoothingLength;
		float radial_mod;
		

	private:
		IMaterial* imageViewingMaterial;
		int imageWidth;
		int imageHeight;

	};

}

#endif