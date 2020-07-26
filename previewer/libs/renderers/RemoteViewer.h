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
 * 		Tim Dykes 
 * 		University of Portsmouth
 *
 */
 
#ifndef SPLOTCH_PREVIEWER_LIBS_RENDERERS_REMOTE_VIEWER
#define SPLOTCH_PREVIEWER_LIBS_RENDERERS_REMOTE_VIEWER

#define GL_GLEXT_PROTOTYPES

#include "IRenderer.h"
#include "previewer/libs/core/Camera.h"
#include "previewer/libs/materials/FF_ParticleMaterial.h"
#include "previewer/libs/core/ParticleSimulation.h"
#include "previewer/libs/core/Fbo.h"
#include "cxxsupport/vec3.h"
#include "previewer/libs/core/WindowManager.h"

#include "splotch/splotch_host.h"
#include "utils/fast_timer.h"
#include <chrono>
#include <thread>

#include "RAWInStream.h"
#include "AsyncClient.h"
#include "SyncQueue.h"

#include "JPEGImage.h"
#include "TJDeCompressor.h"



 namespace previewer
{
#define TEXTURE_WIDTH ParticleSimulation::GetRenderWidth()
#define TEXTURE_HEIGHT ParticleSimulation::GetRenderWidth()

	class RemoteViewer : public IRenderer,
						 public OnKeyReleaseEvent,
						 public OnButtonReleaseEvent
	{
	public:

		void Load(const ParticleData&);
		void Draw();
		void Unload();
		void Update();

		void OnKeyPress(Event);
		void OnKeyRelease(Event);
		void OnMotion(Event);
		void OnButtonPress(Event);
		void OnButtonRelease(Event);

		void Reciever(int size);
		void Sender();
		void SetRenderBrightness(unsigned type, float b);
		void SetSmoothingLength(unsigned type, float sl);
		void PrintCamera();

		int pv2ascii(std::string pvKeyId);
		// Image streaming
		zrf::RAWInStream<> is;
		std::thread recv_thread;
		std::string image_uri;
		SyncQueue<tjpp::Image> image_queue;

		// Event sending
		zrf::AsyncClient<> ac;
		std::string event_uri;
		std::thread event_thread;
		SyncQueue<Event> event_queue;

		// Logging/timing
		fast_timer ft;
		int ctr = 0;
	
	private:
		IMaterial* renderMaterial;
		paramfile* splotchParams;

		int 	xres;
		int 	yres;
		bool 	image_recieved;
		bool 	running;
		bool 	first_image;

		
	};

}

#endif