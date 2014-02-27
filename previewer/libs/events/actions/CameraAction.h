#ifndef SPLOTCH_PREVIEWER_LIBS_EVENTS_ACTIONS_CAMERAACTION
#define SPLOTCH_PREVIEWER_LIBS_EVENTS_ACTIONS_CAMERAACTION

//Debug include
#include "previewer/libs/core/Debug.h"

// Usage includes
#include <list>
#include "cxxsupport/vec3.h"

namespace previewer
{
	// Enumerate list to indicate which coordinate is being dealt with
	enum{
		XCOORD,
		YCOORD,
		ZCOORD
	};

	class CameraAction
	{

	public:
		~CameraAction(void);
		
		virtual void ActionMoveCameraPosition(vec3f, int) = 0;
		virtual void ActionMoveCameraLookAt(vec3f) = 0;
		//virtual void ActionMoveCameraUpVector(vec3f) = 0;
		virtual void ActionGetCameraPos(vec3f&) = 0;
		virtual void ActionGetCameraLookAt(vec3f&) = 0;
		virtual void ActionGetCameraUp(vec3f&) = 0;

		void CameraActionSubscribe();
		void CameraActionUnsubscribe();

		static void CallMoveCameraAction(vec3f, int);
		static void CallGetCameraPosAction(vec3f&);

		static void CallSetCameraLookAtAction(vec3f);
		static void CallGetCameraLookAtAction(vec3f&);
		static void CallGetCameraUpAction(vec3f&);

	private:
		static std::list<CameraAction*>* subscriberList;
	};
}

#endif