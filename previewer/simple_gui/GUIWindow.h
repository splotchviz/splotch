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
 
#ifndef SPLOTCH_PREVIEWER_SIMPLEGUI_GUIWINDOW
#define SPLOTCH_PREVIEWER_SIMPLEGUI_GUIWINDOW

// X11 includes
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/keysymdef.h>

// OpenGL includes
#include <GL/gl.h>
#include <GL/glx.h>

// General includes
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <cstring>

// Usage includes
#include "previewer/libs/core/Event.h"

namespace previewer
{
	namespace simple_gui
	{
		class GUIWindow
		{
		public:
			// Window lifecycle members
			void Create();
			void Create(int _width, int _height);
			void Destroy();

			// Frame lifecycle members
			void SwapBuffers();
			bool IsEventPending();
			Event GetNextEvent();

			// Font members
			void LoadFont(std::string fontName);
			void AddToLabelList(std::string value, float x, float y);
			void DrawLabels();

		private:
			// Window information
			int width;
			int height;

			// X11 event members
			XEvent xEvent;
			Display* display;
			Window window;

			// X11 key handling
			char text[255]; // character buffer
			KeySym key;

			// Window instance
			Window rootWindow;
			XVisualInfo* xVisualInfo;
			XSetWindowAttributes setWindowAttributes;
			XWindowAttributes xWindowAttributes;

			// Window closure controls (little x)
			Atom windowDeleteMessage;

			// OpenGL context and colour map
			GLXContext glContext;
			Colormap colourMap;

			// Keyboard mapping function
			std::string FindKey();
			bool IsKeyID(KeySym);

			// Specific event processors
			void HandleExposeEvent(Event&);
			void HandleKeyPressEvent(Event&);
			void HandleKeyReleaseEvent(Event&);
			void HandleButtonPressEvent(Event&);
			void HandleButtonReleaseEvent(Event&);
			void HandleMotionNotifyEvent(Event&);
			void HandleClientMessageEvent(Event&);
			void HandleIgnoreEvent(Event&);

			// Labels for the text
			struct labelItem {
				std::string str;
				float x;
				float y;
			};

			// 
			std::list<labelItem> labelList;
			GLuint base;
			GLuint list;
			void MakeRasterFont(std::string font);
			void PrintString(std::string str);
		};
	}
}

#endif