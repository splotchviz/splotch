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
 
#include "GUIWindow.h"

namespace previewer
{
	namespace simple_gui
	{
		// Window lifecycle members
		void GUIWindow::Create()
		{
			Create(800, 800);
		}
		void GUIWindow::Create(int _width, int _height)
		{
			// Open the X11 display
			display = XOpenDisplay(NULL);
			if(display == NULL)
			{
				printf("Could not connect to the x server.\n");
				exit(0);
			}

			// Set the root window
			rootWindow = DefaultRootWindow(display);

			// Choose glx visual
			int attr[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
			xVisualInfo = glXChooseVisual(display, 0, attr);
			if(xVisualInfo == NULL)
			{
				printf("Could not find appropriate visual.\n");
				exit(0);
			}

			// Create a colour map
			colourMap = XCreateColormap(display, rootWindow, xVisualInfo->visual, AllocNone);

			// Start setting window attributes
			setWindowAttributes.colormap = colourMap;
			setWindowAttributes.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask | Button1MotionMask | StructureNotifyMask;

			// Create the window
			window = XCreateWindow(display, rootWindow, 0, 0, _width, _height, 0, xVisualInfo->depth, InputOutput, xVisualInfo->visual, CWColormap | CWEventMask, &setWindowAttributes);

			// Map the window and store the window name
			XMapWindow(display, window);
			XStoreName(display, window, "Splotch Previewer");

			// Setup the window delete message (from little x)
			windowDeleteMessage = XInternAtom(display, "WM_DELETE_WINDOW", false);
			if(!XSetWMProtocols(display, window, &windowDeleteMessage, 1))
			{
				printf("WM_Protocols set failed, gui quit window may give a terminal error\n");
			}

			// Create glx context
			glContext = glXCreateContext(display, xVisualInfo, NULL, GL_TRUE);
			glXMakeCurrent(display, window, glContext);

			//Clear window initially
			glClearColor(0.2, 0.2, 0.2, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			SwapBuffers();

			//Clear window for second buffer
			glClearColor(0.2, 0.2, 0.2, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// Set the screen sizes
			width = _width;
			height = _height;

			return;
		}
		void GUIWindow::Destroy()
		{
			glXMakeCurrent(display, None, NULL);
			glXDestroyContext(display, glContext);
			XDestroyWindow(display, window);
			XCloseDisplay(display);
			exit(0);
		}

		// Frame lifecycle members
		void GUIWindow::SwapBuffers()
		{
			glXSwapBuffers(display, window);
		}
		bool GUIWindow::IsEventPending()
		{
			// Check to see if any events are pending from X11
			return (XPending(display)) ? true : false;
		}
		Event GUIWindow::GetNextEvent()
		{
			// Setup a temporary event and get the next X11 event
			Event event;
			XNextEvent(display, &xEvent);

			// See what type of event we have
			switch(xEvent.type)
			{
				case Expose:
					HandleExposeEvent(event);
				break;

				case KeyPress:
					HandleKeyPressEvent(event);
				break;

				case KeyRelease:
					HandleKeyReleaseEvent(event);
				break;

				case ButtonPress:
					HandleButtonPressEvent(event);
				break;

				case ButtonRelease:
					HandleButtonReleaseEvent(event);
				break;

				case MotionNotify:
					HandleMotionNotifyEvent(event);
				break;

				case ClientMessage:
					HandleClientMessageEvent(event);
				break;

				case ConfigureNotify:
				{
					XConfigureEvent xce = xEvent.xconfigure;

					// Test for resize event
					if(xce.width != width || xce.height != height)
					{
						width = xce.width;
						height = xce.height;

						event.eventType = evResize;
						event.mouseX = width;
						event.mouseY = height;
					}
				}
				break;

				default:
					HandleIgnoreEvent(event);
				break;
			}

			return event;
		}

		// Handles the expose event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleExposeEvent(Event& event)
		{
			// Set event type to exposed
			event.eventType = evExposed;

			return;
		}

		// Handles the key press event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleKeyPressEvent(Event& event)
		{
			// Set event type to key press
			event.eventType = evKeyPress;

			// Mouse element is not used
			event.mouseX = 0;
			event.mouseY = 0;

			// Find which key was pressed
			std::string keyPressed = FindKey();

			// Test to see if the key has been pressed
			if(keyPressed == "")
			{
				HandleIgnoreEvent(event);
				return;
			}

			// Setup the event
			event.keyID = keyPressed;

			return;
		}

		// Handles the key release event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleKeyReleaseEvent(Event& event)
		{
			// Set event type to key release
			event.eventType = evKeyRelease;

			// Mouse element is not used
			event.mouseX = 0;
			event.mouseY = 0;

			// Find which key was released
			std::string keyReleased = FindKey();

			// Test to see if the key has been released
			if(keyReleased == "")
			{
				HandleIgnoreEvent(event);
				return;
			}

			// Setup the event
			event.keyID = keyReleased;

			return;
		}

		// Handles the button press event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleButtonPressEvent(Event& event)
		{
			// Set the event type to button press
			event.eventType = evButtonPress;

			// Store the button pressed as a string
			std::stringstream strstream;
			strstream << xEvent.xbutton.button;
			strstream >> event.keyID;

			// Store cursor location at button click
			event.mouseX = xEvent.xbutton.x;
			event.mouseY = xEvent.xbutton.y;

			// Store translated cursor at button click
			event.translatedMouseX = xEvent.xbutton.x;
			event.translatedMouseY = height - xEvent.xbutton.y; // (swap axis)

			return;
		}

		// Handles the button release event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleButtonReleaseEvent(Event& event)
		{
			// Set the event type to button release
			event.eventType = evButtonRelease;

			// Store the button released as a string
			std::stringstream strstream;
			strstream << xEvent.xbutton.button;
			strstream >> event.keyID;

			// Store cursor location at button click
			event.mouseX = xEvent.xbutton.x;
			event.mouseY = xEvent.xbutton.y;

			return;
		}

		// Handles the motion notify event, generating the internal application
		// event to be propogated (when holding down mouse 1 and moving)
		void GUIWindow::HandleMotionNotifyEvent(Event& event)
		{
			// Set the event type and keyID to mouse motion
			event.eventType = evMouseMotion;
			event.keyID = "Motion";

			// Store the new cursor position
			event.mouseX = xEvent.xmotion.x;
			event.mouseY = xEvent.xmotion.y;
			
			// Store translated cursor at button click
			event.translatedMouseX = xEvent.xbutton.x;
			event.translatedMouseY = height - xEvent.xbutton.y;

			return;
		}

		// Handles the client message event, generating the internal application
		// event to be propogated
		void GUIWindow::HandleClientMessageEvent(Event& event)
		{
			// Check for the type of client event
			if((uint)xEvent.xclient.data.l[0] == windowDeleteMessage)
			{
				// Set all the event information
				event.eventType = evQuitApplication;
				event.keyID = "";

				return;
			}
		}

		// Handles an event that we have chosen to ignore, it will create an event
		// with type IgnoreEvent
		void GUIWindow::HandleIgnoreEvent(Event& event)
		{
			// Set all the event information
			event.eventType = evIgnoreEvent;
			event.keyID = "";
			event.mouseX = 0;
			event.mouseY = 0;

			return;
		}

		// Checks to see if a specific key was pressed
		bool GUIWindow::IsKeyID(KeySym k)
		{
			return (XLookupKeysym(&xEvent.xkey, 0) == k) ? true : false;
		}

		// Finds a given key press
		std::string GUIWindow::FindKey()
		{
			// Get the key that was pressed
			std::string keyPressed = "";

			// Process if the key is a char
			if(XLookupString(&xEvent.xkey, text, 255, &key, 0) == 1)
			{
				std::stringstream strstream;
				strstream << text[0];
				keyPressed = strstream.str();
			}

			// Process the special keys
			if(IsKeyID(XK_Return))
				keyPressed = "RETURN";
			else if(IsKeyID(XK_BackSpace))
				keyPressed = "BACKSPACE";
			else if(IsKeyID(XK_Tab))
				keyPressed = "TAB";
			else if(IsKeyID(XK_Escape))
				keyPressed = "ESCAPE";
			else if(IsKeyID(XK_Delete))
				keyPressed = "DELETE";
			else if(IsKeyID(XK_space))
				keyPressed = "SPACE";
			else if(IsKeyID(XK_Shift_L) || IsKeyID(XK_Shift_R))
				keyPressed = "SHIFT";
			else if(IsKeyID(XK_Control_L) || IsKeyID(XK_Control_R))
				keyPressed = "CTRL";
			else if(IsKeyID(XK_Alt_L) || IsKeyID(XK_Alt_R))
				keyPressed = "ALT";
			else if(IsKeyID(XK_Left))
				keyPressed = "LEFT";
			else if(IsKeyID(XK_Right))
				keyPressed = "RIGHT";
			else if(IsKeyID(XK_Up))
				keyPressed = "UP";
			else if(IsKeyID(XK_Down))
				keyPressed = "DOWN";

			return keyPressed;
		}

		void GUIWindow::LoadFont(std::string fontName)
		{
			MakeRasterFont(fontName);
		}

		void GUIWindow::AddToLabelList(std::string label, float x, float y)
		{
			labelItem li;
			li.str = label;
			li.x = x;
			li.y = y;
			labelList.push_back(li);
		}

		void GUIWindow::DrawLabels()
		{
			// Set mModes
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();

			glMatrixMode (GL_MODELVIEW);
			glLoadIdentity ();

			glShadeModel (GL_SMOOTH);

			// If depth testign is enabled, disable it to draw text on top
			bool DTWasEnabled = false;
			if(glIsEnabled(GL_DEPTH_TEST) == GL_TRUE)
			{
				glDisable (GL_DEPTH_TEST);
				DTWasEnabled = true;
			}

			//text colour white
			glColor3f (1.0f, 1.0f, 1.0f);

			//draw all fonts queued
			for(std::list<labelItem>::iterator it = labelList.begin(); it != labelList.end(); it++)
			{
				glRasterPos2f(it->x, it->y);
				PrintString(it->str);
			}

			labelList.clear();

			glFlush(); 

			//Reenable depth testing if necessary
			if(DTWasEnabled)
				glEnable(GL_DEPTH_TEST);
		}

		void GUIWindow::MakeRasterFont(std::string font)
		{
			XFontStruct *fontInfo;
			Font id;
			unsigned int first, last;
			//let user choose font to avoid hardcoding a font they may not have!
			fontInfo = XLoadQueryFont(display, font.c_str());


			if (fontInfo == NULL) 
			{
			    std::cout << "Font not found" << std::endl;
			}

			id = fontInfo->fid;
			first = fontInfo->min_char_or_byte2;
			last = fontInfo->max_char_or_byte2;

			base = glGenLists(last+1);

			if (!base) 
			{
			    std::cout << "out of display lists" << std::endl;
			}

			glXUseXFont(id, first, last-first+1, base+first);	
		}

		void GUIWindow::PrintString(std::string str)
		{
			const char *ch;
			ch = str.c_str();
			glListBase(base);
			glCallLists(strlen(ch), GL_UNSIGNED_BYTE, (GLubyte *)ch);
		}
	}
}