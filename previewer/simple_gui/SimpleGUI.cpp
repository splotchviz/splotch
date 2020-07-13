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
 
#include "SimpleGUI.h"

namespace previewer
{
	namespace simple_gui
	{

		bool SimpleGUI::guiActive = false;

		void SimpleGUI::Load(std::string paramFile)
		{
			// Set terminating flag
			isApplicationTerminating = false;

			// If file exists load application
			if(FileLib::FileExists(const_cast<char*>(paramFile.c_str())))
			{
				application.Load(paramFile);
			}
			// Create a window
			int x = application.GetParameterT<int>("xres", 800);
			int y = application.GetParameterT<int>("yres", 800);
			window.Create( x, y);

			// Once we've loaded the application, check paramfile for a font
			std::string font = application.GetParameter("xfont", "-misc-fixed-medium-r-semicondensed--0-0-75-75-c-0-iso8859-1");
			window.LoadFont(font);

			// Set render size and position
			application.SetRenderSize(x, y);
			application.SetRenderPosition(0,0);

			Run();
		}
		void SimpleGUI::Load()
		{

		}

		void SimpleGUI::Run()
		{
			// Run the current frame
			while(!isApplicationTerminating)
			{
				// Process all the events
				while(window.IsEventPending())
				{
					Event ev = window.GetNextEvent();
// #ifdef CLIENT_SERVER
// 					switch(ev.id)
// #else
					switch(ev.eventType)
//#endif
					{
						case evExposed:
							application.TriggerOnExposedEvent(ev);
						break;

						case evKeyPress:
							// Dont allow application keypress events when gui is active
							if(!guiActive) 
								application.TriggerOnKeyPressEvent(ev);
							command.HandleKeyEvent(ev, application);
						break;

						case evKeyRelease:
							// Dont allow application keyrelease events when gui is active
							if(!guiActive) 
								application.TriggerOnKeyReleaseEvent(ev);
						break;

						case evButtonPress:
							application.TriggerOnButtonPressEvent(ev);
						break;

						case evButtonRelease:
							application.TriggerOnButtonReleaseEvent(ev);
						break;

						case evMouseMotion:
							application.TriggerOnMotionEvent(ev);
						break;

						case evQuitApplication:
							application.TriggerOnQuitApplicationEvent(ev);
							isApplicationTerminating = true;
						break;

						case evResize:
// #ifdef CLIENT_SERVER
// 							application.SetRenderSize(ev.field0, ev.field1);
// #else
							application.SetRenderSize(ev.mouseX, ev.mouseY);
//#endif
						break;

						default:
						break;
					}
				}

				// Run Appication
				application.Run();

				if(guiActive)
				{
					window.AddToLabelList("> " + command.GetCurrentCommandLine(), -0.9, -0.9);
					window.DrawLabels();
				}

				// Swap Buffers
				window.SwapBuffers();

				// See if application is terminating
				if(command.IsTerminating())
				{
					isApplicationTerminating = true;
				}
			}

			Unload();
		}

		void SimpleGUI::Unload()
		{
			// Unload application and destroy the window
			application.Unload();
			window.Destroy();
		}
	}
}