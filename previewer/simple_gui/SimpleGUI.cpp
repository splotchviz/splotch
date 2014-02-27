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

			// Create a window
			window.Create();
			window.LoadFont("-misc-fixed-medium-r-semicondensed--0-0-75-75-c-0-iso8859-1");

			// Set render size and position
			application.SetRenderSize(800, 800);
			application.SetRenderPosition(0,0);

			// If file exists load application
			if(FileLib::FileExists(const_cast<char*>(paramFile.c_str())))
			{
				application.Load(paramFile);
			}

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

					switch(ev.eventType)
					{
						case evExposed:
							application.TriggerOnExposedEvent();
						break;

						case evKeyPress:
							// Dont allow application keypress events when gui is active
							if(!guiActive) 
								application.TriggerOnKeyPressEvent(ev.keyID);
							command.HandleKeyEvent(ev, application);
						break;

						case evKeyRelease:
							// Dont allow application keyrelease events when gui is active
							if(!guiActive) 
								application.TriggerOnKeyReleaseEvent(ev.keyID);
						break;

						case evButtonPress:
							application.TriggerOnButtonPressEvent(ev.keyID, ev.mouseX, ev.mouseY);
						break;

						case evButtonRelease:
							application.TriggerOnButtonReleaseEvent(ev.keyID, ev.mouseX, ev.mouseY);
						break;

						case evMouseMotion:
							application.TriggerOnMotionEvent(ev.mouseX, ev.mouseY);
						break;

						case evQuitApplication:
							application.TriggerOnQuitApplicationEvent();
							isApplicationTerminating = true;
						break;

						case evResize:
							application.SetRenderSize(ev.mouseX, ev.mouseY);
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
					DebugPrint("Command is Terminating");
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