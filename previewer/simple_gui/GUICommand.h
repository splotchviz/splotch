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
 
#ifndef SPLOTCH_PREVIEWER_SIMPLEGUI_GUICOMMAND
#define SPLOTCH_PREVIEWER_SIMPLEGUI_GUICOMMAND

//Debug include
#include "previewer/libs/core/Debug.h"

// Member includes
#include "previewer/Previewer.h"
#include <stack>
#include <string>
#include <sstream>
#include <list>

// Usage includes
#include "previewer/libs/core/Event.h"
#include <algorithm>

namespace previewer
{
	namespace simple_gui
	{
		class GUICommand
		{
		public:
			GUICommand();
			
			void Undo();
			void Redo();

			std::string GetCurrentCommandLine();
			void HandleKeyEvent(Event ev, Previewer& pv);

			bool IsTerminating();

		private:
			std::string currentCommandLine;
			std::stack<std::string> commandStack;
			std::stack<std::string> redoStack;

			void ProcessCommand(Previewer& pv);
			//void ProcessReverseCommand(std::string command);

			bool isTerminating;

			std::string GetArgFromString(std::string str, int arg);
			bool DoesStringBegin(std::string str1, std::string str2);

			std::list<std::string> confirmationsList;
			
		};
	}
}

#endif