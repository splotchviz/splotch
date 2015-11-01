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

#include <stack>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>

#include "previewer/libs/core/Debug.h"
#include "previewer/Previewer.h"
#include "previewer/libs/core/Event.h"

// Max number of particle species
#define MAX_SPECIES 10

namespace previewer
{
	namespace simple_gui
	{
		class GUICommand;

		struct Command{
			std::vector<std::string> cmdTokens;
			std::vector<std::string> argTokens;
			int nArgs;
			void (GUICommand::*func)(Previewer& pv, std::vector<std::string> args);
		};

		class GUICommand
		{
		public:
			GUICommand();
			
			std::string GetCurrentCommandLine();
			void HandleKeyEvent(Event ev, Previewer& pv);

			bool IsTerminating();

		private:
			std::string currentCommandLine;
			//std::stack<std::string> commandStack;
			//std::stack<std::string> redoStack;

			void AddCommand(std::string cmd,void (GUICommand::*func)(Previewer& pv, std::vector<std::string> args), int nArgs);
			void ProcessCommand(Previewer& pv);
			std::vector<std::string> Tokenize(std::string input);
			Command Parse(std::vector<std::string>);
			std::vector<Command> commandList;
			void (GUICommand::*current_func)(Previewer& pv, std::vector<std::string> args);
			bool isTerminating;
			//std::list<std::string> confirmationsList;
			template<typename T> 
			std::string toa(T in)
			{
				std::stringstream sstream;
				sstream << in;
				std::string str;
				sstream >> str;
				return str;
			}

			// List of command functions
			// Internal
			void ClearCommand(Previewer& pv, std::vector<std::string> args);
			void CommandNotFound(Previewer& pv, std::vector<std::string> args);

			// Generic
			void Quit(Previewer& pv, std::vector<std::string> args);	
			void Run(Previewer& pv, std::vector<std::string> args);
			void View(Previewer& pv, std::vector<std::string> args);
			void Stop(Previewer& pv, std::vector<std::string> args);

			// Interface
			void GetFPS(Previewer& pv, std::vector<std::string> args);	
			void SetMoveSpeed(Previewer& pv, std::vector<std::string> args);
			void SetRotateSpeed(Previewer& pv, std::vector<std::string> args);
			void SetXRes(Previewer& pv, std::vector<std::string> args);
			void SetYRes(Previewer& pv, std::vector<std::string> args);
			void SetRes(Previewer& pv, std::vector<std::string> args);
			void SetFOV(Previewer& pv, std::vector<std::string> args);
			void SetTarget(Previewer& pv, std::vector<std::string> args);
			void ResetCamera(Previewer& pv, std::vector<std::string> args);

			// Splotch 
			void WriteParams(Previewer& pv, std::vector<std::string> args);	
			void WriteSceneFile(Previewer& pv, std::vector<std::string> args);	

			// Scene manipulation			
			void SetPalette(Previewer& pv, std::vector<std::string> args);	
			void ReloadColors(Previewer& pv, std::vector<std::string> args);	
			void SetBrightness(Previewer& pv, std::vector<std::string> args);	
			void GetBrightness(Previewer& pv, std::vector<std::string> args);	
			void SetSmoothing(Previewer& pv, std::vector<std::string> args);	
			void GetSmoothing(Previewer& pv, std::vector<std::string> args);	
			void SetParam(Previewer& pv, std::vector<std::string> args);	
			void GetParam(Previewer& pv, std::vector<std::string> args);

			// Animation
			void SetAnimPoint(Previewer& pv, std::vector<std::string> args);	
			void RemoveAnimPoint(Previewer& pv, std::vector<std::string> args);	
			void PreviewAnim(Previewer& pv, std::vector<std::string> args);
			void SaveAnimFile(Previewer& pv, std::vector<std::string> args);
			void LoadAnimFile(Previewer& pv, std::vector<std::string> args);
			void SetCamInterpolation(Previewer& pv, std::vector<std::string> args);
			void SetLookatInterpolation(Previewer& pv, std::vector<std::string> args);
		};
	}
}

#endif