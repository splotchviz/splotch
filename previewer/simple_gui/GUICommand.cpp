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
 
#include "GUICommand.h"
#include "SimpleGUI.h"

namespace previewer
{
	namespace simple_gui
	{
		GUICommand::GUICommand()
		{
			isTerminating = false;
			currentCommandLine = "";
			// Add commands here
			// syntax: command string, command function*, num args
			// -1 means variable arg count

			// Register previewer commands
			// Generic
			AddCommand("quit",&GUICommand::Quit, 0);
			AddCommand("run",&GUICommand::Run, 1);
			AddCommand("view",&GUICommand::View, 1);
			AddCommand("stop",&GUICommand::Stop, 0);

			// Interface
			AddCommand("get fps", &GUICommand::GetFPS, 0);
			AddCommand("set move speed", &GUICommand::SetMoveSpeed, 1);
			AddCommand("set rotate speed", &GUICommand::SetRotateSpeed, 1);
			AddCommand("set xres", &GUICommand::SetXRes, 1);
			AddCommand("set yres", &GUICommand::SetYRes, 1);
			AddCommand("set res", &GUICommand::SetRes, 2);
			AddCommand("set fov", &GUICommand::SetFOV, 1);
			AddCommand("set target",&GUICommand::SetTarget, -1);
			AddCommand("reset camera",&GUICommand::ResetCamera, 0);

			// Splotch 
			AddCommand("write params", &GUICommand::WriteParams, 1);
			AddCommand("write scenefile", &GUICommand::WriteSceneFile, 1);
			AddCommand("set param", &GUICommand::SetParam, 2);
			AddCommand("get param", &GUICommand::GetParam, 1);

			// Scene manipulation
			AddCommand("set palette", &GUICommand::SetPalette, 2);
			AddCommand("reload colors", &GUICommand::ReloadColors, 0);
			AddCommand("set brightness", &GUICommand::SetBrightness, 2);
			AddCommand("get brightness", &GUICommand::GetBrightness, 1);
			AddCommand("set smoothing", &GUICommand::SetSmoothing, 2);
			AddCommand("get smoothing", &GUICommand::SetSmoothing, 1);			


			// Animation
			AddCommand("set point", &GUICommand::SetAnimPoint, 1);
			AddCommand("remove point", &GUICommand::RemoveAnimPoint, 0);
			AddCommand("preview", &GUICommand::PreviewAnim, 0);
			AddCommand("save animation", &GUICommand::SaveAnimFile, 1);
			AddCommand("load animation", &GUICommand::LoadAnimFile, 1);
			AddCommand("set camera interpolation", &GUICommand::SetCamInterpolation, 1);
			AddCommand("set lookat interpolation", &GUICommand::SetLookatInterpolation, 1);

		}

		// AddCommand()

		std::string GUICommand::GetCurrentCommandLine()
		{
			return currentCommandLine;
		}
		void GUICommand::HandleKeyEvent(Event ev, Previewer& pv)
		{
			if(ev.keyID == "#")
			{
				SimpleGUI::guiActive = !SimpleGUI::guiActive;
				currentCommandLine = "";
			}

			else if(SimpleGUI::guiActive)
			{
				if(ev.keyID == "RETURN")
				{
					ProcessCommand(pv);
				}
				else if(ev.keyID == "BACKSPACE")
				{
					currentCommandLine = currentCommandLine.substr(0, currentCommandLine.size()-1);
				}
				else if(ev.keyID == "SPACE")
				{
					currentCommandLine += " ";
				}
				else if(ev.keyID == "UP")
				{
					// Get previous command and display in command line
				}				
				else
				{
					if(ev.keyID.length() == 1)
						currentCommandLine += ev.keyID;
				}
			}
		}

		void GUICommand::AddCommand(std::string inStr,void (GUICommand::*func)(Previewer& pv, std::vector<std::string> args), int nArgs)
		{
			Command cmd;
			cmd.cmdTokens = Tokenize(inStr);
			cmd.func = func;
			cmd.nArgs = nArgs;
			commandList.push_back(cmd);
		}

		void GUICommand::ProcessCommand(Previewer& pv)
		{

			// Tokenize current command on commandline
			std::vector<std::string> tokens = Tokenize(currentCommandLine);

			// Parse for relevant command
			//DebugPrint("CommandList.size(): %u\n",commandList.size());
			//fflush(0);

			//for(unsigned i = 0; i < tokens.size(); i++)
			//{
			//	DebugPrint("token %i: %s\n", i, tokens[i].c_str());
			//}

			if(tokens.size())
			{
				Command cmd = Parse(tokens);

				// Call appropriate function (as defined during AddCommand())
				// SHOULD CHECK nArgs! and allow for -1 for variable arg count
				current_func = (cmd.func);
				(this->*current_func)(pv, cmd.argTokens);
			}
		}

		std::vector<std::string> GUICommand::Tokenize(std::string input)
		{
			// Tokenize a string into a string vector
			bool protectedMode = false; 
			std::string token;
			std::vector<std::string> tokenList;

			for(std::string::iterator it = input.begin(); it != input.end(); it++)
			{
				// Test for space
				if(*it == ' ')
				{
					if(protectedMode)
					{
						token += *it;
					}
					else if(token.size() > 0)
					{
						// if not protected then transform to lowercase
						if(token[token.size()-1] != '"')
							std::transform(token.begin(), token.end(),token.begin(), static_cast<int(*)(int)>(std::tolower));
						tokenList.push_back(token);
						token = "";
					}
				}
				// Test for escape character
				else if( *it == '"')
				{
					protectedMode = !protectedMode;
				}
				// Any other character
				else
				{
					token += *it;
				}
			}
			if(token.size() > 0)
				tokenList.push_back(token);
			
			return tokenList;
		}

		Command GUICommand::Parse(std::vector<std::string> tokens)
		{
			// DebugPrint("tokens.size(): %u\n",tokens.size());
			// for(unsigned i = 0; i < tokens.size(); i++)
			// {
			// 	DebugPrint("Token %i: %s\n", i, tokens[i].c_str());
			// }
			Command cmd;

			// If there are tokens on the command line
			if(tokens.size() > 0)
			{
				// For each command, compare tokens iteratively until we find a full match
				for(unsigned i = 0; i < commandList.size(); i++)
				{
					unsigned j = 0;
					// While current token matches command token
					while(tokens[j] == commandList[i].cmdTokens[j])
					{
						// Check if we are on final token of command
						if(j == commandList[i].cmdTokens.size()-1)
						{
							// We've found a matching command, now give it the arguments and return
							// First clear the arguments list from previous invocations of command
							commandList[i].argTokens.clear();
							for(j++; j < tokens.size(); j++)
								commandList[i].argTokens.push_back(tokens[j]);
							// Clear text from the command line and return command
							currentCommandLine.clear();
							return commandList[i];
						}
						else 
							j++;
					}

				}
			}
			else
			{
				cmd.func = &GUICommand::ClearCommand;
				return cmd;
			}
			// If we get here, we havent found a match
			// Check if this command was already an 'error', 'not found', or 'output' command
			// Just clear screen if so
			if(tokens.size() > 0)
			{
				if(tokens[0] == std::string("error:") ||
				   tokens[0] == std::string("command_not_found:") ||
				   tokens[0] == std::string("output:")	)
				{
					cmd.func = &GUICommand::ClearCommand;
					return cmd;
				}	
			}

			// If its a new command then create and return not found command 
			cmd.argTokens.resize(tokens.size()+1);
			cmd.argTokens[0] = "command_not_found:";
			for(unsigned i = 0; i < tokens.size(); i++)
				cmd.argTokens[i+1] = tokens[i];
			cmd.func = &GUICommand::CommandNotFound;

			return cmd;
		}

		bool GUICommand::IsTerminating()
		{
			return isTerminating;
		}

		//--------------------------------------------------------------------------------
		//  Command functions
		//--------------------------------------------------------------------------------

		// Internal
		void GUICommand::ClearCommand(Previewer& pv, std::vector<std::string> args)
		{
			currentCommandLine.clear();
		}

		void GUICommand::CommandNotFound(Previewer& pv, std::vector<std::string> args)
		{
			currentCommandLine.clear();
			for(unsigned i = 0; i < args.size(); i++)
				currentCommandLine += args[i]+" ";				
		}

		//--------------------------------------------------------------------------------
		// Generic
		void GUICommand::Quit(Previewer& pv, std::vector<std::string> args)
		{
				//currentCommandLine.clear();
				isTerminating = true;
				pv.TriggerOnQuitApplicationEvent();			
		}

		// Pipe to command line
		void GUICommand::Run(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: run takes 1 argument (use quotes eg run \"./exe optional arguments\"";
			else
			{
				std::string cmd;

				for(int i = 0; i < args.size(); i++)
					cmd+=args[i]+" ";

				system(cmd.c_str());
				currentCommandLine = "output: command sent to system, check parent cli for details";
			}
		}

		// View tga image
		void GUICommand::View(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: view takes 1 string argument";
			else
			{
				pv.ViewImage(args[0]);
				currentCommandLine = "output: viewing image, use stop command to return to preview";
			}

		}

		// Stop viewing image
		void GUICommand::GUICommand::Stop(Previewer& pv, std::vector<std::string> args)
		{
			pv.StopViewingImage();
		}

		//--------------------------------------------------------------------------------
		// Interface
		void GUICommand::GetFPS(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 0)
				currentCommandLine = "error: get fps takes 0 arguments";
			else
			{
				std::stringstream sstream;
				sstream << pv.GetFPS();
				std::string str;
				sstream >> str;
				currentCommandLine = "output: FPS " + str;
			}
		}

		void GUICommand::SetMoveSpeed(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set move speed takes 1 int argument";
			else
			{
				pv.SetMovementSpeed(atoi(args[0].c_str()));
				currentCommandLine = "output: Movement speed set to " + args[0];
			}			

		}	

		void GUICommand::SetRotateSpeed(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set rotate speed takes 1 int argument";
			else
			{
				pv.SetRotationSpeed(atoi(args[0].c_str()));
				currentCommandLine = "output: Rotate speed set to " + args[0];
			}
		}	

		void GUICommand::SetXRes(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set xres takes 1 int argument";
			else
			{
				pv.SetXRes(atoi(args[0].c_str()));
			}
		}

		void GUICommand::SetYRes(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set yres  takes 1 int argument";
			else
			{
				pv.SetYRes(atoi(args[0].c_str()));
			}
		}

		void GUICommand::SetRes(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 2)
				currentCommandLine = "error: set res takes 2 int arguments";
			else
			{
				pv.SetXRes(atoi(args[0].c_str()));
				pv.SetYRes(atoi(args[1].c_str()));	
			}
		}

		void GUICommand::SetFOV(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set yres  takes 1 int argument";
			else
			{
				pv.SetFOV(atoi(args[0].c_str()));
			}			
		}

		void GUICommand::SetTarget(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() == 0)
			{
				pv.SetTargetCenter();
			}
			else if(args.size() == 3)
			{
			 	pv.SetTarget(vec3f(atof(args[0].c_str()),atof(args[1].c_str()),atof(args[2].c_str())));				
			}
			else
			{
				currentCommandLine = "error: set target takes 0 args for center or 3 float args for vector position";
			}	
		}

		void GUICommand::ResetCamera(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 0)
				currentCommandLine = "error: reset camera takes 0 arguments";
			else
			{
				pv.ResetCamera();				
			}
		}

		//--------------------------------------------------------------------------------
		// Splotch 
		void GUICommand::WriteParams(Previewer& pv, std::vector<std::string> args)
		{
			//DebugPrint("Args.size(): %i", args.size());
			if(args.size() != 1)
				currentCommandLine = "error: write params takes 1 string argument";
			else
			{
				pv.WriteParameterFile(args[0]);				
			}
	
		}

		void GUICommand::WriteSceneFile(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: write scenefile takes 1 string argument";
			else
			{
			 	pv.Interpolate();
			 	pv.WriteSplotchAnimationFile(args[0]);			
			}			
		}

		void GUICommand::SetParam(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set param takes 2 argument, parametername(string) and value(string)";
			else
			{
				pv.SetParameter(args[0], args[1]);
			}
		}	

		void GUICommand::GetParam(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: get param takes 1 argument, parametername(string)";
			else
			{
				std::string param = pv.GetParameter(args[0]);
				currentCommandLine = args[0]+"="+param;
			}
		}
		//--------------------------------------------------------------------------------
		// Scene manipulation
		void GUICommand::SetPalette(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 2)
				currentCommandLine = "error: set palette takes 2 arguments: integer particle type, palette file";
			else
			{
				int ptype = atoi(args[0].c_str());
				if(ptype<0 || ptype > MAX_SPECIES)
				{
					currentCommandLine = "error: invalid particle type: " + args[0];				
				}
				else if(!FileLib::FileExists(args[1].c_str()))
				{
					currentCommandLine = "error: file not found: " + args[1];
				}
				else
				{
					pv.SetPalette(args[1], ptype);
				}					
			}
		}
	
		void GUICommand::ReloadColors(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 0)
				currentCommandLine = "error: reload colors takes 0 arguments";
			else
			{
				pv.ReloadColorData();	
			}
		}

		void GUICommand::SetBrightness(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 2)
				currentCommandLine = "error: set brightness takes 2 arguments, type(int) float(value)";
			else
			{
				pv.SetRenderBrightness(atoi(args[0].c_str()), (float)atof(args[1].c_str()));
			}
		}

		void GUICommand::GetBrightness(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: get brightness takes 1 argument, type(int)";
			else
			{
				float b = pv.GetRenderBrightness(atoi(args[0].c_str()));
				currentCommandLine = "output: brightness for type "+args[0]+": "+toa(b);	
			}
		}	

		void GUICommand::SetSmoothing(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 2)
				currentCommandLine = "error: set smoothing takes 2 arguments, type(int) float(value)";
			else
			{
				pv.SetSmoothingLength(atoi(args[0].c_str()), (float)atof(args[1].c_str()));
			}
		}

		void GUICommand::GetSmoothing(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: get smoothing takes 1 argument, type(int)";
			else
			{
				float s = pv.GetSmoothingLength(atoi(args[0].c_str()));
				currentCommandLine = "output: smoothing for type "+args[0]+": "+toa(s);	
			}
		}

		//--------------------------------------------------------------------------------	
		// Animation
		void GUICommand::SetAnimPoint(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set point takes 1 argument";
			else
			{			
				pv.SetAnimationPoint(atoi(args[0].c_str()));
				// Animation file is now out of date
				pv.UnloadAnimationFile();
				currentCommandLine = "output: point saved at time: " + args[0];
			}
		}	

		void GUICommand::RemoveAnimPoint(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 0)
				currentCommandLine = "error: remove point takes 0 arguments";
			else
			{	
				pv.RemoveAnimationPoint();
			}
		}	

		void GUICommand::PreviewAnim(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 0)
				currentCommandLine = "error: preview takes 0 arguments";
			else
			{	
				//pv.Interpolate();
				//pv.PreviewAnimation();
				currentCommandLine = "error: animation system currently being worked on";
			}			
		}

		void GUICommand::SaveAnimFile(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: save animation takes 1 string argument";
			else
			{
				//pv.Interpolate();
			 	//pv.SaveAnimationFile(arg[0]);
			 	currentCommandLine = "error: animation system currently being worked on";
			}	
		}

		void GUICommand::LoadAnimFile(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: load animation takes 1 string argument";
			else
			{
				//pv.LoadAnimationFile(arg[0]);
				currentCommandLine = "error: animation system currently being worked on";
			}				
		}

		void GUICommand::SetCamInterpolation(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set camera interpolation takes 1 string argument";
			else
			{
				if(args[0] == "linear")
				{
					pv.SetCameraInterpolation(LINEAR);
				}
				else if(args[0] == "cubic")
				{
					pv.SetCameraInterpolation(CUBIC);
				}
				else 
				{
					currentCommandLine = "error: Camera interpolation type unrecognised (linear or cubic)";
				}
			}	
		}

		void GUICommand::SetLookatInterpolation(Previewer& pv, std::vector<std::string> args)
		{
			if(args.size() != 1)
				currentCommandLine = "error: set lookat interpolation takes 1 string argument";
			else
			{
				// Allow you to set lookat interpolation type
				if(args[0] == "linear")
				{
					pv.SetLookatInterpolation(LINEAR);
				}
				else if(args[0] == "cubic")
				{
					pv.SetLookatInterpolation(CUBIC);
				}
				else 
				{
					currentCommandLine = "error: Lookat interpolation type unrecognised (linear or cubic)";
				}
			}				
		}

	}
}