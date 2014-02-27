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

#include "OnKeyPressEvent.h"

namespace previewer
{
	// Declare the static list
	std::list<OnKeyPressEvent*>* OnKeyPressEvent::eventSubs = 0;

	// Implement the constructor
	OnKeyPressEvent::OnKeyPressEvent()
	{
		DebugPrint("OnKeyPress Event Subscription");

		if(!eventSubs)
		{
			eventSubs = new std::list<OnKeyPressEvent*>();
		}

		// Add the subclass to the subscribers list
		eventSubs->push_back(dynamic_cast<OnKeyPressEvent*>(this));
	}

	// Implement the destructor
	OnKeyPressEvent::~OnKeyPressEvent()
	{
		DebugPrint("OnKeyPress Event Unsubscription");

		// Remove the subclass from the subscribers list
		eventSubs->remove(dynamic_cast<OnKeyPressEvent*>(this));
	}

	// Implement the Caller
	void OnKeyPressEvent::CallEvent(Event ev)
	{
		//DebugPrint("OnKeyPress Event Call");

		if(eventSubs)
		{
			for(std::list<OnKeyPressEvent*>::iterator it = eventSubs->begin(); it!= eventSubs->end(); ++it)
			{
				// Call the OnKeyPress implementation of the sub class
				//*it->OnKeyPress(ev);
				OnKeyPressEvent* thisEvent = dynamic_cast<previewer::OnKeyPressEvent*>(*it);
				thisEvent->OnKeyPress(ev);
			}
		}
	}
}