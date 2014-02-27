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

#include "OnKeyReleaseEvent.h"

namespace previewer
{
	// Declare the static list
	std::list<OnKeyReleaseEvent*>* OnKeyReleaseEvent::eventSubs = 0;

	// Implement the constructor
	OnKeyReleaseEvent::OnKeyReleaseEvent()
	{
		DebugPrint("OnKeyRelease Event Subscription");

		if(!eventSubs)
		{
			eventSubs = new std::list<OnKeyReleaseEvent*>();
		}

		// Add the subclass to the subscribers list
		eventSubs->push_back(dynamic_cast<OnKeyReleaseEvent*>(this));
	}

	// Implement the destructor
	OnKeyReleaseEvent::~OnKeyReleaseEvent()
	{
		DebugPrint("OnKeyRelease Event Unsubscription");
		// Remove the subclass from the subscribers list
		eventSubs->remove(dynamic_cast<OnKeyReleaseEvent*>(this));
	}

	// Implement the Caller
	void OnKeyReleaseEvent::CallEvent(Event ev)
	{
		//DebugPrint("OnKeyRelease Event Call");

		if(eventSubs)
		{
			for(std::list<OnKeyReleaseEvent*>::iterator it = eventSubs->begin(); it!= eventSubs->end(); ++it)
			{
				// Call the OnKeyRelease implementation of the sub class
				//*it->OnKeyRelease(ev);
				OnKeyReleaseEvent* thisEvent = dynamic_cast<previewer::OnKeyReleaseEvent*>(*it);
				thisEvent->OnKeyRelease(ev);
			}
		}
	}
}