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

#include "OnMotionEvent.h"

namespace previewer
{
	// Declare the static list
	std::list<OnMotionEvent*>* OnMotionEvent::eventSubs = 0;

	// Implement the constructor
	OnMotionEvent::OnMotionEvent()
	{
		DebugPrint("OnMotion Event Subscription");

		if(!eventSubs)
		{
			eventSubs = new std::list<OnMotionEvent*>();
		}

		// Add the subclass to the subscribers list
		eventSubs->push_back(dynamic_cast<OnMotionEvent*>(this));
	}

	// Implement the destructor
	OnMotionEvent::~OnMotionEvent()
	{
		DebugPrint("OnMotion Event Unsubscription");
 
		// Remove the subclass from the subscribers list
		eventSubs->remove(dynamic_cast<OnMotionEvent*>(this));
	}

	// Implement the Caller
	void OnMotionEvent::CallEvent(Event ev)
	{
		//DebugPrint("OnMotion Event Call");

		if(eventSubs)
		{
			for(std::list<OnMotionEvent*>::iterator it = eventSubs->begin(); it!= eventSubs->end(); ++it)
			{
				// Call the OnButtonPress implementation of the sub class
				OnMotionEvent* thisEvent = dynamic_cast<previewer::OnMotionEvent*>(*it);
				thisEvent->OnMotion(ev);
			}
		}
	}
}
