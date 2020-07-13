#ifndef ASSIST_H
#define ASSIST_H

// Swap
#include "utility.h" 

template <typename T>
void filter_medians(int* active_lookup, T* medians, int nt, int& n_active)
{
	//Print("filtering medians: nt: %i n_active: %i\n", nt, n_active);
	int idx = nt-1; 
	int i = 0;

	// Move inactive medians to end of the array
	while(i<=idx)
	{
		if(active_lookup[i] == 0)
		{
			tbd::swap2(medians[i], medians[idx]);
			tbd::swap2(active_lookup[i], active_lookup[idx]);
			idx--;
		}
		else 
			i++;
	}
	n_active = i;
}

#endif