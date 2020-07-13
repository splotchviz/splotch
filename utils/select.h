#ifndef TESTBED_COMMON_SELECT_H
#define TESTBED_COMMON_SELECT_H

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
 * 		Tim Dykes 
 *

	File: select.h
	Purpose: Implementation of quick select, hoares selection algorithm for single and multidimensional data
	NOTE: 	Multidimensional data currently must be accessible by subscript operator
			Srand is not initialised in here!

 */

#include "partition.h"
#include "debug.h"
/*

finding nth element, left is left side of array (0), right is right side of array (arr.size()-1)
qselect(data, left, right, n)
check for left = right, i.e. only 1 element

while(1)
	pivotIndex = rand() between left and right
	new pivotIndex = partition by pivotIndex
	if pivotindex is n, break and return it (or in fact the value...) 
	else 
		if pivotindex > n, then n is on left
		make new right side of array pivotIndex-1 (because pivotIndex is in sorted place now)
		else n is on right
		make new left side pivotIndex +1 
		continue while loop to try again

*/

// Quickselect nth element of type T (single dimension)
template<typename T>
T qselect(T* data, int left, int right, int n)
{
	// Out of bounds
	if(n > right || n < left)
	{
		// there is no nth element!
		ErrorMessage("qselect() out of bounds: element %i left boundary: %i right boundary %i\n", n, left, right);
	}
	// We only have one element
	if(left == right)
		return data[left];
	
	//srand()
	int pivotIndex;
	while(1)
	{
		pivotIndex = left + floor(rand()%(right-left+1));
		pivotIndex = partition(data, left, right, pivotIndex);
		if(pivotIndex == n)
		{
			return data[pivotIndex];
		}
		else
		{
			if(pivotIndex > n)
				right = pivotIndex-1;
			else 
				left = pivotIndex+1;
		}
	} 
}

// Quickselect nth element from multidimensional type
template<typename T>
T qselect(T* data, unsigned int left, unsigned int right, unsigned int n, unsigned int dim)
{
	// Out of bounds
	if(n > right || n < left)
	{
		// there is no nth element!
		ErrorMessage("qselect() out of bounds: element %i left boundary: %i right boundary %i\n", n, left, right);
	}
	// We only have one element
	if(left == right)
		return data[left];
	
	//srand()
	unsigned int pivotIndex; 
	while(1)
	{
		pivotIndex = left + floor(rand()%(right-left+1));
		pivotIndex = partition(data, left, right, pivotIndex, dim);
		if(pivotIndex == n)
		{
			return data[pivotIndex];
		}
		else
		{
			if(pivotIndex > n)
				right = pivotIndex-1;
			else 
				left = pivotIndex+1;
		}
	} 

}

// Quickselect nth element from single dimensional type using predicate
template<typename T, typename OP>
T qselect(T* data, unsigned int left, unsigned int right, unsigned int n, OP compare)
{
	// Out of bounds
	if(n > right || n < left)
	{
		// there is no nth element!
		ErrorMessage("qselect() out of bounds: element %i left boundary: %i right boundary %i\n", n, left, right);
	}
	// We only have one element
	if(left == right)
		return data[left];
	
	//srand()
	unsigned int pivotIndex; 
	while(1)
	{
		pivotIndex = left + floor(rand()%(right-left+1));
		pivotIndex = partition(data, left, right, pivotIndex, compare);
		if(pivotIndex == n)
		{
			return data[pivotIndex];
		}
		else
		{
			if(pivotIndex > n)
				right = pivotIndex-1;
			else 
				left = pivotIndex+1;
		}
	} 

}

#endif