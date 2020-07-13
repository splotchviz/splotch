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

	File: partition.h

 */


#ifndef TESTBED_COMMON_PARTITION_H
#define TESTBED_COMMON_PARTITION_H

// Swap
#include "utility.h" 

/*
partition 
get pivot value from data
move it to right hand side
get storeindex (== left)
for each element in data (not including right)
	if lower than pivotvalue
	swap2 to store index, increment store index
swap2 pivot value to storeindex
*/

template<typename T>
int partition(T* data,  int left,  int right,  int pivotIndex)
{
	if(right <= left)
		return left;

	T pivotValue = data[pivotIndex];
	tbd::swap2(data[pivotIndex], data[right]);
	 int storeIndex = left;
	for( int i = left; i < right; i++)
	{
		if(data[i] < pivotValue)
			tbd::swap2(data[i], data[storeIndex++]);
	}
	tbd::swap2(data[right],data[storeIndex]);
	return storeIndex;
}

// for multidimensional data
template<typename T>
int partition(T* data,  int left,  int right,  int pivotIndex,  int dim)
{
	if(right <= left)
		return left;

	T holder;
	holder[dim] = data[pivotIndex][dim];
	tbd::swap2(data[pivotIndex], data[right]);
	 int storeIndex = left;
	for( int i = left; i < right; i++)
	{
		if(data[i][dim] < holder[dim])
			tbd::swap2(data[i], data[storeIndex++]);
	}
	tbd::swap2(data[right],data[storeIndex]);
	return storeIndex;
}

// Partition with comaprison predicate
template<typename T, typename OP>
int partition(T* data,  int left,  int right,  int pivotIndex, OP compare)
{
	if(right <= left)
		return left;

	T pivotValue = data[pivotIndex];
	tbd::swap2(data[pivotIndex], data[right]);
	 int storeIndex = left;
	for( int i = left; i < right; i++)
	{
		if(compare(data[i],pivotValue))
			tbd::swap2(data[i], data[storeIndex++]);
	}
	tbd::swap2(data[right],data[storeIndex]);
	return storeIndex;
}


// Partition on pivot value instead of index
template<typename T>
int partition_by_value(T* data,  int left,  int right, T val)
{
	int storeIndex = left;
	for(int i = left; i <= right; i++)
	{
		if(data[i] < val)
		{
			tbd::swap2(data[i], data[storeIndex++]);
		}
	}
	return storeIndex;
}


// Partition on pivot value instead of index with comaprison predicate
template<typename T, typename OP>
int partition_by_value(T* data, int left, int right, T val, OP compare)
{
	int storeIndex = left;
	for(int i = left; i <= right; i++)
	{
		if(compare(data[i],val))
		{
			tbd::swap2(data[i], data[storeIndex++]);
		}
	}
	return storeIndex;
}

#endif