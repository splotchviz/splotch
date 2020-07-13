#ifndef TESTBED_COMMON_PARALLEL_SELECT_H
#define TESTBED_COMMON_PARALLEL_SELECT_H

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

	File: parallel_select.h
	Purpose: MPI parallel implementation of quick select, hoares selection algorithm for single and multidimensional data
	NOTE: 	srand is not initialised, commented out debug code could be hash deffed as debug build

 */

#include <mpi.h>
#include "select.h"
#include "sort.h"
#include "assist.h"
#include "get_mpi_datatype.h"

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

// If we dont provide a comparison operator, just use standard less than 
template<typename T>
T parqselect(T* data, int left, int right, int n, int rank, int root, int comm_size)
{
	tbd::less_than_op<T> ltop;
	return parqselect(data, left, right, n, rank, root, comm_size, ltop);
}

// Parallel quickselect
template<typename T, typename OP>
T parqselect(T* data, int left, int right, int n, int rank, int root, int comm_size, OP compare)
{
	/*
		take left and right as references because
		median wont be in the middle for parallel version...
	*/

	T selection;
	int T_nbytes = sizeof(T);

	// All tasks start off active
	int task_active = 1;
	// If we didnt have any data, deactivate
	if(data == NULL)
		task_active = 0;

	int n_tasks = comm_size;
	int n_active = comm_size;
	int* active_tasks = NULL;
	T* medians = NULL;

	// First check if we have only 1 rank in our communicator, if so do it serially.
	if(comm_size == 1)
	{
		return qselect(data,left,right,n,compare); 
	}

	// Root processor allocates memory and sets up the active task list
	if(rank==root) 
	{
		medians = new T[comm_size];
		// Get active flags from all tasks, count active, broadcast back to tasks
		active_tasks = new int[comm_size];
	}

	MPI_Gather(&task_active, 1, MPI_INT, active_tasks, 1, MPI_INT, root, MPI_COMM_WORLD);

	//if(rank==root)
	//{
//		Print("qselect(): selecting element %i\n", n);
//	}

	int lcount = 0;
	int rcount = 0;
	int discarded_left = 0;
	int discarded_right = 0;
	int total_l = 0;
	int total_r = 0;
	int local_index;

	while(1)
	{
		/*
			should be 0 not 1 if data == null
		*/
		// How much data does this task have
		int local_size = (right-left)+1;
		int local_nbytes = local_size * T_nbytes;
		if(data == NULL) local_size = 0;
		int global_size = 0;
		// Sum data from tasks, if less than ntasks squared collect data and sum discarded left
		// Then sort, and return element [n-discarded_left_sum]

		// How much data do all tasks have
		MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//
		// Not much? do in serial
		//

		if(global_size < (n_tasks*n_tasks))
		{
			//Print("Global Size: %i, sq(n_tasks): %i, collect and serial compute\n", global_size, n_tasks*n_tasks);
			// Find global 'left'
			int global_left = 0;
			MPI_Reduce(&discarded_left, &global_left, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);

			// Gather data on root
			// First allocate some memory
			int* byte_offsets = NULL;
			T* gathered_data = NULL;
			if(rank == root)
			{
				byte_offsets = new int[n_tasks];
				gathered_data = new T[global_size];
				//for(unsigned i = 0; i < global_size; i++) <----?
				//	gathered_data[i] = -1;
				//Print("GLOBAL LEFT: %i n-global_left: %i\n", global_left, n-global_left);
			}
			// Get local sizes
			MPI_Gather(&local_nbytes, 1, MPI_INT, byte_offsets, 1, MPI_INT, root, MPI_COMM_WORLD);

			// Work out offsets with exclusive scan
			int* displs = NULL;
			if(rank==root)
			{
				displs = new int[n_tasks];
				// in, out, size, op, value[0]
				exclusive_scan(byte_offsets, displs, n_tasks, tbd::sum_op<int>(), 0);
				// for(unsigned i = 0; i < n_tasks; i++)
				// {
				// 	Print("byte_offsets[%i]: %i \nExscan[%i]: %i\n", i, byte_offsets[i], i, displs[i]);
				// }
			}
			//for(unsigned i = 0; i < local_size; i++)
			//{
			//	Print("Rank %i data[%i]: %f\n", rank, i, data[left+i]);
			//}
			// Now gather
			/*
				Test for data==null
			*/
			MPI_Gatherv(&data[left],local_nbytes, MPI_BYTE, gathered_data, byte_offsets, displs, MPI_BYTE, root, MPI_COMM_WORLD);

			// Sort and select global median
			if(rank==root)
			{
				sort(gathered_data, global_size, compare);
				selection = gathered_data[n-global_left];
				// if(selection == -1)
				// {
				// 	for(unsigned i = 0; i < global_size; i++)
				// 		Print("Gathered data[%i]: %f\n",i,gathered_data);
				// }
				delete[] byte_offsets;
				delete[] gathered_data;
				delete[] displs;
			}

			// Sent to all processes
			MPI_Bcast(&selection, T_nbytes, MPI_BYTE, root, MPI_COMM_WORLD);
		
			break;
		}

		//
		// Otherwise pivot and partition to reduce data
		//

		// Find median 
		int half = left+(local_size/2);
		T median;
		T pivot;
		if(task_active)
		{
			median = qselect(data, left, right, half, compare);
			//Print("Rank %i median: %f\n", rank, median);
		}

		// Gather medians from all processors to root
		MPI_Gather(&median, T_nbytes, MPI_BYTE, medians, T_nbytes, MPI_BYTE, root, MPI_COMM_WORLD);
		if(rank==root)
		{
			//Print("Filter medians\n");
			// Get count of active tasks and filter inactive task medians
			filter_medians(active_tasks, medians, n_tasks, n_active);
		}

		// Root gets median of medians (pivot value) for all active tasks
		if(rank==root)
		{
			// for(unsigned i = 0; i < n_active; i++)
			// {
			// 	Print("Rank 0 median %i: %f\n", i, medians[i]);
			// }
			// Print("Rank 0 finding median %i of size %i\n", n_active/2, n_active);
			pivot = qselect(medians, 0, n_active-1, n_active/2, compare);
			//Print("Rank 0 median of medians: %f\n", pivot);

		}

		// Broadcast pivot value to all 
		MPI_Bcast(&pivot, T_nbytes, MPI_BYTE, root, MPI_COMM_WORLD);
		//Print("Rank %i pivot: %f\n", rank, pivot);

		int local_l_total;
		int local_r_total;
		if(task_active)
		{
			// Partition based on pivot & work out left & right counts
			//Print("Rank %i: size: %i left: %i right %i pivot %f\n", rank, local_size, left, right, pivot);
			local_index = partition_by_value(data, left, right, pivot, compare);
			lcount = (local_index-left);
			rcount = (local_size-lcount);
			local_l_total = lcount + discarded_left;
			local_r_total = rcount + discarded_right;
		}
		else
		{
			local_l_total = discarded_left;
			local_r_total = discarded_right;

		}

		//Print("Rank %i: lcount: %i rcount %i\n", rank, lcount, rcount);
		//Print("Rank %i: local_l_total: %i local_r_total %i\n", rank, local_l_total, local_r_total);

		// Reduce left and right count
		MPI_Allreduce(&local_l_total, &total_l, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_r_total, &total_r, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//if(rank==root)
		//{
		//	Print("Rank 0: total_l: %i total_r %i\n", total_l, total_r);
		//}

		// Did we find the median?
		int global_index = total_l;
		if(global_index == n)
		{
			// Who had it (unnecessary?)
			//if(task_active && pivot == median)
			//{
				// 
				//Print("Rank %i found element %i: %f at location %i\n", rank, n, pivot, left+half);
				//MPI_Bcast(&pivot, 1, MPI_FLOAT, root, MPI_COMM_WORLD);
			//}
			//Print("GLOBAL INDEX IS %i, return pivot: %f\n", global_index, pivot);
			selection = pivot;
			break;
		}

		// Didnt find median, discard smaller group
		if(task_active)
		{
			if(global_index > n)
			{
				// Continue with left side
				right -= rcount;
				discarded_right += rcount;
			}
			else
			{
				// Continue with right side
				left += lcount;
				discarded_left += lcount;
			}

			// Check whether we discarded all the processes data
			if(left>right)
			{
				// Remove from group
				//Print("Rank %i: deactivating with left: %i, right %i\n", rank, left, right);
				task_active = 0;
			}
			local_size = (right-left)+1;
		}

		//if(task_active)
		//{
		//Print("Post-loop Rank %i: size: %i left: %i right %i pivot %f\n", rank, local_size, left, right, pivot);
		//}

		// Update list of active tasks
		MPI_Gather(&task_active, 1, MPI_INT, active_tasks, 1, MPI_INT, root, MPI_COMM_WORLD);
	}

	// Clear up
	if(rank==root)
	{
		delete[] medians;
		delete[] active_tasks;
	}

	return selection;
}




#endif