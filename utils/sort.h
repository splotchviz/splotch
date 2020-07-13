#ifndef TESTBED_SORT_H
#define TESTBED_SORT_H

#include "utility.h"
#include <queue>
#include <vector>

/*
	Bubble sort
	
*/

template<typename T, typename OP>
void sort(T* in, int size, OP my_op)
{
	int nb = 1;
	while(nb>0)
	{
		nb = 0; 
		for(int i = 0; i < size-1; i++)
		{
			if(my_op(in[i+1], in[i]))
			{
				tbd::swap2(in[i+1], in[i]);
				nb++;
			}
		}
	}
}


/*
	Topological sort
	Pass in ptr to data, size of data, topological comparison operator.
	Topological comparison operator should return 1/true if p1->p2 dependent
	Data must be directed acyclic graph, any cycles will cause an error.
*/

template<typename T, typename OP>
void toposort(T* in, int size, OP topo_depend)
{
	std::queue<int> to_sort;
	std::vector<T> sorted;

	// IDs of dependent vertices (with edges pointing into inedges[vertex_id])
	std::vector<std::vector<int> > inedges(size);
	// Number of vertex dependencies (edges pointing out)
	std::vector<int> outedges(size, 0);
 	
 	for(int i = 0; i < size; i++)
 		for(int j = i+1; j < size; j++)
 		{
 			int ret = topo_depend(in[i], in[j]);
 			if(ret)
 			{
 				inedges[j].push_back(i);
 				outedges[i]++;
 			}

 			ret = topo_depend(in[j], in[i]);
 			if(ret)
 			{
 				inedges[i].push_back(j);
 				outedges[j]++;
 			}

 		}

 	// Put independent nodes into sort queue
 	for(unsigned i = 0; i < outedges.size(); i++)
 		if(!outedges[i])
 			to_sort.push(i);

 	// Graph is not acyclic, theres no independent nodes
 	if(!to_sort.size())
 	{
 		// Some error (maybe something more understandable...)
 		// ErrorMessage("Error: Cannot toposort cyclic graph\n");
 		std::cout << "Toposort error: no independent nodes\n";
 		return;
 	}

 	// Sort
 	while(to_sort.size())
 	{
 		// Get next independent node
 		int next = to_sort.front();
 		to_sort.pop();
 		// Remove any outgoing edges pointing to this node
 		for(unsigned i = 0; i < inedges[next].size(); i++)
 		{
 			outedges[inedges[next][i]] -= 1;
 			// If that was the last outgoing edge from node i, add node i to sort list
 			if(!outedges[inedges[next][i]])
 				to_sort.push(inedges[next][i]);
 		}
 		// Add to sorted list
 		sorted.push_back(in[next]);
 	}

 	// If any edges are left, the graph had at least one cycle, and the sort will be incorrect
 	for(int i = 0; i < size; i++)
 	{
 		if(outedges[i])
 		{
 			// ErrorMessage("Error: Cannot toposort cyclic graph\n");
 			std::cout << "Toposort error: graph had cycles\n";
 			return;
 		}
 	}

 	// Replace data with sorted list
 	for(int i = 0; i < size; i++)
 	{
 		in[i] = sorted[i];
 	}
}	

/*
	Insertion sort
	 for i = 1 to length(A) - 1
	    x = A[i]
	    j = i
	    while j > 0 and A[j-1] > x
	        A[j] = A[j-1]
	        j = j - 1
	    end while
	    A[j] = x
	 end for

	 		int j;
		for(int i = 1; i < k; ++i)
		{
			temp = knn[i];
			j = i;
			while(j >= 0 && compare(knn[--j], temp))
			{
				knn[j+1] = knn[j];
			}
			knn[++j] = temp;
		}
		
*/

/*
pair<gas_particle, float> temp;
std::list<pair<gas_particle, float> >::iterator jit;
std::list<pair<gas_particle, float> >::iterator jitm1;
for(it = knn.begin()++; it != knn.end(); ++it)
{
	temp = *(it);
	jit = it;
	jitm1 = jit;
	jitm1--;
	while(jit != knn.begin() && compare(*(jitm1), temp))
	{
		*jit = *jitm1;
		jit = jitm1--;
	}
	*jit = temp;
}
*/

/*
	Quicksort

*/


/*
	Mergesort

*/


/*
	Heapsort

*/



/*
	Sort by key

*/
/*
template <typename V, typename K, template<typename A, typename B> bool (*predicate)(A, B) >
void sort_by_key(V values, K keys, P predicate)
{

}
*/
#endif
