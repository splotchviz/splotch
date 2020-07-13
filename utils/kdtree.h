#ifndef KD_TREE_H
#define KD_TREE_H

// For quickselect
#ifdef USE_MPI
#include "parallel_select.h"
#else
#include "select.h"
#endif

// For bounding box when finding longest dimension
#include "box.h"
// For list and pair used in finding k nearest neighbours
#include "utility.h"
#include <list>
// For filename manipulation and output
#include <string>
#include <sstream>
#include <fstream>
#include <climits>
#include <cstring>

// TODO: Have difference functions that dont rely on returning a float
// e.g. return type T with values being differences, can then use larger than predicate 
// to decide which T is larger

// Node for KD tree
template<typename T, int NDIM>
class KDnode{
public:
	// Dimension & location of split
	// -1 for leaf nodes
	short           dim;
	T               loc;
	short           depth;

	T*              data;
	
	// Ghost elements (overlapping)
	// T* ghosts 

	// Size of data in this node on this rank
	unsigned int    local_size;
	// Size of data in this node on all ranks 
	// (in serial local == global)
	unsigned int    global_size;

	// Stores IDs of leaves from 0 -> nleaves 
	// left to right 
	unsigned int 	leaf_id;

	// Store bounding box
	Box<T,NDIM>     bbox;

	// Do we own our data?
	bool            owner;
	// Parent
	KDnode*         parent;
	// Children
	union {
		KDnode*     sons[2];
		struct{
			KDnode* left;
			KDnode* right;
		};      
	};

	// Ghosts
	std::vector<T> ghosts;
	// Number of ghosts merged into data
	unsigned int merged_ghosts;

	KDnode()
	{
		parent      = NULL;
		left        = right = NULL;
		data        = NULL;
		dim         = 0;
		local_size  = 0;
		global_size = 0;
		owner       = false;
		merged_ghosts = 0;
	}
	KDnode(T* inp, unsigned int inlocal_size, Box<T,NDIM>& bb, int inglobal_size = -1, short indepth = -1, KDnode* inparent = NULL,bool own = false)
	{
		left        = right = NULL;
		data        = inp;
		local_size  = inlocal_size;
		global_size = (inglobal_size > -1) ? inglobal_size : inlocal_size;
		depth       = indepth + 1;
		dim         = 0;
		parent      = inparent;
		owner       = own;
		bbox        = bb;
	}
};

template<typename T, int NDIM, typename ACCESSOR_TYPE = float>
class KDtree{
private: 
	// Function pointer for finding the difference in some dimension of T
	//typedef         float   (*Difference_function)(T, T);
	// Function pointer for smaller than predicate
	typedef         bool    (*Smaller_than_function)(const T&,const  T&);
	typedef 		ACCESSOR_TYPE (*Accessor_function)(const T& accessee);
	typedef 		void (*Ghost_setter_function)(T&);
	
	// Disable copy and assignment
	KDtree(const KDtree&) = delete;
	const KDtree& operator=(const KDtree&) = delete;
	
public: 
	
	KDnode<T, NDIM>*        root;
	unsigned short  max_depth;
	unsigned long   max_size_leaf; 
	unsigned long   largest_leaf;
	unsigned long   smallest_leaf;
	short           depth;
	unsigned int    nleaves;
	unsigned int    nNodes;
	bool            active;
	bool            split_longest_axis;
	Box<T,NDIM>     bbox;

	bool 			ghosts;
	int 			ghost_depth;

	// MPI
	unsigned int    mpi_root;
	unsigned int    mpi_rank;
	unsigned int    comm_size;

	// Predicates
	std::vector<Smaller_than_function>   smaller_than;

	// Accessors 
	std::vector<Accessor_function> access;
	//Ghost_setter_function set_ghost;

	// Does the tree (or any node within) own data
	bool owner;

	KDtree()
	{
		root            = NULL;
		max_depth       = 0;
		max_size_leaf   = 0;
		largest_leaf    = 0;
		smallest_leaf   = ULONG_MAX; 
		depth           = 0;
		nleaves         = 0;
		nNodes          = 0;
		active              = false;
		split_longest_axis  = false;
		mpi_rank        = 0;
		mpi_root        = 0;
		comm_size       = 1;
		ghosts 			= false;
		ghost_depth 	= -1;
		owner 			= false;
	}

	~KDtree()
	{
		if(active)
		{
			destroy(root);
		}
	}

/*	
	const KDtree& operator=( const KDtree& copyee)
	{
	
		// If we werent active, allocate a root node
		if(!active)
		{
			root = new KDNode<T,NDIM>();
			active=true;
		}

		// If the copyee owns its data
		if(copyee.owner)
		{
			if copyer is existing already
			{
				Check tree capacity
					if smaller
						Clear tree 
						Re-allocate memory for full size tree
			}
			
		 	Copy all data
		}
		
		Traverse both trees
		  if(copyeenode.owner)
		  {  
		    assert(copyeenode.leaf)
		    allocate memory for node data 
		    copy node data
		  }
		  else
		  {
			update pointer via offsetting from source
		  }

		  copy node information		

		  if copyee extends further but copyer doesnt, make new children
		  if copyer extends further but copyee doesnt, delete copyer children

		copy tree information
		set owner true


	}
*/
	void split_by_longest_axis()
	{
		split_longest_axis = true;
	}

	void split_by_iteration()
	{
		split_longest_axis = false;
	}

	void set_max_leaf(unsigned long m)
	{
		max_size_leaf = m;
	}

	void set_max_depth(short m)
	{
		max_depth = m;
	}

	void add_smaller_than_function(bool (*func)(const T&, const T&))
	{
		smaller_than.push_back(func);
	}

	void add_accessor(Accessor_function func)
	{
		access.push_back(func);
	}

	void set_ghosts(bool onoff/*, Ghost_setter_function gsf*/, int maxghostdepth = -1)
	{
		ghosts = onoff;
		if(onoff)
		{
			ghost_depth = maxghostdepth;
			//set_ghost = gsf;
		}
	}

	void build(T* inp, unsigned int local_size, bool own = false, unsigned int inroot = 0, unsigned int inrank = 0, int incomm_size = 1)
	{
		// Check if we have enough predicates to build
		if(smaller_than.size() < NDIM)
		{
			ErrorMessage("KDTree build(): error, need to add predicates for each dimension with add_smaller_than_function().\nNumber of dimensions: %i, number of functions: %i\n", NDIM, smaller_than.size());
		}

		if((access.size() < NDIM && split_longest_axis) || (access.size() < NDIM+2 && ghosts))
		{
			ErrorMessage("KDTree build(): error, need to add accessors via add_accessor().\nNumber of dimensions: %i, number of accessors: %i\n", NDIM, access.size());
		}

#ifdef USE_MPI
		// Store MPI info
		mpi_root = inroot; 
		mpi_rank = inrank;
		comm_size = incomm_size;

		/* 
			Work out global size and pass this too 
		*/
		unsigned int global_size;
		MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		getbox(bbox, inp, local_size, &smaller_than[0], mpi_rank, mpi_root, comm_size);
		root = new KDnode<T, NDIM>(inp, local_size, bbox, global_size, -1, NULL, own);
#else
		getbox(bbox, inp, local_size, &smaller_than[0]);
		root = new KDnode<T, NDIM>(inp, local_size, bbox); // own should be here

#endif

		build(root);

		active = true;
		root->owner = own;
	};

#ifdef USE_MPI
	// Redistribute the data evenly across ranks
	// If owner==true we delete the original data
	void redistribute_data()
	{

		// Nothing to do if only one rank
		if(comm_size < 2)
		{
			Print("KDTree redistribute_data(): comm_size %i, nothing to do...\n", comm_size);
			return; 
		}

		// Get list of all nodes at the right level 
		// Max depth that has comm_size or more nodes
		int target_level = ceil(log2(comm_size));
		if(target_level != max_depth && target_level != max_depth-1)
			ErrorMessage("KDTree redistribute_data(): error, target level (%i) for comm_size (%i), can only redistribute at max depth (%i) of tree, max_depth should be equal to log2(comm_size) or (log2(comm_size)+1)\n", target_level, comm_size, max_depth);

		std::vector<KDnode<T,NDIM>*> nodes;
		find_all_nodes_at_level(target_level, nodes);
		
		// Check we found some nodes
		// In the case where we have more ranks than 2^max_depth, we will get this error
		// Should then increase max depth of tree...
		if(!nodes.size())
			ErrorMessage("KDTree redistribute_data(): error, couldnt find any nodes at level %i\n", target_level);

		// Allocate either one or two nodes to ranks
		int my_share = 1;
		int spare = nodes.size() % comm_size;
		if(mpi_rank < spare)
			my_share++;

		// Now redistribute
		// Receive and displacement counts for redistribution
		int* recvs  = new int[comm_size];
		int* displs = new int[comm_size];

		int* recvs_ghost  = new int[comm_size];
		int* displs_ghost = new int[comm_size];

		// Node sharing
		unsigned new_size[my_share]{};
		unsigned merged_ghost_size[my_share]{};
		T* new_data[my_share];
		int share_ctr = 0;

		// Each rank simply gets all data for node[rankID]
		// For each tree node, mpi gatherv data from all other tasks
		for(int nid = 0; nid < nodes.size(); nid++)
		{           
			// Ascertain recieving rank (max 2 nodes per rank)
			int receiver = (nid-spare < spare) ? nid/2 : nid-spare;

			// Get local sizes and ghostcounts
			int local_ghost = nodes[nid]->ghosts.size();
			MPI_Allgather(&nodes[nid]->local_size, 1, MPI_INT, recvs, 1, MPI_INT, MPI_COMM_WORLD);
			MPI_Allgather(&local_ghost, 1, MPI_INT, recvs_ghost, 1, MPI_INT, MPI_COMM_WORLD);

			// Get total size, scale recvs to bytecount
			int current_size = 0;
			for(unsigned i = 0; i < comm_size; i++)
			{
				current_size += recvs[i];
				recvs[i] *= sizeof(T);
			}

			// Get total ghosts, scale recvs_ghost to bytecount
			int current_ghosts = 0;
			for(unsigned i = 0; i < comm_size; i++)
			{
				current_ghosts += recvs_ghost[i];
				recvs_ghost[i] *= sizeof(T);
			}

			int total_data = current_size + current_ghosts;

			// Is there any data to send?
			if(total_data)
			{
				// Is this rank getting the data?
				if(receiver==mpi_rank)
				{
					// + ghosts ?
					new_data[share_ctr] = new T[current_size+current_ghosts];
					new_size[share_ctr] = current_size;
					merged_ghost_size[share_ctr] = current_ghosts;
				}

				if(current_size)
				{
					// Work out displacements with exclusive scan
					exclusive_scan(recvs, displs, comm_size, tbd::sum_op<int>(), 0);

					// MPI gather data to rank[nid]
					MPI_Gatherv(nodes[nid]->data, nodes[nid]->local_size*sizeof(T), MPI_BYTE, new_data[share_ctr], recvs, displs, MPI_BYTE, receiver, MPI_COMM_WORLD);					
				}

				if(current_ghosts)
				{
					// Work out displacements with exclusive scan, offsetting by the non-ghost data
					exclusive_scan(recvs_ghost, displs_ghost, comm_size, tbd::sum_op<int>(), int(current_size*sizeof(T)));

					// MPI gather data to rank[nid]
					MPI_Gatherv(&nodes[nid]->ghosts[0], local_ghost*sizeof(T), MPI_BYTE, new_data[share_ctr], recvs_ghost, displs_ghost, MPI_BYTE, receiver, MPI_COMM_WORLD);					
				}

				if(receiver==mpi_rank)
					share_ctr++;
			}
		}

		// Clear local data from all nodes
		clear_local(root);

		// Did we get new nodes?
		int recieved_data = false;
		for(int i = 0; i < my_share; i++)
		{
			if(new_size[i] || merged_ghost_size[i])
			{
				
				int node_id = mpi_rank + i;
				// After nspare, ranks have nodeID = rankID + nspare at target_level
				if(mpi_rank >= spare) 
					node_id += spare;

				replace_local(node_id, target_level, new_data[i], new_size[i], merged_ghost_size[i]);
				recieved_data = true;
			}			
		}

		// If we had data, tree should become owner
		owner = true;
		delete[] recvs;
		delete[] displs;

	}
#else
	void redistribute_data(){ }
#endif

	Box<ACCESSOR_TYPE, NDIM> node_box_to_raw(KDnode<T,NDIM>* node)
	{
		Box<ACCESSOR_TYPE, NDIM> box;
		for(unsigned i = 0; i < NDIM; i++)
		{
			box.min[i] = (*access[i])(node->bbox.min[i]); 
			box.max[i] = (*access[i])(node->bbox.max[i]); 
		}
		return box;
	}

	// Cycle through the tree, invalidate all node local data pointers
	void clear_local(KDnode<T,NDIM>* next)
	{
		if(next->left)
		{
			clear_local(next->left);
			clear_local(next->right);
		}

		// Clear any ghosts forcing deallocation
		if(next->ghosts.size())
			std::vector<T>().swap(next->ghosts);

		// Clear local data
		if(next->data)
		{
			if(next->owner)
				delete[] next->data;
			next->data = NULL;
			next->local_size = 0; 
			next->merged_ghosts = 0;          
		}
	}

	void replace_local(int nid, int level, T* data, int size, int merged_ghost_size)
	{
		
		int node_count = pow(2, level);
		KDnode<T,NDIM>* current_node = root;
		int current_level = 0;
		if(level > max_depth)
			ErrorMessage("KDTree replace(): error, cant replace node at level %i in tree of max depth %i\n", level, max_depth);

		// Find node at level 
		while(current_level < level)
		{
			if(nid < node_count/2)
				current_node = current_node->left;
			else
			{
				current_node = current_node->right;
				nid -= node_count/2;
			}
			node_count /= 2;
			current_level++;            
		}

		// Clear local
		clear_local(current_node);

		// Give new data to node
		give_leaf_data(current_node, data, size, merged_ghost_size);
	}

	void give_leaf_data(KDnode<T,NDIM>* node, T* data, int size, int merged_ghost_size)
	{
		if(node->data != NULL)
			ErrorMessage("KDTree give_data(): error, cant give data to node who already has data...\n");

		// We are giving this data to the node, its now the owner.
		node->owner = true;		

		// Update node
		node->data = data;
		node->local_size = size;
		node->merged_ghosts = merged_ghost_size;

		// Climb to root updating local sizes
		while(node != root)
		{
			node->parent->local_size += size;
			// Update ghost count
			node->parent->merged_ghosts += merged_ghost_size;
			node = node->parent;
		}
	}


	// This backup and restore data functionality is a bit of a hack but useful for fast tree restoration
	// Allows to quickly copy and restore tree data without cost of a safe tree assignment operator
	void backup_data(KDtree<T,NDIM>& backup)
	{
		// Check we have backup root
		 if(!backup.active)
		 {
			backup.root = new KDnode<T,NDIM>;
			backup.active = true;
		 }

		KDnode<T,NDIM>* bakroot = backup.root;

		 // If we own the data, simply backup on a node by node basis
		if(owner)
			 backup_data(root, bakroot);
		else
		{
			// If we dont own the data, just copy all data starting from leftmost node pointer
			KDnode<T,NDIM>* leftmost = find_leftmost(root);
			if(bakroot->owner)
			{
				// Check size is the same
				if(root->local_size != bakroot->local_size)
				{
					// If not, delete and set owner false
					delete[] bakroot->data;
					bakroot->owner = false;
					bakroot->local_size = 0;					
				}		
				// Note previous statement may change ownership status, hence the second check
				if(!bakroot->owner)
				{
					// Allocate 
					bakroot->data = new T[root->local_size];
					// Set owner and size
					bakroot->local_size = bakroot->local_size;
					bakroot->owner = true;
				}		
			}
			memcpy(bakroot->data, leftmost->data, root->local_size*sizeof(T));
		}
	}
	
	// Recursive backup, traverse our tree - whenever we find a data owner, copy that data to the backup tree
	// This allocates nodes in the backup tree as necessary, and reuses data in the backup tree only if it exactly matches size necessary
	void backup_data(KDnode<T,NDIM>* src, KDnode<T,NDIM>* backup)
	{
		if(src->owner)
		{
			if(backup->owner)
			{
				// Check size is the same
				if(backup->local_size != src->local_size)
				{
					// If not, delete and set owner false
					delete[] backup->data;
					backup->owner = false;
					backup->local_size = 0;					
				}
			}

			if(!backup->owner)
			{
				// Allocate 
				backup->data = new T[src->local_size];
				// Set owner and size
				backup->local_size = src->local_size;
				backup->owner = true;
			}
			
			// Copy
			memcpy(backup->data, src->data, src->local_size*sizeof(T));
			return;
		}
		else
		{
			// Recurse
			if(src->left)
			{
				if(!backup->left)
					backup->left = new KDnode<T,NDIM>;
				backup_data(src->left, backup->left);	

				if(!backup->right)
					backup->right = new KDnode<T,NDIM>;
				backup_data(src->right, backup->right);	
			}
		}
	}

	// This restores from a tree previously built via backup_data()
	void restore_data(const KDtree<T,NDIM>& backup)
	{
		KDnode<T,NDIM>* bakroot = backup.root;
		if(bakroot == NULL)
		{
		 	ErrorMessage("KDTree restore_data(): restoring from empty tree...\n");
		}

		// If we own our data, restore on a node by node basis
		if(owner)
			restore_data(root,bakroot);
		// If not, we can quick-restore from root to leftmost
		else
		{
			KDnode<T,NDIM>* leftmost = find_leftmost(root);
			if(!bakroot->owner || bakroot->local_size != root->local_size);
			{
				ErrorMessage("KDTree restore_data(): restoration tree structure does not match backup structure\n");
			}
			memcpy(leftmost->data, bakroot->data, root->local_size*sizeof(T));
		}
	}
	// Recursive backup, traverse our tree - whenever we find a data owner, copy that data to the backup tree
	// This allocates nodes in the backup tree as necessary, and reuses data in the backup tree only if it exactly matches size necessary
	void restore_data(KDnode<T,NDIM>* src, KDnode<T,NDIM>* backup)
	{
		if(src->owner)
		{
			if(!backup->owner || src->local_size != backup->local_size)
			{
				ErrorMessage("KDTree restore_data(): restoration tree structure does not match backup structure\n");
			}

			// Copy
			memcpy(src->data, backup->data, src->local_size*sizeof(T));
			return;
		}
		else
		{
			// Recurse
			if(src->left)
			{
				if(!backup->left)
				{
					ErrorMessage("KDTree restore_data(): restoration tree structure does not match backup structure\n");
				}
				restore_data(src->left, backup->left);	

				if(!backup->right)
				{
					ErrorMessage("KDTree restore_data(): restoration tree structure does not match backup structure\n");
				}
				restore_data(src->right, backup->right);	
			}
		}
	}


/*
// These functions should work but need double checking, do this if they are actually needed
	void give_data(KDnode<T,NDIM>* node, T* data, int size)
	{
		if(node->data != NULL)
			ErrorMessage("KDTree give_data(): error, cant give data to node who already has data...\n");

		// We are giving this data to the node, its now the owner.
		node->owner = true;

		// Sort it to the subtree
		sort_to_existing_subtree(node, data, size);

		// Climb to root updating local sizes
		while(node != root)
		{
			node->parent->local_size += size;
			node = node->parent;
		}
	}

	void sort_to_existing_subtree(KDnode<T,NDIM>* node, T* data, int size)
	{
		// Update node
		node->data = data;
		node->local_size = size;
		
		// Partition and recurse
		if(node->depth < max_depth)
		{
			int split = partition_by_value(data, 0, size-1, node->loc, smaller_than[node->dim]);
			sort_to_existing_subtree(node->left, data, split);
			sort_to_existing_subtree(node->right, (data+split), size-split);    
		}
	}
*/
	void find_all_nodes_at_level(unsigned level, std::vector<KDnode<T,NDIM>* >& nodes)
	{
		if(level == 0)
		{
			nodes.push_back(root);
			return;
		}

		nodes.reserve(pow(2, level));
		
		find_nodes_at_level(level, nodes, root);
	}

	std::vector<KDnode<T,NDIM>*> active_node_list()
	{
		std::vector<KDnode<T,NDIM>*> active_list;
		find_nonempty_leaves(root, active_list);
		
		return active_list;
	}

	KDnode<T,NDIM>* find_leftmost(KDnode<T,NDIM>* node)
	{
		while(node->left!=NULL) node = node->left;
		return node;
	}

	void find_nonempty_leaves(KDnode<T,NDIM>* node, std::vector<KDnode<T,NDIM>*>& nodes)
	{
		if(node->left)
		{
			find_nonempty_leaves(node->left, nodes);
			find_nonempty_leaves(node->right, nodes);
		}
		else
		{
			if(node->local_size > 0)
				nodes.push_back(node);
		}
	}

	void depth_sort_leaves(T& ref, std::vector<int>& sorted_ids)
	{
		sorted_ids.resize(nleaves);
		int counter = 0;
		depth_sort_leaves(root, counter, ref, sorted_ids);

	}

	void depth_sort_leaves(KDnode<T,NDIM>* node, int& counter, T& ref, std::vector<int>& sorted_ids)
	{
		if(node->left)
		{

			// Compare camera on split dimension 
			if(smaller_than[node->dim](ref, node->loc))
			{
				depth_sort_leaves(node->left, counter, ref, sorted_ids);
				depth_sort_leaves(node->right, counter, ref, sorted_ids);
			}
			else
			{
				depth_sort_leaves(node->right, counter, ref, sorted_ids);
				depth_sort_leaves(node->left, counter, ref, sorted_ids);				
			}
		}
		else
		{
			sorted_ids[counter++] = node->leaf_id; 
		}
	}

//------------ status logging --------------
	void status()
	{
		if(mpi_rank==mpi_root) 
		{
			if(root != NULL)
			{
				//Print("root: %p\n", root);
				Print("\nRank %i:\nKD Tree status\n\tMax depth: %i (levels: %i)\n\tNodes: %i\n\tLeaves: %i\n\tLargest Leaf: %i\n\tSmallest Leaf: %i\n", mpi_rank, depth, depth+1, nNodes, nleaves, largest_leaf, smallest_leaf);
				Print("Root node:\n\tDepth: %i Local Size: %i Global Size: %i Split Dim: %i\n", root->depth, root->local_size, root->global_size, root->dim);
				Print("Useful sizes (bytes):\n\tKDnode: %i\n\tKDnode*: %i\n\tKDtree<T,N>: %i\n\tFull tree: %i\n\n", sizeof(KDnode<T,NDIM>), sizeof(KDnode<T,NDIM>*), sizeof(KDtree<T,NDIM>), sizeof(KDtree<T,NDIM>)+sizeof(KDnode<T,NDIM>)*nNodes);
			}
			else
			{
				Print("KDtree status(): Empty tree.\n");
			}
		}
	}

	void full_status_to_file(const char* filename)
	{
		if(root != NULL)
		{
			std::stringstream ss;
			ss << filename << "." << mpi_rank;
			std::string name = ss.str();

			std::ofstream ofile(name.c_str());

			if(ofile.is_open())
			{
				ofile << "KDTree status report for rank " << mpi_rank << "\n";
				ofile << "Max depth: " << depth << "\nNodes: " << nNodes << "\nLeaves: " << nleaves << "\nLargest Leaf: " << largest_leaf << "\nSmallest Leaf: " << smallest_leaf << "\n";
				node_status_to_file(root, ofile);
				ofile.close();
			}
			else
			{

				ErrorMessage("KDTree::full_status_to_file(): Could not open ofstream %s\n", name.c_str());
			}
		}
		else
		{
			if(mpi_rank == mpi_root)
				Print("KDtree full_status_to_file(): Empty tree.\n");
		}
	}

	void all_nodes_status()
	{
		node_status(root);
	}

	void all_leaves_status()
	{
		leaf_status(root);
	}

	void box()
	{
		if(mpi_rank == 0)
		{
			Print("Root boundaries:\n");
			for(unsigned i = 0; i < NDIM; i++)
			{
				T valmin = bbox.min[i];
				T valmax = bbox.max[i];
				Print("Dimension %i: min %f max %f\n", i, (*access[i])(valmin),(*access[i])(valmax));
			}			
		}
	}

private:

	void build(KDnode<T, NDIM>* inNode)
	{
		nNodes++;

		// Check if we should be leaf
		if((max_size_leaf > 0 && inNode->global_size <= max_size_leaf) || (max_depth > 0 && inNode->depth >= max_depth))
		{
			// Create leaf
			smallest_leaf = (inNode->global_size < smallest_leaf) ? inNode->global_size : smallest_leaf;
			largest_leaf  = (inNode->global_size > largest_leaf) ? inNode->global_size : largest_leaf;
			depth = (inNode->depth > depth) ? inNode->depth: depth;
			inNode->leaf_id = nleaves;
			inNode->dim = -1;
			nleaves++;
			return;
		}

		/* 
			if using differences, find longest dimension with difference operator
			if not, iterate through dimensions (i.e. starting ldim = 0, level 1 dim = 1) (modified so levels start at 0, level 0 dim 0..)
		*/
		// Simply iterate to next dimension after whatever used by previous node
		unsigned short ldim = 0;

		if(!split_longest_axis) 
			ldim = (inNode->depth) % NDIM;
		else
		{
			// Find longest dimension using difference functions 
			for(unsigned i = 1; i < NDIM; i++)
			{	
				float d1 = fabs((*access[ldim])(inNode->bbox.max[ldim]) - (*access[ldim])(inNode->bbox.min[ldim]));
				float d2 = fabs((*access[i])(inNode->bbox.max[i]) - (*access[i])(inNode->bbox.min[i]));
				ldim = d1 > d2 ? ldim : i;
			}           
		}
		
		// Find median on longest dimension & partition to left/right 
		int left  = 0;
		int right = inNode->local_size-1;
		int local_split = inNode->local_size/2;
		int global_split = inNode->global_size/2;

#ifdef USE_MPI
		inNode->loc = parqselect(inNode->data,left,right,global_split,mpi_rank, mpi_root, comm_size, smaller_than[ldim]);
		// Parallel quick select doesnt guarantee partitioning, so do partition too and get global split
		local_split = partition_by_value(inNode->data, left, right, inNode->loc, smaller_than[ldim]);
		MPI_Allreduce(&local_split, &global_split, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
		inNode->loc = qselect(inNode->data,left,right,local_split,smaller_than[ldim]);
		global_split = local_split;
#endif

		inNode->dim = ldim;

		// Split array, accounting for possible empty nodes (only happens in parallel)
		T* leftdata = NULL;
		T* rightdata = NULL;
		// Do we have data  at all
		if(inNode->local_size > 0)
		{
			// Do we have data for left node
			if(local_split>0)
				leftdata = &inNode->data[left];
				
			// Do we have data for right node
			if(local_split < (int)inNode->local_size)
				rightdata = &inNode->data[local_split];
		}

		inNode->left  = new KDnode<T, NDIM>(leftdata, local_split, bbox, global_split, inNode->depth, inNode);
		inNode->right = new KDnode<T, NDIM>(rightdata, inNode->local_size - local_split, bbox, inNode->global_size - global_split, inNode->depth, inNode);

		
		// Update bounding box to get longest dimension
		inNode->left->bbox.max[ldim] = inNode->loc;
		inNode->right->bbox.min[ldim] = inNode->loc;      

		//Print("Rank %i about to do ghosts for size %i, ghosts %i ghost_depth %i left->depth %i\n", mpi_rank, inNode->local_size,ghosts,ghost_depth,inNode->left->depth);
		// If we are doing ghosts at this level, then find them
		if(ghosts && ghost_depth >= inNode->left->depth )
		{

			// Partition ghosts, duplicating any that are themselves ghosts
			// Directly accessing ghosts here - type must have .ghost attribute, should have some kind of accessor..
			//int l=0,r=0;
			for(unsigned i = 0; i < inNode->ghosts.size(); i++)
			{
				T g = inNode->ghosts[i];
				// Get difference in split dimension
				float d1 = (*access[inNode->dim])(g) -  (*access[inNode->dim])(inNode->loc);
				// Compare to ghost limit (radius)
				float d2 = (*access[NDIM])(g);

				// Ghost marker modifier
				int gmm = inNode->dim*2;

				// Partition particles, marking+duplicating those who also intersect the new plane
		 		if(d1 < d2)
		 		{
		 			// If its intersecting, mark intersecting plane
		 			if(d1>0) g.ghost |= 1 << gmm;

		 			// Add to ghost list
					inNode->left->ghosts.push_back(g);
					//l++;
		 		}
				
				if(d1 > -d2)
				{
		 			// If its intersecting, mark intersecting plane
		 			if(d1 < 0) g.ghost |= 1 << (gmm+1);

					inNode->right->ghosts.push_back(g);		
					//r++;		
				}
			}
			// Loop through all particles
			// duplicate any where their location in split dimension +- radius overlaps with split location
			// Partition left and right
			// Give to children
			//l=0;r=0;	
			for(unsigned i = 0; i < inNode->local_size; i++)
			{
				// Create ghost copy
				T g = inNode->data[i];

				// Get difference in split dimension
				float d1 = (*access[inNode->dim])(g) -  (*access[inNode->dim])(inNode->loc);
				// Compare to ghost limit (radius)
				float d2 = (*access[NDIM])(g);

				// Get bit ID of intersecting plane
				int gmm = inNode->dim*2;

				// If element is in right side but overlaps border gift it to left & vice versa 
		 		if(d1 > 0 && d1 < d2)
		 		{
		 			// First mark the local particle as a remote ghost (i.e. it is a local particle but considered a ghost to a remote processor)
		 			inNode->data[i].ghost |= (1 << 7);
		 			// Then mark the plane it intersects (the opposite plane to the ghost)
		 			inNode->data[i].ghost |= (1 << gmm+1);

		 			// Mark the ghost intersecting plane
		 			g.ghost |= 1 << gmm;

					inNode->left->ghosts.push_back(g);
					//l++;
		 		}
				
				if(d1 < 0 && d1 > -d2)
				{
		 			// First mark the local particle as a remote ghost (i.e. it is a local particle but considered a ghost to a remote processor)
		 			inNode->data[i].ghost |= (1 << 7);
		 			// Then mark the plane it intersects (the opposite plane)
		 			inNode->data[i].ghost |= (1 << gmm);

		 			// Mark intersecting plane
		 			g.ghost |= 1 << (gmm+1);

					inNode->right->ghosts.push_back(g);		
					//r++;		
				}
			}
			// Clear our ghosts (forcing deallocation)
			std::vector<T>().swap(inNode->ghosts);
		}

		// Build left then right half of tree
		build(inNode->left);
		build(inNode->right);
	}

	void destroy(KDnode<T, NDIM>* inNode)
	{
		// First delete any children
		if(inNode->left != NULL)
		{
			destroy(inNode->left);
		}
		
		if(inNode->right != NULL)
		{
			destroy(inNode->right);
		}
		
		// If we owned any data, delete that
		if(inNode->owner)
			delete[] inNode->data;

		// Then delete node itself
		//Print("Deleting node at depth %i of size %i\n", inNode->depth, inNode->size);
		delete inNode;
		inNode = NULL;

	}


	void find_nodes_at_level(int level, std::vector<KDnode<T, NDIM>*>& nodes, KDnode<T, NDIM>* node)
	{
		// Is this check necessary?
		if(node->depth > level)
			return;
		
		// We're at the right level
		if(node->depth == level)
		{
			nodes.push_back(node);
		}

		// Havent hit the right level yet, recurse
		if(node->depth < level && node->depth < depth)
		{
			find_nodes_at_level(level, nodes, node->left);
			find_nodes_at_level(level, nodes, node->right);
		}
	}

	// Print some info about the tree
	void node_status(KDnode<T, NDIM>* node)
	{
		for(unsigned i = 0; i < comm_size; i++)
		{
			if(mpi_rank==i)
			{
				Print("Rank %i: Node Depth: %i Local Size: %i Global Size: %i nGhost: %i nMergedGhost: %i Dim: %i", mpi_rank, node->depth, node->local_size, node->global_size, node->ghosts.size(), node->merged_ghosts, node->dim);
				if(node->left != NULL)
				{
					Print("\n");
					node_status(node->left);
					node_status(node->right);
				}
				else
					Print(" Leaf\n");

				fflush(0);           
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	void leaf_status(KDnode<T, NDIM>* node)
	{
		for(unsigned i = 0; i < comm_size; i++)
		{
			if(mpi_rank==i)
			{      
			if(node->left != NULL)
			{
				leaf_status(node->left);
				leaf_status(node->right);
			}
			else
			{
				Print("Rank %i: Leaf Node id %u: Depth: %i Size: %i Global Size: %i nGhost: %i nMergedGhost: %i Dim: %i\n", mpi_rank, node->leaf_id, node->depth, node->local_size, node->global_size, node->ghosts.size(), node->merged_ghosts, node->dim);
			}
		}
			fflush(0);  
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}

	void node_status_to_file(KDnode<T, NDIM>* node, std::ofstream& file)
	{
			file << "Node Depth: " << node->depth << " Local Size: " <<  node->local_size << " Dim: " << node->dim;
			if(node->left != NULL)
			{
				file << "\n";
				node_status_to_file(node->left, file);
				node_status_to_file(node->right, file);
			}
			else
				file << " Leaf\n";  
	}



};




#endif