/*
 * Copyright (c) 2004-2014
 *              Tim Dykes University of Portsmouth
 *              Claudio Gheller ETH-CSCS
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
 */

#pragma offload_attribute(push, target(mic))
#include "mic_allocator.h"

AlignedPoolAllocator::AlignedPoolAllocator()
{
	initialised = false;
	firstValidAddr = NULL;
	memoryLimitAddr = NULL;
	alignment = 0;
	sourceAllocations = NULL;
	nSourceAllocations = 0;
	availSourceAllocations = 0;

	// This is the split limit for block mallocs, if the difference
	// between an allocation request size and available block size is
	// less than this the entire block will be provided. 
	// Currently 1KB. Will waste memory if making many small allocations.
	split_limit = 1024;
}

AlignedPoolAllocator::~AlignedPoolAllocator() 
{
	if(initialised)
	{
		// Free memory
		if(alignment == 1)
			free(firstValidAddr);
		else
			_mm_free(firstValidAddr);
	}

	initialised = false;
}

void AlignedPoolAllocator::init(size_t size, size_t align)
{
	// Check if already initialised
	if(initialised)
	{
		printf("Allocator: double initialisation! Exiting\n");
		fflush(0);
		exit(1);
	}		


	// Check for ^2 alignment
	if((align != 1) && (align & (align-1)))
	{
		printf("Allocator: Failed to init, alignment not power of 2, chosen alignment: %zu", align);
		fflush(0);
		exit(1);
	}
	alignment = align;

	// Work out padding to add MCB without losing alignment
	unsigned mcbsize = sizeof(mem_ctrl_block);
	int padding = alignment-(mcbsize%alignment);
	offset = (mcbsize+padding);
	offsetx2 = offset * 2;

	// Space for MCB at start
	unsigned paddedSize = (size + offsetx2);
	

	// Allocate aligned memory
	firstValidAddr = (char*)_mm_malloc(paddedSize, alignment);

	memoryLimitAddr = firstValidAddr + paddedSize;

	if(firstValidAddr == NULL)
	{
		printf("Failed to allocated memory of size: %zu (plus padding of size %zu) with alignment %zu\n", size,offsetx2,alignment);
		fflush(0);
		exit(1);
	}

	// Initialise MCB
	((mem_ctrl_block*)firstValidAddr)->active = false;
	((mem_ctrl_block*)firstValidAddr)->size = size;

	((mem_ctrl_block*)(memoryLimitAddr-offset))->active = false;
	((mem_ctrl_block*)(memoryLimitAddr-offset))->size = size;

	initialised = true;

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: initialised with %u byte buffer aligned to %zu!\n", paddedSize, alignment);
		printf("Allocator: first valid address: %p, last valid address: %p\n", (void*)firstValidAddr, (void*)memoryLimitAddr);
		printf("Allocator: Offset: %zu\n", offset);
		fflush(0);
	#endif
}

void* AlignedPoolAllocator::malloc(size_t bufferSize)
{
	return malloc(bufferSize, alignment);
}

void* AlignedPoolAllocator::malloc(size_t reqSize, size_t reqAlign)
{
	if(!initialised)
	{
		printf("Cannot malloc before initialising allocator!\n");
		fflush(0);
		exit(1);				
	}
	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: malloc called with %zu byte request!\n", reqSize);
		fflush(0);
	#endif

	if(reqAlign > alignment || (reqAlign & (reqAlign-1)))
	{
		printf("Allocator: malloc() requested alignment too large/not power of two!\n");
		printf("Requested alignment: %lu Allocator default alignment: %lu\n", reqAlign, alignment);
		fflush(0);
		exit(1);				
	}		

	char* currentAddr = firstValidAddr;
	mem_ctrl_block* currentMcb = (mem_ctrl_block*)currentAddr;

	// Find next free space thats big enough for allocation
	while(1)
	{
		currentMcb = (mem_ctrl_block*)currentAddr;
		if((currentMcb->active == false) && (currentMcb->size >= reqSize))
			break;

		currentAddr += (currentMcb->size + offsetx2);
		if(currentAddr == memoryLimitAddr)
		{
			printf("Pool overflow: allocating from source for %lu\n",reqSize);
			fflush(0);
			// We dont have enough memory! allocate memory from source
			void* newMemory = _mm_malloc(reqSize, reqAlign);		
			if(newMemory == NULL)
			{
				printf("Pool overflowed and could not get memory from source for allocation %lu\n", reqSize);
				fflush(0);
				exit(1);					
			}	

			// Check if theres a free spot in the source allocation list
			for(unsigned i = 0; i < nSourceAllocations; i++)
			{
				if(sourceAllocations[i] == NULL)
				{
					sourceAllocations[i] = newMemory;
					return newMemory;
				}
			}
			// If not, extend list; reallocating if necessary
			if(nSourceAllocations >= availSourceAllocations)
			{
				if(nSourceAllocations<1) nSourceAllocations = 1;
				void** temp = (void**)_mm_malloc(nSourceAllocations*2*sizeof(void*), 64);
				if(temp == NULL)
				{
					printf("Pool overflowed and could not get memory from source for allocation list\n");
					fflush(0);
					exit(1);					
				}	
				if(sourceAllocations != NULL)
				{
					memcpy(temp, sourceAllocations, nSourceAllocations*sizeof(void*));
					_mm_free(sourceAllocations);
				}
				sourceAllocations = temp;

			}

			// Store new allocation 
			nSourceAllocations++;
			sourceAllocations[nSourceAllocations-1] = newMemory;

			return newMemory;			
		}
	}

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: Found available block at %p of size %zu, allocating block of size %zu\n", currentAddr,currentMcb->size,reqSize);
		fflush(0);
	#endif

	// If block is less than 1KB larger than requested size, give entire block
	unsigned oldSize = currentMcb->size;
	unsigned diff 	= oldSize - reqSize;

	if( diff > split_limit)
	{
		#ifdef _ALLOCATOR_DEBUG_
			printf("Allocator: Diff between available and requested blocks is over 1024, splitting block.\n");
			fflush(0);
		#endif
		// Set requested blocks start MCB
		currentMcb->active = true;
		currentMcb->size = reqSize;

		// Set requested blocks end MCB
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize+offset);
		currentMcb->active = true;
		currentMcb->size = reqSize;

		// Set remainder blocks start MCB
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize+offsetx2);
		currentMcb->active = false;
		currentMcb->size = diff-offsetx2;	

		// Set remainder blocks end MCB
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize + offset+diff);
		currentMcb->active = false;
		currentMcb->size = diff-offsetx2;	

	#ifdef _ALLOCATOR_DEBUG_
		currentMcb = (mem_ctrl_block*)currentAddr;
		printf("Allocator: Allocated new block at %p\n", currentAddr);
		printf("Allocator: Front MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		currentMcb = (mem_ctrl_block*)(currentAddr+currentMcb->size+offset);
		printf("Allocator: End MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize+offsetx2);
		printf("Allocator: Remainder block is at %p\n", currentMcb);
		printf("Allocator: Front MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize + offset+diff);
		printf("Allocator: End MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		fflush(0);
	#endif

	}
	else
	{
		#ifdef _ALLOCATOR_DEBUG_
			printf("Allocator: Diff between available and requested blocks is under 1024, providing block\n");
			fflush(0);
		#endif
		// Set requested blocks start MCB
		currentMcb->active = true;

		// Set requested blocks end MCB
		currentMcb = (mem_ctrl_block*)(currentAddr+reqSize+diff+offset);
		currentMcb->active = true;

	#ifdef _ALLOCATOR_DEBUG_
		currentMcb = (mem_ctrl_block*)currentAddr;
		printf("Allocator: Allocated new block at %p\n", currentAddr);
		printf("Allocator: Front MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		currentMcb = (mem_ctrl_block*)(currentAddr+currentMcb->size+offset);
		printf("Allocator: End MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		fflush(0);
	#endif

	}



	return (void*)currentAddr+offset;
}

void AlignedPoolAllocator::free(void* freeAddr)
{

	if(freeAddr == NULL)
	{
		printf("Allocator attempting to free NULL pointer! Aborting program.\n");
		fflush(0);
		exit(1);			
	}		

	// Check if address came from source instead of pool
	for(unsigned i = 0; i < nSourceAllocations; i++)
	{
		if(sourceAllocations[i] == freeAddr)
		{
			_mm_free(freeAddr);
			sourceAllocations[i] = NULL;
			return;
		}
	}

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: free called with address %p !\n", freeAddr);
		fflush(0);
	#endif

	// Validate MCBs to check this is a previously allocated block
	// Get front MCB
	char* currentAddr = ((char*)freeAddr)-offset;
	mem_ctrl_block* currentMcb = (mem_ctrl_block*) currentAddr;

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: Checking MCB at address %p\n", currentAddr);
		fflush(0);
	#endif

	bool frontActiveFlag = currentMcb->active;
	unsigned frontSizeFlag = currentMcb->size;

	// Check against back MCB
	currentMcb = (mem_ctrl_block*)(currentAddr + frontSizeFlag + offset);

	if(!(currentMcb->active == frontActiveFlag) || !(currentMcb->size == frontSizeFlag) || frontActiveFlag == false)
	{
		// This wasnt a pooled allocation, call source free
		printf("Allocator attempting to free pointer that was not allocated from pool or source! Exiting\n");
		fflush(0);
		exit(1);		
	}

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: Front and back MCBs match. Active flag: %s, size: %u\n", frontActiveFlag ? "true" : "false", frontSizeFlag);
		fflush(0);
	#endif

	// Deactivate block with coalescence 
	coalesce(currentAddr);

	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: Object freed!\n");
		fflush(0);
	#endif
}


bool AlignedPoolAllocator::active()
{
	return initialised;
}

// TODO: Print this in a more readable fashion suited to data structure
void AlignedPoolAllocator::printStatus()
{
	printf("Allocator: Status listing:\n");
	char* currentAddr = firstValidAddr;
	mem_ctrl_block* currentMcb = (mem_ctrl_block*)currentAddr;

	// Find next free space thats big enough for allocation
	while(1)
	{
		currentMcb = (mem_ctrl_block*)currentAddr;
		printf("\t\tFound block at %p\n", currentAddr);
		printf("\t\tFront MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);
		currentMcb = (mem_ctrl_block*)(currentAddr+currentMcb->size+offset);
		printf("\t\tEnd MCB: active: %s, size: %zu\n", currentMcb->active ? "true" : "false", currentMcb->size);

		currentAddr += (currentMcb->size + offsetx2);
		if(currentAddr == memoryLimitAddr)
			break;
	}
	fflush(0);
}

void AlignedPoolAllocator::coalesce(char* currentAddr)
{
	// Bidirectional coalescence
	mem_ctrl_block* currentMcb = (mem_ctrl_block*)currentAddr;

	unsigned currentSize = currentMcb->size;

	// Free our current block
	currentMcb->active = false;
	((mem_ctrl_block*)(currentAddr+currentMcb->size+offset))->active = false;

	#ifdef _ALLOCATOR_DEBUG_
		bool fCoalesce = false, bCoalesce = false;
	#endif

	// If possible, check behind us for free block and merge
	if(currentAddr != firstValidAddr)
	{
		currentMcb = (mem_ctrl_block*)(currentAddr-offset);
		if(currentMcb->active == false)
		{
			unsigned sizeBehind = currentMcb->size;
			char* behindAddr = currentAddr - (sizeBehind + offsetx2);
			currentMcb = (mem_ctrl_block*)behindAddr;
			// New block size is the middle block size + size of block behind it + the now unused MCBs between them
			currentMcb->size = sizeBehind + currentSize + offsetx2;
			// Update end block to reflect new size and active status
			currentMcb = (mem_ctrl_block*)(behindAddr + currentMcb->size + offset);
			currentMcb->size =  sizeBehind + currentSize + offsetx2;
			currentMcb->active = false;
			currentAddr = behindAddr;
		#ifdef _ALLOCATOR_DEBUG_
			bCoalesce = true;
		#endif
		}
	}

	// Now check in front of us and merge if possible
	currentMcb = (mem_ctrl_block*)currentAddr;
	char* nextBlockAddr = (currentAddr + currentMcb->size + offsetx2);
	if(nextBlockAddr != memoryLimitAddr)
	{
	 	mem_ctrl_block* nextBlockMCB = (mem_ctrl_block*)nextBlockAddr;
	 	if(nextBlockMCB->active == false)
	 	{
	 		// Enlarge our block to absorb the one in front, plus two unused MCBs between them
	 		unsigned sizeInFront = nextBlockMCB->size;
	 		currentMcb->size +=  (sizeInFront + offsetx2);
	 		currentMcb->active = false;
	 		// Update the end MCB of the new large block
	 		currentSize = currentMcb->size;
	 		currentMcb = (mem_ctrl_block*)(currentAddr + currentSize + offset);
	 		currentMcb->size = currentSize;
	 		currentMcb->active = false;
	 	#ifdef _ALLOCATOR_DEBUG_
	 		fCoalesce = true;
	 	#endif
	 	}
	}
	#ifdef _ALLOCATOR_DEBUG_
		printf("Allocator: Front coalesce: %s, back coalesce: %s\n", fCoalesce ? "true" : "false", bCoalesce ? "true" : "false");
		fflush(0);
	#endif

}

#pragma offload_attribute(pop)
