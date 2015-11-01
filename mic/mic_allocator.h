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

#ifndef MIC_ALLOCATOR_H
#define MIC_ALLOCATOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Uncomment this to allow (rather verbose) debugging info
//#define _ALLOCATOR_DEBUG_


// Abstract class allocator
class AllocatorBase
{
public:
	AllocatorBase() {}
	virtual ~AllocatorBase() {};

	virtual void* malloc(size_t bufferSize)  = 0;
	virtual void* malloc(size_t bufferSize, size_t alignment) = 0;
	virtual void free(void* ptr)  = 0;
	virtual bool active() = 0;
};

// Standard aligned allocator
class AlignedAllocator : public AllocatorBase
{
public:
	AlignedAllocator() {};
	~AlignedAllocator() {}; 

	void* malloc(size_t bufferSize) { return _mm_malloc(bufferSize, 1); }
	void* malloc(size_t bufferSize, size_t alignment) { /*printf("AlignedAllocator: malloc() size: %d\n", bufferSize);fflush(0);*/ return _mm_malloc(bufferSize, alignment); }
	void free(void* ptr) { _mm_free(ptr); }
	bool active() { return true; }

private:

};

// Aligned pooled allocator (FIXED SIZE)
class AlignedPoolAllocator : public AllocatorBase
{
public:
	AlignedPoolAllocator();
	~AlignedPoolAllocator();

	void init(size_t poolSize, size_t alignment);
	void* malloc(size_t bufferSize);
	void* malloc(size_t bufferSize, size_t alignment);
	void free(void*);
	bool active();
	void printStatus();
	
private: 
	struct mem_ctrl_block{
		bool active; 
		size_t size;
	};

	void coalesce(char*);
	char* firstValidAddr;
	char* memoryLimitAddr;
	size_t alignment;
	size_t offset;
	size_t offsetx2;
	size_t split_limit;
	bool initialised;

	AllocatorBase* source;
	void** sourceAllocations;
	unsigned nSourceAllocations;
	unsigned availSourceAllocations;
};



#endif