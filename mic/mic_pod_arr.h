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

#ifndef MIC_POD_ARR_H
#define MIC_POD_ARR_H

#include "mic_allocator.h"

// Array class for use in offload (only for POD types)
// No construction or destruction is performed
template<typename T>
class Array_POD_T{
public:
	Array_POD_T() { arr_ = NULL; size_ = 0; capacity_ = 0; allocator = &defaultAllocator; }

	Array_POD_T(int i) { 
		if(i>0) {
			allocator = &defaultAllocator;			
			arr_ = (T*)allocator->malloc(i*sizeof(T), 64);
			if(arr_ == NULL){
				printf("Error: malloc for array(%i) returned null\n", i);
				fflush(0);
				exit(1);
			}
			size_ = i; 
			capacity_ = i;
		}
		else {
			arr_ = NULL; 
			size_ = 0;
			capacity_ = 0;
		} 
	}
	
	~Array_POD_T() { 
		if(arr_ != NULL && allocator->active()) {
			allocator->free(arr_);
			arr_ = NULL;
			} 
		size_ = 0;
		capacity_ = 0;
	}

	T& operator[] (int loc) {
		if(loc >= size_){
			printf("Error: array element access out of range [accessed element: %d array size: %d]\n", loc, size_);
			exit(1);
		}
		else
			return arr_[loc];
	}

	T const& operator[] (int loc)const {
		if(loc >= size_){
			printf("Error: array element access out of range [accessed element: %d array size: %d]\n", loc, size_);
			exit(1);
		}
		else
			return arr_[loc];
	}

	void operator= (Array_POD_T<T> const& copyarr) {
		if(copyarr.capacity() > 0)
		{
			if(capacity_ != copyarr.capacity())
			{
				setCapacity(copyarr.capacity());
			}
		}
		if(copyarr.size() > 0)
		{
			if(size_ != copyarr.size())
				resize(copyarr.size());		
			for(unsigned i = 0; i < size_; i++)
				arr_[i] = copyarr[i];
		}
		else
		{
			arr_ = NULL;
			size_ = 0;
			capacity_ = 0;			
		}
	}

	void resize(int newSize) {
		if(newSize>0)
		{
			if(arr_ != NULL && newSize <= capacity_)
			{
				size_ = newSize;
				return;
			}
			if(arr_ != NULL) {
				T* temp = (T*)allocator->malloc(newSize*sizeof(T), 64);
				if(temp == NULL)
				{
					printf("Error: malloc for resize(%i) returned null\n", newSize);
					fflush(0);
					exit(1);
				}
				memcpy(temp, arr_, (newSize > size_ ? size_ : newSize)*sizeof(T) );
				allocator->free(arr_);
				arr_ = temp;
				size_ = newSize;
				capacity_ = newSize;
			}
			else {
				arr_ = (T*)allocator->malloc(newSize*sizeof(T), 64);
				if(arr_ == NULL)
				{
					printf("Error: malloc for resize(%i) returned null\n", newSize);
					fflush(0);
					exit(1);
				}
				size_ = newSize;
				capacity_ = newSize;
			}
		}
		else 
		{
			if(size_ > 0 && allocator->active()) {
				allocator->free(arr_);
			}
			arr_ = NULL; 			
			size_ = 0;
			capacity_ = 0;
		}
	}

	void setCapacity(int newCapacity) {
		if(arr_ != NULL) {
			if(newCapacity < capacity_)
			{
				T* temp = NULL;
				if(newCapacity > 0)
				{
					temp = (T*)allocator->malloc(newCapacity*sizeof(T), 64);
					if(temp == NULL)
					{
						printf("Error: malloc for setCapacity(%i) returned null\n", newCapacity);
						fflush(0);
						exit(1);
					}
					memcpy(temp, arr_, ((newCapacity < size_) ? newCapacity : size_)*sizeof(T) );
				}

				allocator->free(arr_);
				arr_ = temp;
				size_ = (newCapacity < size_) ? newCapacity : size_;
				capacity_ = newCapacity;
			}
			else if(newCapacity > capacity_)
			{
				T* temp;
				temp = (T*)allocator->malloc(newCapacity*sizeof(T), 64);
				if(temp == NULL)
				{
					printf("Error: malloc for setCapacity(%i) returned null\n", newCapacity);
					fflush(0);
					exit(1);
				}
				memcpy(temp, arr_, size_*sizeof(T) );
				allocator->free(arr_);
				arr_ = temp;
				capacity_ = newCapacity;
			}
			else return;

		}
		else {

			if(newCapacity > 0)
			{
				arr_ = (T*)allocator->malloc(newCapacity*sizeof(T), 64);		
				if(arr_ == NULL)
				{
					printf("Error: malloc for setCapacity(%i) returned null\n", newCapacity);
					fflush(0);
					exit(1);
				}	
			}
			capacity_ = newCapacity;
		}		
	}

	void push_back(T element) {
		if(size_ < capacity_){
			size_++;
			arr_[size_-1] = element;
		}
		else {
			if(!capacity_) capacity_++;
			setCapacity(capacity_*2);
			++size_;
			arr_[size_-1] = element;
		}
	}

	T take_back() {
		if(!size_){
			printf("Array pod T: take_back(): Array empty\n");
			exit(1);			
		}
		else
		{
			T temp = arr_[size_-1];
			size_ -= 1;
			return temp;
		}
	}

	void fill(T fillval) {
		for(unsigned i = 0; i < size_; i++)
			arr_[i] = fillval; 
	}

	void giveAllocator(AllocatorBase* newAllocator)
	{
		// If the array already has data, move it to the new allocator's memory domain
		if(arr_ != NULL)
		{

		}
		else 
		{
			allocator = newAllocator;
		}
	}

	T* ptr() { return arr_; }
	T const* ptr() const{ return arr_; }
 	unsigned size() const{ return size_; }
	unsigned capacity() const{ return capacity_; }

	T*& ptr_ref() { return arr_; }
	unsigned& size_ref() { return size_; }
	unsigned& capacity_ref() { return capacity_; }

private:
	AllocatorBase* allocator; 
	AlignedAllocator defaultAllocator;
	T* arr_;
	unsigned size_;
	unsigned capacity_;
};

#endif
