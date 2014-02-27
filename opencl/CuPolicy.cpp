#include "opencl/Policy.h"

CuPolicy::CuPolicy(paramfile &Param) {

	maxregion = Param.find<int>("max_region", 1024);

}

//Get size of device particle data

int CuPolicy::GetMaxRegion() {
	return maxregion;
}

int CuPolicy::GetFBufSize() // return dimension in terms of Megabytes
{
	return fbsize;
}

int CuPolicy::GetGMemSize() // return dimension in terms of Megabytes
{
	int M = 1 << 20;
	int size = gmsize / M;
	return size;
}

void CuPolicy::setGMemSize(size_t s) {
	gmsize = s;
	//if (s > 512 * 1024 * 1024)
	//	fbsize = GetGMemSize() / 2;
	//else
	fbsize = GetGMemSize() / 4;

}

