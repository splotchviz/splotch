
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include<CL/cl.h>
extern "C" cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);


int check_device(int rank)
{
//number of devices
	cl_platform_id cpPlatform;
	oclGetPlatformID (&cpPlatform);
	cl_uint ciDeviceCount=0;
	clGetDeviceIDs (cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);
return (int) ciDeviceCount;


/*cl_uint ciDeviceCount;
		    cl_device_id *devices;
		//Get the devices
		ciErr1 = clGetDeviceIDs (cpPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &ciDeviceCount);




		clGetDeviceIDs (cpPlatform, CL_DEVICE_TYPE_ALL, ciDeviceCount, devices, &ciDeviceCount);

		 for(unsigned int i = 0; i < ciDeviceCount; ++i )
		                {
		                    printf(" dev_count %d---------------------------------\n",ciDeviceCount);
		                    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
		                    printf(" Device %s\n", cBuffer);
		                }*/
}

void print_device_info(int rank, int dev)
{

}
