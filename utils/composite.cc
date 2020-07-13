#include "utils/composite.h"
#include <mpi.h>
// Simple binary swap
void composite(arr2<COLOUR>& pic, arr2<COLOUR>& opac, std::vector<int>& order, int rank, int nranks)
{
	if(rank==0) printf("Compositing %i images...\n",nranks);
	// Number of images to composite
	int nimg = order.size();

	// Composite buffer (+1 for uneven pixel split)
	arr2<COLOUR> buffer(pic.size1() + 1,pic.size2());
	float* recv_pic = (float*)&buffer[0][0].r;
	float* recv_opac = (float*)&buffer[pic.size1()/2 + 1][0].r;


	if(nimg != nranks || nranks&(nranks-1) || nranks>pic.size1())
	{
		// Error - only support 1 image per rank so far
		// Error - only support po2 ranks so far
		// Error - cant have more ranks than horizontal pixels
		planck_fail("composite(): currently only support 1 image per rank, power of 2 rank count, less ranks than horizontal resolution!\n");
	}

	// Position in order
	int id = -1;
	for(unsigned i = 0; i < order.size(); i++)
		if(order[i] == rank)
		{
			id = i;
			break;
		}

	if(id==-1)
	{
		// Error, didnt find rank in order list!
		Print("Composite order:\n");
		for(unsigned i = 0; i < order.size(); i++) Print("%i\n",order[i]);
		planck_fail("composite(): couldnt find rank id in composite order vector!\n");
	}

	int image_size = opac.size();
	int nrounds = log2(nranks);
	int offset = 0;
	int buffer_size = 0;
	int send_size = 0;
	float *target_pic,*target_opac;
	for(unsigned i = 0; i < nrounds; i++)
	{
		//printf("Rank %i round %i\n",rank,i);
		// Composite round
		// Group size
		int g = pow(2,i+1);

		// Local id within group
		int lid = id%g;
		bool right = false;
		if(lid >= g/2) right = true;
		
		// Find partner id from opposing side of group	
		int pid;
		if (right) 	pid = id - g/2;
		else 		pid = id + g/2;

		// Arrange and send buffers
		// Account for non-regular image size
		buffer_size = image_size/g;
		send_size = buffer_size;
		int recv_size = buffer_size;
		int overflow = image_size - g*buffer_size;

		if (id==nranks-1) 			recv_size += overflow;
		else if (pid == nranks-1) 	send_size += overflow;

		float*	send_pic  = (float*)(&( pic[0][0].r)) + offset*3;
		float*	send_opac = (float*)(&(opac[0][0].r)) + offset*3;

		if(right)
		{
			// Send my pic & opacity
			MPI_Send(send_pic , send_size*3, MPI_FLOAT, order[pid], 0, MPI_COMM_WORLD);
			MPI_Send(send_opac, send_size*3, MPI_FLOAT, order[pid], 1, MPI_COMM_WORLD);

			// Recieve partner pic and opacity
			MPI_Recv(recv_pic , recv_size*3, MPI_FLOAT, order[pid], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(recv_opac, recv_size*3, MPI_FLOAT, order[pid], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		else
		{
			// Left side owner must send right side of image
			send_pic  += buffer_size*3;
			send_opac += buffer_size*3;

			// Recieve partner pic and opacity
			MPI_Recv(recv_pic , recv_size*3, MPI_FLOAT, order[pid], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(recv_opac, recv_size*3, MPI_FLOAT, order[pid], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			// Send my pic & opacity
			MPI_Send(send_pic , send_size*3, MPI_FLOAT, order[pid], 0, MPI_COMM_WORLD);
			MPI_Send(send_opac, send_size*3, MPI_FLOAT, order[pid], 1, MPI_COMM_WORLD);
		}

		// Arrange buffers for composite
		// Right side always sends back pic and recieves front pic and vice versa for keft
		float *front_pic,*back_pic,*front_opac,*back_opac;
		if(right)
		{
			// Front is recieved pic
			front_pic   = recv_pic;
			front_opac  = recv_opac;
			// Back is non-sent pic
			back_pic	= send_pic  + buffer_size*3;
			back_opac 	= send_opac + buffer_size*3;
			// Target is non-sent pic
			target_pic  = back_pic; 
			target_opac = back_opac;
		}
		else
		{
			// Front is non-sent pic
			front_pic  = send_pic  - buffer_size*3;
			front_opac = send_opac - buffer_size*3;
			// Back is recieved pic
			back_pic   = recv_pic;
			back_opac  = recv_opac;	
			// Target is non-sent pic
			target_pic  = front_pic; 
			target_opac = front_opac;
		}

		// Do actual composite
		#pragma omp parallel for
		for(unsigned j = 0; j < recv_size*3; j++)
		{
			target_pic[j] = front_pic[j] + back_pic[j] * exp(front_opac[j]);
			target_opac[j] = front_opac[j] + back_opac[j];
		}
		
		// Retain offset for right hand side for next round
		if(right) offset += buffer_size;
	}

	// MPI_Gather to master
	// Recieves are all the same apart from whichever rank is last in the order if the image width isnt multiple of nranks
	int* recvs = new int[nranks];
	for(unsigned i = 0; i < nranks; i++)
	{
		recvs[i] = buffer_size*3;
		if(order[i] == nranks-1)
			recvs[i] += (image_size - nranks*buffer_size)*3;
	}

	// MPI_Gather displacements
	int* displs = new int[nranks];
	int float_offset = offset*3;
	MPI_Gather(&float_offset,1,MPI_INT,displs,1,MPI_INT,0,MPI_COMM_WORLD);


	if(rank==0)
		MPI_Gatherv(MPI_IN_PLACE,0,MPI_FLOAT,&(pic[0][0].r),recvs,displs,MPI_FLOAT,0,MPI_COMM_WORLD);
	else
		MPI_Gatherv(target_pic,send_size*3,MPI_FLOAT,NULL,NULL,NULL,MPI_FLOAT,0,MPI_COMM_WORLD);

	delete[] recvs;
	delete[] displs;
}
