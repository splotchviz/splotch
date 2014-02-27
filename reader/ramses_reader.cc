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


#include "reader.h"
#include "ramses_helper_lib.h"
#include <time.h>
#include "cxxsupport/mpi_support.h"
#include <string.h>
//----------------------------------------------------------------------------
// Ramses file reader for particle or amr data (or both)
// Tim Dykes
//
// Use red,green and blue parameters in parameter file to set the data to visualise. 
// Correct IDs for these can be found in ramses_helper_lib
// 
// amr uses red0,green0,blue0. If drawing just particle data then use the same parameters
// if drawing both types of data use red1,green1,blue1 for particle variables
//
// Note when using red+green+blue set colour_is_vector0=T, if just one colour variable
// is used then set colour_is_vector0=F. (also for just one colour var, use red...)
//
// Parallel read mode 0 for many small files (1cpu per file) mode 1 for small number
// of very large files (multiple cpus per file)
//
//----------------------------------------------------------------------------

#define MIN_GRIDS_PER_READ 100

void ramses_reader(paramfile &params, std::vector<particle_sim> &points)
{
	// Check parameter for read mode, points/amr/both
	int mode = params.find<int>("read_mode",0);

	int parallelmode = params.find<int>("parallel_read_mode",0);
	if(parallelmode != 0 && parallelmode != 1)
		std::cout << "Incorrect parallel_read_mode setting. Set to 0 or 1." << std::endl;

	// Get repository, otherwise assume data is in currentdirectory
	std::string repo = params.find<std::string>("infile","");

	// Get all other params
	float active = params.find<float>("active",1);
	float scale3d = params.find<float>("ramses_scale3d",1);

	int ptypes = params.find<int>("ptypes",1);
	std::vector<float> smooth_factor;
	for(int i = 0; i < ptypes; i++)
		smooth_factor.push_back(params.find<float>("smooth_factor"+dataToString(i),1.0));

	std::vector<float> const_intensity;
	for(int i = 0; i < ptypes; i++)
		const_intensity.push_back(params.find<float>("const_intensity"+dataToString(i),-1.0));	

	std::vector<float> intensity;
	for(int i = 0; i < ptypes; i++)
		intensity.push_back(params.find<float>("intensity"+dataToString(i),-1.0));

	std::vector<float> intense_factor;
	for(int i = 0; i < ptypes; i++)
		intense_factor.push_back(params.find<float>("intense_factor"+dataToString(i),1.0));

	std::vector<float> red;
	for(int i = 0; i < ptypes; i++)
		red.push_back(params.find<float>("red"+dataToString(i),-1));

	std::vector<float> green;
	for(int i = 0; i < ptypes; i++)
		green.push_back(params.find<float>("green"+dataToString(i),-1));

	std::vector<float> blue;
	for(int i = 0; i < ptypes; i++)
		blue.push_back(params.find<float>("blue"+dataToString(i),-1));

	srand((unsigned)time(0));


	// Read info file
	info_file info(repo);

	// Get MPI info
	int ntasks = mpiMgr.num_ranks();
	int rankthistask = mpiMgr.rank();

	// MPI setup
#ifdef USE_MPI
	if(mpiMgr.master())
		std::cout << "RAMSES READER MPI: Reading output from " << info.ncpu << " cpus with " << ntasks << " tasks." << std::endl;
#endif

	// Sampling
	// Sample factor is read in as percentage then converted to factor
	bool doSample = params.find<bool>("sampler",false);
	float sample_factor = params.find<float>("sample_factor",100);
	if(doSample)
	{
		if(sample_factor < 0 || sample_factor > 100)
		{
			if(mpiMgr.master())
				std::cout << "Invalid sample factor: " << sample_factor << "\n Use a percentage to sample, ie sample_factor=50\n";
			exit(0);
		}
		else
		{
			if(mpiMgr.master())
			{
				std::cout << "Sampler: Chosen sample percentage: " << sample_factor << "%\n";
				std::cout << "Sampling nearest available fraction of data: 1/" << (100/sample_factor)<< std::endl; 
			}
		}
	}

	// Storage for sampled points
	std::vector<particle_sim> pointfilter;

	if(mode == 0 || mode == 2)
	{
		// Read amr

		// AMR generated particles are type 0
		int C1 = red[0];
		int C2 = green[0];
		int C3 = blue[0];
		unsigned nlevelmax;

		// Check for constant intensity, if so set i to this value
		bool const_i = false;
		if(const_intensity[0] > -1)
		{
			const_i = true;
			intensity[0] = const_intensity[0];
		}
		// Set default intensity if neither variables are set
		else if(const_intensity[0] == -1 && intensity[0] == -1)
		{
			const_i = true;
			intensity[0] = 1;
		}

		// Check how many colour variables have been set in param file
		// For unset variables, set to density.
		int check = 0;
		if(C1 != -1) ++check;
		else C1 = 0;
		if(C2 != -1) ++check;
		else C2 = 0;
		if(C3 != -1) ++check;
		else C3 = 0;
		if(check == 2) 
		{
			if(mpiMgr.master())
				std::cout << "Ramses reader: must set either one or three param file elements red, green, blue, not two. Aborting." << std::endl;
			exit(0);
		}

		if(info.ndim != 3)
		{
			if(mpiMgr.master())
				std::cout << "Ramses amr reader: non 3 dimension data detected. Aborting" << std::endl;
			exit(0);
		}

		if(mpiMgr.master())
			std::cout << "Reading AMR data... " << std::endl;

		int nfiles = info.ncpu;
		int readsthistask = nfiles;
		int firstread = 0;

		// Parallel mode = 0 means cpu per file (good for generally small files)
		// Parallel mode = 1 means multiple cpus per file (good for generally large files)
		if(parallelmode == 0)
		{
			// If less files than tasks, read one file per task for all available files
			if(nfiles<ntasks)
			{
				if(rankthistask<nfiles)
				{
					firstread = rankthistask;
					readsthistask = 1;
				}
				else
				{
					// This task will read no particles
					firstread = 0;
					readsthistask = 0;
					// Push a fake particle to stop MPI crying about lack of data
					points.push_back(particle_sim(COLOUR(0,0,0),0,0,0,0,0,0,0));
				}
			}
			else
			{
				// Split files amongst tasks
				readsthistask = nfiles/ntasks;
				firstread = readsthistask*rankthistask;

	            // Work out remaining files and distribute
	            int remain = nfiles % ntasks;
	            if(remain)
	            {
	            	if(remain>rankthistask)
	            	{
	            		firstread+=rankthistask;
	            		readsthistask+=1;
	            	}
	            	else
	            		firstread+=remain;
	            }
			}
		}


		for(unsigned icpu = firstread; icpu < (firstread+readsthistask); icpu++)
		{
			// Open hydro file and read metadata
			hydro_file hydro(repo, icpu+1);

			// Validate params
			if( (C1 != -1 && (C1 < 0 || C1 >= hydro.meta.nvar)) || (C2 != -1 && (C2 < 0 || C2 >= hydro.meta.nvar)) ||
				(C3 != -1 && (C3 < 0 || C3 >= hydro.meta.nvar)) || (!const_i && (intensity[0] < 0 || intensity[0] >= hydro.meta.nvar)) )
			{
				if(mpiMgr.master())
				{
					std::cout << "Trying to read variables that are not in file, there are " << hydro.meta.nvar << " hydro variables in file.";
					std::cout << "Check red,green,blue,intensity parameters. Aborting." << std::endl;
				}
				exit(0);
			}

			// Open amr file + read metadata
			amr_file amr(repo, icpu+1);
			amr.file.SkipRecords(13);
			nlevelmax = amr.meta.nlevelmax; 
			
			// Allocated for gridboundarys if necessary
			F90_Arr2D<unsigned> ngridbound;
			if(amr.meta.nboundary>0)
				ngridbound.resize(amr.meta.nboundary,amr.meta.nlevelmax);

			// Read grid numbers
			F90_Arr2D<unsigned> ngridfile;
			ngridfile.resize(amr.meta.ncpu + amr.meta.nboundary,amr.meta.nlevelmax);
			F90_Arr2D<unsigned> ngridfile_noboundary;
			ngridfile_noboundary.resize(amr.meta.ncpu,amr.meta.nlevelmax);
			amr.file.Read2DArray(ngridfile_noboundary);
			memcpy( (void*)&ngridfile(0,0),(const void*)&ngridfile_noboundary(0,0), (ngridfile_noboundary.xdim*ngridfile_noboundary.ydim*sizeof(unsigned)) );
			ngridfile_noboundary.Delete();

			amr.file.SkipRecord();

			if(amr.meta.nboundary>0)
			{
				amr.file.SkipRecords(2);
				amr.file.Read2DArray(ngridbound);
				memcpy( (void*)&ngridfile(amr.meta.ncpu,0),(const void*)&ngridbound(0,0), (ngridbound.xdim*ngridbound.ydim*sizeof(unsigned)) );
			}

			amr.file.SkipRecords(2);

			if(info.ordering == "bisection")
				amr.file.SkipRecords(5);
			else 
				amr.file.SkipRecords(4);

			// Storage
			// Not using vectors to avoid performance hit of initialisations on resize
			double* gridvars = 0;
			F90_Arr2D<double> gridcoords;
			F90_Arr3D<double> gridcellhydro;
			F90_Arr2D<unsigned> gridsons;
			int* sonindex = 0;

			unsigned gridsthistask = 0;
			unsigned firstgrid = 0;
			unsigned gridsthisdomain = 0;
			unsigned currentworkproc = 0;

			// Loop over levels within file
			for(unsigned ilevel = 0; ilevel < amr.meta.nlevelmax; ilevel++)
			{
				// Loop over domains within level (this would be ncpu+nboundary if nboundary>0)
				for(unsigned idomain = 0; idomain < amr.meta.ncpu+amr.meta.nboundary; idomain++)
				{

					// Check there are grids for this domain, process dependant on parallel read mode
					if(parallelmode == 0)
						gridsthistask = ngridfile(idomain,ilevel);
					else
					{
						if(ngridfile(idomain,ilevel) > 0)
						{
							// Work out how many grids this task should read
							gridsthisdomain = ngridfile(idomain,ilevel);

							// If less grids than MIN_GRIDS_PER_READ*tasks, reduce tasks until we have a useful share ratio 
							if(gridsthisdomain < MIN_GRIDS_PER_READ*ntasks)
							{
								// Each task should read minimum of MIN_GRIDS_PER_READ grids, work out how many tasks to use
								int subtasks = gridsthisdomain/MIN_GRIDS_PER_READ;
								
								// If we only need one processor
								if(subtasks<2) 
								{
									subtasks = 1;
									if(rankthistask==currentworkproc)
									{
										firstgrid = 0;
										gridsthistask = gridsthisdomain;
									}
									else 
									{
										gridsthistask = 0;
										firstgrid = 0;
									}
								}
								// Is this task is amongst subtasks designated with reading?
								else if( (rankthistask >= currentworkproc && rankthistask < (currentworkproc+subtasks)) ||
								    (currentworkproc+subtasks >= ntasks && rankthistask < (currentworkproc+subtasks-ntasks)) ) 
								{
									// Check how many grids to read, and where to start reading
									gridsthistask = gridsthisdomain/subtasks;
									// Work out subrank (rank relative to number of subtasks)
									int subrank = rankthistask;
									while(subrank >= subtasks)
										subrank -= subtasks;
									firstgrid = gridsthistask*subrank;
									// Final process reads remainder
									if(subrank == subtasks-1)
										gridsthistask = gridsthisdomain-firstgrid;

								}
								else 
								{
									gridsthistask = 0;
									firstgrid = 0;
								}
								// Move currentworkproc
								currentworkproc+=subtasks;
								if(currentworkproc>=ntasks)
									currentworkproc -= ntasks;
							}
							// In this case we have enough grids to allocate at least 100 grids per task
							else
							{
								// Split grids amongst tasks
								gridsthistask = gridsthisdomain/ntasks;
								firstgrid = gridsthistask*rankthistask;

	                            // Work out remaining grids and add to task's load
	                            int remain = gridsthisdomain % ntasks;
	                            if(remain)
	                            {
	                            	if(remain>rankthistask)
	                            	{
	                            		firstgrid+=rankthistask;
	                            		gridsthistask+=1;
	                            	}
	                            	else
	                            		firstgrid+=remain;
	                            }
							}

							//std::cout <<" ilevel: "<<ilevel<< " Idomain: "<<idomain<<" Rank "<<rankthistask<<": Reading "<<gridsthistask<<" grids out of "<<gridsthisdomain<<" first: "<<firstgrid<<std::endl;
						}
						else
							gridsthistask = 0;
					}

					// If this task is allocated grids to read
					if(gridsthistask > 0)
					{

						// Allocate arrays
						// Used to tempstore one type of variable for n grids (n grids refers to all grids in this domain and level) (xdp)
						gridvars = new double[gridsthistask];
						// Used to store 3 coords for n grids (xxdp)
						gridcoords.resize(gridsthistask, amr.meta.ndim);
						// Used to store set of hydro variables per cell (8) for n grids (vvdp)
						gridcellhydro.resize(gridsthistask,amr.meta.twotondim,hydro.meta.nvar);
						// Used to store son indices of n grids (8 indices per grid) (sdp)
						gridsons.resize(gridsthistask,amr.meta.twotondim);
						// Tempstore particular cell son index for n grids when reading (idp)
						sonindex = new int[gridsthistask]; 

						// Start reading AMR data
						// Skip grid index, next index and prev index
						amr.file.SkipRecords(3);
						
						// Read centre point of grid for n grids
						for(unsigned idim = 0; idim < amr.meta.ndim; idim++)
						{
							// Read n grid coordinates for 1 dimension and store in gridcoords
							// Read subarray
							amr.file.Read1DArray(gridvars, firstgrid, gridsthistask);
							for(unsigned igrid = 0; igrid < gridsthistask; igrid++)
								gridcoords(igrid,idim) = gridvars[igrid];
						}
						// Skip father (1) and neighbour (2ndim) indices
						amr.file.SkipRecords(1+(2*amr.meta.ndim));
						
						// Read son (cell) indices, 8 per grid
						for(unsigned icell = 0; icell < amr.meta.twotondim; icell++)
						{
							// Read subarray
							amr.file.Read1DArray(sonindex, firstgrid, gridsthistask);
							for(unsigned igrid = 0; igrid < gridsthistask; igrid++)
								gridsons(igrid,icell) = sonindex[igrid];
						}
						// Skip cpu and refinement maps
						amr.file.SkipRecords(amr.meta.twotondim*2);
					}
					// Otherwise skip these grids (only in parallelmode 1)
					else if((ngridfile(idomain,ilevel) > 0) && (gridsthistask == 0) && (parallelmode == 1))
						amr.file.SkipRecords(3+amr.meta.ndim+1+(2*amr.meta.ndim)+amr.meta.twotondim+(amr.meta.twotondim*2));

					// Start reading hydro data
					hydro.file.SkipRecords(2);

					// If this task is allocated grids to read
					if(gridsthistask > 0)
					{
						// Loop over cells (8 per grid)
						for(unsigned icell = 0; icell < amr.meta.twotondim; icell++)
						{
							// Loop over hydro variables
							for(unsigned ivar = 0; ivar < hydro.meta.nvar; ivar++)
							{
								// Read set of hydro variables for a particular cell for n grids
								// Read subarray
								hydro.file.Read1DArray(gridvars, firstgrid, gridsthistask);
								// Loop through n grids for this domain at this level
								for(unsigned igrid = 0; igrid < gridsthistask; igrid++)
								{
									
									// Store hydro variables in correct location
									gridcellhydro(igrid,icell,ivar) = gridvars[igrid];
									// Conditions: store data only if final loop of hydros (ie, read all hydro data)
									// current domain must match current file..?
									// Current cell must be at highest level of refinement (ie son indexes are 0) or ilevel == lmax
									if((ivar==hydro.meta.nvar-1) && (idomain==icpu) && (gridsons(igrid,icell)==0 || ilevel == amr.meta.nlevelmax))
									{
										double dx = pow(0.5, ilevel);
										int ix, iy, iz;
										iz = icell/4;
										iy = (icell - (4*iz))/2;
										ix = icell - (2*iy) - (4*iz);
										// Calculate absolute coordinates + jitter, and generate particle
										// randomize location within cell using same method as io_ramses.f90:
										// call ranf(localseed,xx)
                                        // xp(pc,l)=xx*boxlen*dx+xc(l)-boxlen*dx/2
                                        // float r = (float)rand()/(float)RAND_MAX;
										particle_sim p;
										p.x = ((((float)rand()/(float)RAND_MAX) * amr.meta.boxlen * dx) +(amr.meta.boxlen * (gridcoords(igrid,0) + (double(ix)-0.5) * dx )) - (amr.meta.boxlen*dx/2)) * scale3d;
										p.y = ((((float)rand()/(float)RAND_MAX) * amr.meta.boxlen * dx) +(amr.meta.boxlen * (gridcoords(igrid,1) + (double(iy)-0.5) * dx )) - (amr.meta.boxlen*dx/2)) * scale3d;
										p.z = ((((float)rand()/(float)RAND_MAX) * amr.meta.boxlen * dx) +(amr.meta.boxlen * (gridcoords(igrid,2) + (double(iz)-0.5) * dx )) - (amr.meta.boxlen*dx/2)) * scale3d;
										// Smoothing length set by box resolution
										// Can be scaled via scale3d if box has been scaled up
										// Can be modified with multiplicative factor smooth_factor
										p.r = (amr.meta.boxlen * scale3d * dx * smooth_factor[0]);
										p.I = (const_i) ? intensity[0] : gridcellhydro(igrid,icell,intensity[0]);
										p.e.r = gridcellhydro(igrid,icell,C1);
										p.e.g = gridcellhydro(igrid,icell,C2);
										p.e.b = gridcellhydro(igrid,icell,C3);
										p.type = 0;
										p.active = active;

										if(doSample)
											pointfilter.push_back(p);
										else
											points.push_back(p);
									}
								} // End loop over grids
							} // End loop over hydro vars
						} // End loop over cells
						if(gridvars) delete[] gridvars;
						gridcoords.Delete();
						gridcellhydro.Delete();
						gridsons.Delete();
						if(sonindex) delete[] sonindex;
					} 
					// Skip grids from a domain that contains grids but none of which are read by this task (only for parallel mode 1)
					else if(ngridfile(idomain,ilevel) > 0 && gridsthistask == 0 && parallelmode == 1)
						hydro.file.SkipRecords(hydro.meta.nvar*amr.meta.twotondim);

					// Sampling
					if(doSample)
					{
						int stride = (100/sample_factor);
						for(int i = 0; i < pointfilter.size(); i += stride)
							points.push_back(pointfilter[i]);
						pointfilter.clear();
					}

				} // End loop over domains
			} // End loop over levels
		} // End loop over files

		if(mpiMgr.master())
			std::cout << "Read " << nlevelmax << " levels for " << info.ncpu << " domains." << std::endl;
	}

	// read points
	if(mode == 1 || mode == 2)
	{
		int type = (mode==1) ? 0 : 1;

		int C1 = params.find<int>("red"+dataToString(type),-1);
		int C2 = params.find<int>("green"+dataToString(type),-1);
		int C3 = params.find<int>("blue"+dataToString(type),-1);

		// Check for constant intensity, if so set i to this value
		bool const_i = false;
		if(const_intensity[type] > -1)
		{
			const_i = true;
			intensity[type] = const_intensity[type];
		}
		// Set default intensity if neither variables are set
		else if(const_intensity[type] == -1 && intensity[type] == -1)
		{
			const_i = true;
			intensity[type] = 1;
		}

		// Check how many colour variables have been set in param file
		// For unset variables, set to velocity.
		int check = 0;
		if(C1 != -1) ++check;
		else C1 = 0;
		if(C2 != -1) ++check;
		else C2 = 1;
		if(C3 != -1) ++check;
		else C3 = 2;
		if(check == 2) 
		{
			std::cout << "Ramses particle reader: must set either one or three param file elements red,green,blue, not two. Aborting." << std::endl;
			exit(0);
		}

		// Validate params
		if( (C1 != -1 && (C1 < 0 || C1 > 4)) || (C2 != -1 && (C2 < 0 || C2 > 4)) ||
			(C3 != -1 && (C3 < 0 || C3 > 4)) || (!const_i && (intensity[type] < 0 || intensity[type] > 4)) )
		{
			std::cout << "Trying to read variables that are not readable, check red,green,blue,intensity parameters. Aborting. ";
			exit(0);
		}

		if(mpiMgr.master())
			std::cout << "Reading particle data..." << std::endl;

		int nfiles = info.ncpu;
		int readsthistask = nfiles;
		int firstread = 0;

		// Parallel mode = 0 means cpu per file (good for generally small files)
		// Parallel mode = 1 means multiple cpus per file (good for generally large files)
		if(parallelmode == 0)
		{
			// If less files than tasks, read one file per task for all available files
			if(nfiles<ntasks)
			{
				if(rankthistask<nfiles)
				{
					firstread = rankthistask;
					readsthistask = 1;
				}
				else 
				{
					// This task will read no particles
					firstread = 0;
					readsthistask = 0;
					// Push a fake particle to stop MPI crying about lack of data
					points.push_back(particle_sim(COLOUR(0,0,0),0,0,0,0,0,0,0));
				}
			}
			else
			{
				// Split files amongst tasks
				readsthistask = nfiles/ntasks;
				firstread = readsthistask*rankthistask;

	            // Work out remaining files and distribute
	            int remain = nfiles % ntasks;
	            if(remain)
	            {
	            	if(remain>rankthistask)
	            	{
	            		firstread+=rankthistask;
	            		readsthistask+=1;
	            	}
	            	else
	            		firstread+=remain;
	            }
			}
		}

		
		for(unsigned ifile = firstread; ifile < (firstread+readsthistask); ifile++)
		{
			// Open file and check header
			part_file part(repo, ifile+1);

			if(part.meta.ndim != 3)
			{
				std::cout << "Ramses particle reader: non 3 dimension data detected. Aborting" << std::endl;
				exit(0);
			}

			// Check number of particles to be read, divide up amongst tasks
			unsigned partsthisfile = part.meta.npart;
			unsigned partsthistask = partsthisfile/ntasks;
			unsigned firstpart = partsthistask*rankthistask;

			if(parallelmode == 0)
			{
				firstpart = 0;
				partsthistask = partsthisfile;
			}
			else
			{
				// Spread remainder of particles amongst tasks
				int remain = partsthisfile%ntasks;
				if(remain)
				{
					if(remain>rankthistask)
					{
						firstpart+=rankthistask;
						partsthistask+=1;
					}
					else
						firstpart+=remain;				
				}

				// Handle the rare occurance that there are less particles than tasks. 
				// In this case rank 0 reads all particles.
				if(partsthisfile < ntasks)
				{
					if(mpiMgr.master())
					{
						firstpart = 0;
						partsthistask = partsthisfile;
					}
					else
						continue;
				}
			}
			//std::cout<<"ifile: "<<ifile<<" rank: "<<rankthistask<<" P_thisfile: "<<partsthisfile<<" P_thistask: "<<partsthistask<<" P_1: "<<firstpart<<std::endl; 


			// Resize for extra particles
			//int previousSize = points.size();
			//points.resize(points.size()+partsthistask);

			// Doubel check pointfilter is empty
			pointfilter.clear();
			pointfilter.resize(partsthistask);

			float* fstorage = 0;
			double* dstorage = 0;

			// Check data type, following reads depend on this...
			char dType;
			int dSize;
			part.file.Peek(dSize);
			if(dSize/sizeof(float) == part.meta.npart)
			{
				fstorage = new float[partsthistask];
				dType = 'f';
			}
			else if(dSize/sizeof(double) == part.meta.npart)
			{
				dstorage = new double[partsthistask];
				dType = 'd';
			}
			else
			{
				std::cout << "Ramses particle reader: Checking data type failed, data size: " << dSize << std::endl;
				std::cout << "npart: " << part.meta.npart << std::endl;
				std::cout << "Aborting." << std::endl;

				exit(0);
			}

			// Read positions
			if(dType=='f')
			{
				part.file.Read1DArray(fstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].x = fstorage[i] * scale3d;

				part.file.Read1DArray(fstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].y = fstorage[i] * scale3d;

				part.file.Read1DArray(fstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].z = fstorage[i] * scale3d;
			}
			else
			{
				part.file.Read1DArray(dstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].x = dstorage[i] * scale3d;

				part.file.Read1DArray(dstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].y = dstorage[i] * scale3d;

				part.file.Read1DArray(dstorage, firstpart, partsthistask);
				for(unsigned i = 0; i < pointfilter.size(); i++)
					pointfilter[i].z = dstorage[i] * scale3d;
			}

			// Read appropriate data
			for(int idata = 0; idata < 5; idata++)
			{
				if(C1 == idata)
				{
					// If metallicity is requested, skip to metallicity position.
					// Dont request metallicity if it is not in the file...
					if(idata==4) part.file.SkipRecords(3);
					if(dType=='f')
					{
						part.file.Read1DArray(fstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.r = fstorage[i];		
					}
					else
					{
						part.file.Read1DArray(dstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.r = dstorage[i];							
					}		
				}
				else if(C2 == idata)
				{
					if(idata==4) part.file.SkipRecords(3);
					if(dType=='f')
					{
						part.file.Read1DArray(fstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.g = fstorage[i];		
					}
					else
					{
						part.file.Read1DArray(dstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.g = dstorage[i];							
					}	
				}
				else if(C3 == idata)
				{
					if(idata==4) part.file.SkipRecords(3);
					if(dType=='f')
					{
						part.file.Read1DArray(fstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.b = fstorage[i];		
					}
					else
					{
						part.file.Read1DArray(dstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].e.b = dstorage[i];							
					}	
				}
				else if(!const_i && intensity[type] == idata)
				{
					if(idata==4) part.file.SkipRecords(3);
					if(dType=='f')
					{
						part.file.Read1DArray(fstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].I = fstorage[i] * intense_factor[type];		
					}
					else
					{
						part.file.Read1DArray(dstorage, firstpart, partsthistask);
						for(unsigned i = 0; i < pointfilter.size(); i++)
							pointfilter[i].I = dstorage[i] * intense_factor[type];							
					}	
				}
				// Skip to next record if not final read
				else if(idata < 4) part.file.SkipRecord();				
			}


			// Insert param data
			for(unsigned i = 0; i < pointfilter.size(); i++)
			{
				pointfilter[i].r = smooth_factor[type];
				pointfilter[i].type = type;
				pointfilter[i].active = active;
				if(const_i) pointfilter[i].I = intensity[type] * intense_factor[type];
			}

			// If sampling, do sample, else memcpy into correct place
			if(doSample)
			{
				int stride = (100/sample_factor);
				for(int i = 0; i < pointfilter.size(); i += stride)
					points.push_back(pointfilter[i]);
				pointfilter.clear();				
			}
			else
			{
				int size = points.size(); 
				points.resize(size+pointfilter.size());
				memcpy(&points[size],&pointfilter[0],pointfilter.size()*sizeof(particle_sim));
			}

			// Clean up memory
			if(fstorage) delete[] fstorage;
			if(dstorage) delete[] dstorage;
		}

	}
	
	if(mode!= 0  && mode != 1 && mode != 2)
	{
		if(mpiMgr.master())
		{
			// Explain mode parameter usage and quit
			std::cout << "Reader parameters incorrect. Please set read_mode in parameter file (default 0).\n";
			std::cout << "read_mode=0 for amr data only.\n";
			std::cout << "read_mode=1 for point data only.\n";
			std::cout << "read_mode=2 for both amr + point data.\n";
			std::cout << "Note: You must have amr_ and hydro_ files for opt 0, part_ files for opt1, and all 3 for opt 2." << std::endl;
		}
		exit(0);
	}

}
