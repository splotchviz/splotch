
/*

Bonsai reader created using 
https://github.com/egaburov/bonsaiRender/blob/master/render/main.cpp
stored under GNU Public Licence V2

*/

#include "reader.h"
#include "cxxsupport/error_handling.h"

#ifdef USE_MPI

#include "BonsaiIO.h"
#include <math.h>

void bonsai_reader(paramfile &params, std::vector<particle_sim> &points)
{
  const int reduceDM = 1;
  const int reduceS = 1;

    // Get MPI info
  const int nranks = mpiMgr.num_ranks();
  const int rank = mpiMgr.rank();

  MPI_Comm comm = MPI_COMM_WORLD;

  std::string fileName = params.find<std::string>("infile","");

  fprintf(stderr, "Splotch reader for Bonsai data (in-progress)\n");

  // Setup IO object and print header
  if (rank == 0)
    fprintf(stderr, " ----------- \n");
  BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::READ, fileName);
  if (rank == 0)
    out.getHeader().printFields();

  // Datatypes for start particles
  typedef float float4[4];
  typedef float float3[3];
  typedef float float2[2];

  BonsaiIO::DataType<BonsaiIO::IDType> IDListS("Stars:IDType");
  BonsaiIO::DataType<float4> posS("Stars:POS:real4");
  BonsaiIO::DataType<float3> velS("Stars:VEL:float[3]");
  BonsaiIO::DataType<float2> rhohS("Stars:RHOH:float[2]");

  // If reduceS, read star particles
  if (reduceS > 0)
  {
    if (rank  == 0)
      fprintf(stderr, " Reading star data \n");
    if (!out.read(IDListS, true, reduceS)) printf("(!out.read(IDListS, true, reduceS))\n");
    assert(out.read(posS,    true, reduceS));
    assert(out.read(velS,    true, reduceS));
    bool renderDensity = true;
    if (!out.read(rhohS,  true, reduceS))
    {
      if (rank == 0)
      {
        fprintf(stderr , " -- Stars RHOH data is found \n");
        fprintf(stderr , " -- rendering stars w/o density info \n");
      }
      renderDensity = false;
    }
    assert(IDListS.getNumElements() == posS.getNumElements());
    assert(IDListS.getNumElements() == velS.getNumElements());
    if (renderDensity)
      assert(IDListS.getNumElements() == posS.getNumElements());
  }

  // Datatypes for dark matter particles
  BonsaiIO::DataType<BonsaiIO::IDType> IDListDM("DM:IDType");
  BonsaiIO::DataType<float4> posDM("DM:POS:real4");
  BonsaiIO::DataType<float3> velDM("DM:VEL:float[3]");
  BonsaiIO::DataType<float2> rhohDM("DM:RHOH:float[2]");

  // If reduceDM, read dm particles
  if (reduceDM > 0)
  {
    if (rank  == 0)
      fprintf(stderr, " Reading DM data \n");
    if(!out.read(IDListDM, true, reduceDM)) printf("(!out.read(IDListDM, true, reduceDM))\n");
    assert(out.read(posDM,    true, reduceDM));
    assert(out.read(velDM,    true, reduceDM));
    bool renderDensity = true;
    if (!out.read(rhohDM,  true, reduceDM))
    {
      if (rank == 0)
      {
        fprintf(stderr , " -- DM RHOH data is found \n");
        fprintf(stderr , " -- rendering stars w/o density info \n");
      }
      renderDensity = false;
    }
    assert(IDListS.getNumElements() == posS.getNumElements());
    assert(IDListS.getNumElements() == velS.getNumElements());
    if (renderDensity)
      assert(IDListS.getNumElements() == posS.getNumElements());
  }

  // MPI reduce to get total particles
  const int nS  = IDListS.getNumElements();
  const int nDM = IDListDM.getNumElements();
  long long int nSloc = nS, nSglb;
  long long int nDMloc = nDM, nDMglb;

  MPI_Allreduce(&nSloc, &nSglb, 1, MPI_LONG, MPI_SUM, comm);
  MPI_Allreduce(&nDMloc, &nDMglb, 1, MPI_LONG, MPI_SUM, comm);
  if (rank == 0)
  {
    fprintf(stderr, "nStars = %lld\n", nSglb);
    fprintf(stderr, "nDM    = %lld\n", nDMglb);
  }

  // Temporary compute bounding box for ease of Splotch camera placement
  // Max and mins for stars
  double maxx,minx;
  maxx = posS[0][0];
  minx = posS[0][0];
  double maxy,miny;
  maxy = posS[0][1];
  miny = posS[0][1];
  double maxz,minz;
  maxz = posS[0][2];
  minz = posS[0][2];

  // Max and mins for DM
  double maxdmx,mindmx;
  maxdmx = posDM[0][0];
  mindmx = posDM[0][0];
  double maxdmy,mindmy;
  maxdmy = posDM[0][1];
  mindmy = posDM[0][1];
  double maxdmz,mindmz;
  maxdmz = posDM[0][2];
  mindmz = posDM[0][2];

  // Get some parameters for Splotch data
  float starRadius = params.find<float>("bonsai_r_star",1);
  float starIntensity = params.find<float>("bonsai_I_star",1);
  float dmRadius = params.find<float>("bonsai_r_dm",1);
  float dmIntensity = params.find<float>("bonsai_I_dm",1);

  // Convert to Splotch data
  // Use mass as colour
  points.resize(nS+nDM);
  for (int i = 0; i < nS; i++)
  {

  	points[i].x = posS[i][0];
  	points[i].y = posS[i][1];
  	points[i].z = posS[i][2];

    // Get max/min
  	maxx = (maxx > points[i].x) ? maxx : points[i].x;
  	minx = (minx < points[i].x) ? minx : points[i].x;
  	maxy = (maxy > points[i].x) ? maxy : points[i].y;
  	miny = (miny < points[i].x) ? miny : points[i].y;
  	maxz = (maxz > points[i].x) ? maxz : points[i].z;
  	minz = (minz < points[i].x) ? minz : points[i].z;

  	points[i].type = 0;
  	//points[i].r = starRadius;
  	points[i].r = rhohS[i][1]*starRadius;
        points[i].I = starIntensity*log10(rhohS[i][1]); 
  	points[i].e.r = log10(rhohS[i][0]+0.0000001);
        points[i].e.g = 1;
        points[i].e.b = 1;


    // For veloc & rhoh etc

    // const int ip = i;
    // rData.posx(ip) = posS[i][0];
    // rData.posy(ip) = posS[i][1];
    // rData.posz(ip) = posS[i][2];
    // rData.ID  (ip) = IDListS[i].getID();
    // rData.type(ip) = IDListS[i].getType();
    // assert(rData.type(ip) == 1); /* sanity check */
    //rData.attribute(RendererData::MASS, ip) = posS[i][3];
    // rData.attribute(RendererData::VEL,  ip) =
    //   std::sqrt(
    //       velS[i][0]*velS[i][0] +
    //       velS[i][1]*velS[i][1] +
    //       velS[i][2]*velS[i][2]);
    // if (rhohS.size() > 0)
    // {
    //   rData.attribute(RendererData::RHO, ip) = rhohS[i][0];
    //   rData.attribute(RendererData::H,  ip)  = rhohS[i][1];
    // }
    // else
    // {
    //   rData.attribute(RendererData::RHO, ip) = 0.0;
    //   rData.attribute(RendererData::H,   ip) = 0.0;
    // }
  }

  // Do the same for dark matter
  long index=0;
  for (int i = nS; i < nS+nDM; i++)
  {
  	points[i].x = posDM[index][0];
  	points[i].y = posDM[index][1];
  	points[i].z = posDM[index][2];

	maxdmx = (maxdmx > points[i].x) ? maxdmx : points[i].x;
  	mindmx = (mindmx < points[i].x) ? mindmx : points[i].x;
  	maxdmy = (maxdmy > points[i].x) ? maxdmy : points[i].y;
  	mindmy = (mindmy < points[i].x) ? mindmy : points[i].y;
  	maxdmz = (maxdmz > points[i].x) ? maxdmz : points[i].z;
  	mindmz = (mindmz < points[i].x) ? mindmz : points[i].z;

  	points[i].type = 1;
  	points[i].r = rhohDM[index][1]*dmRadius;
        points[i].I = dmIntensity*log10(rhohDM[index][1]); 
  	//points[i].e.r = posS[i][3];
  	points[i].e.r = log10(rhohDM[index][0]+0.0000001);
        points[i].e.g = 1;
        points[i].e.b = 1;

        index++;

    // For veloc & rhoh etc

    // const int ip = i + nS;
    // rData.posx(ip) = posDM[i][0];
    // rData.posy(ip) = posDM[i][1];
    // rData.posz(ip) = posDM[i][2];
    // rData.ID  (ip) = IDListDM[i].getID();
    // rData.type(ip) = IDListDM[i].getType();
    // assert(rData.type(ip) == 0); /* sanity check */
    // rData.attribute(RendererData::MASS, ip) = posDM[i][3];
    // rData.attribute(RendererData::VEL,  ip) =
    //   std::sqrt(
    //       velDM[i][0]*velDM[i][0] +
    //       velDM[i][1]*velDM[i][1] +
    //       velDM[i][2]*velDM[i][2]);
    // if (rhohDM.size() > 0)
    // {
    //   rData.attribute(RendererData::RHO, ip) = rhohDM[i][0];
    //   rData.attribute(RendererData::H,   ip) = rhohDM[i][1];
    // }
    // else
    // {
    //   rData.attribute(RendererData::RHO, ip) = 0.0;
    //   rData.attribute(RendererData::H,   ip) = 0.0;
    // }
  }

  printf("Star physical boundaries: \n");
  printf("minx: %f maxx: %f\nminy: %f maxy: %f\nminz: %f maxz: %f\n",minx,maxx,miny,maxy,minz,maxz);

  printf("DM physical boundaries: \n");
  printf("minx: %f maxx: %f\nminy: %f maxy: %f\nminz: %f maxz: %f\n",mindmx,maxdmx,mindmy,maxdmy,mindmz,maxdmz);
}

#else

void bonsai_reader(paramfile &params, std::vector<particle_sim> &points)
{
  planck_fail("Bonsai reader currently only available with MPI Splotch (enable USE_MPI in Makefile)\n");
}


#endif
