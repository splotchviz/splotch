/*
 *  This file is part of libcxxsupport.
 *
 *  libcxxsupport is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libcxxsupport is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libcxxsupport; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libcxxsupport is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Copyright (C) 2009-2011 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifdef USE_MPI
#include "mpi.h"
#else
#include <cstring>
#include <cstdlib>
#endif
#include "mpi_support.h"

MPI_Manager mpiMgr;

using namespace std;

namespace {

void assert_unequal (const void *a, const void *b)
  { planck_assert (a!=b,"input and output buffers must not be identical"); }

} // unnamed namespace

#ifdef USE_MPI

#define LS_COMM MPI_COMM_WORLD

namespace {

MPI_Datatype ndt2mpi (NDT type)
  {
  switch (type)
    {
    case NAT_CHAR: return MPI_CHAR;
    case NAT_INT: return MPI_INT;
    case NAT_UINT: return MPI_UNSIGNED;
    case NAT_LONG: return MPI_LONG;
    case NAT_ULONG: return MPI_UNSIGNED_LONG;
    case NAT_LONGLONG: return MPI_LONG_LONG;
    case NAT_ULONGLONG: return MPI_UNSIGNED_LONG_LONG;
    case NAT_FLOAT: return MPI_FLOAT;
    case NAT_DOUBLE: return MPI_DOUBLE;
    case NAT_FCMPLX: return MPI_COMPLEX;
    case NAT_DCMPLX: return MPI_DOUBLE_COMPLEX;
    case NAT_LONGDOUBLE: return MPI_LONG_DOUBLE;
    default: planck_fail ("Unsupported type");
    }
  }
MPI_Op op2mop (MPI_Manager::redOp op)
  {
  switch (op)
    {
    case MPI_Manager::Min : return MPI_MIN;
    case MPI_Manager::Max : return MPI_MAX;
    case MPI_Manager::Sum : return MPI_SUM;
    case MPI_Manager::Prod: return MPI_PROD;
    default: planck_fail ("unsupported reduction operation");
    }
  }

} // unnamed namespace

#endif

#ifdef USE_MPI

void MPI_Manager::gatherv_helper1_m (int nval_loc, arr<int> &nval,
  arr<int> &offset, int &nval_tot) const
  {
  gather_m (nval_loc, nval);
  nval_tot=0;
  for (tsize i=0; i<nval.size(); ++i)
    nval_tot+=Int(Ts(nval_tot)+Ts(nval[i]));
  offset.alloc(num_ranks_);
  offset[0]=0;
  for (tsize i=1; i<offset.size(); ++i)
    offset[i]=Int(Ts(offset[i-1])+Ts(nval[i-1]));
  }

void MPI_Manager::all2allv_easy_prep (tsize insz, const arr<int> &numin,
  arr<int> &disin, arr<int> &numout, arr<int> &disout) const
  {
  tsize n=num_ranks_;
  planck_assert (numin.size()==n,"array size mismatch");
  numout.alloc(n); disin.alloc(n); disout.alloc(n);
  all2all (numin,numout);
  disin[0]=disout[0]=0;
  for (tsize i=1; i<n; ++i)
    {
    disin [i]=Int(Ts(disin [i-1])+Ts(numin [i-1]));
    disout[i]=Int(Ts(disout[i-1])+Ts(numout[i-1]));
    }
  planck_assert(insz==tsize(disin[n-1]+numin[n-1]), "incorrect array size");
  }

#else

void MPI_Manager::gatherv_helper1_m (int nval_loc, arr<int> &nval,
  arr<int> &offset, int &nval_tot) const
  {
  nval.alloc(1);
  nval[0]=nval_tot=nval_loc;
  offset.alloc(1);
  offset[0]=0;
  }

void MPI_Manager::all2allv_easy_prep (tsize insz, const arr<int> &numin,
  arr<int> &disin, arr<int> &numout, arr<int> &disout) const
  {
  planck_assert (numin.size()==1,"array size mismatch");
  numout=numin; disin.allocAndFill(1,0); disout.allocAndFill(1,0);
  planck_assert(insz==tsize(numin[0]), "incorrect array size");
  }

#endif

#ifdef USE_MPI

MPI_Manager::MPI_Manager ()
  {
  int flag;
  MPI_Initialized(&flag);
  if (!flag)
    {
    MPI_Init(0,0);
    MPI_Errhandler_set(LS_COMM, MPI_ERRORS_ARE_FATAL);
    }
  MPI_Comm_size(LS_COMM, &num_ranks_);
  MPI_Comm_rank(LS_COMM, &rank_);
  }
MPI_Manager::~MPI_Manager ()
  {
  int flag;
  MPI_Finalized(&flag);
  if (!flag)
    MPI_Finalize();
  }

void MPI_Manager::abort() const
  { MPI_Abort(LS_COMM, 1); }

void MPI_Manager::barrier() const
  { MPI_Barrier(LS_COMM); }

#else

MPI_Manager::MPI_Manager ()
  : num_ranks_(1), rank_(0) {}
MPI_Manager::~MPI_Manager () {}

void MPI_Manager::abort() const
  { exit(1); }

void MPI_Manager::barrier() const {}

#endif

#ifdef USE_MPI
void MPI_Manager::sendRawVoid (const void *data, NDT type, tsize num,
  tsize dest) const
  { MPI_Send(const_cast<void *>(data),num,ndt2mpi(type),dest,0,LS_COMM); }
void MPI_Manager::recvRawVoid (void *data, NDT type, tsize num, tsize src) const
  { MPI_Recv(data,num,ndt2mpi(type),src,0,LS_COMM,MPI_STATUS_IGNORE); }
void MPI_Manager::sendrecvRawVoid (const void *sendbuf, tsize sendcnt,
  tsize dest, void *recvbuf, tsize recvcnt, tsize src, NDT type) const
  {
  assert_unequal(sendbuf,recvbuf);

  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Sendrecv (const_cast<void *>(sendbuf),sendcnt,dtype,dest,0,
    recvbuf,recvcnt,dtype,src,0,LS_COMM,MPI_STATUS_IGNORE);
  }
void MPI_Manager::sendrecv_replaceRawVoid (void *data, NDT type, tsize num,
  tsize dest, tsize src) const
  {
  MPI_Sendrecv_replace (data,num,ndt2mpi(type),dest,0,src,0,LS_COMM,
    MPI_STATUS_IGNORE);
  }

void MPI_Manager::gatherRawVoid (const void *in, tsize num, void *out, NDT type,
  int root) const
  {
  assert_unequal(in,out);
  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Gather(const_cast<void *>(in),1,dtype,out,num,dtype,root,LS_COMM);
  }
void MPI_Manager::gathervRawVoid (const void *in, tsize num, void *out,
  const int *nval, const int *offset, NDT type) const
  {
  assert_unequal(in,out);
  MPI_Datatype dtype = ndt2mpi(type);
  MPI_Gatherv(const_cast<void *>(in),num,dtype,out,const_cast<int *>(nval),
    const_cast<int *>(offset),dtype,0,LS_COMM);
  }

void MPI_Manager::allgatherRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  {
  assert_unequal(in,out);
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Allgather (const_cast<void *>(in),num,tp,out,num,tp,LS_COMM);
  }
void MPI_Manager::allreduceRawVoid (void *data, NDT type,
  tsize num, redOp op) const
  { MPI_Allreduce (MPI_IN_PLACE,data,num,ndt2mpi(type),op2mop(op),LS_COMM); }
void MPI_Manager::allreduceRawVoid (const void *in, void *out, NDT type,
  tsize num, redOp op) const
  {
  assert_unequal(in,out);
  MPI_Allreduce (const_cast<void *>(in),out,num,ndt2mpi(type),op2mop(op),
    MPI_COMM_WORLD);
  }
void MPI_Manager::reduceRawVoid (const void *in, void *out, NDT type, tsize num,
  redOp op, int root) const
  {
  assert_unequal(in,out);
  MPI_Reduce (const_cast<void *>(in),out,num,ndt2mpi(type),op2mop(op), root,
    LS_COMM);
  }

void MPI_Manager::bcastRawVoid (void *data, NDT type, tsize num, int root) const
  { MPI_Bcast (data,num,ndt2mpi(type),root,LS_COMM); }

void MPI_Manager::all2allRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  {
  assert_unequal(in,out);
  planck_assert (num%num_ranks_==0,
    "array size is not divisible by number of ranks");
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Alltoall (const_cast<void *>(in),num/num_ranks_,tp,out,num/num_ranks_,
    tp,LS_COMM);
  }

void MPI_Manager::all2allvRawVoid (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, NDT type)
  const
  {
  assert_unequal(in,out);
  MPI_Datatype tp = ndt2mpi(type);
  MPI_Alltoallv (const_cast<void *>(in), const_cast<int *>(numin),
    const_cast<int *>(disin), tp, out, const_cast<int *>(numout),
    const_cast<int *>(disout), tp, LS_COMM);
  }

void MPI_Manager::all2allvRawVoidBlob (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, int sz)
  const
  {
  assert_unequal(in,out);
  MPI_Datatype tp;
  MPI_Type_contiguous (sz,MPI_CHAR,&tp);
  MPI_Type_commit(&tp);
  MPI_Alltoallv (const_cast<void *>(in), const_cast<int *>(numin),
    const_cast<int *>(disin), tp, out, const_cast<int *>(numout),
    const_cast<int *>(disout), tp, LS_COMM);
  MPI_Type_free(&tp);
  }

#else

void MPI_Manager::sendRawVoid (const void *, NDT, tsize, tsize) const
  { planck_fail("not supported in scalar code"); }
void MPI_Manager::recvRawVoid (void *, NDT, tsize, tsize) const
  { planck_fail("not supported in scalar code"); }
void MPI_Manager::sendrecvRawVoid (const void *sendbuf, tsize sendcnt,
  tsize dest, void *recvbuf, tsize recvcnt, tsize src, NDT type) const
  {
  assert_unequal(sendbuf,recvbuf);
  planck_assert ((dest==0) && (src==0), "inconsistent call");
  planck_assert (sendcnt==recvcnt, "inconsistent call");
  memcpy (recvbuf, sendbuf, sendcnt*ndt2size(type));
  }
void MPI_Manager::sendrecv_replaceRawVoid (void *, NDT, tsize, tsize dest,
  tsize src) const
  { planck_assert ((dest==0) && (src==0), "inconsistent call"); }

void MPI_Manager::gatherRawVoid (const void *in, tsize num, void *out, NDT type,
  int root) const
  {
  planck_assert(root==0, "invalid root task");
  assert_unequal(in,out);
  memcpy (out, in, num*ndt2size(type));
  }
void MPI_Manager::gathervRawVoid (const void *in, tsize num, void *out,
  const int *, const int *, NDT type) const
  { assert_unequal(in,out); memcpy (out, in, num*ndt2size(type)); }
void MPI_Manager::allgatherRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  { assert_unequal(in,out); memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::allreduceRawVoid (const void *in, void *out, NDT type,
  tsize num, redOp) const
  { assert_unequal(in,out); memcpy (out, in, num*ndt2size(type)); }
void MPI_Manager::allreduceRawVoid (void *, NDT, tsize, redOp) const
  {}
void MPI_Manager::reduceRawVoid (const void *in, void *out, NDT type, tsize num,
  redOp, int) const
  { assert_unequal(in,out); memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::bcastRawVoid (void *, NDT, tsize, int) const
  {}

void MPI_Manager::all2allRawVoid (const void *in, void *out, NDT type,
  tsize num) const
  { assert_unequal(in,out); memcpy (out, in, num*ndt2size(type)); }

void MPI_Manager::all2allvRawVoid (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, NDT type)
  const
  {
  assert_unequal(in,out);
  planck_assert (numin[0]==numout[0],"message size mismatch");
  const char *in2 = static_cast<const char *>(in);
  char *out2 = static_cast<char *>(out);
  tsize st=ndt2size(type);
  memcpy (out2+disout[0]*st,in2+disin[0]*st,numin[0]*st);
  }
void MPI_Manager::all2allvRawVoidBlob (const void *in, const int *numin,
  const int *disin, void *out, const int *numout, const int *disout, int sz)
  const
  {
  assert_unequal(in,out);
  planck_assert (numin[0]==numout[0],"message size mismatch");
  const char *in2 = static_cast<const char *>(in);
  char *out2 = static_cast<char *>(out);
  memcpy (out2+disout[0]*sz,in2+disin[0]*sz,numin[0]*sz);
  }

#endif
