/*
Copyright (c) 2009, David Cheng, Viral B. Shah.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
   Original code obtained from  http://gauss.cs.ucsb.edu/code/index.shtml

   Adapted for use within the Planck Simulation package by Martin Reinecke
*/

#ifndef PSORT_H
#define PSORT_H

#include <algorithm>
#include <numeric>
#include <queue>
#ifdef USE_MPI
#include "mpi.h"
#endif
#include "arr.h"
#include "safe_cast.h"

namespace psort {

using namespace std;

#ifdef USE_MPI

template <typename T, typename Tcomp> class PermCompare
  {
  private:
    T *weights;
    Tcomp comp;
  public:
    PermCompare(T *w, Tcomp c) : weights(w), comp(c) {}
    bool operator()(int a, int b)
      { return comp (weights[a], weights[b]); }
  };

template<typename RAIter, typename Tcomp, typename Tdist>
void median_split (RAIter first, RAIter last, const arr<Tdist> &dist,
  Tcomp comp, arr2<Tdist> &right_ends, MPI_Datatype MPI_valueType,
  MPI_Datatype MPI_distanceType, MPI_Comm comm)
  {
  typedef typename iterator_traits<RAIter>::value_type T;

  int nproc, rank;
  MPI_Comm_size (comm, &nproc);
  MPI_Comm_rank (comm, &rank);

  int n_real = nproc;
  for (int i=0; i<nproc; ++i)
    if (dist[i]==0) { n_real=i; break; }

  copy (dist.begin(), dist.end(), &right_ends[nproc][0]);

  // union of [0, right_end[i+1]) on each task produces dist[i] total values
  arr<Tdist> targets(nproc-1);
  partial_sum (dist.begin(), dist.end()-1, &targets[0]);

  // keep a list of ranges, trying to "activate" them at each branch
  arr<pair<RAIter, RAIter> > d_ranges(nproc-1);
  arr<pair<Tdist *, Tdist *> > t_ranges(nproc-1);
  d_ranges[0] = make_pair(first, last);
  t_ranges[0] = make_pair(targets.begin(), targets.end());

  // invariant: subdist[i][rank] == d_ranges[i].second - d_ranges[i].first
  // amount of data each proc still has in the search
  arr2<Tdist> subdist(nproc-1,nproc);
  copy (dist.begin(), dist.end(), &subdist[0][0]);

  // for each task, d_ranges - first
  arr2<Tdist> outleft(nproc-1,nproc,0);

  for (int n_act=1; n_act>0; )
    {
    for (int k=0; k<n_act; ++k)
      planck_assert(subdist[k][rank] == d_ranges[k].second-d_ranges[k].first,
        "subdist error");

    //------- generate n_act guesses

    // for the allgather, make a flat array of nproc chunks, each with n_act
    // elements
    arr<T> medians(nproc*n_act);
    for (int k=0; k<n_act; ++k)
      {
      T *ptr = d_ranges[k].first;
      Tdist index = subdist[k][rank]/2;
      medians[rank*n_act+k] = ptr[index];
      }

    MPI_Allgather (&medians[rank*n_act], n_act, MPI_valueType,
                   &medians[0], n_act, MPI_valueType, comm);

    // compute the weighted median of medians
    arr<T> queries(n_act);

    for (int k=0; k<n_act; ++k)
      {
      arr<Tdist> ms_perm(n_real);
      for (int i=0; i<n_real; ++i) ms_perm[i] = i*n_act+k;
      sort (ms_perm.begin(), ms_perm.end(),
            PermCompare<T,Tcomp> (medians.begin(), comp));

      Tdist mid = accumulate(&subdist[k][0],&subdist[k][0]+nproc,0) / 2;
      Tdist query_ind = -1;
      for (int i=0; i<n_real; ++i)
        {
        if (subdist[k][ms_perm[i]/n_act] == 0) continue;

        mid -= subdist[k][ms_perm[i] / n_act];
        if (mid<=0)
          { query_ind = ms_perm[i]; break; }
        }

      planck_assert(query_ind>=0,"query_ind error");
      queries[k] = medians[query_ind];
      }

    //------- find min and max ranks of the guesses
    arr<Tdist> ind_local(2*n_act);
    for (int k=0; k<n_act; ++k)
      {
      pair<RAIter, RAIter> ind_local_p
        = equal_range (d_ranges[k].first, d_ranges[k].second, queries[k], comp);

      ind_local[2*k] = ind_local_p.first - first;
      ind_local[2*k+1] = ind_local_p.second - first;
      }

    arr<Tdist> ind_all(2*n_act*nproc);
    MPI_Allgather (&ind_local[0], 2*n_act, MPI_distanceType,
                   &ind_all[0], 2*n_act, MPI_distanceType, comm);
    // sum to get the global range of indices
    arr<pair<Tdist, Tdist> > ind_global(n_act);
    for (int k=0; k<n_act; ++k)
      {
      ind_global[k] = make_pair(0, 0);
      for (int i=0; i<nproc; ++i)
        {
        ind_global[k].first += ind_all[2*(i*n_act+k)];
        ind_global[k].second += ind_all[2*(i*n_act+k)+1];
        }
      }

    // state to pass on to next iteration
    arr<pair<RAIter, RAIter> > d_ranges_x(nproc-1);
    arr<pair<Tdist *, Tdist *> > t_ranges_x(nproc-1);
    arr2<Tdist> subdist_x(nproc-1, nproc), outleft_x(nproc-1, nproc, 0);
    int n_act_x=0;

    for (int k=0; k<n_act; ++k)
      {
      Tdist *split_low = lower_bound
        (t_ranges[k].first, t_ranges[k].second, ind_global[k].first);
      Tdist *split_high = upper_bound
        (t_ranges[k].first, t_ranges[k].second, ind_global[k].second);

      // iterate over targets we hit
      for (Tdist *s=split_low; s!=split_high; ++s)
        {
        planck_assert (*s>0,"*s error");
        // a bit sloppy: if more than one target in range, excess won't zero out
        Tdist excess = *s - ind_global[k].first;
        // low procs to high take excess for stability
        for (int i=0; i<nproc; ++i)
          {
          Tdist amount = min (ind_all[2*(i*n_act+k)] + excess,
                              ind_all[2*(i*n_act+k) + 1]);
          right_ends[(s-&targets[0])+1][i] = amount;
          excess -= amount-ind_all[2*(i*n_act+k)];
          }
        }

      if ((split_low-t_ranges[k].first) > 0)
        {
        t_ranges_x[n_act_x] = make_pair(t_ranges[k].first, split_low);
        // lop off local_ind_low..end
        d_ranges_x[n_act_x] = make_pair(d_ranges[k].first,first+ind_local[2*k]);
        for (int i=0; i<nproc; ++i)
          {
          subdist_x[n_act_x][i] = ind_all[2*(i*n_act+k)] - outleft[k][i];
          outleft_x[n_act_x][i] = outleft[k][i];
          }
        ++n_act_x;
        }

      if ((t_ranges[k].second-split_high) > 0)
        {
        t_ranges_x[n_act_x] = make_pair (split_high, t_ranges[k].second);
        // lop off begin..local_ind_high
        d_ranges_x[n_act_x] = make_pair (first+ind_local[2*k+1],
                                         d_ranges[k].second);
        for (int i=0; i<nproc; ++i)
          {
          subdist_x[n_act_x][i] = outleft[k][i]
            + subdist[k][i] - ind_all[2*(i*n_act+k)+1];
          outleft_x[n_act_x][i] = ind_all[2*(i*n_act+k)+1];
          }
        ++n_act_x;
        }
      }

    t_ranges = t_ranges_x;
    d_ranges = d_ranges_x;
    subdist = subdist_x;
    outleft = outleft_x;
    n_act = n_act_x;
    }
  }

// out-of-place tree merge
template<typename RAIter, typename Tcomp, typename Tdist> void oop_tree_merge
  (RAIter in, RAIter out, const arr<Tdist> &disps, int nproc, Tcomp comp)
  {
  if (nproc==1)
    { copy (in, in+disps[nproc], out); return; }

  RAIter bufs[2] = { in, out };
  arr<Tdist> locs(nproc,0);

  Tdist next = 1;
  while (true)
    {
    Tdist stride = next*2;
    if (stride>=nproc)
      break;

    for (Tdist i=0; i+next<nproc; i+=stride)
      {
      Tdist end_ind = min (i+stride, Tdist(nproc));

      std::merge (bufs[locs[i]]+disps[i], bufs[locs[i]]+disps[i+next],
        bufs[locs[i+next]]+disps[i+next], bufs[locs[i+next]]+disps[end_ind],
        bufs[1-locs[i]]+disps[i], comp);
      locs[i] = 1-locs[i];
      }

    next = stride;
    }

  // now we have 4 cases for final merge
  if (locs[0]==0) // 00, 01 => out of place
    std::merge (in, in+disps[next], bufs[locs[next]]+disps[next],
      bufs[locs[next]]+disps[nproc], out, comp);
  else if (locs[next]==0) // 10 => backwards out of place
    std::merge (reverse_iterator<RAIter> (in+disps[nproc]),
                reverse_iterator<RAIter> (in+disps[next]),
                reverse_iterator<RAIter> (out+disps[next]),
                reverse_iterator<RAIter> (out),
                reverse_iterator<RAIter> (out+disps[nproc]),
                not2 (comp));
  else // 11 => in-place
    std::inplace_merge (out, out+disps[next], out+disps[nproc], comp);
  }

template <typename RAIter, typename T, typename Tdist>
static void alltoall (arr2<Tdist> &right_ends, RAIter first, RAIter last,
  arr<T> &trans_data, arr<Tdist> &boundaries, MPI_Datatype MPI_valueType,
  MPI_Comm comm)
  {
  int nproc, rank;
  MPI_Comm_size (comm, &nproc);
  MPI_Comm_rank (comm, &rank);

  // Should be Tdist, but MPI wants ints
  int n_loc = safe_cast<int>(last-first);

  // Calculate the counts for redistributing data
  arr<int> send_counts(nproc), send_disps(nproc),
           recv_counts(nproc), recv_disps(nproc);
  send_disps[0] = recv_disps[0] = 0;
  for (int i=0; i<nproc; ++i)
    {
    send_counts[i] = safe_cast<int>(right_ends[i+1][rank]-right_ends[i][rank]);
    recv_counts[i] = safe_cast<int>(right_ends[rank+1][i]-right_ends[rank][i]);
    }
  partial_sum (send_counts.begin(), send_counts.end()-1, send_disps.begin()+1);
  partial_sum (recv_counts.begin(), recv_counts.end()-1, recv_disps.begin()+1);

  planck_assert (accumulate(&recv_counts[0], &recv_counts[0]+nproc, 0) == n_loc,
    "accumulate error");

  // Do the transpose
  MPI_Alltoallv (first, &send_counts[0], &send_disps[0], MPI_valueType,
    trans_data.begin(), &recv_counts[0], &recv_disps[0], MPI_valueType, comm);

  for (int i=0; i<nproc; ++i) boundaries[i] = Tdist (recv_disps[i]);
  boundaries[nproc] = Tdist(n_loc);  // for the merging
  }

#endif // USE_MPI

template<typename RAIter, typename Tcomp>
void parallel_sort (RAIter first, RAIter last, Tcomp comp)
  {
  typedef typename iterator_traits<RAIter>::value_type T;
  typedef typename iterator_traits<RAIter>::difference_type Tdist;

  // Sort the data locally
  const bool stablesort=false;
  stablesort ? std::stable_sort (first, last, comp)
             : std::sort (first, last, comp);

#ifdef USE_MPI
  MPI_Comm comm = MPI_COMM_WORLD;

  int nproc;
  MPI_Comm_size (comm, &nproc);

  if (nproc==1) return;

  MPI_Datatype MPI_valueType, MPI_distanceType;
  MPI_Type_contiguous (sizeof(T), MPI_CHAR, &MPI_valueType);
  MPI_Type_commit (&MPI_valueType);
  MPI_Type_contiguous (sizeof(Tdist), MPI_CHAR, &MPI_distanceType);
  MPI_Type_commit (&MPI_distanceType);

  Tdist nelems = last-first;
  arr<Tdist> dist(nproc);
  MPI_Allgather (&nelems, 1, MPI_distanceType, &dist[0], 1, MPI_distanceType,
    comm);

  // Find splitters
  arr2<Tdist> right_ends(nproc+1,nproc,0);
  median_split (first, last, dist, comp, right_ends,
                 MPI_valueType, MPI_distanceType, comm);

  // Communicate to destination
  Tdist n_loc = last-first;
  arr<T> trans_data(n_loc);
  arr<Tdist> boundaries(nproc+1);
  alltoall (right_ends, first, last, trans_data, boundaries,
            MPI_valueType, comm);

  // Merge streams from all tasks
  oop_tree_merge (trans_data.begin(), first, boundaries, nproc, comp);

  MPI_Type_free (&MPI_valueType);
  MPI_Type_free (&MPI_distanceType);
#endif // USE_MPI
  }

template<typename RAIter>
inline void parallel_sort (RAIter first, RAIter last)
  {
  typedef typename iterator_traits<RAIter>::value_type T;
  parallel_sort (first, last, less<T>());
  }

}

#endif /* PSORT_H */
