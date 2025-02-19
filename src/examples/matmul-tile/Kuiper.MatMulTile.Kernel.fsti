module Kuiper.MatMulTile.Kernel

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open Kuiper
open Kuiper.Math

(* trigger crossing the fsti *)
inline_for_extraction let x = 1

module Impure = Kuiper.MatMul.Impure
module Barrier = Kuiper.MatMulTile.Barrier
module SZ = FStar.SizeT
module Layout4 = Kuiper.MatMulTile.Layout4

[@@pulse_unfold]
let kpre (rows shared columns : nat)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1 : erased (seq u64))
  (#s2 : erased (seq u64))
  (nthr : erased nat { nthr > 0 })
  (tid : nat{ tid < rows * columns})
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nthr s2
  ** gpu_pts_to_array1 r tid

[@@pulse_unfold]
let kpost (rows shared columns : nat)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1 : erased (seq u64))
  (#s2 : erased (seq u64))
  (nthr : erased nat { nthr > 0 })
  (tid : nat {  tid < rows * columns })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nthr s2
  ** gpu_pts_to_array1 r tid

let permute (rows_tile columns_tile bdim : pos)
: GTot (permutation (i : nat { 0 <= i /\ i < rows_tile * columns_tile * bdim * bdim }))
= Layout4.titi_permutation bdim bdim rows_tile columns_tile

let tid_to_idx
  (rows shared columns : pos)
  (bdim : pos{bdim /? rows /\ bdim /? columns})
  (tid : nat { 0 <= tid /\ tid < rows * columns })
: GTot (tid : nat { 0 <= tid /\ tid < rows * columns })
= calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { () }
    rows * columns;
  };
  lemma_divides_exact rows bdim;
  lemma_divides_exact columns bdim;
  assert (rows / bdim >= 1);
  assert (columns / bdim >= 1);
  let r = (permute (rows / bdim) (columns / bdim) bdim).f tid in
  r

[@@CPrologue "__global__"]
fn kernel
  (rows shared columns : szp)
  (bdim : szp { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1 : erased (seq u64) {len s1 == rows * shared})
  (#s2 : erased (seq u64) {len s2 == shared * columns})
  (nblk : erased sz { SZ.v nblk == (rows / bdim) * (columns / bdim) })
  (nthr : erased sz { SZ.v nthr == bdim * bdim })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (smem_sz : erased nat { smem_sz == 2 * SZ.v nthr })
  (ear : erased (gpu_array u64 smem_sz))
  (etid : tid_t { gdim_x etid == SZ.v nblk /\ bdim_x etid == SZ.v nthr })
  requires gpu
    ** thread_id etid
    ** shmem_tok ear
    ** Barrier.shared_pre nthr 0 ear (bidx_x etid) (tidx_x etid)
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr)
         (tid_to_idx rows shared columns bdim (thread_index etid))
  ensures  gpu
    ** thread_id etid
    ** Barrier.shared_pre nthr (2 * (shared / bdim)) ear (bidx_x etid) (tidx_x etid)
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nblk * SZ.v nthr)
         (tid_to_idx rows shared columns bdim (thread_index etid))
