module Kuiper.MatMulTileF32.Kernel

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

open Kuiper
open Kuiper.Math

(* trigger crossing the fsti *)
inline_for_extraction let x = 1

module P = Kuiper.MatMul.Pure
module I = Kuiper.MatMul.Impure

module Barrier = Kuiper.MatMulTileF32.Barrier
module SZ = FStar.SizeT
module U64 = FStar.UInt64
module Layout4 = Kuiper.MatMulTileF32.Layout4

[@@pulse_unfold]
let kpre (rows shared columns: nat)
  (ga1: gpu_array f32 (rows * shared))
  (ga2: gpu_array f32 (shared * columns))
  (r: gpu_array f32 (rows * columns))
  (#s1: erased (seq f32) )
  (#s2: erased (seq f32))
  (nthr: erased nat { nthr > 0 })
  (tid : nat{ tid < rows * columns})
  : slprop
  =
  I.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** I.gpu_pts_to_matrix shared columns ga2 nthr s2
  ** gpu_pts_to_array1 r tid

[@@pulse_unfold]
let kpost (rows shared columns: nat)
  (ga1: gpu_array f32 (rows * shared))
  (ga2: gpu_array f32 (shared * columns))
  (r: gpu_array f32 (rows * columns))
  (#s1: erased (seq f32) {len s1 == rows * shared})
  (#s2: erased (seq f32) {len s2 == shared * columns})
  (nthr: erased nat { nthr > 0 })
  (tid : nat {  tid < rows * columns })
  : slprop
  =
  I.gpu_pts_to_matrix rows shared ga1 nthr s1
  ** I.gpu_pts_to_matrix shared columns ga2 nthr s2
  ** gpu_pts_to_array1 r tid
  // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)

let permute (rows_tile columns_tile bdim: pos)
: GTot (permutation (i: nat { 0 <= i /\ i < rows_tile * columns_tile * bdim * bdim }))
= Layout4.titi_permutation bdim bdim rows_tile columns_tile

let tid_to_idx
  (rows shared columns : pos)
  (bdim: pos{bdim /? rows /\ bdim /? columns})
  (tid: nat { 0 <= tid /\ tid < rows * columns })
: GTot (tid: nat { 0 <= tid /\ tid < rows * columns })
= calc (==) {
    (rows / bdim) * (columns / bdim) * bdim * bdim;
    == { admit() } // fixme, boring proof (we have divisibility)
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
  (ga1 : gpu_array f32 (rows * shared))
  (ga2 : gpu_array f32 (shared * columns))
  (r : gpu_array f32 (rows * columns))
  (#s1: erased (seq f32) {len s1 == rows * shared})
  (#s2: erased (seq f32) {len s2 == shared * columns})
  (nblk : erased sz { SZ.v nblk == (rows / bdim) * (columns / bdim) })
  (nthr : erased sz { SZ.v nthr == bdim * bdim
                     /\ SZ.v nblk * SZ.v nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (smem_sz : erased nat { smem_sz == 2 * SZ.v nthr })
  (ear: erased (gpu_array f32 smem_sz))
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
