module GPU.MatMul.Kernel

#lang-pulse

#push-options "--fuel 1 --ifuel 1"

(* sigh. Need this to trigger dependency scan to look at fst. *)
inline_for_extraction let x = 1

open GPU

module Impure = GPU.MatMul.Impure
module Pure = GPU.MatMul.Pure
module SZ = FStar.SizeT
module U64 = FStar.UInt64

[@@pulse_unfold]
let kpre_pair (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (#s1: erased (seq u64))
  (#s2: erased (seq u64))
  (nth: erased nat { nth > 0 })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nth s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2

[@@pulse_unfold]
let kpre (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) )
  (#s2: erased (seq u64))
  (nth: erased nat { reveal nth == rows * columns })
  (tid : nat{ tid < rows * columns})
  : slprop
  =
  kpre_pair rows shared columns ga1 ga2 #s1 #s2 nth
  ** (exists* sr. gpu_pts_to_array_slice r tid (tid+1) sr)

[@@pulse_unfold]
let kpost (rows shared columns: nat)
  (ga1: gpu_array u64 (rows * shared))
  (ga2: gpu_array u64 (shared * columns))
  (r: gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nth: erased nat { reveal nth == rows * columns })
  (tid : nat {  tid < rows * columns })
  : slprop
  =
  Impure.gpu_pts_to_matrix rows shared ga1 nth s1
  ** Impure.gpu_pts_to_matrix shared columns ga2 nth s2
  ** gpu_pts_to_array_slice r tid (tid+1) (seq![Pure.matmul_single rows shared columns s1 s2 (tid / columns) (tid % columns) shared])
  // ** (exists* s. gpu_pts_to_array_slice r tid (tid+1) s)



[@@CPrologue "__global__"]
fn kernel
  (rows: sz) (shared: sz) (columns: sz{rows * columns < pow2 64})
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (r : gpu_array u64 (rows * columns))
  (#s1: erased (seq u64) {Seq.length s1 == rows * shared})
  (#s2: erased (seq u64) {Seq.length s2 == shared * columns})
  (nth : erased sz { SZ.v nth == SZ.v SZ.(rows *^ columns) })
  (etid : erased tid_t { gdim_x etid == SZ.v nth /\ bdim_x etid == 1 })
  requires gpu
    ** thread_id etid
    ** kpre rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nth) (thread_index etid)
  ensures  gpu
    ** thread_id etid
    ** kpost rows shared columns ga1 ga2 r #s1 #s2 (SZ.v nth) (thread_index etid)
