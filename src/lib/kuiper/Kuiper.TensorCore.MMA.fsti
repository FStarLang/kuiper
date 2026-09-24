module Kuiper.TensorCore.MMA

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Spec.GEMM

(* Assumed hardware interface for one
   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 operation.
   Requires sm_80+. All 32 live threads of a warp must execute the same
   operation sequence convergently, with matching matrix views and fill
   values. As with WMMA/WGMMA, participation and publication of input
   writes are not modelled.
   The gpu capability excludes CPU calls; it does not prove convergence. *)
new
val fragment : Type0

val fragment_pts_to
  ([@@@mkey] fr : fragment)
  (v : chest2 float 16 8)
  : slprop

unfold
instance has_pts_to_fragment : has_pts_to fragment (chest2 float 16 8) = {
  pts_to = (fun r #f v -> fragment_pts_to r v);
}

(* A distinct hardware result, not WMMA's emma or a scalar FMA loop.
   PTX leaves accumulation order, rounding and subnormal handling unspecified. *)
val emma
  (mc : chest2 float 16 8)
  (ma : chest2 bf16 16 16)
  (mb : chest2 bf16 16 8)
  : chest2 float 16 8

val emma_approx_lemma
  (mc : chest2 float 16 8)
  (ma : chest2 bf16 16 16)
  (mb : chest2 bf16 16 8)
  (rc : chest2 real 16 8)
  (ra : chest2 real 16 16)
  (rb : chest2 real 16 8)
  : Lemma (requires mc %~ rc /\ ma %~ ra /\ mb %~ rb)
          (ensures emma mc ma mb %~ matplus rc (matmul ra rb))

fn alloc_fragment ()
  preserves gpu
  returns fr : fragment
  ensures exists* v. fr |-> v

fn fill
  (fr : fragment)
  (x : float)
  (#v : chest2 float 16 8)
  preserves gpu
  requires fr |-> v
  ensures fr |-> const (16 @| 8 @| INil) x

fn load_accum
  (fr : fragment)
  (#l : layout2 16 8) {| strided_row_major l |}
  (c : array2 float l)
  (#f : perm)
  (#vc #v : chest2 float 16 8)
  preserves gpu
  preserves c |-> Frac f vc
  requires fr |-> v
  ensures fr |-> vc

fn mma_sync
  (#la : layout2 16 16) {| strided_row_major la |}
  (a : array2 bf16 la)
  (#lb : layout2 16 8) {| strided_col_major lb |}
  (b : array2 bf16 lb)
  (fr : fragment)
  (#fa #fb : perm)
  (#va : chest2 bf16 16 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires fr |-> vc
  ensures fr |-> emma vc va vb

fn store
  (fr : fragment)
  (#l : layout2 16 8) {| strided_row_major l |}
  (c : array2 float l)
  (#v #vc : chest2 float 16 8)
  preserves gpu
  preserves fr |-> v
  requires c |-> Frac (1.0R /. warp_size) vc
  ensures c |-> Frac (1.0R /. warp_size) v

inline_for_extraction let () = ()
