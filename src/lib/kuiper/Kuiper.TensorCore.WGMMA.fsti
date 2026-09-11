module Kuiper.TensorCore.WGMMA

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Spec.GEMM
include Kuiper.TensorCore.WGMMA.Layout

(* Assumed hardware interface, like TensorCore.Base. This first form is
   wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 (shared/shared).
   Requires CUDA 12+ and sm_90a. All 128 threads of an aligned warpgroup
   must call the same operations with the same operands. Participation,
   convergence and publication of shared-memory writes are NOT modelled.
   In particular, this API cannot be used by a 32-thread kernel.

   mma_sync fences registers, issues one operation, commits and waits for
   completion. There are no outstanding operations on return; no async
   synchronization protocol is exposed or modelled here. *)
let warpgroup_size : pos = 128

(* [custard_c_reference] for the same reason as [wmma_fragment]: every
   operation below takes [kpr_wgmma_fragment &], so a fragment bound by copy
   would silently lose every write to it rather than fail to compile. *)
new [@@FStar.Attributes.custard_extern "kpr_wgmma_fragment";
     FStar.Attributes.custard_c_reference;
     FStar.Attributes.custard_c_header "kuiper/wgmma.h"]
val fragment : Type0

val fragment_pts_to
  ([@@@mkey] fr : fragment)
  (v : chest2 float 64 8)
  : slprop

unfold
instance has_pts_to_fragment : has_pts_to fragment (chest2 float 64 8) = {
  pts_to = (fun r #f v -> fragment_pts_to r v);
}

(* A distinct opaque hardware result. PTX does not specify accumulation
   order, rounding or subnormal handling. Do not equate this with emma,
   a scalar FMA loop, or a real-valued matmul to claim bitwise equivalence. *)
val ewgmma
  (mc : chest2 float 64 8)
  (ma : chest2 bf16 64 16)
  (mb : chest2 bf16 16 8)
  : chest2 float 64 8

val ewgmma_approx_lemma
  (mc : chest2 float 64 8)
  (ma : chest2 bf16 64 16)
  (mb : chest2 bf16 16 8)
  (rc : chest2 real 64 8)
  (ra : chest2 real 64 16)
  (rb : chest2 real 16 8)
  : Lemma (requires mc %~ rc /\ ma %~ ra /\ mb %~ rb)
          (ensures ewgmma mc ma mb %~ matplus rc (matmul ra rb))

fn alloc_fragment ()
  returns fr : fragment
  ensures exists* v. fr |-> v

fn fill
  (fr : fragment)
  (x : float)
  (#v : chest2 float 64 8)
  requires fr |-> v
  ensures fr |-> const (64 @| 8 @| INil) x

fn load_accum
  (fr : fragment)
  (#l : layout2 64 8) {| strided_row_major l |}
  (c : array2 float l)
  (#f : perm)
  (#vc #v : chest2 float 64 8)
  preserves c |-> Frac f vc
  requires fr |-> v
  ensures fr |-> vc

fn mma_sync
  (a : array2 bf16 a_layout)
  (b : array2 bf16 b_layout)
  (fr : fragment)
  (#fa #fb : perm)
  (#va : chest2 bf16 64 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 64 8)
  requires pure (Kuiper.SHMem.is_block_array (core a) /\
                 Kuiper.SHMem.is_block_array (core b) /\
                 Kuiper.Array.aligned 16 (core a) /\
                 Kuiper.Array.aligned 16 (core b))
  preserves a |-> Frac fa va
  preserves b |-> Frac fb vb
  requires fr |-> vc
  ensures fr |-> ewgmma vc va vb

(* Like WMMA's collective store, each participating thread owns a fraction
   of the tile. WGMMA uses 1/128, not WMMA's 1/32. *)
fn store
  (fr : fragment)
  (#l : layout2 64 8) {| strided_row_major l |}
  (c : array2 float l)
  (#v #vc : chest2 float 64 8)
  preserves fr |-> v
  requires c |-> Frac (1.0R /. warpgroup_size) vc
  ensures c |-> Frac (1.0R /. warpgroup_size) v

inline_for_extraction let () = ()
