module Kuiper.Example.TensorCore.WGMMA

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.TensorCore.WGMMA
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major }

inline_for_extraction noextract
instance c8 : concrete_sz 8 = { x = 8sz; }

(* Device entry points for the CUDA driver. Shared-memory initialization
   and warpgroup participation are supplied by the caller, like WMMA. *)
[@@CPrologue "inline"; CPrologue "__device__"]
fn accumulate_twice
  (a : array2 bf16 a_layout)
  (b : array2 bf16 b_layout)
  (c : array2 float (row_major 64 8))
  (#va : chest2 bf16 64 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 64 8)
  requires pure (Kuiper.SHMem.is_block_array (core a) /\
                 Kuiper.SHMem.is_block_array (core b) /\
                 Kuiper.Array.aligned 16 (core a) /\
                 Kuiper.Array.aligned 16 (core b))
  preserves a |-> Frac (1.0R /. 128) va
  preserves b |-> Frac (1.0R /. 128) vb
  requires c |-> Frac (1.0R /. 128) vc
  ensures c |-> Frac (1.0R /. 128) (ewgmma (ewgmma vc va vb) va vb)
{
  let fr = alloc_fragment ();
  load_accum fr c;
  mma_sync a b fr;
  mma_sync a b fr;
  store fr c;
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}

[@@CPrologue "inline"; CPrologue "__device__"]
fn multiply
  (a : array2 bf16 a_layout)
  (b : array2 bf16 b_layout)
  (c : array2 float (row_major 64 8))
  (#va : chest2 bf16 64 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 64 8)
  requires pure (Kuiper.SHMem.is_block_array (core a) /\
                 Kuiper.SHMem.is_block_array (core b) /\
                 Kuiper.Array.aligned 16 (core a) /\
                 Kuiper.Array.aligned 16 (core b))
  preserves a |-> Frac (1.0R /. 128) va
  preserves b |-> Frac (1.0R /. 128) vb
  requires c |-> Frac (1.0R /. 128) vc
  ensures c |-> Frac (1.0R /. 128)
    (ewgmma (const (64 @| 8 @| INil) (zero #float)) va vb)
{
  let fr = alloc_fragment ();
  fill fr zero;
  mma_sync a b fr;
  store fr c;
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}
