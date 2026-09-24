module Kuiper.Example.TensorCore.MMA

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.TensorCore.MMA
open Kuiper.Tensor.Layout.Alg { l2_row_major as row_major, l2_col_major as col_major }
open Kuiper.Tensor.Tiling { subtile_layout }

inline_for_extraction noextract
instance c8 : concrete_sz 8 = { x = 8sz; }

[@@expect_failure [228]]
fn cannot_allocate_on_cpu ()
  preserves cpu
  returns fr : fragment
  ensures exists* v. fr |-> v
{
  alloc_fragment ()
}

[@@expect_failure [228]]
fn cannot_multiply_on_cpu
  (a : array2 bf16 (row_major 16 16))
  (b : array2 bf16 (col_major 16 8))
  (fr : fragment)
  (#va : chest2 bf16 16 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 16 8)
  preserves cpu
  preserves a |-> Frac (1.0R /. 32) va
  preserves b |-> Frac (1.0R /. 32) vb
  requires fr |-> vc
  ensures fr |-> emma vc va vb
{
  mma_sync a b fr
}

(* The CUDA driver supplies collective execution; the gpu capability alone
   does not establish warp convergence or publication of input writes. *)
[@@CPrologue "inline"; CPrologue "__device__"]
fn multiply
  (a : array2 bf16 (row_major 16 16))
  (b : array2 bf16 (col_major 16 8))
  (c : array2 float (row_major 16 8))
  (#va : chest2 bf16 16 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac (1.0R /. 32) va
  preserves b |-> Frac (1.0R /. 32) vb
  requires c |-> Frac (1.0R /. 32) vc
  ensures c |-> Frac (1.0R /. 32)
    (emma (const (16 @| 8 @| INil) (zero #float)) va vb)
{
  let fr = alloc_fragment ();
  fill fr zero;
  mma_sync a b fr;
  store fr c;
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}

let a_layout = subtile_layout (row_major 32 32) 16 16 1 1
let b_layout = subtile_layout (col_major 32 16) 16 8 1 1
let c_layout = subtile_layout (row_major 32 16) 16 8 1 1

[@@CPrologue "inline"; CPrologue "__device__"]
fn accumulate_twice
  (a : array2 bf16 a_layout)
  (b : array2 bf16 b_layout)
  (c : array2 float c_layout)
  (#va : chest2 bf16 16 16)
  (#vb : chest2 bf16 16 8)
  (#vc : chest2 float 16 8)
  preserves gpu
  preserves a |-> Frac (1.0R /. 32) va
  preserves b |-> Frac (1.0R /. 32) vb
  requires c |-> Frac (1.0R /. 32) vc
  ensures c |-> Frac (1.0R /. 32) (emma (emma vc va vb) va vb)
{
  let fr = alloc_fragment ();
  load_accum fr c;
  mma_sync a b fr;
  mma_sync a b fr;
  store fr c;
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}
