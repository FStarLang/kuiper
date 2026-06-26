module Kuiper.Example.CellRef

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT

// #set-options "--print_implicits"

(* Read cell [i] by obtaining a ref into it and dereferencing. *)
fn cell_get
  (a : array1 u32 (l1_forward 16))
  (i : szlt 16)
  (#f : perm)
  (#v : erased u32)
  preserves
    tensor_pts_to_cell a #f ((i <: natlt 16), ()) v
  returns
    r : u32
  ensures
    pure (r == reveal v)
{
  tensor_cell_to_ref a ((SZ.v i <: natlt 16), ());
  let p = get_ref_of_tensor_cell a ((i <: szlt 16), ());
  let x = !p;
  tensor_cell_from_ref a ((SZ.v i <: natlt 16), ());
  x
}

(* Write [w] into cell [i] by obtaining a ref into it and assigning. *)
fn cell_set
  (a : array1 u32 (l1_forward 16))
  (i : szlt 16)
  (w : u32)
  (#v : erased u32)
  requires
    tensor_pts_to_cell a ((i <: natlt 16), ()) v
  ensures
    tensor_pts_to_cell a ((i <: natlt 16), ()) w
{
  tensor_cell_to_ref a ((SZ.v i <: natlt 16), ());
  let p = get_ref_of_tensor_cell a ((i <: szlt 16), ());
  p := w;
  tensor_cell_from_ref a ((SZ.v i <: natlt 16), ());
}

(* End-to-end: take a full array, explode it into per-cell ownership,
   focus on cell [j], write through the [cell_set] reference setter, and
   reassemble the array. The only runtime effect is the [cell_set] call;
   everything else is ghost. *)
fn array_set_via_ref
  (a : array1 u32 (l1_forward 16))
  (j : szlt 16)
  (w : u32)
  (#s : chest1 u32 16)
  requires
    a |-> s
  ensures
    a |-> upd1 s (SZ.v j <: natlt 16) w
{
  tensor_explode a;
  forevery_extract' #(abs (ICons 16 INil)) ((SZ.v j <: natlt 16), ()) _;
  cell_set a j w;
  Pulse.Lib.Forall.elim_forall
    (fun (i : abs (ICons 16 INil)) ->
      Cell a i |-> Frac 1.0R (acc (upd s ((SZ.v j <: natlt 16), ()) w) i));
  Pulse.Lib.Trade.elim_trade _ _;
  tensor_implode a;
}
