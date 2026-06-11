module Kuiper.Example.CellRef

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Index
module SZ = Kuiper.SizeT

let layout (m : nat) : layout m =
  pack <|
  major_on 0 m <|
  lunit

inline_for_extraction noextract
instance blah
  (m : SZ.t{SZ.fits m})
  : ctlayout (layout m)
  =
  c_pack #_ #_ <|
  c_major_on 0sz _ #_ #{v = 1sz} <|
  cunit

inline_for_extraction noextract
instance _crutch : ctlayout (layout 16) = blah 16sz

(* Read cell [i] by obtaining a ref into it and dereferencing. *)
fn cell_get
  (a : array1 u32 (layout 16))
  (i : szlt 16)
  (#f : perm)
  (#v : erased u32)
  preserves
    Cell a (i <: natlt 16) |-> Frac f v
  returns
    r : u32
  ensures
    pure (r == reveal v)
{
  array1_cell_to_ref a i;
  let p = get_ref_of_array_cell a i;
  rewrite each ref_of_array_cell a (SZ.v i) as p;
  let x = !p;
  rewrite each p as ref_of_array_cell a (SZ.v i);
  array1_cell_from_ref a i;
  x
}

(* Write [w] into cell [i] by obtaining a ref into it and assigning. *)
fn cell_set
  (a : array1 u32 (layout 16))
  (i : szlt 16)
  (w : u32)
  (#v : erased u32)
  requires
    Cell a (i <: natlt 16) |-> v
  ensures
    Cell a (i <: natlt 16) |-> w
{
  array1_cell_to_ref a i;
  let p = get_ref_of_array_cell a i;
  rewrite each ref_of_array_cell a (SZ.v i) as p;
  p := w;
  rewrite each p as ref_of_array_cell a (SZ.v i);
  array1_cell_from_ref a i;
}

(* End-to-end: take a full array, explode it into per-cell ownership,
   focus on cell [j], write through the [cell_set] reference setter, and
   reassemble the array. The only runtime effect is the [cell_set] call;
   everything else is ghost. *)
fn array_set_via_ref
  (a : array1 u32 (layout 16))
  (j : szlt 16)
  (w : u32)
  (#s : erased (lseq u32 16))
  requires
    a |-> reveal s
  ensures
    a |-> lseq_upd (reveal s) (SZ.v j <: natlt 16) w
{
  explode a;
  forevery_extract' #(natlt 16) (SZ.v j <: natlt 16) _;
  cell_set a j w;
  Pulse.Lib.Forall.elim_forall
    (fun (x : natlt 16) ->
      Cell a x |-> Frac 1.0R (Seq.index (lseq_upd (reveal s) (SZ.v j <: natlt 16) w) x));
  Pulse.Lib.Trade.elim_trade _ _;
  implode a;
}
