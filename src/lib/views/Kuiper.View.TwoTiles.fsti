module Kuiper.View.TwoTiles

#lang-pulse

open Kuiper
open Kuiper.EMatrix { ematrix }
open Kuiper.View { aview }
open Kuiper.IView { ciview }
open Kuiper.Matrix {} // instances

// "Definition Kuiper.View.fits cannot be found." ????
// open Kuiper.View { cview }

module SZ = Kuiper.SizeT
module MC = Kuiper.Matrix.Common
module R  = Kuiper.Matrix.Reprs
module Concat = Kuiper.View.Concat

let valid_tile = x:sz{SZ.fits (2 * x * x)}

(**************
Here is how we to define the shared memory view for the SHmem GEMM, just pasting
two row-major views together. The one problem with this is that trying
to use indices like Inl (i / tile, i % tile) below does not work well:
apparently bidirectionality is not kicking in and
the types are inferred to be int/SZ.t instead of the proper refinements.
So, instead, we define mkAIdx and mkCIdx functions that return the right
types, and use them in the code below.
***************)

let aview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : aview et (ematrix et tile tile & ematrix et tile tile)
= Kuiper.View.Concat.aview_concat
    (MC.aview_from_mlayout et (R.row_major tile tile))
    (MC.aview_from_mlayout et (R.row_major tile tile))

inline_for_extraction noextract
instance ciview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : ciview (aview_2tile2 et tile).iview =
  (*neat *)
  Kuiper.View.Concat.cview_concat
    _ solve
    _ solve
    ()

(**************)

type ait (tile : valid_tile) = either (natlt tile & natlt tile) (natlt tile & natlt tile)

inline_for_extraction noextract
type cit (tile : valid_tile) = either (szlt tile & szlt tile) (szlt tile & szlt tile)

let chk1 et (tile : valid_tile) = assert ((aview_2tile2 et tile).iview.sch.ait == ait tile)
let chk2 et (tile : valid_tile) = assert_norm ((ciview_2tile2 et tile).sch.cit == cit tile)

let mkAIdx (#tile:valid_tile) (i : natlt 2) (j : natlt tile) (k : natlt tile) : ait tile =
  match i with
  | 0 -> Inl (j, k)
  | 1 -> Inr (j, k)

inline_for_extraction noextract
let mkCIdx (#tile:valid_tile) (i : szlt 2) (j : szlt tile) (k : szlt tile) : cit tile =
  (* FIXME!!!! A match with machine integers does not reduce.
     We have to use the if-then-else form to get C code out. *)
  if i = 0sz
  then Inl (j, k)
  else Inr (j, k)
  // match i with
  // | 0sz -> Inl (j, k)
  // | 1sz -> Inr (j, k)
