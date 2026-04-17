module Kuiper.View.TwoTiles

#lang-pulse

open Kuiper
open Kuiper.View { aview }
open Kuiper.IView { ciview }
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg

open Kuiper.Index
module Chest = Kuiper.Chest
module SZ = Kuiper.SizeT
module Concat = Kuiper.View.Concat

let valid_tile = x:sz{SZ.fits (2 * x * x)}

(**************
Here is how we used to define the shared memory view for the SHmem GEMM, just
pasting two row-major views together. The one problem with this is that trying
to use indices like Inl (i / tile, i % tile) below does not work well:
apparently bidirectionality is not kicking in and the types are inferred to be
int/SZ.t instead of the proper refinements. So, instead, we defined mkAIdx and
mkCIdx functions that return the right types, and use them in the code below.

We now have separate SHMem arrays according to the shmem description, so this is
no longer needed, but we keep it around for reference.
***************)

let st et tile = Chest.t (tile @| tile @| INil) et

let aview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : aview et (st et tile & st et tile)
= Kuiper.View.Concat.aview_concat
    (tensor_aview et (l2_row_major tile tile))
    (tensor_aview et (l2_row_major tile tile))

inline_for_extraction noextract
instance ciview_2tile2
  (et : Type0)
  (tile : valid_tile)
  : ciview (aview_2tile2 et tile).iview =
  (* FIXME: annotation needed since otherwise we bump into
  ghost due to the implicit in the Mkconcrete_sz constructor. *)
  Kuiper.View.Concat.cview_concat
    solve solve () #({x = tile *^ tile} <: concrete_sz (SZ.v tile * SZ.v tile))

(**************)

type ait (tile : valid_tile) = either (natlt tile & (natlt tile & unit)) (natlt tile & (natlt tile & unit))

inline_for_extraction noextract
type cit (tile : valid_tile) = either (szlt tile & (szlt tile & unit)) (szlt tile & (szlt tile & unit))

let chk1 et (tile : valid_tile) = assert ((aview_2tile2 et tile).iview.ait == ait tile)
let chk2 et (tile : valid_tile) = assert_norm ((ciview_2tile2 et tile).sch.cit == cit tile)

let mkAIdx (#tile:valid_tile) (i : natlt 2) (j : natlt tile) (k : natlt tile) : ait tile =
  match i with
  | 0 -> Inl (j, (k, ()))
  | 1 -> Inr (j, (k, ()))

inline_for_extraction noextract
let mkCIdx (#tile:valid_tile) (i : szlt 2) (j : szlt tile) (k : szlt tile) : cit tile =
  (* FIXME!!!! A match with machine integers does not reduce.
     We have to use the if-then-else form to get C code out. *)
  if i = 0sz
  then Inl (j, (k, ()))
  else Inr (j, (k, ()))
  // match i with
  // | 0sz -> Inl (j, (k, ()))
  // | 1sz -> Inr (j, (k, ()))
