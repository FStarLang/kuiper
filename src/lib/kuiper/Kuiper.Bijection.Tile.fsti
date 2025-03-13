module Kuiper.Bijection.Tile

open Kuiper.Bijection
open Kuiper.Common
open Kuiper.SizeT
open FStar.Ghost
module SZ = FStar.SizeT

inline_for_extraction noextract
let tile_ff
  (mrows mcols : erased nat)
  (brows bcols : SZ.t)
  (_ : squash (SZ.fits (mrows * brows) /\ SZ.fits (mcols * bcols)))
  : (szlt mrows & szlt mcols & szlt brows & szlt bcols) ->
    szlt (mrows * brows) & szlt (mcols * bcols)
= fun (bi, bj, si, sj) ->
    let i = s_undivmod brows (bi, si) in
    let j = s_undivmod bcols (bj, sj) in
    i, j

inline_for_extraction noextract
let tile_gg
  (mrows mcols : erased nat)
  (brows bcols : SZ.t)
  (_ : squash (SZ.fits (mrows * brows) /\ SZ.fits (mcols * bcols)))
  : szlt (mrows * brows) & szlt (mcols * bcols) ->
    (szlt mrows & szlt mcols & szlt brows & szlt bcols)
= fun (i, j) ->
    let bi, si = s_divmod brows i in
    let bj, sj = s_divmod bcols j in
    (bi, bj, si, sj)

inline_for_extraction noextract
let tile_bij
  (mrows mcols : erased nat)
  (brows bcols : SZ.t)
  (_ : squash (SZ.fits (mrows * brows) /\ SZ.fits (mcols * bcols)))
  : (  (szlt mrows & szlt mcols &
        szlt brows & szlt bcols)
     =~
        (szlt (mrows * brows) & szlt (mcols * bcols)) )
= {
    ff = tile_ff _ _ _ _ ();
    gg = tile_gg _ _ _ _ ();
    ff_gg = ez;
    gg_ff = ez;
}
