module Kuiper.Tensor.Tiling.CollectApprox
#lang-pulse

open Kuiper
open Kuiper.EMatrix
module SZ = Kuiper.SizeT
open Kuiper.Tensor
open Kuiper.Tensor.Tiling

#set-options "--admit_smt_queries true"
(* Combinator for teardown of approximate kernels.
   Collects per-cell existentials into a matrix-level existential,
   and transforms per-cell approximation facts into matrix-level approximation.

   Usage: After separating gA/gB, call this on the remaining forall+ of cells
   to get a single existential matrix that approximates the spec.

   TODO: this needs a thourough review. *)
ghost
fn array2_collect_approx_tiled
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (trows : pos { trows /? rows })
  (tcols : pos { tcols /? cols })
  (ntr : nat { ntr == rows / trows })
  (ntc : nat { ntc == cols / tcols })
  (spec_fn : natlt rows -> natlt cols -> et -> prop)
  requires
    pure (SZ.fits (l.ulen))
  requires
    forall+ (bid : natlt (ntr * ntc)) (tid : natlt (trows * tcols)).
      exists* (v : et).
        tensor_pts_to_cell
          (array2_subtile gm trows tcols (bid / ntc) (bid % ntc))
          (idx2 (tid / tcols <: natlt trows) (tid % tcols <: natlt tcols)) v **
        pure (spec_fn ((bid / ntc) * trows + (tid / tcols))
                      ((bid % ntc) * tcols + (tid % tcols)) v)
  returns vf : (natlt (ntr * ntc) -> natlt (trows * tcols) -> GTot et)
  ensures
    gm |-> mkM (fun (row : natlt rows) (col : natlt cols) ->
      vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))) **
    pure (forall (row : natlt rows) (col : natlt cols).
      spec_fn row col
        (vf ((row / trows) * ntc + (col / tcols)) ((row % trows) * tcols + (col % tcols))))
