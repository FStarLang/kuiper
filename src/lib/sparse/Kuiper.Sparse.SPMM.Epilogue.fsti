module Kuiper.Sparse.SPMM.Epilogue

#lang-pulse
#set-options "--z3rlimit 20"

open Kuiper
open Kuiper.Tensor
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Sparse.SPMM.Maps
open Kuiper.Array.Vectorized
open Kuiper.Array2.Strided
open Kuiper.Chest

(* Retain the initial C model while distributing its cells to threads. *)
ghost
fn scatter
  (#tc : Type0) {| scalar tc, sized tc, has_vec_cpy tc, real_like tc |}
  (#rows #cols : pos { chunk tc /? cols }) (#l : layout2 rows cols)
  (gC : array2 tc l) (row : natlt rows)
  (col : natlt cols { chunk tc /? col })
  (eC : chest2 tc rows cols) (rC : chest2 real rows cols { eC %~ rC })
  (nthr : pos) (n : nat { chunk tc /? n })
  requires pts_to_tile_in_matrix gC row col eC (n * nthr)
  ensures forall+ (tid : natlt nthr).
    thread_tile_approx gC row (thread_offset tc col tid) rC n nthr

(* Apply the combination locally, then copy a complete aligned vector into C.
   The bounds check covers the whole vector, including on partial tiles. *)
inline_for_extraction noextract
fn combine_store
  (#tc #tacc : Type0)
  {| scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (#rows #cols : szp { chunk tc /? cols })
  (#lC : layout2 rows cols) {| ctlayout lC, strided : strided_row_major lC |}
  (gC : array2 tc lC)
  (row : szlt rows) (col : sz { chunk tc /? col })
  (rC rProduct : chest2 real rows cols)
  (n : sz { chunk tc /? n })
  (out : larray tacc n) (#values : lseq tacc n)
  (nthr : szp)
  preserves gpu ** out |-> values
  requires thread_tile_approx gC row col rC n nthr
  requires pure (aligned 16 (core gC))
  requires pure (aligned_strided_row_major (chunk tc) strided)
  requires pure (fits (col + n * nthr))
  requires pure (tile_values_approx (chunk tc) rProduct row col values nthr)
  ensures thread_tile_approx gC row col (chest_comb rcomb rC rProduct) n nthr
