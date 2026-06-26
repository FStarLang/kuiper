module Kuiper.Array2.Vectorized
#lang-pulse

(* Vectorized read for Array2, analogous to Kuiper.Matrix.Vectorized. *)

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.EMatrix

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.Array2.Strided
module T = Kuiper.Tensor

inline_for_extraction noextract
fn array2_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : layout2 rows cols) {| T.ctlayout l, strided : strided_row_major l |}
  (gm : array2 et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  pure (aligned' 16 (T.core gm) (cell_of_pos l i j))
  requires  pure (aligned 16 arr)
  requires  arr |-> s
  requires  pure (Pulse.Lib.Array.length arr == chunk et)
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))
