module Kuiper.Matrix.Vectorized
#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn gpu_matrix_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  arr |-> s
  ensures   arr |-> Seq.init_ghost (chunk et) (fun x -> macc em i (j + x))

inline_for_extraction noextract
fn gpu_matrix_vec_read'
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided_row_major l |}
  (gm : gpu_matrix et l)
  (i : szlt rows)
  (j : szlt (cols - chunk et + 1))
  (#f : perm)
  (#em : ematrix et rows cols)
  (arr : array et)
  (#s : erased (seq et))
  preserves gpu
  preserves gm |-> Frac f em
  requires  arr |-> s
  ensures   exists* (s': lseq et (chunk et)). arr |-> s' **
    pure (forall x. Seq.index s' x == macc em i (j + x))
