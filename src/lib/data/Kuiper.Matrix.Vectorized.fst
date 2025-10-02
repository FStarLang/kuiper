module Kuiper.Matrix.Vectorized
#lang-pulse

friend Kuiper.Matrix

open Kuiper.Matrix.Common

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

inline_for_extraction noextract
fn gpu_matrix_vec_read
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| clayout l, strided : strided_row_major l |}
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
{
  // Pretty fake for now
  let p = core gm;
  assume (live p);
  with ps. assert (p |-> ps);
  assume (pure (Seq.length s >= chunk et));
  assume pure False;
  assert (pure (chunk et >= 1));
  strided.pf i j;
  strided.pf i (j + chunk et - 1);
  let offset = strided.offset +^ strided.stride *^ i +^ j;
  gpu_array_vec_cpy_dh arr 0sz p offset;
  drop_ (live p);
  ();
}
