module Kuiper.Matrix.Vectorized
#lang-pulse

module T = FStar.Tactics.V2

open Kuiper
open Kuiper.Array.Vectorized
open Kuiper.Matrix
open Kuiper.Matrix.Reprs.Type

val gpu_matrix_pts_to_4cells
  (#et:Type) (#rows #cols : nat)
  (#l : mlayout rows cols)
  ([@@@mkey] gm : gpu_matrix et l)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey]i : natlt rows)
  ([@@@mkey]j : natlt cols)
  (v : et & et & et & et)
  : slprop

inline_for_extraction noextract
fn gpu_matrix_vec4_read_cells
  // (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix float l)
  (i : szlt rows)
  (j : szlt cols)
  (#f : perm)
  (#v0 : float & float & float & float)
  preserves gpu
  preserves gpu_matrix_pts_to_4cells gm #f i j v0
  returns e : float4
  ensures
    pure (e == make_float4 v0._1 v0._2 v0._3 v0._4)


inline_for_extraction noextract
fn gpu_matrix_vec4_write_cells
  (#et:Type0)
  (#rows #cols : erased nat)
  (#l : mlayout rows cols) {| c : clayout l |}
  (gm : gpu_matrix float l)
  (i : szlt rows)
  (j : szlt cols)
  (v : float4)
  (#v0 : float & float & float & float)
  preserves gpu
  requires  gpu_matrix_pts_to_4cells gm i j v0
  ensures
    (exists* v1. gpu_matrix_pts_to_4cells gm i j v1 **
                 pure(v1 == (getx v, gety v, getz v, getw v)))
