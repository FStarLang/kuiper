module Kuiper.IArray.Vectorized
#lang-pulse

module T = FStar.Tactics.V2

open Kuiper
open Kuiper.IArray
open Kuiper.IView
open Kuiper.Array.Vectorized
open Kuiper.VectorType

val iarray_pts_to_4cells
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (ai : vw.sch.ait)
  (v : et & et & et & et)
  : slprop

inline_for_extraction noextract
fn iarray_vec4_read_cells
  // (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  (a : iarray float vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v : erased (float & float & float & float))
  preserves gpu
  preserves iarray_pts_to_4cells #float a #f (ci_to_ai vw ci) v
  returns
    e : float4
  ensures
    pure (e == make_float4 (reveal v)._1 (reveal v)._2 (reveal v)._3 (reveal v)._4)

inline_for_extraction noextract
fn iarray_vec4_write_cells
  // (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  // (a : iarray et vw)
  (a : iarray float vw)
  (ci : cw.sch.cit)
  (v : float4)
  (#v0 : erased (float & float & float & float))
  preserves gpu
  requires  iarray_pts_to_4cells #float a (ci_to_ai vw ci) v0
  ensures   (exists* v1. iarray_pts_to_4cells #float a (ci_to_ai vw ci) v1 **
                         pure (v1 == (getx v, gety v, getz v, getw v)))
