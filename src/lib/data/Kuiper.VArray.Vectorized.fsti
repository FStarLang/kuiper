module Kuiper.VArray.Vectorized
#lang-pulse

module T = FStar.Tactics.V2

open Kuiper
open Kuiper.Array.Vectorized

open Kuiper.VArray

val varray_pts_to_4cells
  (#et:Type0) (#st : Type)
  (#vw : aview et st)
  (a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (ai : vw.iview.sch.ait)
  (v : et & et & et & et)
  : slprop

inline_for_extraction noextract
fn varray_vec4_read_cells
  // (#et : Type)
  (#st : Type)
  (#vw : aview float st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (#v : float & float & float & float)
  preserves gpu
  preserves varray_pts_to_4cells #float a #f (ci_to_ai vw ci) v
  returns
    e : float4
  ensures
    pure (e == make_float4 v._1 v._2 v._3 v._4)

inline_for_extraction noextract
fn varray_vec4_write_cells
  // (#et : Type)
  (#st : Type)
  (#vw : aview float st)
  {| cw : cview vw |}
  (a : varray vw)
  (ci : cw.sch.cit)
  (v : float4)
  (#v0 : float & float & float & float)
  preserves gpu
  requires
    varray_pts_to_4cells #float a (ci_to_ai vw ci) v0
  ensures
    (exists* v1. varray_pts_to_4cells #float a (ci_to_ai vw ci) v1 **
                 pure (v1 == (getx v, gety v, getz v, getw v)))
