module Kuiper.VArray.Vectorized
#lang-pulse

friend Kuiper.VArray

open Kuiper.View
module IVec = Kuiper.IArray.Vectorized


let varray_pts_to_4cells
  (#et:Type0) (#len : erased nat) (#st : Type)
  (#vw : aview et len st)
  (a : varray vw)
  (#[T.exact (`1.0R)] f : perm)
  (ai : vw.iview.sch.ait)
  (v : et & et & et & et)
  : slprop
  =
    IVec.iarray_pts_to_4cells (VA?._0 a) #f ai v

inline_for_extraction noextract
fn varray_vec4_read_cells
  // (#et : Type) 
  (#len : erased nat) (#st : Type)
  (#vw : aview float len st)
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
{
  unfold varray_pts_to_4cells #float a #f (ci_to_ai vw ci) v;
  let res = IVec.iarray_vec4_read_cells (VA?._0 a) ci;
  fold varray_pts_to_4cells #float a #f (ci_to_ai vw ci) v;
  res
}

inline_for_extraction noextract
fn varray_vec4_write_cells
  // (#et : Type)
  (#len : erased nat) (#st : Type)
  (#vw : aview float len st)
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
{
  unfold varray_pts_to_4cells #float a (ci_to_ai vw ci) v0;
  IVec.iarray_vec4_write_cells (VA?._0 a) ci v;
  fold varray_pts_to_4cells #float a (ci_to_ai vw ci)
    (getx v, gety v, getz v, getw v);
}
