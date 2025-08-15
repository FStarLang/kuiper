module Kuiper.IArray.Vectorized

#lang-pulse

friend Kuiper.IArray

open FStar.Seq
open Kuiper.IArray
open Kuiper.Injection

module SZ = FStar.SizeT

let ai_add
  (vw : aiview)
  (ai : vw.sch.ait)
  (x : nat{in_image vw.step.imap.f ((it_to_nat vw ai) + x)})
  : GTot vw.sch.ait  = it_of_nat vw ((it_to_nat vw ai) + x)

let iarray_pts_to_4cells
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  (ai : vw.sch.ait)
  (v : et & et & et & et)
  : slprop
  =
    Pulse.Lib.WithPure.with_pure
      (forall (x : natlt 4). in_image vw.step.imap.f ((it_to_nat vw ai) + x))
      (fun _ ->
          iarray_pts_to_cell a #f ai               v._1 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 1) v._2 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 2) v._3 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 3) v._4)

let gpu_pts_to_4slice
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : (et & et & et & et))
  : slprop
  =
    pure (forall (x : natlt 4). in_image #vw.sch.ait #nat vw.step.imap.f ((it_to_nat vw ai) + x)) **
    gpu_pts_to_slice (core a) #f
      (it_to_nat vw ai) (it_to_nat vw ai + 4) seq![v._1; v._2; v._3; v._4]

ghost
fn iarray_4cells_pts_to_gpu_4slice
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : (et & et & et & et))
  requires iarray_pts_to_4cells a #f ai v
  ensures
    (* WHY DOES THIS NOT WORK? WHAT AM I MISSING? pulling it out into its own definition works no problem *)
      pure (forall (x : natlt 4). in_image #_ #nat vw.step.imap.f ((it_to_nat vw ai) + x)) **
      gpu_pts_to_slice (core a) #f
        (it_to_nat vw ai) (it_to_nat vw ai + 4) seq![v._1; v._2; v._3; v._4]
    // gpu_pts_to_4slice a #f ai v 
{
  unfold iarray_pts_to_4cells a #f ai v;

  (* make index look the same for concatenation *)
  rewrite each iarray_pts_to_cell a #f ai v._1
  as (iarray_pts_to_cell a #f (ai_add vw ai 0) v._1);

  (* get gpu_pts_to_slice *)
  unfold iarray_pts_to_cell a #f (ai_add vw ai 0) v._1;
  unfold iarray_pts_to_cell a #f (ai_add vw ai 1) v._2;
  unfold iarray_pts_to_cell a #f (ai_add vw ai 2) v._3;
  unfold iarray_pts_to_cell a #f (ai_add vw ai 3) v._4;

  (* concatenate into a single slice *)
  gpu_slice_concat (core a) #f (it_to_nat vw (ai_add vw ai 0)) (it_to_nat vw (ai_add vw ai 1)) _;
  gpu_slice_concat (core a) #f (it_to_nat vw (ai_add vw ai 0)) (it_to_nat vw (ai_add vw ai 2)) _;
  gpu_slice_concat (core a) #f (it_to_nat vw (ai_add vw ai 0)) (it_to_nat vw (ai_add vw ai 3)) _;

  (* appending the values gives the goal sequence *)
  with i j s. assert gpu_pts_to_slice (core a) #f i j s;
  assert (pure (Seq.equal s seq![v._1; v._2; v._3; v._4]));
  rewrite each s
  as seq![v._1; v._2; v._3; v._4];

  (* make indices look the way that they are expected in the goal *)
  rewrite each (ai_add vw ai 0) as ai;
  ()
}

ghost
fn gpu_array_slice_pts_to_iarray_4cells
  (#et:Type0)
  (#vw : aiview)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : seq et{Seq.length v >= 4})
  requires gpu_pts_to_slice (core a) #f
            (it_to_nat vw ai) (it_to_nat vw ai + 4) v
  ensures  iarray_pts_to_4cells a #f ai (v @! 0, v@! 1, v @! 2, v @! 3)
{
  admit();
}

// #push-options "--debug SMTFail --split_queries always"
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
{
  iarray_4cells_pts_to_gpu_4slice a #f (ci_to_ai vw ci) v;

  (* vectorized read from array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  // assert pure (it_to_nat vw (ci_to_ai vw ci) + 3 < len);
  // This assertion isn't proven without the above :(
  // assert pure (it_to_nat vw (ci_to_ai vw ci) + 4 <= len);
  // gpu_pts_to_slice_ref (core a) (it_to_nat vw (ci_to_ai vw ci)) (it_to_nat vw (ci_to_ai vw ci) + 4);

  (* read from array *)
  let v' = gpu_array_vec4_read (core a) flat_idx;

  gpu_array_slice_pts_to_iarray_4cells a #f (ci_to_ai vw ci) _;
  
  (* return float4 *)
  v'
}

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print-implicits"
inline_for_extraction noextract
fn iarray_vec4_write_cells
  // (#et:Type0)
  (#vw : aiview) {| cw : ciview vw |}
  // (a : iarray et vw)
  (a : iarray float vw)
  (ci : cw.sch.cit)
  (v : float4)
  (#v0 : (float & float & float & float))
  preserves gpu
  requires  iarray_pts_to_4cells #float a (ci_to_ai vw ci) v0
  ensures   (exists* v1. iarray_pts_to_4cells #float a (ci_to_ai vw ci) v1 **
                         pure (v1 == (getx v, gety v, getz v, getw v)))
{
  iarray_4cells_pts_to_gpu_4slice a (ci_to_ai vw ci) v0;

  (* vectorized write to array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  gpu_array_vec4_write (core a) flat_idx v;

  gpu_array_slice_pts_to_iarray_4cells a (ci_to_ai vw ci) _;

  ()
}
