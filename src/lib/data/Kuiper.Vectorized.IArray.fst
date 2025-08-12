module Kuiper.Vectorized.IArray

#lang-pulse

module T = FStar.Tactics.V2
open FStar.Seq

open Kuiper
open Kuiper.IView
open Kuiper.Vectorized

friend Kuiper.IArray
open Kuiper.IArray

module SZ = FStar.SizeT

open Kuiper.Injection

let ai_add
  (#len : nat)
  (vw : aiview len)
  (ai : vw.sch.ait)
  (x : nat{in_image vw.step.imap.f ((it_to_nat vw ai) + x)})
  : GTot vw.sch.ait  = it_of_nat vw ((it_to_nat vw ai) + x)

let iarray_pts_to_4cells
  (#et:Type0)
  (#len : erased nat) (#vw : aiview len)
  ([@@@mkey] a : iarray et vw)
  (#[T.exact (`1.0R)] f : perm)
  ([@@@mkey] ai : vw.sch.ait)
  // Should probably be restricted to only the elements that are accessed?
  //  this: (#v : (ai: vw.sch.ait{0 <= vw.sch.bij.ff ai /\ vw.sch.bij.ff ai < 4} -> GTot float))
  (v : et & et & et & et)
  : slprop
  =
    Pulse.Lib.WithPure.with_pure
      ((forall (x : natlt 4). in_image vw.step.imap.f ((it_to_nat vw ai) + x)))
      (fun _ ->
        // pure (SZ.fits len) **
          iarray_pts_to_cell a #f ai               v._1 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 1) v._2 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 2) v._3 ** 
          iarray_pts_to_cell a #f (ai_add vw ai 3) v._4)

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print-implicits"
fn iarray_vec4_read_cells
  // (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : ciview vw |}
  // (a : iarray et vw)
  (a : iarray float vw)
  (ci : cw.sch.cit)
  (#f : perm)
  // Should probably be restricted to only the elements that are accessed?
  //  this: (#v : (ai: vw.sch.ait{0 <= vw.sch.bij.ff ai /\ vw.sch.bij.ff ai < 4} -> GTot float))
  (v : erased (float & float & float & float))
  // (_ : squash (forall (x : natlt 4). in_image vw.step.imap.f (it_to_nat vw (ci_to_ai vw ci) + x)))
  preserves gpu
  preserves iarray_pts_to_4cells #float a #f (ci_to_ai vw ci) v
  returns
    e : float4
  ensures
    pure (e == make_float4 (reveal v)._1 (reveal v)._2 (reveal v)._3 (reveal v)._4)
{
  (* get gpu_pts_to_slice *)
  unfold iarray_pts_to_4cells a #f (ci_to_ai vw ci) v;

  unfold iarray_pts_to_cell a #f (ci_to_ai vw ci) (reveal v)._1;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 1) (reveal v)._2;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 2) (reveal v)._3;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 3) (reveal v)._4;

  (* make index look the same for concatenation *)
  rewrite each gpu_pts_to_slice (core a) #f (ci_to_ai vw ci |~> vw.step.imap) ((ci_to_ai vw ci |~> vw.step.imap) + 1)  seq![(reveal v)._1]
  as gpu_pts_to_slice (core a) #f (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap) ((ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap) + 1) seq![(reveal v)._1];

  rewrite each ((ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap) + 1) as ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap));
  rewrite each ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) + 1) as ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap));
  rewrite each ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap) + 1) as ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap));

  (* concatenate into a single slice *)
  gpu_slice_concat (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) + 1);
  gpu_slice_concat (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap) + 1);
   gpu_slice_concat (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1);

  (* read from array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  let v' = gpu_array_read_vec4 (core a) flat_idx;

  (* split full slice back into multiple *)
  gpu_slice_split (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1);
  gpu_slice_split (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap) + 1);
  gpu_slice_split (core a) #f
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) + 1);

  (* reverse how index looks *)
  rewrite each gpu_pts_to_slice (core a) #f (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap) (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) seq![(reveal v)._1]
  as gpu_pts_to_slice (core a) #f (ci_to_ai vw ci |~> vw.step.imap) ((ci_to_ai vw ci |~> vw.step.imap) + 1)  seq![(reveal v)._1];

  (* fold back *)
  fold iarray_pts_to_cell a #f (ci_to_ai vw ci) (reveal v)._1;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 1) (reveal v)._2;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 2) (reveal v)._3;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 3) (reveal v)._4;

  fold iarray_pts_to_4cells a #f (ci_to_ai vw ci) v;
  
  (* return float4 *)
  v'
}
