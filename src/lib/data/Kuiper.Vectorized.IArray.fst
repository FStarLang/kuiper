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

fn iarray_vec4_read_cells
  // (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : ciview vw |}
  (a : iarray float vw)
  (ci : cw.sch.cit)
  (#f : perm)
  (v : erased (float & float & float & float))
  preserves gpu
  preserves iarray_pts_to_4cells #float a #f (ci_to_ai vw ci) v
  returns
    e : float4
  ensures
    pure (e == make_float4 (reveal v)._1 (reveal v)._2 (reveal v)._3 (reveal v)._4)
{
  unfold iarray_pts_to_4cells a #f (ci_to_ai vw ci) v;

  (* make index look the same for concatenation *)
  rewrite each iarray_pts_to_cell a #f (ci_to_ai vw ci) (reveal v)._1
  as (iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1);

  (* get gpu_pts_to_slice *)
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 1) (reveal v)._2;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 2) (reveal v)._3;
  unfold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 3) (reveal v)._4;

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

  (* vectorized read from array *)
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

  (* fold back *)
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 1) (reveal v)._2;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 2) (reveal v)._3;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 3) (reveal v)._4;

  (* reverse how index looks *)
  rewrite each iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1
  as iarray_pts_to_cell a #f (ci_to_ai vw ci) (reveal v)._1;

  fold iarray_pts_to_4cells a #f (ci_to_ai vw ci) v;
  
  (* return float4 *)
  v'
}

// #push-options "--debug SMTFail --split_queries always"
// #push-options "--print-implicits"
fn iarray_vec4_write_cells
  // (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : ciview vw |}
  // (a : iarray et vw)
  (a : iarray float vw)
  (ci : cw.sch.cit)
  // Should probably be restricted to only the elements that are accessed?
  (v : float4)
  (#v0 : (float & float & float & float))
  preserves gpu
  requires  iarray_pts_to_4cells #float a (ci_to_ai vw ci) v0
  ensures   (exists* v1. iarray_pts_to_4cells #float a (ci_to_ai vw ci) v1 **
                         pure (v1 == (getx v, gety v, getz v, getw v)))
{
  (* get gpu_pts_to_slice *)
  unfold iarray_pts_to_4cells a (ci_to_ai vw ci) v0;

  (* make index look the same for concatenation *)
  rewrite each iarray_pts_to_cell a (ci_to_ai vw ci) v0._1
  as (iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 0) v0._1);

  (* get gpu_pts_to_slice *)
  unfold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 0) v0._1;
  unfold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 1) v0._2;
  unfold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 2) v0._3;
  unfold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 3) v0._4;

  (* concatenate into a single slice *)
  gpu_slice_concat (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) + 1);
  gpu_slice_concat (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap) + 1);
   gpu_slice_concat (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1);

  (* vectorized write to array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  gpu_array_write_vec4 (core a) flat_idx v;

  with s'.
  rewrite gpu_pts_to_slice (core a)
      (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
      ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1) s'
  as gpu_pts_to_slice (core a)
      (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
      ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1)
     (upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
       (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v); 

  // unfold (upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
  //       (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v);
  admit();
  assert pure
    (append seq![getx v; gety v; getz v] seq![getw v] ==
      (upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
        (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v));

  (* split full slice back into multiple *)
  gpu_slice_split (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1);
  gpu_slice_split (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 2 |~> vw.step.imap) + 1);
  gpu_slice_split (core a)
    (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
    (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap)
    ((ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) + 1);

  // rewrite 



  // rewrite each gpu_pts_to_slice (core a) #f
  //   (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
  //   (ai_add vw (ci_to_ai vw ci) 1 |~> vw.step.imap) seq![(reveal v)._1]
  // as gpu_pts_to_slice (core a) #f
  //   (ci_to_ai vw ci |~> vw.step.imap)
  //   ((ci_to_ai vw ci |~> vw.step.imap) + 1)  seq![(reveal v)._1];

  (* fold back *)
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 1) (reveal v)._2;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 2) (reveal v)._3;
  fold iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 3) (reveal v)._4;

  (* reverse how index looks *)
  rewrite each iarray_pts_to_cell a #f (ai_add vw (ci_to_ai vw ci) 0) (reveal v)._1
  as iarray_pts_to_cell a #f (ci_to_ai vw ci) (reveal v)._1;

  fold iarray_pts_to_4cells a #f (ci_to_ai vw ci) v;
 
}