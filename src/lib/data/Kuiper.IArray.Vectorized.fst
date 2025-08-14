module Kuiper.IArray.Vectorized

#lang-pulse

friend Kuiper.IArray

open FStar.Seq
open Kuiper.IArray
open Kuiper.Injection

module SZ = FStar.SizeT

let ai_add
  (#len : nat)
  (vw : aiview len)
  (ai : vw.sch.ait)
  (x : nat{in_image vw.step.imap.f ((it_to_nat vw ai) + x)})
  : GTot vw.sch.ait  = it_of_nat vw ((it_to_nat vw ai) + x)

let iarray_pts_to_4cells
  (#et:Type0)
  (#len : erased nat) (#vw : aiview len)
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
  (#len : nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : (et & et & et & et))
  : slprop
  =
    pure (forall (x : natlt 4). in_image vw.step.imap.f ((it_to_nat vw ai) + x)) **
    gpu_pts_to_slice (core a) #f
      (it_to_nat vw ai) (it_to_nat vw ai + 4) seq![v._1; v._2; v._3; v._4]

ghost
fn iarray_4cells_pts_to_gpu_4slice
  (#et:Type0)
  (#len : nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : (et & et & et & et))
  requires iarray_pts_to_4cells a #f ai v
  ensures
    (* WHY DOES THIS NOT WORK? WHAT AM I MISSING? pulling it out into its own definition works no problem *)
      // pure (forall (x : natlt 4). in_image vw.step.imap.f ((it_to_nat vw ai) + x)) **
      // gpu_pts_to_slice (core a) #f
      //   (it_to_nat vw ai) (it_to_nat vw ai + 4) seq![v._1; v._2; v._3; v._4]
    gpu_pts_to_4slice a #f ai v 
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
  gpu_slice_concat (core a) #f
    (it_to_nat vw (ai_add vw ai 0))
    (it_to_nat vw (ai_add vw ai 1))
    (it_to_nat vw (ai_add vw ai 1) + 1);
  gpu_slice_concat (core a) #f
    (it_to_nat vw (ai_add vw ai 0))
    (it_to_nat vw (ai_add vw ai 2))
    (it_to_nat vw (ai_add vw ai 2) + 1);
  gpu_slice_concat (core a) #f
    (it_to_nat vw (ai_add vw ai 0))
    (it_to_nat vw (ai_add vw ai 3))
    (it_to_nat vw (ai_add vw ai 3) + 1);

  (* appending the values gives the goal sequence *)
  assert pure (Seq.equal
    seq![v._1; v._2; v._3; v._4]
    (append (append (append seq![v._1] seq![v._2]) seq![v._3]) seq![v._4]));
  rewrite each (append (append (append seq![v._1] seq![v._2]) seq![v._3]) seq![v._4])
  as seq![v._1; v._2; v._3; v._4];

  (* make indices look the way that they are expected in the goal *)
  rewrite each (ai_add vw ai 0) as ai;
  fold gpu_pts_to_4slice a #f ai v;
  ()
}

ghost
fn gpu_array_slice_pts_to_iarray_4cells
  (#et:Type0)
  (#len : nat)
  (#vw : aiview len)
  (a : iarray et vw)
  (#f : perm)
  (ai : vw.sch.ait)
  (v : (et & et & et & et))
  requires gpu_pts_to_slice (core a) #f
            (it_to_nat vw ai) (it_to_nat vw ai + 4) seq![v._1; v._2; v._3; v._4]
  requires iarray_pts_to_4cells a #f ai v
{
  admit();
}

// #push-options "--debug SMTFail --split_queries always"
inline_for_extraction noextract
fn iarray_vec4_read_cells
  // (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : ciview vw |}
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
  // iarray_4cells_pts_to_gpu_4slice a #f (ci_to_ai vw ci) v;
  // unfold gpu_pts_to_4slice a #f (ci_to_ai vw ci) v;
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
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1) + 1);
  gpu_slice_concat (core a) #f
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2) + 1);
  gpu_slice_concat (core a) #f
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3) + 1);

  (* vectorized read from array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  // assert pure (it_to_nat vw (ci_to_ai vw ci) + 3 < len);
  // This assertion isn't proven without the above :(
  // assert pure (it_to_nat vw (ci_to_ai vw ci) + 4 <= len);
  let v' = gpu_array_vec4_read (core a) flat_idx;

  (* split full slice back into multiple *)
  gpu_slice_split (core a) #f
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3) + 1);
  gpu_slice_split (core a) #f
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2) + 1);
  gpu_slice_split (core a) #f
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1) + 1);

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
inline_for_extraction noextract
fn iarray_vec4_write_cells
  // (#et:Type0)
  (#len : erased nat)
  (#vw : aiview len) {| cw : ciview vw |}
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
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1) + 1);
  gpu_slice_concat (core a)
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2) + 1);
  gpu_slice_concat (core a)
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3) + 1);

  (* vectorized write to array *)
  cw.step.compat (ci |> cw.sch.bij.gg);
  let flat_idx = ci |~> cw.step.cimap;
  gpu_array_vec4_write (core a) flat_idx v;

  with s'.
  rewrite gpu_pts_to_slice (core a)
      (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
      ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1) s'
  as gpu_pts_to_slice (core a)
      (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)
      ((ai_add vw (ci_to_ai vw ci) 3 |~> vw.step.imap) + 1)
     (upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
       (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v); 

  (* make sure the sequence is in a shape that makes it obvious how to split the slice *)
  assert pure
    (Seq.equal
      (upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
        (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v)
      (append (append (append seq![getx v] seq![gety v]) seq![getz v]) seq![getw v]));
  rewrite each
    upd_seq_vec4 (append (append (append seq![v0._1] seq![v0._2]) seq![v0._3]) seq![v0._4])
        (SZ.v flat_idx - (ai_add vw (ci_to_ai vw ci) 0 |~> vw.step.imap)) v
  as (append (append (append seq![getx v] seq![gety v]) seq![getz v]) seq![getw v]);

  (* split full slice back into multiple *)
  gpu_slice_split (core a)
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 3) + 1);
  gpu_slice_split (core a)
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 2) + 1);
  gpu_slice_split (core a)
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 0))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1))
    (it_to_nat vw (ai_add vw (ci_to_ai vw ci) 1) + 1);

  (* fold back *)
  fold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 0) (getx v);
  fold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 1) (gety v);
  fold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 2) (getz v);
  fold iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 3) (getw v);

  (* reverse how index looks *)
  rewrite each iarray_pts_to_cell a (ai_add vw (ci_to_ai vw ci) 0) (getx v)
  as iarray_pts_to_cell a (ci_to_ai vw ci) (getx v);

  fold iarray_pts_to_4cells a (ci_to_ai vw ci) (getx v, gety v, getz v, getw v);
  ()
}