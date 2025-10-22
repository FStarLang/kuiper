module RefArray

#lang-pulse
open Kuiper
open Pulse.Lib.Array
open FStar.UInt32

let array_ref_u32_pts_to
  (arr : array (ref u32)) (l : seq u32)
: slprop =
  exists* (s : lseq (ref u32) (len l)).
    arr |-> s **
    forall+ (i : natlt (Seq.length s)).
      (s @! i) |-> (l @! i)

fn test (arr : array (ref u32))
  requires array_ref_u32_pts_to arr seq![1ul;2ul;3ul]
  ensures  array_ref_u32_pts_to arr seq![1ul;3ul;3ul]
{
  unfold array_ref_u32_pts_to;
  with s. assert arr |-> s;
  let r0 = arr.(1sz);
  forevery_extract' #(natlt (Seq.length s)) 1 _;
  r0 := !r0 +^ 1ul;
  Pulse.Lib.Forall.elim_forall (fun i -> (s @! i) |-> (seq![1ul;3ul;3ul] @! i));
  Pulse.Lib.Trade.elim_trade _ _;
  fold array_ref_u32_pts_to;
  ()
}


ghost
fn helper (p : natlt 3 -> slprop)
  requires p 0 ** p 1 ** p 2
  ensures  forall+ (i : natlt 3). p i
{
  forevery_intro_false p;
  forevery_insert p 0;
  forevery_insert p 1;
  forevery_insert p 2;
  forevery_unrefine p;
}

fn use ()
{
  let mut arr = [| Pulse.Lib.Reference.null #u32; 3sz |];
  let mut r0 = 1ul; arr.(0sz) <- r0;
  let mut r1 = 2ul; arr.(1sz) <- r1;
  let mut r2 = 3ul; arr.(2sz) <- r2;

  let s = hide (seq![r0; r1; r2]);
  let p = (fun (i : natlt (Seq.length s)) ->
    (s @! i) |-> (seq![1ul;2ul;3ul] @! i));

  rewrite r0 |-> 1ul as p 0;
  rewrite r1 |-> 2ul as p 1;
  rewrite r2 |-> 3ul as p 2;
  helper p;

  with foo.
    assert arr |-> foo;

  assert pure (Seq.equal foo s);

  rewrite arr |-> foo
       as arr |-> s;

  forevery_rw_size 3 (Seq.length s);
  forevery_ext p
    (fun i -> (s @! i) |-> (seq![1ul;2ul;3ul] @! i));

  fold array_ref_u32_pts_to arr seq![1ul;2ul;3ul];

  test arr;

  admit()
}
