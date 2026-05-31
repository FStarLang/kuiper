module Kuiper.Array.Extra

#lang-pulse

(* Extra functions over Pulse's normal arrays. *)
open Pulse
open Pulse.Lib.Array
open FStar.Seq
open Kuiper.ForEvery
open Kuiper.Common

ghost
fn rec array_share
  (#t:Type0)
  (a : array t)
  (#s : seq t)
  (#f : perm)
  (n : pos)
  requires
    a |-> Frac f s
  ensures
    forall+ (_ : natlt n).
      a |-> Frac (f /. Real.of_int n) s
  decreases n
{
  if (n = 1) {
    rewrite (pts_to a #f s)
         as (pts_to a #(f /. Real.of_int n) s);
    forevery_intro_false #(natlt n) (fun _ -> a |-> Frac (f /. Real.of_int n) s);
    forevery_insert #(natlt n) (fun _ -> a |-> Frac (f /. Real.of_int n) s) 0;
    forevery_unrefine #(natlt n) (fun _ -> a |-> Frac (f /. Real.of_int n) s);
  } else {
    Pulse.Lib.Array.PtsTo.to_mask a;
    with s_opt . assert (pts_to_mask a #f s_opt (fun _ -> True));
    let f' = f -. (f /. Real.of_int n);
    mask_share_gen a (f /. Real.of_int n) f';
    Pulse.Lib.Array.PtsTo.from_mask #_ a #(f /. Real.of_int n);
    with v1 . assert (Pulse.Lib.Array.PtsTo.pts_to a #(f /. Real.of_int n) v1);
    assert pure (Seq.equal v1 s);
    rewrite (Pulse.Lib.Array.PtsTo.pts_to a #(f /. Real.of_int n) v1)
         as (Pulse.Lib.Array.PtsTo.pts_to a #(f /. Real.of_int n) s);
    Pulse.Lib.Array.PtsTo.from_mask #_ a #f';
    with v2 . assert (Pulse.Lib.Array.PtsTo.pts_to a #f' v2);
    assert pure (Seq.equal v2 s);
    rewrite (Pulse.Lib.Array.PtsTo.pts_to a #f' v2) as (Pulse.Lib.Array.PtsTo.pts_to a #f' s);
    array_share #_ a #s #f' (n - 1);
    forevery_ext #(natlt (n - 1)) _ (fun _ -> a |-> Frac (f /. Real.of_int n) s);
    forevery_natlt_push n (fun _ -> a |-> Frac (f /. Real.of_int n) s);
  }
}

ghost
fn rec array_gather
  (#t:Type0)
  (a : array t)
  (#s : seq t)
  (#f : perm)
  (n : pos)
  requires
    forall+ (_ : natlt n).
      a |-> Frac (f /. Real.of_int n) s
  ensures
    a |-> Frac f s
  decreases n
{
  if (n = 1) {
    forevery_singleton_elim' #(natlt n) _ 0;
  } else {
    forevery_natlt_pop n _;
    let f' = f -. (f /. Real.of_int n);
    forevery_ext #(natlt (n - 1)) _ (fun _ -> a |-> Frac (f' /. Real.of_int (n - 1)) s);
    array_gather #_ a #s #f' (n - 1);
    Pulse.Lib.Array.PtsTo.to_mask a #(f /. Real.of_int n);
    Pulse.Lib.Array.PtsTo.to_mask a #f';
    mask_gather a;
    Pulse.Lib.Array.PtsTo.from_mask a;
    with v . assert (Pulse.Lib.Array.PtsTo.pts_to a #(f /. Real.of_int n +. f') v);
    assert pure (Seq.equal v s);
    rewrite (Pulse.Lib.Array.PtsTo.pts_to a #(f /. Real.of_int n +. f') v)
         as (Pulse.Lib.Array.PtsTo.pts_to a #f s);
  }
}
