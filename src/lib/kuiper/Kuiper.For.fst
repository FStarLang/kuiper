module Kuiper.For

#lang-pulse

open Kuiper
open Kuiper.ForEvery
module SZ = Kuiper.SizeT

fn for_loop' (lo hi : SZ.t)
  (pre post : between lo hi -> slprop)
  (frame : slprop)
  (fn f (x:SZ.t{lo <= x /\ x < hi})
       requires frame ** pre (SZ.v x)
       ensures  frame ** post (SZ.v x))
  requires pure (lo <= hi)
  preserves frame
  requires forall+ (x : between lo hi). pre x
  ensures  forall+ (x : between lo hi). post x
{
  let mut i : SZ.t = lo;
  forevery_intro_empty
    #(x : between lo hi{x < lo})
    post;

  forevery_refine_ext
    #(between lo hi)
    #(fun _ -> True)
    (fun x -> x >= lo)
    pre;

  while (!i <^ hi)
    invariant
      exists* (vi : SZ.t).
        i |-> vi **
        (forall+ (x:between lo hi {x >= !i}). pre x) **
        (forall+ (x:between lo hi {x <  !i}). post x) **
        pure (lo <= SZ.v vi /\ SZ.v vi <= SZ.v hi)
  {
    with vi. assert i |-> vi;
    forevery_remove' #(between lo hi) (fun x -> x >= vi) pre vi;
    forevery_refine_ext
      #(between lo hi)
      #(fun x -> x >= vi /\ x =!= vi)
      (fun x -> x >= vi +^ 1sz) // Have to use machine addition here...
      pre;
    f !i;
    forevery_insert #(between lo hi) #(fun x -> x < vi) post vi;
    forevery_refine_ext
      #(between lo hi)
      #(fun (x : between lo hi) -> x < vi \/ eq2 #(between lo hi) (SZ.v vi) x)
      (fun x -> x < vi +^ 1sz) // Have to use machine addition here...
      post;
    i := !i +^ 1sz;
    ();
  };

  rewrite each !i as hi;

  forevery_elim_empty
    #(x : between lo hi{x >= hi})
    pre;

  forevery_unrefine
    #(between lo hi)
    #(fun x -> x < hi)
    post;

  ()
}

fn for_loop (lo hi : SZ.t)
  (pre post : between lo hi -> slprop)
  (fn f (x:SZ.t{lo <= x /\ x < hi})
       requires pre (SZ.v x)
       ensures  post (SZ.v x))
  requires pure (lo <= hi)
  requires forall+ (x : between lo hi). pre x
  ensures  forall+ (x : between lo hi). post x
{
  for_loop' lo hi pre post emp fn x { f x };
}
