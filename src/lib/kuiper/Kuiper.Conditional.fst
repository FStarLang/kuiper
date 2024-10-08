module Kuiper.Conditional

#lang-pulse

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open FStar.Tactics.V2

// REWRITE

ghost
fn if_rewrite_bool (b1 b2: bool) (#_: squash (b1 == b2)) (p: slprop)
  requires if_ b1 p
  ensures  if_ b2 p
{ () }

ghost
fn if_rewrite (#b: bool) (#p1 p2: slprop) (#e: (squash b -> squash (p1 == p2)))
  requires if_ b p1
  ensures  if_ b p2
{
  if b {
    e ();
    rewrite each p1 as p2;
  } else {
    ()
  }
}

// INTRODUCTION and ELIMINATION RULES

ghost
fn if_intro_true (p: slprop)
  requires p
  ensures  if_ true p
{
  rewrite p as if_ true p;
}

ghost
fn if_intro_false (p: slprop)
  requires emp
  ensures  if_ false p
{
  rewrite emp as if_ false p;
}

ghost
fn if_elim_true (p: slprop)
  requires if_ true p
  ensures  p
{
  rewrite if_ true p as p;
}

ghost
fn if_elim_false (p: slprop)
  requires if_ false p
  ensures  emp
{
  rewrite if_ false p as emp;
}

// SPLIT and JOIN RULES

ghost
fn case_split (b: bool) (p: slprop)
  requires p
  ensures  if_ b p ** if_ (not b) p
{
  if b {
    if_intro_true p;
    if_rewrite_bool true b p;
    if_intro_false p;
    if_rewrite_bool false (not b) p;
  } else {
    if_intro_true p;
    if_rewrite_bool true (not b) p;
    if_intro_false p;
    if_rewrite_bool false b p;
  }
}

ghost
fn case_join (b: bool) (p: slprop)
  requires if_ b p ** if_ (not b) p
  ensures  p
{
  if b {
    if_elim_true p;
    if_elim_false p;
  } else {
    if_elim_true p;
    if_elim_false p;
  }
}

// COMBINE and SPLIT RULES

ghost
fn combine (b: bool) (p1 p2: slprop)
  requires if_ b p1 ** if_ b p2
  ensures  if_ b (p1 ** p2)
{
  if b {
    if_elim_true p1;
    if_elim_true p2;
    if_intro_true (p1 ** p2);
  } else {
    if_elim_false p1;
    if_elim_false p2;
    if_intro_false (p1 ** p2);
  }
}

ghost
fn split (b: bool) (p1 p2: slprop)
  requires if_ b (p1 ** p2)
  ensures  if_ b p1 ** if_ b p2
{
  if b {
    if_elim_true (p1 ** p2);
    if_intro_true p1;
    if_rewrite_bool true b p1;
    if_intro_true p2;
    if_rewrite_bool true b p2;
  } else {
    if_elim_false (p1 ** p2);
    if_intro_false p1;
    if_rewrite_bool false b p1;
    if_intro_false p2;
    if_rewrite_bool false b p2;
  }
}

// MAP

ghost
fn if_map (#b: bool) (#p #q: slprop) (f: unit -> (stt_ghost unit emp_inames (p) (fun _ -> q)))
  requires if_ b p
  ensures  if_ b q
{
  if b {
    if_elim_true p;
    f ();
    if_intro_true q;
  } else {
    if_elim_false p;
    if_intro_false q;
  }
}

ghost
fn if_flatten (#b1 #b2: bool) (#p: slprop)
  requires if_ b1 (if_ b2 p)
  ensures  if_ (b1 && b2) p
{
  if b1 {
    if_elim_true (if_ b2 p);
    if b2 {
      if_elim_true  p;
      if_intro_true p;
    } else {
      if_elim_false p;
      if_intro_false p;
    }
  } else {
    if_elim_false (if_ b2 p);
    if_intro_false p;
  }
}

// BIGSTAR

ghost
fn bigstar_if_elim
  (#u1 : int)
  (#m: nat)
  (#n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i:nat { m <= i /\ i < n }) -> slprop)
  requires bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i))
  ensures  p x
{
  rewrite (bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i)))
       as (bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp));
  Pulse.Lib.BigStar.bigstar_if_elim #u1 #m #n x p;
}

ghost
fn bigstar_if_intro
  (#[Tactics.exact (`0)]u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i:nat { m <= i /\ i < n }) -> slprop)
  requires p x
  ensures  bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i))
{
  Pulse.Lib.BigStar.bigstar_if_intro #u1 m n x p;
  rewrite (bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> cond (i = x) (p i) emp))
       as (bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i)));
}
