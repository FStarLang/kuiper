module Kuiper.Conditional

#lang-pulse

open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar

[@@no_mkeys]
let if_ (b: bool) (p: slprop): slprop = cond b p emp

// REWRITE

ghost
fn if_rewrite_bool (b1 b2: bool) (#_: squash (b1 == b2)) (p: slprop)
  requires if_ b1 p
  ensures  if_ b2 p

ghost
fn if_rewrite (#b: bool) (#p1 p2: slprop) (#e: (squash b -> squash (p1 == p2)))
  requires if_ b p1
  ensures  if_ b p2

// INTRODUCTION and ELIMINATION RULES

ghost
fn if_intro_true (p: slprop)
  requires p
  ensures  if_ true p

ghost
fn if_intro_true' (b:bool) (p: slprop)
  requires pure b ** p
  ensures  if_ b p

ghost
fn if_intro_false (p: slprop)
  requires emp
  ensures  if_ false p

ghost
fn if_elim_true (p: slprop)
  requires if_ true p
  ensures  p

ghost
fn if_elim_false (p: slprop)
  requires if_ false p
  ensures  emp

// SPLIT and JOIN RULES

ghost
fn case_split (b: bool) (p: slprop)
  requires p
  ensures  if_ b p ** if_ (not b) p

ghost
fn case_join (b: bool) (p: slprop)
  requires if_ b p ** if_ (not b) p
  ensures  p

// COMBINE and SPLIT RULES

ghost
fn combine (b: bool) (p1 p2: slprop)
  requires if_ b p1 ** if_ b p2
  ensures  if_ b (p1 ** p2)

ghost
fn split (b: bool) (p1 p2: slprop)
  requires if_ b (p1 ** p2)
  ensures  if_ b p1 ** if_ b p2

// MAP

ghost
fn if_map (#b: bool) (#p #q: slprop) (f: unit -> (stt_ghost unit emp_inames (p) (fun _ -> q)))
  requires if_ b p
  ensures  if_ b q

ghost
fn if_flatten (#b1 #b2: bool) (#p: slprop)
  requires if_ b1 (if_ b2 p)
  ensures  if_ (b1 && b2) p

ghost
fn bigstar_if_elim
  (#u1 : int)
  (#m: nat)
  (#n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i:nat { m <= i /\ i < n }) -> slprop)
  requires bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i))
  ensures  p x

ghost
fn bigstar_if_intro
  (#[Tactics.exact (`0)]u1 : int)
  (m: nat)
  (n : nat {m <= n})
  (x : nat { m <= x /\ x < n })
  (p: (i: nat { m <= i /\ i < n }) -> slprop)
  requires p x
  ensures bigstar #u1 m n (fun (i:nat { m <= i /\ i < n }) -> if_ (i = x) (p i))
