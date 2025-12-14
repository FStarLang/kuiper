module Kuiper.Bitmask

open Kuiper
#lang-pulse

module U32 = FStar.UInt32

val bitmask_len (n : nat) : (nb : nat{nb * 32 >= n})

val bitmask (n:nat) : Type0

val bitmask_pts_to (#n:_) (b:bitmask n) (p : GSet.set nat)
  : slprop


fn init_empty (n:nat) (a : larray u32 (bitmask_len n))
  requires
    exists* v_a.
      a |-> v_a **
      pure (forall i. v_a @! i == 0ul)
  returns  b : bitmask n
  ensures  bitmask_pts_to b GSet.empty


let full #a : GSet.set a = GSet.comprehend (fun _ -> true)
let full_mask = U32.uint_to_t 0xffffffff

fn init_full (n:nat) (a : larray u32 (bitmask_len n))
  requires
    exists* v_a.
      a |-> v_a **
      pure (forall i. v_a @! i == full_mask)
  returns  b : bitmask n
  ensures  bitmask_pts_to b full

fn get (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  preserves bitmask_pts_to b p
  requires  pure (i < n)
  returns   v : bool
  ensures   pure (v = GSet.mem (SizeT.v i) p)

let add #a (x:a) (s: GSet.set a) : GSet.set a =
  GSet.union (GSet.singleton x) s

let remove #a (x:a) (s: GSet.set a) : GSet.set a =
  GSet.intersect (GSet.complement (GSet.singleton x)) s

fn set (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  requires  bitmask_pts_to b p
  requires  pure (i < n)
  ensures   bitmask_pts_to b (add (SizeT.v i) p) 

fn unset (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  requires  bitmask_pts_to b p
  requires  pure (i < n)
  ensures   bitmask_pts_to b (remove (SizeT.v i) p)