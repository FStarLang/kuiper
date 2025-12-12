module Kuiper.Bitmask

open Kuiper
#lang-pulse

val bitmask (n:nat) : Type0

val bitmask_pts_to (#n:_) (b:bitmask n) (p : FStar.GSet.set nat)
  : slprop

fn alloc (n:nat)
  requires emp
  returns  b : bitmask n
  ensures  bitmask_pts_to b FStar.GSet.empty

fn get (#n:nat) (b:bitmask n) (i:sz) (#p : GSet.set nat)
  preserves bitmask_pts_to b p
  requires  pure (i < n)
  returns   v : bool
  ensures   pure (v = FStar.GSet.mem (SizeT.v i) p)

let add #a (x:a) (s: GSet.set a) : GSet.set a =
  GSet.union (GSet.singleton x) s

let remove #a (x:a) (s: GSet.set a) : GSet.set a =
  GSet.intersect (GSet.complement (GSet.singleton x)) s

fn set (#n:nat) (b:bitmask n) (i:sz) (v:bool) (#p : GSet.set nat)
  requires  bitmask_pts_to b p
  requires  pure (i < n)
  ensures   exists* p'.
              bitmask_pts_to b p' ** 
                pure (p' == (if v then add (SizeT.v i) p else remove (SizeT.v i) p))
