module KPlay.Proof.Base.WarpCache

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.ForEvery
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module BF = Kuiper.BFloat16
module Every = Kuiper.ForEvery
module Warp = Kuiper.Barrier.Warp
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.CacheNative
module Head = KPlay.Proof.Base.Head
module Hidden = KPlay.Proof.Base.WarpHidden

let lane = Head.lane

ghost
fn rec share
  (cache:larray BF.t 12582912) (#before:erased Cache.snapshot) (#fraction:perm) (count:pos)
  requires A.pts_to_mask cache #fraction before Init.all_cells
  ensures forall+ (_:Kuiper.Common.natlt count).
    A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells
  decreases count
{
  if (count = 1) {
    rewrite A.pts_to_mask cache #fraction before Init.all_cells
      as A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells;
    Every.forevery_intro_false #(Kuiper.Common.natlt count)
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells);
    Every.forevery_insert #(Kuiper.Common.natlt count)
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells) 0;
    Every.forevery_unrefine #(Kuiper.Common.natlt count)
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells);
  } else {
    let remainder = fraction -. (fraction /. Real.of_int count);
    let values:erased (Seq.seq (option BF.t)) = hide #(Seq.seq (option BF.t)) (reveal #Cache.snapshot before);
    A.mask_share_gen cache #values #fraction (fraction /. Real.of_int count) remainder #Init.all_cells;
    rewrite A.pts_to_mask cache #(fraction /. Real.of_int count) values Init.all_cells
      as A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells;
    rewrite A.pts_to_mask cache #remainder values Init.all_cells
      as A.pts_to_mask cache #remainder before Init.all_cells;
    share cache #before #remainder (count - 1);
    Every.forevery_ext #(Kuiper.Common.natlt (count - 1))
      (fun _ -> A.pts_to_mask cache #(remainder /. Real.of_int (count - 1)) before Init.all_cells)
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells);
    Every.forevery_natlt_push count
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells);
  }
}

ghost
fn rec gather_same
  (cache:larray BF.t 12582912) (#before:erased Cache.snapshot) (#fraction:perm) (count:pos)
  requires forall+ (_:Kuiper.Common.natlt count).
    A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells
  ensures A.pts_to_mask cache #fraction before Init.all_cells
  decreases count
{
  if (count = 1) {
    Every.forevery_singleton_elim' #(Kuiper.Common.natlt count)
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells) 0;
  } else {
    Every.forevery_natlt_pop count
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells);
    let remainder = fraction -. (fraction /. Real.of_int count);
    Every.forevery_ext #(Kuiper.Common.natlt (count - 1))
      (fun _ -> A.pts_to_mask cache #(fraction /. Real.of_int count) before Init.all_cells)
      (fun _ -> A.pts_to_mask cache #(remainder /. Real.of_int (count - 1)) before Init.all_cells);
    gather_same cache #before #remainder (count - 1);
    A.mask_gather cache;
    with values. assert A.pts_to_mask cache #(fraction /. Real.of_int count +. remainder) values Init.all_cells;
    assert pure (Seq.equal values before);
    Seq.lemma_eq_elim values before;
    rewrite A.pts_to_mask cache #(fraction /. Real.of_int count +. remainder) values Init.all_cells
      as A.pts_to_mask cache #fraction before Init.all_cells;
  }
}

inline_for_extraction noextract
let full (cache:larray BF.t 12582912) (property:Cache.snapshot -> prop) : slprop =
  exists* (values:Cache.snapshot). A.pts_to_mask cache values Init.all_cells ** pure (property values)

inline_for_extraction noextract
let shared (cache:larray BF.t 12582912) (property:Cache.snapshot -> prop) (i:lane) : slprop =
  exists* (values:Cache.snapshot). A.pts_to_mask cache #(1.0R /. 32) values Init.all_cells **
    pure (i = 0 ==> property values)

inline_for_extraction noextract
let anchor (cache:larray BF.t 12582912) (values:Cache.snapshot) : slprop =
  A.pts_to_mask cache #(1.0R /. 32) values Init.all_cells

ghost
fn agreement
  (cache:larray BF.t 12582912) (#left #right:erased Cache.snapshot) (#fl #fr:perm)
  preserves A.pts_to_mask cache #fl left Init.all_cells ** A.pts_to_mask cache #fr right Init.all_cells
  ensures pure (left == right)
{
  A.pts_to_mask_injective_eq cache;
  assert pure (Seq.equal left right);
  Seq.lemma_eq_elim left right;
}

ghost
fn gather
  (cache:larray BF.t 12582912) (property:Cache.snapshot -> prop) ()
  requires forall+ (i:lane). shared cache property i
  ensures forall+ (i:lane). Head.lane_only i (full cache property)
{
  Every.forevery_extract_if_eqtype #lane (0 <: lane) (shared cache property);
  unfold shared cache property (0 <: lane);
  with first. assert A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells;
  fold anchor cache first;
  Every.forevery_ext #lane
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else shared cache property i)
    (fun i -> Hidden.nonzero i (shared cache property i));
  Every.forevery_map_extra #lane (anchor cache first)
    (fun i -> Hidden.nonzero i (shared cache property i))
    (fun i -> Hidden.nonzero i (A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells))
    fn i {
      if (( = ) #lane i (0 <: lane)) {
        rewrite Hidden.nonzero i (shared cache property i)
          as Hidden.nonzero i (A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells);
      } else {
        rewrite Hidden.nonzero i (shared cache property i) as shared cache property i;
        unfold shared cache property i;
        with other. assert A.pts_to_mask cache #(1.0R /. 32) other Init.all_cells;
        unfold anchor cache first;
        agreement cache #first #other #(1.0R /. 32) #(1.0R /. 32);
        rewrite A.pts_to_mask cache #(1.0R /. 32) other Init.all_cells
          as A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells;
        fold anchor cache first;
        rewrite A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells
          as Hidden.nonzero i (A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells);
      }
    };
  Every.forevery_ext #lane
    (fun i -> Hidden.nonzero i (A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells))
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells);
  unfold anchor cache first;
  Every.forevery_unextract_if_eqtype #lane (0 <: lane)
    (fun (_:lane) -> A.pts_to_mask cache #(1.0R /. 32) first Init.all_cells);
  gather_same cache #first #1.0R 32;
  fold full cache property;
  Head.give_lane_zero (full cache property) ();
}

ghost
fn scatter
  (cache:larray BF.t 12582912) (property:Cache.snapshot -> prop) ()
  requires forall+ (i:lane). Head.lane_only i (full cache property)
  ensures forall+ (i:lane). shared cache property i
{
  Head.take_lane_zero (full cache property) ();
  unfold full cache property;
  with values. assert A.pts_to_mask cache values Init.all_cells;
  share cache #values #1.0R 32;
  Every.forevery_map #lane
    (fun (_:lane) -> A.pts_to_mask cache #(1.0R /. 32) values Init.all_cells)
    (shared cache property)
    fn i { fold shared cache property i; };
}
