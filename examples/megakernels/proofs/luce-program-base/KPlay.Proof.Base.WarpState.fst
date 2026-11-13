module KPlay.Proof.Base.WarpState

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery

module A = Pulse.Lib.Array
module T = Kuiper.Tensor
module BF = Kuiper.BFloat16
module Every = Kuiper.ForEvery
module Warp = Kuiper.Barrier.Warp
module Native = KPlay.Proof.Base.NativeProjection
module Init = KPlay.Proof.Base.Initialization
module Cache = KPlay.Proof.Base.CacheNative
module Head = KPlay.Proof.Base.Head
module Hidden = KPlay.Proof.Base.WarpHidden
module CacheShare = KPlay.Proof.Base.WarpCache

let lane = Head.lane
let property = Hidden.snapshot -> Cache.snapshot -> prop

inline_for_extraction noextract
let full (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property) : slprop =
  exists* (h:Hidden.snapshot) (k:Cache.snapshot).
    hidden |-> h ** A.pts_to_mask cache k Init.all_cells ** pure (p h k)

inline_for_extraction noextract
let pair (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (h:Hidden.snapshot) (k:Cache.snapshot) : slprop =
  hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells

inline_for_extraction noextract
let shared (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property) (i:lane) : slprop =
  exists* (h:Hidden.snapshot) (k:Cache.snapshot).
    pair hidden cache h k ** pure (i = 0 ==> p h k)

inline_for_extraction noextract
let anchor (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (h:Hidden.snapshot) (k:Cache.snapshot) : slprop =
  pair hidden cache h k

ghost
fn scatter
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property) ()
  requires forall+ (i:lane). Head.lane_only i (full hidden cache p)
  ensures forall+ (i:lane). shared hidden cache p i
{
  Head.take_lane_zero (full hidden cache p) ();
  unfold full hidden cache p;
  with h k. assert hidden |-> h ** A.pts_to_mask cache k Init.all_cells;
  T.tensor_share_n hidden 32;
  CacheShare.share cache #k #1.0R 32;
  Every.forevery_zip #lane
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) h)
    (fun (_:lane) -> A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells);
  Every.forevery_map #lane
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells)
    (shared hidden cache p)
    fn i {
      fold pair hidden cache h k;
      fold shared hidden cache p i;
    };
}

ghost
fn gather
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property) ()
  requires forall+ (i:lane). shared hidden cache p i
  ensures forall+ (i:lane). Head.lane_only i (full hidden cache p)
{
  Every.forevery_extract_if_eqtype #lane (0 <: lane) (shared hidden cache p);
  unfold shared hidden cache p (0 <: lane);
  with h k. assert pair hidden cache h k;
  fold anchor hidden cache h k;
  Every.forevery_ext #lane
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else shared hidden cache p i)
    (fun i -> Hidden.nonzero i (shared hidden cache p i));
  Every.forevery_map_extra #lane (anchor hidden cache h k)
    (fun i -> Hidden.nonzero i (shared hidden cache p i))
    (fun i -> Hidden.nonzero i (pair hidden cache h k))
    fn i {
      if (( = ) #lane i (0 <: lane)) {
        rewrite Hidden.nonzero i (shared hidden cache p i) as Hidden.nonzero i (pair hidden cache h k);
      } else {
        rewrite Hidden.nonzero i (shared hidden cache p i) as shared hidden cache p i;
        unfold shared hidden cache p i;
        with other_h other_k. assert pair hidden cache other_h other_k;
        unfold pair hidden cache other_h other_k;
        unfold anchor hidden cache h k;
        unfold pair hidden cache h k;
        Hidden.agreement hidden #h #other_h #(1.0R /. 32) #(1.0R /. 32);
        CacheShare.agreement cache #k #other_k #(1.0R /. 32) #(1.0R /. 32);
        rewrite hidden |-> Frac (1.0R /. 32) other_h as hidden |-> Frac (1.0R /. 32) h;
        rewrite A.pts_to_mask cache #(1.0R /. 32) other_k Init.all_cells
          as A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells;
        fold pair hidden cache h k;
        fold anchor hidden cache h k;
        fold pair hidden cache h k;
        rewrite pair hidden cache h k as Hidden.nonzero i (pair hidden cache h k);
      }
    };
  Every.forevery_ext #lane
    (fun i -> Hidden.nonzero i (pair hidden cache h k))
    (fun i -> if ( = ) #lane i (0 <: lane) then emp else pair hidden cache h k);
  unfold anchor hidden cache h k;
  Every.forevery_unextract_if_eqtype #lane (0 <: lane) (fun (_:lane) -> pair hidden cache h k);
  Every.forevery_map #lane (fun (_:lane) -> pair hidden cache h k)
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) h ** A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells)
    fn _ { unfold pair hidden cache h k; };
  Every.forevery_unzip #lane
    (fun (_:lane) -> hidden |-> Frac (1.0R /. 32) h)
    (fun (_:lane) -> A.pts_to_mask cache #(1.0R /. 32) k Init.all_cells);
  T.tensor_gather_n hidden 32;
  CacheShare.gather_same cache #k #1.0R 32;
  fold full hidden cache p;
  Head.give_lane_zero (full hidden cache p) ();
}

inline_for_extraction noextract
let scatter_wait (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** Head.lane_only tid (full hidden cache p))
      (ensures fun _ -> thread_id 32 tid ** shared hidden cache p tid) =
  Warp.warp_barrier_wait ()
    (fun i -> Head.lane_only i (full hidden cache p)) (shared hidden cache p)
    (scatter hidden cache p ()) #32 #tid

inline_for_extraction noextract
let gather_wait (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912) (p:property)
  (#tid:Kuiper.Common.enatlt 32)
  : stt unit
      (requires thread_id 32 tid ** shared hidden cache p tid)
      (ensures fun _ -> thread_id 32 tid ** Head.lane_only tid (full hidden cache p)) =
  Warp.warp_barrier_wait () (shared hidden cache p)
    (fun i -> Head.lane_only i (full hidden cache p))
    (gather hidden cache p ()) #32 #tid
