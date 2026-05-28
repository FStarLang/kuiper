module Kuiper.Base

#lang-pulse
include Pulse.Lib.Send
include Kuiper.Locs
open Pulse.Lib.Core
module SZ = Kuiper.SizeT

(* This should be 2^31-1, or 2^30. We constrain this more than normal due to our
hack about interpreting size_t as uint32_t in karamel (see Kuiper.SizeT). When
that is gone, this should be increased. *)
unfold
let max_blocks : SZ.t = SZ.uint_to_t 2097152

(* Help F* *)
let max_blocks_explicit : squash (SZ.v max_blocks == 2097152 /\ SZ.v max_blocks == pow2 21) =
  assert_norm (SZ.v max_blocks == 2097152);
  assert_norm (SZ.v max_blocks == pow2 21)

(* Hard CUDA limit *)
unfold
let max_threads : SZ.t = 1024sz

unfold let warp_size = 32sz

(* Get a concrete value for the number of blocks (~ gridDim.x) *)
fn get_gdim ()
  preserves block_id 'nblk 'bid
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nblk)

(* Get a concrete value for the number of threads (~ blockDim.x) *)
fn get_bdim ()
  preserves thread_id 'nthr 'tid
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nthr)
