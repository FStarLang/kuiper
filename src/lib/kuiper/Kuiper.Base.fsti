module Kuiper.Base

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Main
module SZ = Kuiper.SizeT

type mode_t = | CPU | GPU

val mode : mode_t -> slprop

(* Token for being in CPU code *)
unfold
let cpu : slprop = mode CPU

(* Token for being in GPU code *)
unfold
let gpu : slprop = mode GPU

(* Token allowing to create a barrier for n threads. Only
   available while in the block_setup of a kernel. *)
[@@no_mkeys]
val can_create_barrier (nthr : nat) : slprop

(* Token marking we have in fact created a barrier, or ditched the
   token.  This makes sure the token is not stashed away somewhere. *)
[@@no_mkeys]
val consumed_can_create_barrier : slprop

(* A function that can be called to consume the token
   without actually creating a barrier. We could also
   define consumed_can_create_barrier as a trivial
   barrier token. *)
ghost
fn no_mk_barrier (#n:nat) ()
  requires can_create_barrier n
  ensures  consumed_can_create_barrier

(* This should be 2^31-1, or 2^30. We constrain this more than normal due to our
hack about interpreting size_t as uint32_t in karamel (see Kuiper.SizeT). When
that is gone, this should be increased. *)
unfold
let max_blocks : erased int = 2097152 // 2^21

(* Help F* *)
let max_blocks_explicit : squash (reveal max_blocks == 2097152 /\ reveal max_blocks == pow2 21) =
  assert_norm (reveal max_blocks == 2097152);
  assert_norm (reveal max_blocks == pow2 21)

(* Hard CUDA limit *)
unfold
let max_threads : erased int = 1024

inline_for_extraction noextract
unfold let warp_sz = 32sz
inline_for_extraction noextract
unfold let warp_size = 32

(* Token given to a particular block within a grid. Both here
and in thread_id, the first argument is always positive
when this resource is actually live, but not placing that refinement
here helps with inference in some places. *)
[@@no_mkeys]
val block_id (nblk : int) (bid : int) : slprop

(* Token given to a particular thread within a block *)
[@@no_mkeys]
val thread_id (nthr : int) (tid : int) : slprop

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
