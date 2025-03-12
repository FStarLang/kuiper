module Kuiper.Base

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Main
module SZ = FStar.SizeT

type mode_t = | CPU | GPU

val mode : mode_t -> slprop

(* Token for being in CPU code *)
unfold
let cpu : slprop = mode CPU

(* Token for being in GPU code *)
unfold
let gpu : slprop = mode GPU

(* Arbitrary *)
let max_blocks : erased int = pow2 30

let max_blocks_explicit : squash (reveal max_blocks == 1073741824) =
  assert_norm (reveal max_blocks == 1073741824)

(* Hard CUDA limit *)
let max_threads : erased int = 1024

(* Token for being in GPU block setup code *)
val block_setup (nthr : nat) : slprop

(* Token given to a particular block within a grid. Both here
and in thread_id, the first argument is always positive
when this resource is actually live, but not placing that refinement
here helps with inference in some places. *)
val block_id (nblk : nat) (bid : nat) : slprop

(* Token given to a particular thread within a block *)
val thread_id (nthr : nat) (tid : nat) : slprop

(* Get a concrete value for the number of blocks (~ gridDim.x) *)
fn get_gdim () 
  preserves block_id 'nblk 'bid
  requires  emp
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nblk)

fn get_bid ()
  preserves block_id 'nblk 'bid
  requires  emp
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'bid)

fn get_bdim ()
  preserves thread_id 'nthr 'tid
  requires  emp
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nthr)

fn get_tid ()
  preserves thread_id 'nthr 'tid
  requires  emp
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'tid)


// let thread_index (n: tid_t): GTot (i: nat { i < gdim_x n * bdim_x n }) = (
//   assert ((bidx_x n + 1) * bdim_x n <= gdim_x n * bdim_x n);
//   bidx_x n * bdim_x n + tidx_x n
// )
// let thread_count (n: tid_t): GTot pos = gdim_x n * bdim_x n
