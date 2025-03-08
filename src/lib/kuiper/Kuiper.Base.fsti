module Kuiper.Base

#lang-pulse

open Kuiper.Common
open Kuiper.SizeT
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

[@@erasable]
val tid_t : Type0

(* Arbitrary *)
let max_blocks : erased int = pow2 30

let max_blocks_explicit : squash (reveal max_blocks == 1073741824) =
  assert_norm (reveal max_blocks == 1073741824)

(* Hard CUDA limit *)
let max_threads : erased int = 1024

(* Token for being in GPU block setup code *)
val block_setup (nthr: SZ.t { 0 < nthr /\ nthr <= max_threads })
  : slprop

(* Token for being a particular thread *)
val thread_id (tid : tid_t)
  : slprop

(* How many blocks total in the grid *)
val gdim_x (tid : tid_t)
  : GTot (r:pos{r <= max_blocks})

(* Which block am I in? *)
val bidx_x (tid : tid_t)
  : GTot (r:nat{r < gdim_x tid})

(* How many threads per block *)
val bdim_x (tid : tid_t)
  : GTot (r:pos{r <= max_threads})

(* Which thread am I in? *)
val tidx_x (tid : tid_t)
  : GTot (r:nat{r < bdim_x tid})

let thread_index (n: tid_t): GTot (i: nat { i < gdim_x n * bdim_x n }) = (
  assert ((bidx_x n + 1) * bdim_x n <= gdim_x n * bdim_x n);
  bidx_x n * bdim_x n + tidx_x n
)
let thread_count (n: tid_t): GTot pos = gdim_x n * bdim_x n

fn block_idx_x () (#n: tid_t)
  preserves thread_id n
  requires  emp
  returns   id : SZ.t
  ensures   pure (SZ.v id == bidx_x n)

fn block_dim_x () (#n: tid_t)
  preserves thread_id n
  requires  emp
  returns   id : SZ.t
  ensures   pure (SZ.v id == bdim_x n)

fn thread_idx_x () (#n: tid_t)
  preserves thread_id n
  requires  emp
  returns   id : SZ.t
  ensures   pure (SZ.v id == tidx_x n)
