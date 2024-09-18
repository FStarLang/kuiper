module GPU.Base

#lang-pulse

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Seq
open Pulse.Lib.BigStar
open GPU.SizeT
open FStar.Mul
module U32 = FStar.UInt32
module SZ = FStar.SizeT

type mode_t = | CPU | GPU

val mode : mode_t -> slprop

(* Token for being in CPU code *)
[@@pulse_unfold]
let cpu : slprop = mode CPU

(* Token for being in GPU code *)
[@@pulse_unfold]
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
  : GTot (r:nat{0 < r /\ r <= max_blocks})

(* Which block am I in? *)
val bidx_x (tid : tid_t)
  : GTot (r:nat{r < gdim_x tid})

(* How many threads per block *)
val bdim_x (tid : tid_t)
  : GTot (r:nat{0 < r /\ r <= max_threads})

(* Which thread am I in? *)
val tidx_x (tid : tid_t)
  : GTot (r:nat{r < bdim_x tid})

let thread_index (n: tid_t): GTot (i: nat { i < gdim_x n * bdim_x n }) = (
  assert ((bidx_x n + 1) * bdim_x n <= gdim_x n * bdim_x n);
  bidx_x n * bdim_x n + tidx_x n
)
let thread_count (n: tid_t): GTot pos = gdim_x n * bdim_x n

fn block_idx_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (U32.v id == bidx_x n)

fn block_dim_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (U32.v id == bdim_x n)

fn thread_idx_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (U32.v id == tidx_x n)

let lemma_mul_lt (a b: nat) (c: nat { a < c }) (d: nat { b <= d /\ d > 0 }): Lemma (a * b < c * d) = ()

noextract inline_for_extraction
fn thread_idx_all () (#n: tid_t)
  requires thread_id n
  returns  id : SZ.t // FIXME: do we use SZ.t or U32.t for thread/block indices? Be consistent
  ensures  thread_id n ** pure (SZ.v id == thread_index n /\ SZ.v id < max_blocks * max_threads)
{
  assert (pure (bidx_x n < 1024 * 1024 * 1024 /\ tidx_x n < 1024 /\ bdim_x n <= 1024));
  lemma_mul_lt (bidx_x n) (bdim_x n) (1024 * 1024 * 1024) 1024;
  assert (pure (bidx_x n * tidx_x n < 1024 * 1024 * 1024 * 1024 /\ bdim_x n <= 1024));
  let bid = block_idx_x ();
  let bdim = block_dim_x ();
  let tid = thread_idx_x ();
  open FStar.SizeT;
  let r =
    (SZ.uint32_to_sizet bid *^ SZ.uint32_to_sizet bdim) +^ SZ.uint32_to_sizet tid;
  r
}
