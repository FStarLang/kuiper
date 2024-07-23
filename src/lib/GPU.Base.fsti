module GPU.Base

open Pulse.Lib.Pervasives
open FStar.Tactics.V2
open FStar.Seq
open Pulse.Lib.BigStar
open GPU.SizeT
open FStar.Mul
module U32 = FStar.UInt32
module SZ = FStar.SizeT

(* Token for being in CPU code *)
val cpu : slprop

(* Token for being in GPU code *)
val gpu : slprop

[@@erasable]
val tid_t : Type0

(* Arbitrary *)
let max_blocks = 1024 * 1024
(* Hard CUDA limit *)
let max_threads = 1024

(* Token for being in GPU block setup code *)
val block_setup : (nthr: SZ.t { 0 < nthr /\ nthr <= max_threads }) -> slprop

(* Token for being a particular thread *)
val thread_id : tid_t -> slprop

(* How many blocks total in the grid *)
val gdim_x : tid_t -> (r:SZ.t { 0 < r /\ r <= max_blocks })
(* Which block am I in? *)
val bidx_x : (etid:tid_t) -> (r:SZ.t { r < gdim_x etid })

(* How many threads per block *)
val bdim_x : tid_t -> (r:SZ.t { 0 < r /\ r <= max_threads })
(* Which thread am I in? *)
val tidx_x : (etid:tid_t) -> (r:SZ.t { r < bdim_x etid })

let thread_index (n: tid_t): GTot (i: nat { i < gdim_x n * bdim_x n }) = (
  assert ((bidx_x n + 1) * bdim_x n <= gdim_x n * bdim_x n);
  bidx_x n * bdim_x n + tidx_x n
)
let thread_count (n: tid_t): GTot pos = gdim_x n * bdim_x n

```pulse
val
fn block_idx_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (SZ.uint32_to_sizet id == bidx_x n)
```

```pulse
val
fn block_dim_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (SZ.uint32_to_sizet id == bdim_x n)
```

```pulse
val
fn thread_idx_x () (#n: tid_t)
  requires thread_id n
  returns  id : U32.t
  ensures  thread_id n ** pure (SZ.uint32_to_sizet id == tidx_x n)
```

let lemma_mul_lt (a b: nat) (c: nat { a < c }) (d: nat { b <= d /\ d > 0 }): Lemma (a * b < c * d) = ()

noextract inline_for_extraction
```pulse
fn thread_idx_all () (#n: tid_t)
  requires thread_id n
  returns  id : SZ.t
  ensures  thread_id n ** pure (SZ.v id == thread_index n /\ SZ.v id < max_blocks * max_threads)
{
  assert (pure (bidx_x n < 1024 * 1024 /\ tidx_x n < 1024 /\ bdim_x n <= 1024));
  lemma_mul_lt (SZ.v (bidx_x n)) (SZ.v (bdim_x n)) (1024 * 1024) 1024;
  // assert (pure (bidx_x n * tidx_x n < 1024 * 1024 * 1024 /\ bdim_x n <= 1024));
  let bid = block_idx_x ();
  let bdim = block_dim_x ();
  let tid = thread_idx_x ();
  SZ.add (SZ.mul (SZ.uint32_to_sizet bid) (SZ.uint32_to_sizet bdim)) (SZ.uint32_to_sizet tid) 
}
```
