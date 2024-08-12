module GPU.MatMulOpt.Barrier

#lang-pulse

open FStar.Mul
open Pulse.Lib.Pervasives
open Pulse.Lib.BigStar
open GPU
module Pure = GPU.MatMulOpt.Pure
module SZ = FStar.SizeT
module U64 = FStar.UInt64

#push-options "--admit_smt_queries true"
let barrier_mm_share
    (n: nat)
    (s1 s2: Seq.Base.seq (Seq.Base.seq U64.t))
    (ar: gpu_array U64.t (2 * n))
    (it: nat { it < Seq.Base.length s1 /\ it < Seq.Base.length s2 })
    (from: nat { 0 <= from /\ from < n })
    (to: nat { 0 <= to /\ to < n })
    : slprop
=
  gpu_pts_to_array_slice ar #(1.0R /. Real.of_int n)
    from (from + 1) (Pure.singleton (Seq.Base.index (Seq.Base.index s1 it) from)) **
  gpu_pts_to_array_slice ar #(1.0R /. Real.of_int n)
    (from + n) (from + n + 1) (Pure.singleton (Seq.Base.index (Seq.Base.index s2 it) from))
#pop-options

let barrier_mm_gather
    (n: nat)
    (ar: gpu_array U64.t (2 * n))
    (it: nat)
    (from: nat { 0 <= from /\ from < n })
    (to: nat { 0 <= to /\ to < n })
    : slprop
= bigstar 0 n (fun i -> cond (i = to) (gpu_pts_to_array1 ar #(1.0R /. Real.of_int n) i ** gpu_pts_to_array1 ar #(1.0R /. Real.of_int n) (i + n)) emp)

let barrier_mm
    (n: nat)
    (s1 s2: Seq.Base.seq (Seq.Base.seq U64.t)) //  { Seq.Base.length i == n }))
    (ar: gpu_array U64.t (2 * n))
    (it: nat)
    (from: nat { 0 <= from /\ from < n })
    (to: nat { 0 <= to /\ to < n })
    : slprop
=
  if (it / 2 < Seq.Base.length s1 && it / 2 < Seq.Base.length s2)
  then (cond (it % 2 = 0) (barrier_mm_share n s1 s2 ar (it / 2) from to) (barrier_mm_gather n ar (it / 2) from to))
  else emp

let shared_pre (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
    (s1 s2: Seq.Base.seq (Seq.Base.seq U64.t))
    (it: nat) (ar: gpu_array U64.t SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr })
: slprop
= gpu_pts_to_array1 ar i **
  gpu_pts_to_array1 ar (i + nthr) ** mbarrier_tok nthr (barrier_mm nthr s1 s2 ar) it i

ghost
fn block_setup_ghost
  (nblk : SZ.t { 0 < reveal nblk /\ reveal nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (smem_sz : SZ.t { smem_sz == SZ.(2sz *^ nthr) })
  (ar: gpu_array U64.t smem_sz)
  (bid: SZ.t { 0 <= bid /\ SZ.v bid < SZ.v nblk })
  requires block_setup nthr ** (exists* v. gpu_pts_to_array #U64.t #smem_sz ar #1.0R v)
  ensures block_setup nthr ** bigstar 0 nthr (shared_pre nthr Seq.Base.empty Seq.Base.empty 0 ar)
{
  with v. assert (gpu_pts_to_array #U64.t #smem_sz ar #1.0R v);
  unfold gpu_pts_to_array ar v;
  gpu_slice_slice_1_underspec #1 ar #1.0R 0 smem_sz nthr;
  drop_   (bigstar #1 0 (SZ.v nthr - 0) (fun x -> gpu_pts_to_array1 ar (x + 0)));
  assume_ (bigstar #1 0 nthr            (fun x -> gpu_pts_to_array1 ar x));

  gpu_slice_slice_1_underspec #2 ar #1.0R nthr smem_sz smem_sz;
  drop_   (bigstar #2 0 (smem_sz - nthr) (fun x -> gpu_pts_to_array1 ar (x + nthr)));
  assume_ (bigstar #2 0 nthr             (fun x -> gpu_pts_to_array1 ar (x + nthr)));

  bigstar_zip #1 #2 #1 0 nthr _ _;

  mk_mbarrier nthr (barrier_mm nthr Seq.Base.empty Seq.Base.empty ar);
  bigstar_zip #1 #0 #0 0 nthr _ _;

  // FOLD:
  drop_   (bigstar #0 0 nthr (fun x -> gpu_pts_to_array1 ar x ** gpu_pts_to_array1 ar (x + nthr) ** mbarrier_tok nthr (barrier_mm nthr Seq.Base.empty Seq.Base.empty ar) 0 x));
  assume_ (bigstar #0 0 nthr (fun x -> shared_pre nthr Seq.Base.empty Seq.Base.empty 0 ar x));

  bigstar_uneta();
  gpu_slice_empty_elim ar smem_sz;
}

let shared_post (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads }) (ar: gpu_array U64.t SZ.(2sz *^ nthr)) (i: nat { 0 <= i /\ i < nthr }): slprop =
  exists* it. shared_pre nthr Seq.Base.empty Seq.Base.empty it ar i
