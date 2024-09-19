module Kuiper.MatMulTile.Prep
#lang-pulse

#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Math

module SZ   = FStar.SizeT
module Kernel = Kuiper.MatMulTile.Kernel

ghost
fn setup
  (rows shared columns : pos)
  (bdim : pos { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : nat { nblk == (rows / bdim) * (columns / bdim) })
  (nthr : nat { nthr == bdim * bdim
                     /\ nblk * nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1 : seq u64 { Seq.length v1 == rows * shared })
  (v2 : seq u64 { Seq.length v2 == shared * columns })
  (#s : seq u64)
  requires gpu_pts_to_array gr s **
           gpu_pts_to_array ga1 v1 **
           gpu_pts_to_array ga2 v2
  ensures  bigstar 0 (nblk * nthr) (fun i ->
             Kernel.kpre rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr)
               (Kernel.tid_to_idx rows shared columns bdim i))


ghost
fn breakdown
  (rows shared columns : pos)
  (bdim : pos { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : nat { nblk == (rows / bdim) * (columns / bdim) })
  (nthr : nat { nthr == bdim * bdim
                     /\ nblk * nthr == rows * columns
                     /\ 2 * (shared / bdim) >= 0
                     })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga1 : gpu_array u64 (rows * shared))
  (ga2 : gpu_array u64 (shared * columns))
  (gr  : gpu_array u64 (nblk * nthr))
  (v1 : seq u64 { Seq.length v1 == rows * shared })
  (v2 : seq u64 { Seq.length v2 == shared * columns })
  requires
    bigstar 0 (nblk * nthr) (fun i ->
      Kernel.kpost rows shared columns ga1 ga2 gr #v1 #v2 (nblk * nthr) (Kernel.tid_to_idx rows shared columns bdim i))
  ensures
    (exists* vr. gpu_pts_to_array gr vr) **
    gpu_pts_to_array ga1 v1 **
    gpu_pts_to_array ga2 v2
