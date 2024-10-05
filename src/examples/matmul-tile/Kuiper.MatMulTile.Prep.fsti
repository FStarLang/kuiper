module Kuiper.MatMulTile.Prep
#set-options "--fuel 1 --ifuel 1 --z3rlimit 40"
#lang-pulse

open Kuiper
open Kuiper.Math

module SZ = FStar.SizeT
module K  = Kuiper.MatMulTile.Kernel

ghost
fn setup
  (rows shared columns : pos)
  (bdim : pos { bdim /? rows /\ bdim /? columns /\ bdim /? shared /\ bdim < pow2 30})
  (nblk : nat { nblk == (rows / bdim) * (columns / bdim) })
  (nthr : nat { nthr == bdim * bdim })
  (* ^ 2nd and 3rd conjunct above just to help verifying this spec, sigh. *)
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  requires
    (ga |-> 'va) **
    (gb |-> 'vb) **
    (gr |-> 'vr)
  ensures 
    bigstar 0 (nblk * nthr) (fun i ->
      K.kpre rows shared columns ga gb gr #'va #'vb (nblk * nthr)
        (K.tid_to_idx rows shared columns bdim i))

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
  (ga : gpu_array u64 (rows * shared))
  (gb : gpu_array u64 (shared * columns))
  (gr : gpu_array u64 (rows * columns))
  requires
    bigstar 0 (nblk * nthr) (fun i ->
      K.kpost rows shared columns ga gb gr #'va #'vb (nblk * nthr) (K.tid_to_idx rows shared columns bdim i))
  ensures
    (exists* vr. gr |-> vr) **
    (ga |-> 'va) **
    (gb |-> 'vb)