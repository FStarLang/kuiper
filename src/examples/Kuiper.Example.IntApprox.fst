module Kuiper.Example.IntApprox

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Chest1.Helpers { chest1_fold_left }
module SZ = Kuiper.SizeT
module U32 = FStar.UInt32

(* Strengthen the spec of a reduction *)

noextract
fn reduce (#et:Type0) {| scalar et, real_like et |}
  (len : szp)
  (a : array1 u32 (l1_forward len) { is_global a })
  (#s : chest1 u32 len)
  (r : chest1 real len)
  preserves cpu
  preserves on gpu_loc (a |-> s)
  requires  pure (s %~ r) **
            pure (SZ.fits (len + 1024sz)) // Almost impossible to falsify
  returns   res : u32
  ensures   pure (res %~ chest1_rsum r)
{
  assert pure (equal (chest_map id r) r);
  Kuiper.Kernel.Reduce.reduce1 id id len 1024sz a r;
}

(* Move these 3 away *)
let to_real_chest (#a:Type) {| scalar a, real_like a |}
  (#r : nat) (#d : shape r)
  (s : chest d a) : GTot (chest d real)
  = chest_map to_real s

let to_real_chest_is_approx (#a:Type) (#n:nat) {| scalar a, real_like a |}
  (s : chest1 a n)
  : Lemma (s %~ to_real_chest s)
          [SMTPat (chest_approximates s (to_real_chest s))]
  = ()

let sum_is_approx #a #n {| scalar a, real_like a |}
  (s : chest1 a n) (s' : chest1 real n)
  : Lemma (requires s %~ s')
          (ensures chest1_fold_left add zero s %~ chest1_rsum s')
  = admit() // induction

(* A stronger exact spec for reduce on u32s, proven from the approximate spec. *)
noextract
fn reduce_u32
  (len : szp)
  (a : array1 u32 (l1_forward len) { is_global a })
  (#s : chest1 u32 len)
  preserves cpu
  preserves on gpu_loc (a |-> s)
  requires  pure (SZ.fits (len + 1024sz)) // Almost impossible to falsify
  returns   res : u32
  ensures   pure (U32.v res == chest1_fold_left add zero s)
{
  to_real_chest_is_approx s;
  let res = reduce #u32 len a #s (chest_map to_real s);
  let rr = chest1_rsum (chest_map to_real s);
  assert pure (res %~ rr);
  sum_is_approx s (chest_map to_real s);
  v_approximates_inj res (chest1_fold_left add zero s) rr;
  res
}
