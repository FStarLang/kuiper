module Kuiper.Chest1.Helpers

(* Numeric helper lemmas relating [chest1] slicing/concatenation
   ([chest1_sub], [chest1_append]) to [chest1_rsum] and to approximation
   ([%~]/[add]).  These bridge the chest1-based [Kuiper.Kernel.HReduce]
   interface to the seq-level [rsum]/[seq_approximates_append] lemmas. *)

open Kuiper
open Kuiper.Real
open Kuiper.Approximates
open Kuiper.Seq.Common
open Kuiper.Chest
module Seq = FStar.Seq

let rec chest1_fold_left'
  (#et #at : Type) (#n : nat)
  (f : at -> et -> at)
  (z : at)
  (c : chest1 et n)
  (k : natle n)
  : GTot at (decreases n - k)
  = if k = n then z
    else chest1_fold_left' f (f z (acc1 c k)) c (k + 1)

let chest1_fold_left
  (#et #at : Type) (#n : nat)
  (f : at -> et -> at)
  (z : at)
  (c : chest1 et n)
  : GTot at
  = chest1_fold_left' f z c 0

(* [chest1_to_seq] turns concatenation into [Seq.append]. *)
let chest1_to_seq_append
  (#et : Type) (#n #m : nat)
  (c1 : chest1 et n) (c2 : chest1 et m)
  : Lemma (chest1_to_seq (chest1_append c1 c2)
           == Seq.append (chest1_to_seq c1) (chest1_to_seq c2))
  = Seq.lemma_eq_elim
      (chest1_to_seq (chest1_append c1 c2))
      (Seq.append (chest1_to_seq c1) (chest1_to_seq c2))

(* Sum of a concatenation is the sum of the parts. *)
let chest1_rsum_append
  (#n #m : nat)
  (c1 : chest1 real n) (c2 : chest1 real m)
  : Lemma (chest1_rsum (chest1_append c1 c2)
           == chest1_rsum c1 +. chest1_rsum c2)
  = chest1_to_seq_append c1 c2;
    rsum_append (chest1_to_seq c1) (chest1_to_seq c2)

(* Approximation is compatible with concatenation: if [s1] and [s2]
   approximate the two partial sums, then [add s1 s2] approximates the
   sum of the concatenation. *)
let chest1_approximates_append
  (#et : Type) {| scalar et |} {| real_like et |}
  (#n #m : nat)
  (s1 s2 : et)
  (c1 : chest1 real n) (c2 : chest1 real m)
  : Lemma (requires s1 %~ chest1_rsum c1 /\ s2 %~ chest1_rsum c2)
          (ensures  (s1 `add` s2) %~ chest1_rsum (chest1_append c1 c2))
  = chest1_to_seq_append c1 c2;
    seq_approximates_append s1 s2 (chest1_to_seq c1) (chest1_to_seq c2)

(* A sub-slice [i, k) splits into [i, j) followed by [j, k). *)
let chest1_sub_split_eq
  (#et : Type) (#n : nat)
  (i j k : natle n{i <= j /\ j <= k})
  (s : chest1 et n)
  : Lemma (chest1_to_seq (chest1_sub i k s)
           == chest1_to_seq (chest1_append (chest1_sub i j s) (chest1_sub j k s)))
  = Seq.lemma_eq_elim
      (chest1_to_seq (chest1_sub i k s))
      (chest1_to_seq (chest1_append (chest1_sub i j s) (chest1_sub j k s)))

(* Sum of a sub-slice splits additively. *)
let chest1_rsum_sub_split
  (#n : nat)
  (i j k : natle n{i <= j /\ j <= k})
  (s : chest1 real n)
  : Lemma (chest1_rsum (chest1_sub i k s)
           == chest1_rsum (chest1_sub i j s) +. chest1_rsum (chest1_sub j k s))
  = chest1_sub_split_eq i j k s;
    chest1_rsum_append (chest1_sub i j s) (chest1_sub j k s)

(* A length-one sub-slice sums to its single element. *)
let chest1_rsum_sub_one
  (#et : Type)
  (#n : nat)
  (i : nat{i < n})
  (s : chest1 real n)
  : Lemma (chest1_rsum (chest1_sub i (i + 1) s) == acc1 s i)
  = let c = chest1_sub i (i + 1) s in
    Seq.lemma_eq_elim (chest1_to_seq c) (Seq.create 1 (acc1 s i));
    let SCons hd tl = view_seq (chest1_to_seq c) in
    assert (Seq.equal tl (Seq.empty #real))

(* A full sub-slice [0, n) is the whole chest. *)
let chest1_rsum_sub_full
  (#n : nat)
  (s : chest1 real n)
  : Lemma (chest1_rsum (chest1_sub 0 n s) == chest1_rsum s)
  = Seq.lemma_eq_elim (chest1_to_seq (chest1_sub 0 n s)) (chest1_to_seq s)

(* [acc1] of a [mk1] reduces; exposed as an SMT pattern so the framing prover
   can compute singleton/literal chest contents. *)
let acc1_mk1 (#et : Type) (#n : nat) (f : natlt n -> GTot et) (i : natlt n)
  : Lemma (acc1 (mk1 f) i == f i)
          [SMTPat (acc1 (mk1 f) i)]
  = ()

(* Chest-level approximation transfers to the underlying sequences. *)
let chest1_to_seq_approx
  (#et : Type) {| scalar et |} {| real_like et |}
  (#n : nat)
  (va : chest1 et n) (vr : chest1 real n)
  : Lemma (requires va %~ vr)
          (ensures  chest1_to_seq va %~ chest1_to_seq vr)
  = assert (forall (i : natlt n). acc1 va i %~ acc1 vr i)

