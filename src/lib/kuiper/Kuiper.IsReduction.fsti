module Kuiper.IsReduction

open Pulse.Lib.Core
open Kuiper.Functions
open FStar.Seq
open Kuiper.Scalars
open Kuiper.Seq.Common
open Kuiper.Len
open Kuiper.Array
open Kuiper.Conditional

noeq
type is_reduction (#a:Type0) (z:a) (f : a -> a -> a) : (s : seq a) -> (r : a) -> Type0 =
  | Emp :
    seq![] `is_reduction z f` z
  | Singl :
    r:a ->
    seq![r] `is_reduction z f` r
  | Split :
    s1:seq a -> s2:seq a -> r1:a -> r2:a ->
    s1 `is_reduction z f` r1 ->
    s2 `is_reduction z f` r2 ->
    (s1 `Seq.append` s2) `is_reduction z f` (r1 `f` r2)
    (* ^ FIXME: cannot use `@+` above, bad inference. *)

(*
  [1;2] `is_reduction` 1 + 2
  [1;2] `is_reduction` (1 + 2) + 0
  [1;2] `is_reduction` 0 + (1 + 2)
  ....
 *)

  // Add at some point:
  // | Perm :
  //   s1:seq a -> s2:seq a ->
  //   r:a ->
  //   is_permutation s1 s2 ->
  //   is_reduction z f s1 r ->
  //   is_reduction z f s2 r

val lemma_Singl (#a:Type0) (z:a) (f : a -> a -> a) (r : a)
  : Lemma (is_reduction z f seq![r] r)
          [SMTPat (is_reduction z f seq![r] r)]

val ac_eq_foldl
  (#a:Type) (z : a) (f : a -> a -> a) (s : seq a) (r : a)
  : Lemma (requires is_comm_semigroup z f /\ is_reduction z f s r)
          (ensures r == Kuiper.Seq.Common.seq_fold_left f z s)

val assoc_uniq_reduction
  (#a:Type) (z:a) (f : a -> a -> a) (xs : seq a) (r1 r2 : a)
: Lemma (requires is_comm_semigroup z f /\ is_reduction z f xs r1 /\ is_reduction z f xs r2)
        (ensures r1 == r2)

val op_is_reduction
  (#a:Type0) (z:a) (f : a -> a -> a)
  (s1 : seq a) (r1 : a)
  (s2 : seq a) (r2 : a)
: Lemma (requires is_reduction z f s1 r1 /\ is_reduction z f s2 r2)
        (ensures is_reduction z f (s1 @+ s2) (f r1 r2))
        [SMTPat (is_reduction z f (s1 @+ s2) (f r1 r2))]


(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
unfold
let gpu_pts_to_slice_sum_inner
  (#et:Type0) {| scalar et |}
  (#sz:nat)
  (r : gpu_array et sz)
  (i j :nat)
  (v : seq et)
  (s : seq et)
: slprop
= gpu_pts_to_slice r i j s
  ** pure (i < j /\ j <= sz /\
           len v = sz /\
           len s = j - i /\
           squash (is_reduction zero add (Seq.slice v i j) (s @! 0))) // SQUASH VERY IMPORTANT!!

(* Not easy to mark this unfold as it has a lambda (in the exists) *)
let gpu_pts_to_slice_sum
  (#et:Type0) {| scalar et |}
  (#sz:nat)
  ([@@@mkey] r: gpu_array et sz)
  ([@@@mkey] i : nat)
  (j:nat)
  (v: seq et)
: slprop
= exists* s. gpu_pts_to_slice_sum_inner r i j v s
