module Klas.Reduce.Argmax

(* Argmax as an instance of the generic relational reduction (Klas.Reduce):
   this is what makes cuBLAS amax "just a reduction with a different
   accumulator type". The accumulator is [key & nat & nat] = (best value, its
   index within the reduced range, range length); the length lets [combine]
   offset the right operand's index, so the relation is preserved by
   concatenation for a *tree* reduction (not just a left fold).

   Generic over a total preorder [le] (so the algebra is clean); amax
   instantiates [le a b := lte (fabs a) (fabs b)], amin the reverse. *)

open Klas.Reduce

let is_total (#k:Type0) (le : k -> k -> bool) = forall a b. le a b \/ le b a
let is_trans (#k:Type0) (le : k -> k -> bool) = forall a b c. le a b /\ le b c ==> le a c
let is_refl  (#k:Type0) (le : k -> k -> bool) = forall a. le a a

let acc3 (key:Type0) = key & nat & nat

let lift3 (#key:Type0) (e:key) : acc3 key = (e, 0, 1)

let combine3 (#key:Type0) (le : key -> key -> bool) (p q : acc3 key) : acc3 key =
  let (pv, pi, pl) = p in
  let (qv, qi, ql) = q in
  if le qv pv then (pv, pi, pl + ql)        (* pv >= qv: keep left value+index *)
  else (qv, qi + pl, pl + ql)               (* qv >  pv: take right, offset its index *)

(* "[a] is a correct argmax of [s]". *)
let r3 (#key:Type0) (le : key -> key -> bool) (s : Seq.seq key) (a : acc3 key) : prop =
  let (v, i, l) = a in
  l == Seq.length s /\ i < l /\ v == Seq.index s i /\
  (forall (j:nat{j < l}). le (Seq.index s j) v) /\
  (forall (j:nat{j < l}). le v (Seq.index s j) ==> i <= j)

let argmax_singleton_ok
  (#key:Type0) (le : key -> key -> bool) (_refl : squash (is_refl le))
  : singleton_ok lift3 (r3 le)
  = fun e ->
      assert (Seq.index (Seq.create 1 e) 0 == e)

#push-options "--z3rlimit 60 --fuel 1 --ifuel 1"
let argmax_concat_lemma
  (#key:Type0) (le : key -> key -> bool)
  (tot : squash (is_total le)) (trans : squash (is_trans le))
  (s1 s2 : Seq.seq key) (x1 x2 : acc3 key)
  : Lemma (requires r3 le s1 x1 /\ r3 le s2 x2)
          (ensures r3 le (Seq.append s1 s2) (combine3 le x1 x2))
  = let s = Seq.append s1 s2 in
    let (v1, i1, l1) = x1 in
    let (v2, i2, l2) = x2 in
    (* every index of s lands in s1 (< l1) or s2 (offset by l1) *)
    assert (forall (j:nat{j < l1}). Seq.index s j == Seq.index s1 j);
    assert (forall (j:nat{l1 <= j /\ j < l1 + l2}). Seq.index s j == Seq.index s2 (j - l1))
#pop-options

let argmax_concat_ok
  (#key:Type0) (le : key -> key -> bool)
  (tot : squash (is_total le)) (trans : squash (is_trans le))
  : concat_ok (combine3 le) (r3 le)
  = fun s1 s2 x1 x2 ->
      introduce (r3 le s1 x1 /\ r3 le s2 x2) ==>
                r3 le (Seq.append s1 s2) (combine3 le x1 x2)
      with _h. argmax_concat_lemma le tot trans s1 s2 x1 x2

(* The reduction and its correctness, packaged. *)
let argmax_reduce
  (#key:Type0) (le : key -> key -> bool) (s : nonempty_seq key)
  : acc3 key
  = reduce lift3 (combine3 le) s

let argmax_reduce_correct
  (#key:Type0) (le : key -> key -> bool)
  (tot : squash (is_total le)) (trans : squash (is_trans le)) (refl : squash (is_refl le))
  (s : nonempty_seq key)
  : Lemma (ensures r3 le s (argmax_reduce le s))
  = reduce_sound lift3 (combine3 le) (r3 le)
      (argmax_singleton_ok le refl) (argmax_concat_ok le tot trans) s
