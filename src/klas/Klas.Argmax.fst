module Klas.Argmax

(* Formalization of an argmax as a *reduction with a (value, index)
   accumulator*. This is the pure/spec foundation for cuBLAS amax/amin
   (Isamax/Isamin): the index of the first element with the largest key.

   The point of this module is to show that argmax is a commutative monoid
   reduction - exactly like sum (add/zero) or max (fmax) - just over the
   accumulator type [key & nat] (a running best value together with its
   index). A data-race-free tree reduction over this monoid therefore computes
   the argmax, independent of the association/order in which threads combine.

   We keep it generic over a key type with a total preorder [le], so the
   algebra is clean order theory; cuBLAS amax instantiates [key := et] and
   [le a b := lte (fabs a) (fabs b)] (a total preorder on non-NaN floats),
   amin uses the reverse. *)

let acc (key:Type0) = key & nat

(* p combined with q: keep the larger key; on a tie keep the smaller index.
   [le a b] reads "a <= b". *)
let combine
  (#key:Type0) (le : key -> key -> bool)
  (p q : acc key)
  : acc key
  = let (pk, pi) = p in
    let (qk, qi) = q in
    if le qk pk
    then (if le pk qk then (if pi <= qi then p else q) else p)  (* pk >= qk *)
    else q                                                       (* pk <  qk *)

(* The three preorder/total-order facts we need from [le], passed explicitly
   so the lemmas are self-contained. *)
let is_total  (#key:Type0) (le : key -> key -> bool) = forall a b. le a b \/ le b a
let is_trans  (#key:Type0) (le : key -> key -> bool) = forall a b c. le a b /\ le b c ==> le a c
let is_refl   (#key:Type0) (le : key -> key -> bool) = forall a. le a a

let combine_comm
  (#key:Type0) (le : key -> key -> bool)
  (p q : acc key)
  : Lemma (requires is_total le /\ snd p =!= snd q)
          (ensures combine le p q == combine le q p)
  = ()

let combine_assoc
  (#key:Type0) (le : key -> key -> bool)
  (p q r : acc key)
  : Lemma (requires is_total le /\ is_trans le /\
                    snd p =!= snd q /\ snd q =!= snd r /\ snd p =!= snd r)
          (ensures combine le (combine le p q) r == combine le p (combine le q r))
  = ()

(* The reduction: fold [combine] over the indexed elements of a sequence.
   [reduce s k] is the running argmax over [s[0..k)]. *)
let rec reduce
  (#key:Type0) (le : key -> key -> bool)
  (s : Seq.seq key) (k : nat { 0 < k /\ k <= Seq.length s })
  : Tot (acc key) (decreases k)
  = if k = 1 then (Seq.index s 0, 0)
    else combine le (reduce le s (k - 1)) (Seq.index s (k - 1), k - 1)

let seq_argmax
  (#key:Type0) (le : key -> key -> bool)
  (s : Seq.seq key { 0 < Seq.length s })
  : nat
  = snd (reduce le s (Seq.length s))

(* Correctness: the chosen index is in range, its key is maximal, and it is
   the *first* such index (cuBLAS's tie-break). *)
let rec reduce_correct
  (#key:Type0) (le : key -> key -> bool)
  (s : Seq.seq key) (k : nat { 0 < k /\ k <= Seq.length s })
  : Lemma (requires is_total le /\ is_trans le /\ is_refl le)
          (ensures (let (bv, bi) = reduce le s k in
                    bi < k /\
                    bv == Seq.index s bi /\
                    (forall (j:nat{j < k}). le (Seq.index s j) bv) /\
                    (forall (j:nat{j < k}). le bv (Seq.index s j) ==> bi <= j)))
          (decreases k)
  = if k = 1 then ()
    else reduce_correct le s (k - 1)
