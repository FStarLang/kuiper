module Klas.Reduce

(* Generic *relational* reduction contract.

   This is the abstraction that lets one reduction kernel serve any
   intermediate/accumulator type: instead of fixing the operation (add, fmax,
   ...) and a concrete spec (rsum, seq_max, ...), a reduction is parameterized
   by

     - an element type [et] and an accumulator type [acc];
     - [lift    : et -> acc]            -- a single element as an accumulator;
     - [combine : acc -> acc -> acc]    -- the (GPU-computed) binary combine;
     - a correctness relation [r : seq et -> acc -> prop] read as
       "[x] is a correct reduction of the sequence [s]".

   The only facts the kernel needs from a caller are that [r] holds for
   singletons and is *preserved by concatenation*:

     r_single :  r [e] (lift e)
     r_concat :  r s1 x1 /\ r s2 x2  ==>  r (s1 @+ s2) (combine x1 x2)

   Given these, *any* reduction that combines contiguous adjacent sub-ranges
   (a left fold, or a data-parallel tree) yields a result satisfying [r] for
   the whole input - this module proves it for the reference left fold, and a
   contiguous tree reduction follows by applying [r_concat] at each node (no
   commutativity required, since nodes merge adjacent ranges).

   Callers instantiate this to whatever they need, e.g.:
     - approximate sum: acc = et, lift = pre_map, combine = add,
       r s x = (x %~ rsum (... s));            (HReduce is this instance)
     - argmax (cuBLAS amax): acc = et & nat & nat  (value, index, length),
       combine = "max value / min index on ties, lengths add",
       r s (v,i,l) = (l = length s /\ i < l /\ v = s[i] /\ v maximal /\ i first).
       (see Klas.Argmax for the monoid.) *)

let nonempty_seq (et:Type0) = s:Seq.seq et { 0 < Seq.length s }

(* Reference reduction: left fold of [lift]/[combine] over [s[0..k)]. *)
let rec fold
  (#et #acc : Type0)
  (lift : et -> acc) (combine : acc -> acc -> acc)
  (s : Seq.seq et) (k : nat { 0 < k /\ k <= Seq.length s })
  : Tot acc (decreases k)
  = if k = 1 then lift (Seq.index s 0)
    else combine (fold lift combine s (k - 1)) (lift (Seq.index s (k - 1)))

let reduce
  (#et #acc : Type0)
  (lift : et -> acc) (combine : acc -> acc -> acc)
  (s : nonempty_seq et)
  : acc
  = fold lift combine s (Seq.length s)

(* The two contract obligations, as squash-returning functions (the codebase
   idiom for passing proofs as arguments). *)
let singleton_ok
  (#et #acc : Type0) (lift : et -> acc) (r : Seq.seq et -> acc -> prop)
  = e:et -> squash (r (Seq.create 1 e) (lift e))

let concat_ok
  (#et #acc : Type0) (combine : acc -> acc -> acc) (r : Seq.seq et -> acc -> prop)
  = s1:Seq.seq et -> s2:Seq.seq et -> x1:acc -> x2:acc ->
    squash (r s1 x1 /\ r s2 x2 ==> r (Seq.append s1 s2) (combine x1 x2))

(* Soundness of the contract: from the two obligations, the reference reduction
   of any non-empty prefix satisfies [r]. *)
let rec fold_sound
  (#et #acc : Type0)
  (lift : et -> acc) (combine : acc -> acc -> acc)
  (r : Seq.seq et -> acc -> prop)
  (r_single : singleton_ok lift r)
  (r_concat : concat_ok combine r)
  (s : Seq.seq et) (k : nat { 0 < k /\ k <= Seq.length s })
  : Lemma (ensures r (Seq.slice s 0 k) (fold lift combine s k))
          (decreases k)
  = if k = 1
    then
      (r_single (Seq.index s 0);
       assert (Seq.equal (Seq.slice s 0 1) (Seq.create 1 (Seq.index s 0))))
    else
      (fold_sound lift combine r r_single r_concat s (k - 1);
       r_single (Seq.index s (k - 1));
       r_concat (Seq.slice s 0 (k - 1)) (Seq.create 1 (Seq.index s (k - 1)))
                (fold lift combine s (k - 1)) (lift (Seq.index s (k - 1)));
       assert (Seq.equal
                 (Seq.append (Seq.slice s 0 (k - 1)) (Seq.create 1 (Seq.index s (k - 1))))
                 (Seq.slice s 0 k)))

(* Corollary: [reduce] of the whole sequence satisfies [r]. *)
let reduce_sound
  (#et #acc : Type0)
  (lift : et -> acc) (combine : acc -> acc -> acc)
  (r : Seq.seq et -> acc -> prop)
  (r_single : singleton_ok lift r)
  (r_concat : concat_ok combine r)
  (s : nonempty_seq et)
  : Lemma (ensures r s (reduce lift combine s))
  = fold_sound lift combine r r_single r_concat s (Seq.length s);
    assert (Seq.equal (Seq.slice s 0 (Seq.length s)) s)
