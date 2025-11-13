module Kuiper.IntApprox

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Kuiper.Approximates.U32
open Kuiper.Seq.Common
module U32 = FStar.UInt32

(* To expose the definitions of to_real and v_approximates. *)
friend Kuiper.Approximates.U32

(* If we have a u32 z approximating x+y, it must be that z is *exactly*
   (x+y)%2^32, and that x+y is an integer. *)

let lem (z : u32) (x y : real)
  : Lemma (requires z %~ (x +. y))
          (ensures  exists (z' : int). x+.y == FStar.Real.of_int z' /\ z' % (pow2 32) == U32.v z)
  = ()

(* And if x and y were obtained from to_real, then we can specialize a bit more. *)
let lem' (z x y : u32)
  : Lemma (requires z %~ (to_real x +. to_real y))
          (ensures  FStar.UInt32.(z == x +%^ y))
  = ()

(* Strenghten the spec of a reduction *)

fn reduce (#et:Type0) {| scalar et, real_like et |}
  (len:nat) (a : larray u32 len) (#s : seq u32)
  (#r : erased (seq real))
  preserves a |-> s ** pure (s %~ r)
  returns   res : u32
  ensures   pure (res %~ seq_fold_left (+.) 0.0R r)
{
  admit(); // Intentional, to not importa HReduce here.
}

let get_int_from_real_approx (x:u32) (r:real)
  : Ghost (erased int)
          (requires x %~ r)
          (ensures  fun i -> r == Real.of_int i /\ UInt32.v x == i % pow2 32)
  = FStar.IndefiniteDescription.indefinite_description_ghost int
      (fun i -> r == Real.of_int i /\ UInt32.v x == i % pow2 32)

#push-options "--z3rlimit 30"
let rec lem_seq_approx (s : seq u32) (i : int)
  : Lemma (requires Real.of_int i == seq_fold_left (+.) 0.0R (seq_map to_real s))
          (ensures  i % (pow2 32) == U32.v (seq_fold_left add zero s))
          (decreases Seq.length s)
  = match view_seq s with
    | SNil -> ()
    | SCons hd tl ->
      let i' = i - U32.v hd in
      lemma_seq_fold_left_sum 0.0R (+.) seq![to_real hd] (seq_map to_real tl);
      assert (s == seq![hd] `Seq.append` tl);
      assert (Seq.equal (seq_map to_real s)
                         (seq![to_real hd] `Seq.append` seq_map to_real tl));
      assert (Seq.equal (seq_map to_real seq![hd])
                         seq![to_real hd]);
      assert (seq_fold_left (+.) 0.0R seq![to_real hd] 
              == to_real hd);
      assert (seq_fold_left (+.) 0.0R (seq_map to_real s)
               == to_real hd +. seq_fold_left (+.) 0.0R (seq_map to_real tl));
      lem_seq_approx tl (i - U32.v hd);
      calc (==) {
        i % pow2 32;
        == {}
        (i' + U32.v hd) % pow2 32;
        == { Math.Lemmas.modulo_distributivity i' (U32.v hd) (pow2 32) }
        (i' % pow2 32 + U32.v hd % pow2 32) % pow2 32;
        == {}
        (U32.v (seq_fold_left add zero tl) % pow2 32 + U32.v hd % pow2 32) % pow2 32;
        == {}
        (U32.v hd % pow2 32 + U32.v (seq_fold_left add zero tl) % pow2 32) % pow2 32;
        == { Math.Lemmas.modulo_distributivity
                 (U32.v hd) (U32.v (seq_fold_left add zero tl)) (pow2 32) }
        (U32.v hd + U32.v (seq_fold_left add zero tl)) % pow2 32;
        == {}
        U32.v (hd `add` seq_fold_left add zero tl);
        == { lemma_seq_fold_left_sum #u32 zero add seq![hd] tl }
        U32.v (seq_fold_left add zero s);
        == { Math.Lemmas.small_mod (U32.v (seq_fold_left add zero s)) (pow2 32) }
        U32.v (seq_fold_left add zero s) % pow2 32;
      };
      ()
#pop-options

(* A stronger exact spec for reduce on u32s, proven from the approximate spec. *)
fn reduce_u32 (len:nat) (a : larray u32 len) (#s : seq u32)
  preserves a |-> s 
  returns   res : u32
  ensures   pure (U32.v res == seq_fold_left add zero s)
{
  let res = reduce #u32 len a #s #(seq_map to_real s);
  ();
  let rr = seq_fold_left (+.) 0.0R (seq_map to_real s);
  assert pure (res %~ rr);
  assert pure (exists (i:int). rr == Real.of_int i /\ i % pow2 32 == U32.v res);
  let i = get_int_from_real_approx res rr;
  lem_seq_approx s i;
  assert pure (res %~ rr);
  res
}
