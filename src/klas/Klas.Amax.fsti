module Klas.Amax

(* cuBLAS I<t>amax: index of the element of largest absolute value (the first
   such element on ties). cuBLAS is 1-based; we return the 0-based index.

   A one-pass argmax reduction. Correctness rests on absolute value being a
   total preorder, in particular transitivity of [lte] on non-NaN floats
   (Kuiper.Floating.Base.lte_trans), so the input must contain no NaNs.

   The spec [is_amax] below matches (the amax instance of) the generic
   relational reduction in Klas.Reduce.Argmax. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module U64 = FStar.UInt64

(* ----------------------------------------------------------------------- *)
(* Pure specification                                                        *)
(* ----------------------------------------------------------------------- *)

let all_not_nan (#et:Type0) {| floating et |} (s : Seq.seq et) : prop =
  forall (j:nat). j < Seq.length s ==> ~(NaN? (kind (Seq.index s j)))

(* Index of the max-abs element among the first [k]. Switch only on a strict
   increase, so the earliest maximizer wins ties. *)
let rec amax_pre (#et:Type0) {| floating et |}
  (s : Seq.seq et) (k:nat{1 <= k /\ k <= Seq.length s})
  : Tot (i:nat{i < k}) (decreases k) =
  if k = 1 then 0
  else
    let p = amax_pre s (k-1) in
    if lt (abs (Seq.index s p)) (abs (Seq.index s (k-1))) then k-1 else p

(* "[i] is a correct argmax of the first [k] elements of [s]": [i] attains the
   max absolute value and is the earliest index that does so. *)
let is_amax_pre (#et:Type0) {| floating et |}
  (s : Seq.seq et) (k:nat{1 <= k /\ k <= Seq.length s}) (i:nat) : prop =
  i < k /\
  (forall (j:nat). j < k ==> lte (abs (Seq.index s j)) (abs (Seq.index s i))) /\
  (forall (j:nat). j < k ==> (lte (abs (Seq.index s i)) (abs (Seq.index s j)) ==> i <= j))

let is_amax (#et:Type0) {| floating et |}
  (s : Seq.seq et{Seq.length s >= 1}) (i:nat) : prop =
  is_amax_pre s (Seq.length s) i

(* ----------------------------------------------------------------------- *)
(* Device entry points                                                       *)
(* ----------------------------------------------------------------------- *)

inline_for_extraction noextract
type amax_ty (et:Type0) {| floating et |} =
  fn (lena : szp)
     (a : array1 et (l1_forward lena) { is_global a })
     (#va : erased (lseq et lena))
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (all_not_nan va)
  returns
    res : U64.t
  ensures
    pure (is_amax va (U64.v res))

val amax_f32 : amax_ty f32
val amax_f64 : amax_ty f64
