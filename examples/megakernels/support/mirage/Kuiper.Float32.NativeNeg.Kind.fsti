module Kuiper.Float32.NativeNeg.Kind

module F = Kuiper.Floating.Base
module F32 = Kuiper.Float32
module NativeNeg = Kuiper.Float32.NativeNeg

(* Explicit trusted classification of the same native -x operation.
   Finite, Infinite and NaN are classes, not signs or payloads. *)
val neg_kind (x : F32.t)
  : Lemma (F.kind (NativeNeg.neg x) == F.kind x)
