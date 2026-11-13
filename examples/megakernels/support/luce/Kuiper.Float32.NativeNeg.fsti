module Kuiper.Float32.NativeNeg

open Kuiper.Approximates.Base
open Kuiper.Real

module F32 = Kuiper.Float32

(* Trusted native CUDA unary minus, not zero subtraction. Its external C
   definition is Kuiper_Float32_NativeNeg_neg in the accompanying header.
   No NaN payload/sign or exact decoded-real representation is specified. *)
val neg : F32.t -> F32.t

(* An explicit native API contract in the existing approximation model;
   not a derived implementation theorem. *)
val neg_approx (x : F32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (neg x) (0.0R -. r))
