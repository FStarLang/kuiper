module Kuiper.Complex32

(* Single-precision complex as a [scalar] instance, so the polymorphic
   Kuiper/Klas kernels instantiate to complex and extract to CUDA's
   cuFloatComplex arithmetic. The arithmetic comes from Kuiper.Complex32.Base. *)

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base
open Kuiper.Complex32.Base

(* Public name for the type. *)
unfold let cf32 = Complex32.Base.t

(* A cuFloatComplex is two floats, so 8 bytes. *)
inline_for_extraction noextract
instance sized_cf32 : sized Complex32.Base.t = { size = 8sz; default = czero }

inline_for_extraction noextract
instance scalar_cf32 : scalar Complex32.Base.t = {
  is_sized = solve;
  add = cadd;
  mul = cmul;
  zero = czero;
  one = cone;
  lt = clt;
  lte = clte;
  eq = ceq;
}
