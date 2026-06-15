module Kuiper.Complex64

(* Double-precision complex as a [scalar] instance, so the polymorphic
   Kuiper/Klas kernels instantiate to complex and extract to CUDA's
   cuDoubleComplex arithmetic. The arithmetic comes from Kuiper.Complex64.Base. *)

open FStar.Tactics.Typeclasses { solve }
open Kuiper.Sized
open Kuiper.Scalars.Base
open Kuiper.Complex64.Base

(* Public name for the type. *)
unfold let cf64 = Complex64.Base.t

(* A cuDoubleComplex is two doubles, so 16 bytes. *)
inline_for_extraction noextract
instance sized_cf64 : sized Complex64.Base.t = { size = 16sz; default = czero }

inline_for_extraction noextract
instance scalar_cf64 : scalar Complex64.Base.t = {
  is_sized = solve;
  add = cadd;
  mul = cmul;
  zero = czero;
  one = cone;
  lt = clt;
  lte = clte;
  eq = ceq;
}
