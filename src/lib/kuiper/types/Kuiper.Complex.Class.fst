module Kuiper.Complex.Class

(* A typeclass that recognizes a [scalar] type [c] as a complex-number type whose
   components are of (floating) type [r]. It exposes the constructor [cmk], the
   projections [cre]/[cim], the imaginary unit [cunit], and the conjugate
   [cconj], so complex kernels can be written polymorphically over precision.

   The [fundeps] hint lets instance resolution pick [r] from [c]. *)

open FStar.Tactics.Typeclasses { tcinstance, solve }
open Kuiper.Scalars.Base
open Kuiper.Floating.Base

[@@FStar.Tactics.Typeclasses.fundeps [0]]
inline_for_extraction noextract
class complex (c : Type) (r : Type) = {
  [@@@tcinstance] cx_scalar : scalar c;
  [@@@tcinstance] cx_real   : floating r;

  cmk   : r -> r -> c;   (* make a complex from (re, im) *)
  cre   : c -> r;        (* real part *)
  cim   : c -> r;        (* imaginary part *)
  cunit : c;             (* the imaginary unit i *)
  cconj : c -> c;        (* complex conjugate *)
}

(* Lift a real to a complex (imaginary part 0). *)
inline_for_extraction noextract
let of_real (#c #r : Type) {| complex c r |} (x : r) : c = cmk x zero

(* ----------------------------------------------------------------------- *)
(* Single-precision instance: cf32 over f32.                                 *)
(* ----------------------------------------------------------------------- *)

module C = Kuiper.Complex32.Base
module F = Kuiper.Float32

inline_for_extraction noextract
instance complex_cf32 : complex C.t F.t = {
  cx_scalar = Kuiper.Complex32.scalar_cf32;
  cx_real   = F.is_floating;
  cmk   = C.cmk;
  cre   = C.re;
  cim   = C.im;
  cunit = C.cci;
  cconj = C.cconj;
}

(* ----------------------------------------------------------------------- *)
(* Double-precision instance: cf64 over f64.                                 *)
(* ----------------------------------------------------------------------- *)

module C64 = Kuiper.Complex64.Base
module F64 = Kuiper.Float64

inline_for_extraction noextract
instance complex_cf64 : complex C64.t F64.t = {
  cx_scalar = Kuiper.Complex64.scalar_cf64;
  cx_real   = F64.is_floating;
  cmk   = C64.cmk;
  cre   = C64.re;
  cim   = C64.im;
  cunit = C64.cci;
  cconj = C64.cconj;
}
