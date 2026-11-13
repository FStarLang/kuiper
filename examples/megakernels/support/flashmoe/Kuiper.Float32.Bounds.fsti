module Kuiper.Float32.Bounds

module F = Kuiper.Floating
module F32 = Kuiper.Float32
module B = Kuiper.Floating.Bounds

(* Trusted native binary32 classification, separate from approximation.
   Integer envelopes are exactly representable; no rounding-error equality,
   reverse approximation, saturation, or FTZ substitution is asserted. *)
val integer_bounds (n:B.bound)
  : Lemma (B.nonnegative n (B.integer #F32.t n))

val integer_order (m n:B.bound)
  : Lemma (
      (F.lte (B.integer #F32.t m) (B.integer n) <==> m <= n) /\
      (F.lt (B.integer #F32.t m) (B.integer n) <==> m < n))

val add_bounds (a:B.bound) (b:B.bound{a + b <= B.limit}) (x y:F32.t)
  : Lemma
      (requires B.magnitude a x /\ B.magnitude b y)
      (ensures B.magnitude (a + b) (F.add x y) /\
        (F.lte F.zero x /\ F.lte F.zero y ==>
          F.lte F.zero (F.add x y) /\
          F.lte x (F.add x y) /\ F.lte y (F.add x y)))

val sub_bounds (x y:F32.t)
  : Lemma (B.sub_post x y (F.sub x y))

val mul_unit (x y:F32.t)
  : Lemma (requires B.unit x /\ B.unit y)
          (ensures B.unit (F.mul x y))

val div_unit (x y:F32.t)
  : Lemma (B.divide_post x y (F.div x y))

(* expf, not __expf. Range is a native guarantee, not an implication of
   the real exponential model or of a generic ULP error bound. *)
val exp_nonpositive (x:F32.t)
  : Lemma (B.exp_post x (F.fexp x))

let zero ()
  : Lemma (B.magnitude 0 (F.zero #F32.t) /\ B.nonnegative 0 (F.zero #F32.t))
  =
  B.integer_zero #F32.t ();
  integer_bounds 0;
  B.nonnegative_magnitude 0 (F.zero #F32.t)

let one ()
  : Lemma (B.magnitude 1 (F.one #F32.t) /\ B.unit (F.one #F32.t))
  =
  B.integer_one #F32.t ();
  integer_bounds 1;
  B.nonnegative_magnitude 1 (F.one #F32.t);
  B.nonnegative_unit (F.one #F32.t)
