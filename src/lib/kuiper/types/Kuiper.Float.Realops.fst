module Kuiper.Float.Realops

(* Shared mathematical models for every floating format. Domains belong here,
   rather than in the primitive catalogue. These specifications are ghost. *)
open Kuiper.Real
module Exp = FStar.Math.Exp
module Pow = FStar.Math.Pow
module Sqrt = FStar.Math.Sqrt
module Trig = Kuiper.Real.Trigonometry

unfold let r_zero : real = 0.0R
unfold let r_one : real = 1.0R
unfold let r_add (x y:real) : real = x +. y
unfold let r_mul (x y:real) : real = x *. y
unfold let r_sub (x y:real) : real = x -. y
unfold let r_div (x:real) (y:real{y =!= 0.0R}) : real = x /. y
unfold let r_rcp (x:real{x =!= 0.0R}) : real = 1.0R /. x
unfold let r_of_int (x:FStar.Int64.t) : real = of_int (FStar.Int64.v x)
unfold let r_exp (x:real) : real = Exp.exp x
unfold let r_log (x:real{x >. 0.0R}) : real = Exp.log x
unfold let r_expm1 (x:real) : real = Exp.exp x -. 1.0R
unfold let r_log1p (x:real{x >. 0.0R -. 1.0R}) : real = Exp.log (1.0R +. x)
unfold let r_sqrt (x:Sqrt.rnonneg) : real = Sqrt.sqrt x
unfold let r_rsqrt (x:Sqrt.rpos) : real = Sqrt.rsqrt x
unfold let r_pow (x:Pow.rpos) (y:real) : real = Pow.pow x y
unfold let r_exp2 (x:real) : real = Pow.exp2 x
unfold let r_fma (x y z:real) : real = x *. y +. z
unfold let r_sin (x:real) : real = Trig.sin x
unfold let r_cos (x:real) : real = Trig.cos x
unfold let r_tan (x:real{Trig.cos x =!= 0.0R}) : real = Trig.sin x /. Trig.cos x

unfold let r_fmax (x y:real) : real = rmax x y
unfold let r_fmin (x y:real) : real = if t2b (x <. y) then x else y
unfold let r_fabs (x:real) : real = if t2b (x >=. 0.0R) then x else 0.0R -. x
unfold let r_sinh (x:real) : real = (Exp.exp x -. Exp.exp (0.0R -. x)) /. 2.0R
unfold let r_cosh (x:real) : real = (Exp.exp x +. Exp.exp (0.0R -. x)) /. 2.0R
unfold let r_tanh (x:real) : real =
  Exp.exp_positive x;
  Exp.exp_positive (0.0R -. x);
  r_sinh x /. r_cosh x

let log_base_nonzero (b:real{b >. 1.0R}) : Lemma (Exp.log b =!= 0.0R) =
  Exp.exp_base ();
  Exp.exp_log b

unfold let r_log2 (x:real{x >. 0.0R}) : real =
  log_base_nonzero 2.0R;
  Exp.log x /. Exp.log 2.0R
unfold let r_log10 (x:real{x >. 0.0R}) : real =
  log_base_nonzero 10.0R;
  Exp.log x /. Exp.log 10.0R

(* The remaining mathematical models are intentionally opaque until their
   definitions are supplied. All floating types refer to these same names. *)
assume val r_of_literal : string -> real
assume val r_asin : x:real{0.0R -. 1.0R <=. x /\ x <=. 1.0R} -> real
assume val r_acos : x:real{0.0R -. 1.0R <=. x /\ x <=. 1.0R} -> real
assume val r_atan : real -> real
assume val r_ceil : real -> real
assume val r_floor : real -> real
assume val r_round : real -> real
assume val r_erf : real -> real
assume val r_atan2 : real -> real -> real
assume val r_fmod : real -> y:real{y =!= 0.0R} -> real
assume val r_copysign : real -> real -> real
