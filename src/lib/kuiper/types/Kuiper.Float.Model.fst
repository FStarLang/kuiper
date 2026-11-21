module Kuiper.Float.Model

open Kuiper.Real
module R = Kuiper.Float.Realops

(* Only the representation survives extraction. The public float interfaces
   hide this record so clients cannot assign arbitrary ideals to runtime bits. *)
noeq
type value (raw:Type0) = {
  repr : raw;
  ideal : real;
}

inline_for_extraction noextract
let unary #raw (f:raw -> raw) (g:real -> real) (x:value raw) : value raw =
  { repr = f x.repr; ideal = g x.ideal }

inline_for_extraction noextract
let binary #raw (f:raw -> raw -> raw) (g:real -> real -> real)
  (x y:value raw) : value raw =
  { repr = f x.repr y.repr; ideal = g x.ideal y.ideal }

inline_for_extraction noextract
let ternary #raw (f:raw -> raw -> raw -> raw) (g:real -> real -> real -> real)
  (x y z:value raw) : value raw =
  { repr = f x.repr y.repr z.repr; ideal = g x.ideal y.ideal z.ideal }

(* Outside a real specification's domain its ideal result is unconstrained.
   Choosing zero makes the implementation total without strengthening clients'
   partial-domain contracts. These choices do not affect runtime bits. *)
let div (x y:real) : real =
  if t2b (y =!= 0.0R) then R.r_div x y else 0.0R

let log (x:real) : real =
  if t2b (x >. 0.0R) then R.r_log x else 0.0R

let log1p (x:real) : real =
  if t2b (x >. 0.0R -. 1.0R) then R.r_log1p x else 0.0R

let sqrt (x:real) : real =
  if t2b (x >=. 0.0R) then R.r_sqrt x else 0.0R

let rsqrt (x:real) : real =
  if t2b (x >. 0.0R) then R.r_rsqrt x else 0.0R

let rcp (x:real) : real =
  if t2b (x =!= 0.0R) then R.r_rcp x else 0.0R

let tan (x:real) : real =
  if t2b (Kuiper.Real.Trigonometry.cos x =!= 0.0R) then R.r_tan x else 0.0R

let asin (x:real) : real =
  if t2b (0.0R -. 1.0R <=. x /\ x <=. 1.0R) then R.r_asin x else 0.0R

let acos (x:real) : real =
  if t2b (0.0R -. 1.0R <=. x /\ x <=. 1.0R) then R.r_acos x else 0.0R

let log2 (x:real) : real =
  if t2b (x >. 0.0R) then R.r_log2 x else 0.0R

let log10 (x:real) : real =
  if t2b (x >. 0.0R) then R.r_log10 x else 0.0R

let pow (x y:real) : real =
  if t2b (x >. 0.0R) then R.r_pow x y else 0.0R

let fmod (x y:real) : real =
  if t2b (y =!= 0.0R) then R.r_fmod x y else 0.0R
