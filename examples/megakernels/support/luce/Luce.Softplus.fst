module Luce.Softplus

open Kuiper
module A = Kuiper.Approximates.Base
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module R = Kuiper.Real

inline_for_extraction noextract
let softplus20 (x : f32) : f32 =
  if lt (F.of_int 20L) x then x else F32.flog1p (F.fexp x)

let model (r : real) : GTot real =
  if t2b (r >. 20.0R) then r
  else (R.exp_positive r; R.log (1.0R +. R.exp r))

(* LUCE-local numerical assumption, not a derived theorem about CUDA or the
   individual arms. It provides no branch agreement or error bound. Keep this
   module explicit in the proof's trusted dependencies, not witness sources. *)
assume val softplus20_approx
  (x : f32)
  (r : real)
  : Lemma
      (requires A.v_approximates x r)
      (ensures A.v_approximates (softplus20 x) (model r))
