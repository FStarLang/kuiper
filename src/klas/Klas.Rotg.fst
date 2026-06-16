module Klas.Rotg

(* See Klas.Rotg.fsti for the specification and the big note on the sign /
   negative-zero caveat. *)

#lang-pulse
open Kuiper
module AB = Kuiper.Approximates.Base

(* The radius is non-negative and (under the precondition) non-zero, so its
   square equals a*a + b*b. That single algebraic fact turns the Givens
   rotation property into real field arithmetic that Z3 can close. *)
let s_rotg_rotates ra rb =
  let sm = ra *. ra +. rb *. rb in
  assert (ra *. ra >=. 0.0R);
  assert (rb *. rb >=. 0.0R);
  realsqrt_nonneg_sq sm

inline_for_extraction noextract
let rotg_gen (#et:Type0) {| floating et |} {| real_like et |} {| floating_real_like et |}
  (a b : et)
  : Pure (rotg_out et)
      (requires s_rotg_r (to_real a) (to_real b) =!= 0.0R)
      (ensures fun o ->
         o.rr %~ s_rotg_r (to_real a) (to_real b) /\
         o.rc %~ s_rotg_c (to_real a) (to_real b) /\
         o.rs %~ s_rotg_s (to_real a) (to_real b))
  = let ra = to_real a in
    let rb = to_real b in
    let aa = mul a a in
    let bb = mul b b in
    let sumsq = add aa bb in
    let r = sqrt sumsq in
    AB.to_real_ok a;
    AB.to_real_ok b;
    AB.a_mul a a ra ra;                       (* aa    %~ ra*.ra            *)
    AB.a_mul b b rb rb;                        (* bb    %~ rb*.rb            *)
    AB.a_add aa bb (ra *. ra) (rb *. rb);      (* sumsq %~ ra*.ra +. rb*.rb *)
    AB.sqrt_approx sumsq (ra *. ra +. rb *. rb); (* r   %~ s_rotg_r ra rb   *)
    AB.div_approx a r ra (s_rotg_r ra rb);     (* div a r %~ s_rotg_c ra rb *)
    AB.div_approx b r rb (s_rotg_r ra rb);     (* div b r %~ s_rotg_s ra rb *)
    { rc = div a r; rs = div b r; rr = r }

(* No rotg_f16: rotg is a host-side scalar fn and half arithmetic is
   __device__-only (and cuBLAS has no half rotg). See Klas.Rotg.fsti. *)
let rotg_f32 = rotg_gen #f32
let rotg_f64 = rotg_gen #f64
