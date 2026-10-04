module Kuiper.Example.Float32Math

#lang-pulse

open Kuiper
module F32 = Kuiper.Float32
module Casts = Kuiper.Float.Casts
module SZ = Kuiper.SizeT
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

let ordinary_contracts
  (x y : f32) (xr : Pow.rpos) (yr : real)
  : Lemma
      (requires v_approximates x xr /\ v_approximates y yr)
      (ensures
        v_approximates (pow x y) (Pow.pow xr yr) /\
        v_approximates (sin y) (Trig.sin yr) /\
        v_approximates (cos y) (Trig.cos yr)) =
  F32.pow_approx x y xr yr;
  F32.sin_approx y yr;
  F32.cos_approx y yr

inline_for_extraction noextract
let rotary_angle (base coordinate position : f32) : f32 =
  mul position (div one (pow base (div coordinate (of_int #f32 64L))))

let rotary_contract
  (base coordinate position : f32)
  (br : Pow.rpos) (cr pr : real)
  : Lemma
      (requires
        v_approximates base br /\ v_approximates coordinate cr /\
        v_approximates position pr)
      (ensures (
        let angle = rotary_angle base coordinate position in
        let real_angle = pr *. (1.0R /. Pow.pow br (cr /. 64.0R)) in
        v_approximates (Casts.fcast #f32 #bf16 (sin angle)) (Trig.sin real_angle) /\
        v_approximates (Casts.fcast #f32 #bf16 (cos angle)) (Trig.cos real_angle))) =
  let width = of_int #f32 64L in
  let exponent = div coordinate width in
  let power = pow base exponent in
  let frequency = div one power in
  let angle = mul position frequency in
  let real_power = Pow.pow br (cr /. 64.0R) in
  let real_angle = pr *. (1.0R /. real_power) in
  of_int_approx #f32 64L;
  div_approx coordinate width cr 64.0R;
  F32.pow_approx base exponent br (cr /. 64.0R);
  div_approx one power 1.0R real_power;
  a_mul position frequency pr (1.0R /. real_power);
  F32.sin_approx angle real_angle;
  F32.cos_approx angle real_angle;
  Casts.fcast_approx #f32 #bf16 (sin angle) (Trig.sin real_angle);
  Casts.fcast_approx #f32 #bf16 (cos angle) (Trig.cos real_angle)

[@@expect_failure [19]]
let pow_requires_positive_base
  (x y : f32) (xr yr : real)
  : Lemma (requires v_approximates x xr /\ v_approximates y yr) (ensures True) =
  F32.pow_approx x y xr yr

[@@expect_failure [19]]
let pow_requires_base_relationship
  (x y : f32) (xr : Pow.rpos) (yr : real)
  : Lemma
      (requires v_approximates y yr)
      (ensures v_approximates (pow x y) (Pow.pow xr yr)) =
  F32.pow_approx x y xr yr

[@@expect_failure [19]]
let pow_requires_exponent_relationship
  (x y : f32) (xr : Pow.rpos) (yr : real)
  : Lemma
      (requires v_approximates x xr)
      (ensures v_approximates (pow x y) (Pow.pow xr yr)) =
  F32.pow_approx x y xr yr

[@@expect_failure [19]]
let sin_requires_input_relationship (x : f32) (xr : real)
  : Lemma (v_approximates (sin x) (Trig.sin xr)) =
  F32.sin_approx x xr

[@@expect_failure [19]]
let cos_requires_input_relationship (x : f32) (xr : real)
  : Lemma (v_approximates (cos x) (Trig.cos xr)) =
  F32.cos_approx x xr

inline_for_extraction noextract
fn kernel
  (n : sz{SZ.v n <= 536870911})
  (inputs : larray f32 (3 * SZ.v n))
  (outputs : larray f32 (8 * SZ.v n))
  (#xs : erased (lseq f32 (3 * SZ.v n)))
  (#before : erased (seq f32))
  preserves gpu ** inputs |-> xs
  requires outputs |-> before
  ensures exists* after. outputs |-> after
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j : sz{j <= n}). i |-> j
    invariant exists* values. outputs |-> values
    decreases (n - !i)
  {
    let j = !i;
    let base = inputs.(3sz *^ j);
    let coordinate = inputs.(3sz *^ j +^ 1sz);
    let position = inputs.(3sz *^ j +^ 2sz);
    let power = pow base (div coordinate (of_int #f32 64L));
    let angle = rotary_angle base coordinate position;
    let sn = sin angle;
    let cs = cos angle;
    let bsn = Casts.fcast #f32 #bf16 sn;
    let bcs = Casts.fcast #f32 #bf16 cs;
    pts_to_len outputs;
    outputs.(8sz *^ j) <- power;
    outputs.(8sz *^ j +^ 1sz) <- sin position;
    outputs.(8sz *^ j +^ 2sz) <- cos position;
    outputs.(8sz *^ j +^ 3sz) <- angle;
    outputs.(8sz *^ j +^ 4sz) <- sn;
    outputs.(8sz *^ j +^ 5sz) <- cs;
    outputs.(8sz *^ j +^ 6sz) <- Casts.fcast #bf16 #f32 bsn;
    outputs.(8sz *^ j +^ 7sz) <- Casts.fcast #bf16 #f32 bcs;
    i := !i +^ 1sz;
  }
}

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global_array arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn run
  (n : sz{SZ.v n <= 536870911})
  (inputs : larray f32 (3 * SZ.v n){is_global_array inputs})
  (outputs : larray f32 (8 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (3 * SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}
