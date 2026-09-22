module Kuiper.Example.Float32GPU

#lang-pulse

open Kuiper
module F32 = Kuiper.Float32
module Pow = FStar.Math.Pow

inline_for_extraction noextract
fn approximation_contract
  (xr yr : erased real)
  (x y : f32)
  preserves gpu
  requires pure (x %~ xr /\ y %~ yr)
  returns result : f32
  ensures pure (result %~ (Pow.exp2 (reveal xr *. reveal yr) <: real))
{
  let product = F32.mul_rn_ftz x y;
  F32.exp2_approx_ftz product
}

[@@expect_failure [228]]
fn cannot_multiply_on_cpu (x y : f32)
  preserves cpu
  returns f32
{
  F32.mul_rn_ftz x y
}

[@@expect_failure [228]]
fn cannot_exponentiate_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  F32.exp2_approx_ftz x
}

inline_for_extraction noextract
fn multiply_kernel (out : gpu_ref f32) (#old : erased f32) (x y : f32)
  preserves gpu
  requires out |-> old
  ensures exists* result. out |-> result
{
  let result = F32.mul_rn_ftz x y;
  out := result;
}

inline_for_extraction noextract
fn exponentiate_kernel (out : gpu_ref f32) (#old : erased f32) (x : f32)
  preserves gpu
  requires out |-> old
  ensures exists* result. out |-> result
{
  let result = F32.exp2_approx_ftz x;
  out := result;
}

fn multiply (x y : f32)
  preserves cpu
  returns f32
{
  let mut out : f32 = zero;
  let device = alloc0 #f32 ();
  with old. assert on gpu_loc (device |-> old);
  launch_kernel_1 (fun () -> multiply_kernel device #old x y);
  Kuiper.Ref.memcpy_device_to_host out device;
  let result = !out;
  free device;
  result
}

fn exponentiate (x : f32)
  preserves cpu
  returns f32
{
  let mut out : f32 = zero;
  let device = alloc0 #f32 ();
  with old. assert on gpu_loc (device |-> old);
  launch_kernel_1 (fun () -> exponentiate_kernel device #old x);
  Kuiper.Ref.memcpy_device_to_host out device;
  let result = !out;
  free device;
  result
}
