module Kuiper.Example.Float32GPU

#lang-pulse

open Kuiper
module F32 = Kuiper.Float32
module SZ = Kuiper.SizeT
module Pow = FStar.Math.Pow
module Sqrt = FStar.Math.Sqrt

inline_for_extraction noextract
fn accumulation_contract
  (xr ar br : erased real)
  (x accumulator bias : f32)
  preserves gpu
  requires pure (x %~ xr /\ accumulator %~ ar /\ bias %~ br)
  returns result : f32
  ensures pure (result %~
    (reveal xr *. reveal xr +. reveal ar +. reveal br <: real))
{
  let square_sum = F32.fma_rn_ftz x x accumulator;
  F32.add_rn_ftz square_sum bias
}

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

inline_for_extraction noextract
fn division_contract
  (xr : erased real)
  (yr : erased real{reveal yr =!= 0.0R})
  (x y : f32)
  preserves gpu
  requires pure (x %~ xr /\ y %~ (yr <: erased real))
  returns result : f32
  ensures pure (result %~ (reveal xr *. (1.0R /. reveal yr) <: real))
{
  let reciprocal = F32.rcp_approx_ftz y;
  F32.mul_rn_ftz x reciprocal
}

inline_for_extraction noextract
fn normalization_contract
  (xr : erased real)
  (yr : erased real{reveal yr >. 0.0R})
  (x y : f32)
  preserves gpu
  requires pure (x %~ xr /\ y %~ (yr <: erased real))
  returns result : f32
  ensures pure (result %~ (reveal xr *. (1.0R /. Sqrt.sqrt (reveal yr)) <: real))
{
  let inverse_root = F32.rsqrt_approx_ftz y;
  F32.mul_rn_ftz x inverse_root
}

[@@expect_failure [228]]
fn cannot_add_on_cpu (x y : f32)
  preserves cpu
  returns f32
{
  F32.add_rn_ftz x y
}

[@@expect_failure [228]]
fn cannot_fma_on_cpu (x y z : f32)
  preserves cpu
  returns f32
{
  F32.fma_rn_ftz x y z
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

[@@expect_failure [228]]
fn cannot_reciprocate_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  F32.rcp_approx_ftz x
}

[@@expect_failure [228]]
fn cannot_inverse_root_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  F32.rsqrt_approx_ftz x
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

inline_for_extraction noextract
fn reciprocal_kernel (out : gpu_ref f32) (#old : erased f32) (x : f32)
  preserves gpu
  requires out |-> old
  ensures exists* result. out |-> result
{
  let result = F32.rcp_approx_ftz x;
  out := result;
}

inline_for_extraction noextract
fn inverse_root_kernel (out : gpu_ref f32) (#old : erased f32) (x : f32)
  preserves gpu
  requires out |-> old
  ensures exists* result. out |-> result
{
  let result = F32.rsqrt_approx_ftz x;
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

fn reciprocal (x : f32)
  preserves cpu
  returns f32
{
  let mut out : f32 = zero;
  let device = alloc0 #f32 ();
  with old. assert on gpu_loc (device |-> old);
  launch_kernel_1 (fun () -> reciprocal_kernel device #old x);
  Kuiper.Ref.memcpy_device_to_host out device;
  let result = !out;
  free device;
  result
}

fn inverse_root (x : f32)
  preserves cpu
  returns f32
{
  let mut out : f32 = zero;
  let device = alloc0 #f32 ();
  with old. assert on gpu_loc (device |-> old);
  launch_kernel_1 (fun () -> inverse_root_kernel device #old x);
  Kuiper.Ref.memcpy_device_to_host out device;
  let result = !out;
  free device;
  result
}

inline_for_extraction noextract
fn arithmetic_kernel
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (3 * SZ.v n))
  (outputs : larray f32 (2 * SZ.v n))
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
    let x = inputs.(3sz *^ j);
    let y = inputs.(3sz *^ j +^ 1sz);
    let z = inputs.(3sz *^ j +^ 2sz);
    let sum = F32.add_rn_ftz x y;
    let fused = F32.fma_rn_ftz x y z;
    pts_to_len outputs;
    outputs.(2sz *^ j) <- sum;
    outputs.(2sz *^ j +^ 1sz) <- fused;
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

fn arithmetic
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (3 * SZ.v n){is_global_array inputs})
  (outputs : larray f32 (2 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (3 * SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> arithmetic_kernel n inputs outputs #xs #before);
}
