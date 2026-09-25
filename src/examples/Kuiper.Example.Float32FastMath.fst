module Kuiper.Example.Float32FastMath

#lang-pulse

open Kuiper
module Fast = Kuiper.Float32.FastMath
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn exp_contract (x : f32) (xr : erased real)
  preserves gpu
  requires pure (x %~ xr)
  returns result : f32
  ensures pure (result %~ Kuiper.Real.exp (reveal xr))
{
  Fast.exp x
}

inline_for_extraction noextract
fn divide_contract (x y : f32) (xr : erased real)
  (yr : erased real{reveal yr =!= 0.0R})
  preserves gpu
  requires pure (x %~ xr /\ y %~ (yr <: erased real))
  returns result : f32
  ensures pure (result %~ (reveal xr /. reveal yr <: real))
{
  Fast.divide x y
}

inline_for_extraction noextract
fn fma_contract (x y z : f32) (xr yr zr : erased real)
  preserves gpu
  requires pure (x %~ xr /\ y %~ yr /\ z %~ zr)
  returns result : f32
  ensures pure (result %~ (reveal xr *. reveal yr +. reveal zr <: real))
{
  Fast.fma_rn x y z
}

inline_for_extraction noextract
fn sub_contract (x y : f32) (xr yr : erased real)
  preserves gpu
  requires pure (x %~ xr /\ y %~ yr)
  returns result : f32
  ensures pure (result %~ (reveal xr -. reveal yr <: real))
{
  Fast.sub_rn x y
}

inline_for_extraction noextract
fn normalized_exp_contract
  (x maximum denominator : f32)
  (xr mr : erased real)
  (dr : erased real{reveal dr =!= 0.0R})
  preserves gpu
  requires pure (x %~ xr /\ maximum %~ mr /\ denominator %~ (dr <: erased real))
  returns result : f32
  ensures pure (result %~
    (Kuiper.Real.exp (reveal xr -. reveal mr) /. reveal dr <: real))
{
  let difference = Fast.sub_rn x maximum;
  let numerator = Fast.exp difference;
  Fast.divide numerator denominator
}

[@@expect_failure [228]]
fn cannot_exp_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  Fast.exp x
}

[@@expect_failure [228]]
fn cannot_divide_on_cpu (x y : f32)
  preserves cpu
  returns f32
{
  Fast.divide x y
}

[@@expect_failure [228]]
fn cannot_fma_on_cpu (x y z : f32)
  preserves cpu
  returns f32
{
  Fast.fma_rn x y z
}

[@@expect_failure [228]]
fn cannot_sub_on_cpu (x y : f32)
  preserves cpu
  returns f32
{
  Fast.sub_rn x y
}

inline_for_extraction noextract
fn kernel
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (3 * SZ.v n))
  (outputs : larray f32 (4 * SZ.v n))
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
    let e = Fast.exp x;
    let d = Fast.divide x y;
    let f = Fast.fma_rn x y z;
    let s = Fast.sub_rn x y;
    pts_to_len outputs;
    outputs.(4sz *^ j) <- e;
    outputs.(4sz *^ j +^ 1sz) <- d;
    outputs.(4sz *^ j +^ 2sz) <- f;
    outputs.(4sz *^ j +^ 3sz) <- s;
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
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (3 * SZ.v n){is_global_array inputs})
  (outputs : larray f32 (4 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (3 * SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}
