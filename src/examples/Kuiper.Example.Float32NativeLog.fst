module Kuiper.Example.Float32NativeLog

#lang-pulse

open Kuiper
module Fast = Kuiper.Float32.FastMath
module SZ = Kuiper.SizeT

inline_for_extraction noextract
fn logarithm_contract (x : f32) (xr : real{xr >. 0.0R})
  preserves gpu
  requires pure (x %~ (xr <: real))
  returns result : f32
  ensures pure (result %~ Kuiper.Real.log xr)
{
  Fast.log x
}

[@@expect_failure [228]]
fn cannot_log_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  Fast.log x
}

inline_for_extraction noextract
fn kernel
  (n : sz)
  (inputs outputs : larray f32 (SZ.v n))
  (#xs : erased (lseq f32 (SZ.v n)))
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
    let x = inputs.(j);
    let value = Fast.log x;
    pts_to_len outputs;
    outputs.(j) <- value;
    i := !i +^ 1sz;
  }
}

(* The generic instance targets visibility_of arr; launch_kernel_1 needs gpu_of. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global_array arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn run
  (n : sz)
  (inputs : larray f32 (SZ.v n){is_global_array inputs})
  (outputs : larray f32 (SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}
