module Kuiper.Example.Float32Trigonometry

#lang-pulse

open Kuiper
module Fast = Kuiper.Float32.FastMath
module Trig = Kuiper.Real.Trigonometry
module SZ = Kuiper.SizeT

let base_cases ()
  : Lemma (Trig.sin 0.0R == 0.0R /\ Trig.cos 0.0R == 1.0R) =
  Trig.sin_zero ();
  Trig.cos_zero ()

let nonconstant ()
  : Lemma (Trig.sin 1.0R >. 0.0R /\ Trig.cos 1.0R <. 1.0R) =
  Trig.sin_remainder 1.0R 1;
  Trig.cos_remainder 1.0R 2

inline_for_extraction noextract
fn sine_contract (x : f32) (xr : real)
  preserves gpu
  requires pure (x %~ xr)
  returns result : f32
  ensures pure (result %~ Trig.sin xr)
{
  Fast.sin x
}

inline_for_extraction noextract
fn cosine_contract (x : f32) (xr : real)
  preserves gpu
  requires pure (x %~ xr)
  returns result : f32
  ensures pure (result %~ Trig.cos xr)
{
  Fast.cos x
}

[@@expect_failure [228]]
fn cannot_sin_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  Fast.sin x
}

[@@expect_failure [228]]
fn cannot_cos_on_cpu (x : f32)
  preserves cpu
  returns f32
{
  Fast.cos x
}

inline_for_extraction noextract
fn kernel
  (n : sz{SZ.v n <= 2147483647})
  (inputs : larray f32 (SZ.v n))
  (outputs : larray f32 (2 * SZ.v n))
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
    let s = Fast.sin x;
    let c = Fast.cos x;
    pts_to_len outputs;
    outputs.(2sz *^ j) <- s;
    outputs.(2sz *^ j +^ 1sz) <- c;
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
  (n : sz{SZ.v n <= 2147483647})
  (inputs : larray f32 (SZ.v n){is_global_array inputs})
  (outputs : larray f32 (2 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}
