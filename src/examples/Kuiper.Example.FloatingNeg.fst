module Kuiper.Example.FloatingNeg

#lang-pulse

open Kuiper

module SZ = Kuiper.SizeT

let approximation #t {| floating t, real_like t, floating_real_like t |} (x:t) (r:real)
  : Lemma (requires x %~ r)
          (ensures neg x %~ (0.0R -. r <: real)) = ()

let classification #t {| floating t |} (x:t)
  : Lemma (kind (neg x) == kind x) = neg_kind x

let double_negation #t {| floating t |} (x:t{not_nan x})
  : Lemma (ieee_eq (neg (neg x)) x) = neg_neg x

[@@expect_failure [19]]
let cannot_replace_with_zero_subtraction #t {| floating t |} (x:t)
  : Lemma (bit_eq (neg x) (sub zero x)) = ()

[@@expect_failure [19]]
let cannot_infer_value_preservation #t
  {| floating t, real_like t, floating_real_like t |} (x:t) (r:real)
  : Lemma (requires x %~ r)
          (ensures neg x == x) = ()

let host_f16 (x:f16) : f16 = neg x
let host_bf16 (x:bf16) : bf16 = neg x
let host_f32 (x:f32) : f32 = neg x
let host_f64 (x:f64) : f64 = neg x

inline_for_extraction noextract
fn kernel
  (#t:Type0) {| floating t |}
  (n:sz)
  (inputs outputs:larray t (SZ.v n))
  (#xs:erased (lseq t (SZ.v n)))
  (#before:erased (seq t))
  preserves gpu ** inputs |-> xs
  requires outputs |-> before
  ensures exists* after. outputs |-> after
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}). i |-> j
    invariant exists* values. outputs |-> values
    decreases (n - !i)
  {
    let j = !i;
    let x = inputs.(j);
    pts_to_len outputs;
    outputs.(j) <- neg x;
    i := !i +^ 1sz;
  }
}

instance send_global_array_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
fn run
  (#t:Type0) {| floating t |}
  (n:sz)
  (inputs:larray t (SZ.v n){is_global_array inputs})
  (outputs:larray t (SZ.v n){is_global_array outputs})
  (#xs:erased (lseq t (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}

let run_f16 = run #f16
let run_bf16 = run #bf16
let run_f32 = run #f32
let run_f64 = run #f64
