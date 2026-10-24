module Kuiper.Example.Float32NativeNeg

#lang-pulse

open Kuiper

module F = Kuiper.Float32
module SZ = Kuiper.SizeT

let approximation (x:f32) (r:real)
  : Lemma (requires x %~ r)
          (ensures F.native_neg x %~ (0.0R -. r <: real)) = ()

let classification (x:f32)
  : Lemma (kind (F.native_neg x) == kind x) = ()

[@@expect_failure [19]]
let cannot_replace_with_zero_subtraction (x:f32)
  : Lemma (bit_eq (F.native_neg x) (sub zero x)) = ()

[@@expect_failure [19]]
let cannot_infer_value_preservation (x:f32) (r:real)
  : Lemma (requires x %~ r)
          (ensures F.native_neg x == x) = ()

fn host_negate (x:f32)
  preserves cpu
  returns f32
{
  F.native_neg x
}

inline_for_extraction noextract
fn kernel
  (n:sz)
  (inputs outputs:larray f32 (SZ.v n))
  (#xs:erased (lseq f32 (SZ.v n)))
  (#before:erased (seq f32))
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
    outputs.(j) <- F.native_neg x;
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

fn run
  (n:sz)
  (inputs:larray f32 (SZ.v n){is_global_array inputs})
  (outputs:larray f32 (SZ.v n){is_global_array outputs})
  (#xs:erased (lseq f32 (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}
