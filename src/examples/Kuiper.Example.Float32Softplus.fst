module Kuiper.Example.Float32Softplus

#lang-pulse

open Kuiper
module A = Kuiper.Approximates.Base
module C = Kuiper.Float.Casts
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module R = Kuiper.Real
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let softplus (x : f32) : f32 =
  if lt (F.of_int 20L) x then x else F32.flog1p (F.fexp x)

let model (r : real) : GTot real =
  if t2b (r >. 20.0R) then r
  else (R.exp_positive r; R.log (1.0R +. R.exp r))

inline_for_extraction noextract
let argument (x y : bf16) : f32 =
  add (C.fcast x <: f32) (C.fcast y <: f32)

let softplus_contract (x : f32) (r : real)
  : Lemma
      (requires A.v_approximates x r)
      (ensures A.v_approximates (softplus x) (model r))
  = F32.softplus20_approx x r

let cast_add_contract (x y : bf16) (xr yr : real)
  : Lemma
      (requires A.v_approximates x xr /\ A.v_approximates y yr)
      (ensures A.v_approximates (softplus (argument x y)) (model (xr +. yr)))
  = C.fcast_approx #bf16 #f32 x xr;
    C.fcast_approx #bf16 #f32 y yr;
    A.a_add (C.fcast x <: f32) (C.fcast y <: f32) xr yr;
    F32.softplus20_approx (argument x y) (xr +. yr)

let scaled_exp_contract (x scale : f32) (xr sr : real)
  : Lemma
      (requires A.v_approximates x xr /\ A.v_approximates scale sr)
      (ensures A.v_approximates (F.fexp (mul scale (softplus x)))
        (R.exp (sr *. model xr)))
  = F32.softplus20_approx x xr;
    A.a_mul scale (softplus x) sr (model xr);
    A.exp_approx (mul scale (softplus x)) (sr *. model xr)

[@@expect_failure [19]]
let cannot_approximate_without_input (x : f32) (r : real)
  : Lemma (A.v_approximates (softplus x) (model r))
  = F32.softplus20_approx x r

[@@expect_failure [19]]
let cannot_preserve_branch (x : f32) (r : real)
  : Lemma
      (requires A.v_approximates x r)
      (ensures lt (F.of_int 20L) x == t2b (r >. 20.0R))
  = F32.softplus20_approx x r

inline_for_extraction noextract
fn kernel
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (SZ.v n))
  (pairs : larray bf16 (2 * SZ.v n))
  (outputs : larray f32 (4 * SZ.v n))
  (#xs : erased (lseq f32 (SZ.v n)))
  (#ys : erased (lseq bf16 (2 * SZ.v n)))
  (#before : erased (seq f32))
  preserves gpu ** inputs |-> xs ** pairs |-> ys
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
    let left = pairs.(2sz *^ j);
    let right = pairs.(2sz *^ j +^ 1sz);
    let a = argument left right;
    let selected = if lt (F.of_int 20L) a {
      F.of_int #f32 1L
    } else {
      F.of_int #f32 0L
    };
    pts_to_len outputs;
    outputs.(4sz *^ j) <- softplus x;
    outputs.(4sz *^ j +^ 1sz) <- a;
    outputs.(4sz *^ j +^ 2sz) <- selected;
    outputs.(4sz *^ j +^ 3sz) <- softplus a;
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
  (inputs : larray f32 (SZ.v n){is_global_array inputs})
  (pairs : larray bf16 (2 * SZ.v n){is_global_array pairs})
  (outputs : larray f32 (4 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (SZ.v n)))
  (#ys : erased (lseq bf16 (2 * SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs) ** on gpu_loc (pairs |-> ys)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs pairs outputs #xs #ys #before);
}
