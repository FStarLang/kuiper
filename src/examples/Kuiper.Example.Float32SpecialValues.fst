module Kuiper.Example.Float32SpecialValues

#lang-pulse

open Kuiper
module Fast = Kuiper.Float32.FastMath
module F = Kuiper.Floating.Base
module SZ = Kuiper.SizeT

let ordinary_zero_exponentials #t {| floating t |} (x : t{is_zero x})
  : Lemma
      (fexp (F.zero #t) == F.one /\
       fexp (F.neg (F.zero #t)) == F.one /\
       fexp x == F.one)
  = F.exp_special_values (F.zero #t);
    F.neg_is_zero (F.zero #t);
    F.exp_special_values (F.neg (F.zero #t));
    F.exp_special_values x

let ordinary_infinite_exponentials #t {| floating t |} ()
  : Lemma
      (fexp (F.neg F.infinity) == (F.zero #t) /\
       fexp (F.infinity #t) == F.infinity)
  = F.exp_special_values (F.neg (F.infinity #t));
    F.exp_special_values (F.infinity #t)

let ordinary_nan_exponential #t {| floating t |} (x : t{NaN? (kind x)})
  : Lemma (NaN? (kind (fexp x)))
  = F.exp_special_values x

let infinite_subtractions
  (x : f32{Infinite? (kind x)}) (y : f32{Finite? (kind y)})
  : Lemma
      (Fast.pure_sub_rn x y == x /\
       Fast.pure_sub_rn y x == F.neg x /\
       NaN? (kind (Fast.pure_sub_rn x x)))
  = ()

let opposite_infinities
  (x : f32{Infinite? (kind x)})
  (y : f32{Infinite? (kind y) /\ x =!= y})
  : Lemma (Fast.pure_sub_rn x y == x)
  = ()

let zero_subtractions (x : f32{is_zero x}) (y : f32{is_zero y})
  : Lemma
      (Fast.pure_sub_rn x x == F.zero /\
       (x =!= y ==> Fast.pure_sub_rn x y == x))
  = is_zero_spec x; is_zero_spec y

inline_for_extraction noextract
fn repeated_subtraction (x y : f32)
  preserves gpu
  returns result : f32
  ensures pure (
    result == Fast.pure_sub_rn x y /\
    (forall (xr yr : real).
      v_approximates x xr /\ v_approximates y yr ==>
      v_approximates result (Kuiper.Float.Realops.r_sub xr yr)))
{
  let first = Fast.sub_rn x y;
  let second = Fast.sub_rn x y;
  assert pure (first == second);
  first
}

[@@expect_failure [19]]
let cannot_preserve_subtraction_nan_payload
  (x : f32{NaN? (kind x)}) (y : f32)
  : Lemma (Fast.pure_sub_rn x y == x)
  = ()

inline_for_extraction noextract
fn nan_subtraction (x y : f32)
  preserves gpu
  requires pure (NaN? (kind x) \/ NaN? (kind y))
  returns result : f32
  ensures pure (NaN? (kind result))
{
  Fast.sub_rn x y
}

inline_for_extraction noextract
fn padding_exp (maximum : f32)
  preserves gpu
  returns result : f32
  ensures pure (
    (Finite? (kind maximum) \/ maximum == F.infinity ==>
      result == F.zero /\ v_approximates result 0.0R) /\
    (maximum == F.neg F.infinity \/ NaN? (kind maximum) ==>
      NaN? (kind result)))
{
  ordinary_infinite_exponentials #f32 ();
  let shifted = Fast.sub_rn (F.neg F.infinity) maximum;
  F.exp_special_values shifted;
  fexp shifted
}

inline_for_extraction noextract
fn finite_padding (maximum : f32)
  preserves gpu
  requires pure (Finite? (kind maximum))
  returns result : f32
  ensures pure (result == F.zero /\ v_approximates result 0.0R)
{
  padding_exp maximum
}

inline_for_extraction noextract
fn positive_infinite_maximum ()
  preserves gpu
  returns result : f32
  ensures pure (result == F.zero /\ v_approximates result 0.0R)
{
  padding_exp F.infinity
}

inline_for_extraction noextract
fn negative_infinite_maximum ()
  preserves gpu
  returns result : f32
  ensures pure (NaN? (kind result))
{
  padding_exp (F.neg F.infinity)
}

inline_for_extraction noextract
fn nan_maximum (maximum : f32)
  preserves gpu
  requires pure (NaN? (kind maximum))
  returns result : f32
  ensures pure (NaN? (kind result))
{
  padding_exp maximum
}

[@@expect_failure [19]]
fn cannot_drop_maximum_condition (maximum : f32)
  preserves gpu
  returns result : f32
  ensures pure (result == F.zero)
{
  padding_exp maximum
}

[@@expect_failure [19]]
fn negative_infinite_padding_is_not_zero ()
  preserves gpu
  returns result : f32
  ensures pure (result == F.zero)
{
  padding_exp (F.neg F.infinity)
}

[@@expect_failure [19]]
let cannot_preserve_nan_payload (x : f32{NaN? (kind x)})
  : Lemma (fexp x == x)
  = F.exp_special_values x

[@@expect_failure [228]]
fn cannot_compute_padding_on_cpu (maximum : f32)
  preserves cpu
  returns f32
{
  padding_exp maximum
}

inline_for_extraction noextract
fn kernel
  (n : sz{SZ.v n <= 1073741823})
  (inputs : larray f32 (SZ.v n))
  (outputs : larray f32 (4 * SZ.v n))
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
    let shifted = Fast.sub_rn (F.neg F.infinity) x;
    let padded = padding_exp x;
    let difference = Fast.sub_rn x F.one;
    pts_to_len outputs;
    outputs.(4sz *^ j) <- fexp x;
    outputs.(4sz *^ j +^ 1sz) <- shifted;
    outputs.(4sz *^ j +^ 2sz) <- padded;
    outputs.(4sz *^ j +^ 3sz) <- difference;
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
  (outputs : larray f32 (4 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> kernel n inputs outputs #xs #before);
}

inline_for_extraction noextract
fn exp_kernel
  (#t : Type0) {| floating t |}
  (n : sz)
  (inputs outputs : larray t (SZ.v n))
  (#xs : erased (lseq t (SZ.v n)))
  (#before : erased (seq t))
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
    pts_to_len outputs;
    outputs.(j) <- fexp x;
    i := !i +^ 1sz;
  }
}

inline_for_extraction noextract
fn run_exp
  (#t : Type0) {| floating t |}
  (n : sz)
  (inputs : larray t (SZ.v n){is_global_array inputs})
  (outputs : larray t (SZ.v n){is_global_array outputs})
  (#xs : erased (lseq t (SZ.v n)))
  preserves cpu ** on gpu_loc (inputs |-> xs)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1 (fun () -> exp_kernel n inputs outputs #xs #before);
}

let run_exp_f16 = run_exp #f16
let run_exp_bf16 = run_exp #bf16
let run_exp_f64 = run_exp #f64

inline_for_extraction noextract
fn subtraction_kernel
  (n : sz{SZ.v n <= 2147483647})
  (left right : larray f32 (SZ.v n))
  (outputs : larray f32 (2 * SZ.v n))
  (#xs : erased (lseq f32 (SZ.v n)))
  (#ys : erased (lseq f32 (SZ.v n)))
  (#before : erased (seq f32))
  preserves gpu ** left |-> xs ** right |-> ys
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
    let x = left.(j);
    let y = right.(j);
    let first = repeated_subtraction x y;
    let second = Fast.sub_rn x y;
    pts_to_len outputs;
    outputs.(2sz *^ j) <- first;
    outputs.(2sz *^ j +^ 1sz) <- second;
    i := !i +^ 1sz;
  }
}

fn run_subtraction
  (n : sz{SZ.v n <= 2147483647})
  (left : larray f32 (SZ.v n){is_global_array left})
  (right : larray f32 (SZ.v n){is_global_array right})
  (outputs : larray f32 (2 * SZ.v n){is_global_array outputs})
  (#xs : erased (lseq f32 (SZ.v n)))
  (#ys : erased (lseq f32 (SZ.v n)))
  preserves cpu ** on gpu_loc (left |-> xs ** right |-> ys)
  requires exists* before. on gpu_loc (outputs |-> before)
  ensures exists* after. on gpu_loc (outputs |-> after)
{
  with before. assert on gpu_loc (outputs |-> before);
  launch_kernel_1
    (fun () -> subtraction_kernel n left right outputs #xs #ys #before);
}
