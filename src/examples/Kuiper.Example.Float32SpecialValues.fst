module Kuiper.Example.Float32SpecialValues

#lang-pulse

open Kuiper
module F32 = Kuiper.Float32
module Fast = Kuiper.Float32.FastMath
module F = Kuiper.Floating.Base
module SZ = Kuiper.SizeT

let ordinary_infinite_exponentials ()
  : Lemma
      (fexp (F.sub F.zero F.infinity) == (F.zero #f32) /\
       fexp (F.infinity #f32) == F.infinity)
  = F32.exp_special_values (F.sub F.zero F.infinity);
    F32.exp_special_values F.infinity

let ordinary_nan_exponential (x : f32{NaN? (kind x)})
  : Lemma (NaN? (kind (fexp x)))
  = F32.exp_special_values x

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
    (maximum == F.sub F.zero F.infinity \/ NaN? (kind maximum) ==>
      NaN? (kind result)))
{
  let shifted = Fast.sub_rn (F.sub F.zero F.infinity) maximum;
  F32.exp_special_values shifted;
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
  padding_exp (F.sub F.zero F.infinity)
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
  padding_exp (F.sub F.zero F.infinity)
}

[@@expect_failure [19]]
let cannot_preserve_nan_payload (x : f32{NaN? (kind x)})
  : Lemma (fexp x == x)
  = F32.exp_special_values x

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
    let shifted = Fast.sub_rn (F.sub F.zero F.infinity) x;
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
