module Kuiper.Example.SigmoidOptimized

#lang-pulse

open Kuiper
module F = Kuiper.Float32.Exact
module S = Kuiper.Spec.SigmoidOptimized
module U = FStar.UInt32

inline_for_extraction noextract
fn flush_subnormal (x : F.t)
  returns r : F.t
  ensures pure (F.to_bits r == S.flush_bits (F.to_bits x) /\
                S.is_flushed (F.to_bits r))
{
  S.flush_properties (F.to_bits x);
  let b = F.bits x;
  assert pure (b == F.to_bits x);
  let result_bits = (if U.eq (U.logand b 0x7f800000ul) 0ul
                     then U.logand b 0x80000000ul else b);
  F.of_bits result_bits
}

(* Every constant is constructed from its checked binary32 encoding.
   No assumptions or admits are needed in this function. *)
[@@CPrologue "__device__"]
fn sigmoid (input : F.t)
  returns r : F.t
  ensures pure (F.to_bits r == S.sigmoid_bits (F.to_bits input) /\
                S.is_flushed (F.to_bits r))
{
  let x = flush_subnormal input;
  if (F.ge x (F.of_bits S.minus_one) && F.le x (F.of_bits S.one)) {
    let x2 = flush_subnormal (F.mul_rn x x);
    let p4 = flush_subnormal (F.fma (F.of_bits S.c5) x2 (F.of_bits S.c4));
    let p3 = flush_subnormal (F.fma p4 x2 (F.of_bits S.c3));
    let p2 = flush_subnormal (F.fma p3 x2 (F.of_bits S.c2));
    let p1 = flush_subnormal (F.fma p2 x2 (F.of_bits S.c1));
    let p0 = flush_subnormal (F.fma p1 x2 (F.of_bits S.c0));
    flush_subnormal (F.fma x p0 (F.of_bits S.half))
  } else {
    let exponential = flush_subnormal (F.exp_fast (F.neg x));
    let denominator = flush_subnormal (F.add_rn F.one exponential);
    flush_subnormal (F.div_fast F.one denominator)
  }
}
