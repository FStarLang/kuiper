module Kuiper.Example.Add

(* Testing basic polymorphism *)

#lang-pulse

open Kuiper

inline_for_extraction
fn incr (#t:Type0) {| scalar t |}
  (x : t)
  returns t
{
  add x one;
}

let pincr_f32 x = incr #f32 #_ x
let pincr_u64 x = incr #u64 #_ x

fn incr_f32
  (x : f32)
  returns f32
{
  add x one;
}

fn incr_f32'
  (x : f32)
  returns f32
{
  incr x;
}

fn incr_u64
  (x : u64)
  returns u64
{
  add x one;
}

fn incr_u64'
  (x : u64)
  returns u64
{
  incr x;
}
