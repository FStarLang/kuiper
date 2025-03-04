module Kuiper.Add2

(* Testing basic polymorphism *)

#lang-pulse

open Kuiper

inline_for_extraction
fn padd (#t:Type0) {| simple_scalar t |}
  (x y : t)
  returns t
{
  add x y;
}

let padd_f32 x = add #f32 #_ x
let padd_u64 x = add #u64 #_ x

fn add_f32
  (x y : f32)
  returns f32
{
  add x y;
}

fn add_f32'
  (x y : f32)
  returns f32
{
  padd x y;
}

fn add_u64
  (x y : u64)
  returns u64
{
  add x y
}

fn add_u64'
  (x y : u64)
  returns u64
{
  padd x y;
}
