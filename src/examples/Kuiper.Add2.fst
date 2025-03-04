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

(* FIXME: Eta-expanding with just x and not y
   causes a karamel failure due to a partially
   applied Add. Probably a pure F* issue. *)
let padd_f32 x y = add #f32 #_ x y
let padd_u64 x y = add #u64 #_ x y

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
