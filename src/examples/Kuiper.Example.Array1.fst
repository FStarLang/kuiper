module Kuiper.Example.Array1

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg

let layout (m : nat) : layout1 m =
  pack <|
  major_on 0 m <|
  lunit

fn test1 (m : array1 u32 (layout 100))
  preserves m |-> 's
  returns u32
{
  m.((1sz <: szlt 100), ());
}

fn test2 (m : array1 u32 (layout 100))
  requires m |-> 's
  ensures  m |-> upd1 's 1 42ul
{
  m.((1sz <: szlt 100), ()) <- 42ul;
  ()
}
