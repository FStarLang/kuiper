module Kuiper.Example.Array2

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT

let layout (d0 d1 : nat) : layout2 d0 d1 =
  pack <|
  major_on 0 d0 <|
  major_on 0 d1 <|
  lunit

inline_for_extraction noextract
instance blah
  (d0 d1 : sz)
  (#_ : squash (SZ.fits (d0 * d1)))
  : ctlayout (layout d0 d1)
  =
  c_pack <|
  c_major_on 0sz _ #_ #{v = d1} <|
  c_major_on 0sz _ #_ #{v = 1sz} <|
  solve

fn test0 (m : array2 u32 (layout 3 5))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (layout 10 10) = blah 10sz 10sz

fn test1 (m : array2 u32 (layout 10 10))
  preserves m |-> 's
  returns u32
{
  m.((1sz <: szlt _), ((2sz <: szlt _), ()));
}

fn test2 (m : array2 u32 (layout 10 10))
  requires m |-> 's
  ensures  m |-> upd2 's 1 2 42ul
{
  m.((1sz <: szlt _), ((2sz <: szlt _), ())) <- 42ul;
  ()
}
