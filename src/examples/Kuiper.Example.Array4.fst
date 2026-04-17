module Kuiper.Example.Array4

#lang-pulse
open Kuiper
open Kuiper.Array4
module Array4 = Kuiper.Array4
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg
open Kuiper.Index
module SZ = Kuiper.SizeT

let layout (d0 d1 d2 d3 : nat) : layout d0 d1 d2 d3 =
  pack <|
  major_on 0 d0 <|
  major_on 0 d1 <|
  major_on 0 d2 <|
  major_on 0 d3 <|
  lunit

inline_for_extraction noextract
instance blah
  (d0 : SZ.t{SZ.fits d0})
  (d1 : SZ.t{SZ.fits d1})
  (d2 : SZ.t{SZ.fits d2})
  (d3 : SZ.t{SZ.fits d3})
  (#_ : squash (SZ.fits (d0 * d1 * d2 * d3) /\ SZ.fits (d1 * d2 * d3) /\ SZ.fits (d2 * d3)))
  : ctlayout (layout d0 d1 d2 d3)
  =
  c_pack #_ #_ <|
  c_major_on 0sz _ #_ #{v = d1 *^ (d2 *^ d3)} <|
  c_major_on 0sz _ #_ #{v = d2 *^ d3} <|
  c_major_on 0sz _ #_ #{v = d3} <|
  c_major_on 0sz _ #_ #{v = 1sz} <|
  solve

fn test0 (m : array4 u32 (layout 3 5 4 2))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (layout 10 10 10 10) = blah 10sz 10sz 10sz 10sz

fn test1 (m : array4 u32 (layout 10 10 10 10))
  preserves m |-> 's
  returns u32
{
  let v = Array4.(m.(1sz, 2sz, 3sz, 4sz));
  v
}

fn test2 (m : array4 u32 (layout 10 10 10 10))
  requires m |-> 's
  ensures  m |-> Kuiper.EMatrix4.mupd 's 1 2 3 4 42ul
{
  Array4.(m.(1sz, 2sz, 3sz, 4sz) <- 42ul);
  ()
}
