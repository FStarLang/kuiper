module Kuiper.Example.Array3

#lang-pulse
open Kuiper
open Kuiper.Array3
module Array3 = Kuiper.Array3
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.TensorLayout
open Kuiper.Index
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let layout (d0 d1 d2 : nat) : layout d0 d1 d2 =
  pack <|
  g_grouped_by 0 d0 <|
  g_grouped_by 0 d1 <|
  g_grouped_by 0 d2 <|
  lunit

// VERY brittle postprocessing to make sure we get a 1st-order function. Would
// not be needed if strict_on_arguments worked properly on recursive functions
// (it seems not to).
[@@Tac.(postprocess_with (fun () ->
           norm [iota; delta; zeta_full; zeta; primops];
           trefl ()))]
inline_for_extraction noextract
instance blah
  (d0 : SZ.t{SZ.fits d0})
  (d1 : SZ.t{SZ.fits d1})
  (d2 : SZ.t{SZ.fits d2})
  (#_ : squash (SZ.fits (d0 * d1 * d2) /\ SZ.fits (d1 * d2)))
  : ctlayout (layout d0 d1 d2)
  =
  close _ <|
  c_grouped_by 0sz _ #_ #{v = d1 *^ d2} <|
  c_grouped_by 0sz _ #_ #{v = d2} <|
  c_grouped_by 0sz _ #_ #{v = 1sz} <|
  cunit

fn test0 (m : array3 u32 (layout 3 5 4))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (layout 10 10 10) = blah 10sz 10sz 10sz

fn test1 (m : array3 u32 (layout 10 10 10))
  preserves m |-> 's
  returns u32
{
  let v = Array3.(m.(1sz, 2sz, 3sz));
  v
}

fn test2 (m : array3 u32 (layout 10 10 10))
  requires m |-> 's
  ensures  m |-> Kuiper.EMatrix3.mupd 's 1 2 3 42ul
{
  Array3.(m.(1sz, 2sz, 3sz) <- 42ul);
  ()
}
