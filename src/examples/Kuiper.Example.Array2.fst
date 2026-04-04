module Kuiper.Example.Array2

#lang-pulse
open Kuiper
open Kuiper.Array2
module Array2 = Kuiper.Array2
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.TensorLayout
open Kuiper.Index
open Kuiper.EMatrix
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let layout (rows cols : nat) : layout rows cols =
  pack <|
  g_grouped_by 0 rows <|
  g_grouped_by 0 cols <|
  lunit

// VERY brittle postprocessing to make sure we get a 1st-order function. Would
// not be needed if strict_on_arguments worked properly on recursive functions
// (it seems not to).
[@@Tac.(postprocess_with (fun () ->
           norm [iota; delta; zeta_full; zeta; primops];
           trefl ()))]
inline_for_extraction noextract
instance blah
  (rows : SZ.t{SZ.fits rows})
  (cols : SZ.t{SZ.fits cols})
  (#_ : squash (SZ.fits (rows * cols)))
  : ctlayout (layout rows cols)
  =
  close _ <|
  c_grouped_by 0sz _ #_ #{v = cols} <|
  c_grouped_by 0sz _ #_ #{v = 1sz} <|
  cunit

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
  let v = Array2.(m.(1sz, 2sz));
  v
}

fn test2 (m : array2 u32 (layout 10 10))
  requires m |-> 's
  ensures  m |-> (mupd 's (1 <: natlt 10) (2 <: natlt 10) 42ul <: ematrix u32 10 10)
{
  Array2.(m.(1sz, 2sz) <- 42ul);
  ()
}
