module Kuiper.Example.Array1

#lang-pulse
open Kuiper
open Kuiper.Array1
module Array1 = Kuiper.Array1
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.TensorLayout
open Kuiper.Index
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let layout (m : nat) : layout m =
  pack <|
  g_grouped_by 0 m <|
  lunit

// VERY brittle postprocessing to make sure we get a 1st-order function. Would
// not be needed if strict_on_arguments worked properly on recursive functions
// (it seems not to).
[@@Tac.(postprocess_with (fun () ->
           norm [iota; delta; zeta_full; zeta; primops];
           trefl ()))]
inline_for_extraction noextract
instance blah
  (m : SZ.t{SZ.fits m})
  : ctlayout (layout m)
  =
  close _ <|
  c_grouped_by 0sz _ #_ #{v = 1sz} <|
  cunit

fn test0 (m : array1 u32 (layout 10))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (layout 100) = blah 100sz

fn test1 (m : array1 u32 (layout 100))
  preserves m |-> 's
  returns u32
{
  let v = Array1.(m.(1sz));
  v
}

fn test2 (m : array1 u32 (layout 100))
  requires m |-> 's
  // ensures  m |-> Kuiper.Chest.upd 's ((1 <: natlt 10), ((2 <: natlt 20), ())) 42ul
  ensures  m |-> Kuiper.Seq.Common.lseq_upd 's (1 <: natlt 100) 42ul
{
  Array1.(m.(1sz) <- 42ul);
  ()
}
