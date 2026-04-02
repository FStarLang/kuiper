module Kuiper.Example.Array2

#lang-pulse
open Kuiper
open Kuiper.Array2
module Array2 = Kuiper.Array2
open Kuiper.Bijection
open Kuiper.Injection
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

let layout (rows cols : nat) : layout rows cols = {
  ulen = rows * cols;
  imap = inj_bij bij_nat_prod; // row major
}

inline_for_extraction noextract
instance blah (rows : SZ.t{SZ.fits rows}) (cols : SZ.t{SZ.fits cols /\ SZ.fits (SZ.v rows * SZ.v cols)}) : clayout (layout rows cols) =
  {
    culen = rows `SZ.mul` cols;
    all_fit = ();
    cimap = (fun i j -> i `SZ.mul` cols `SZ.add` j);
  }

fn test0 (m : array2 u32 (layout 3 5))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : clayout (layout 10 10) = blah 10sz 10sz

fn test1 (m : array2 u32 (layout 10 10))
  preserves m |-> 's
  returns u32
{
  let v = Array2.read m 1sz 2sz;
  v
}

fn test2 (m : array2 u32 (layout 10 10))
  requires m |-> 's
  ensures  m |-> (mupd 's (1 <: natlt 10) (2 <: natlt 10) 42ul <: ematrix u32 10 10)
{
  Array2.write m 1sz 2sz 42ul;
  ()
}
