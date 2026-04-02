module Kuiper.Example.Array1

#lang-pulse
open Kuiper
open Kuiper.Array1
module Array1 = Kuiper.Array1
open Kuiper.Injection
module SZ = Kuiper.SizeT

let layout (m : nat) : layout m = {
  ulen = m;
  imap = mk_injection (fun i -> i) ez;
}

inline_for_extraction noextract
instance blah (m : SZ.t{SZ.fits m}) : clayout (layout m) =
  {
    culen = m;
    all_fit = ();
    cimap = (fun i -> i);
  }

fn test0 (m : array1 u32 (layout 10))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : clayout (layout 100) = blah 100sz

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
