module Kuiper.Example.Array3

#lang-pulse
open Kuiper
open Kuiper.Array3
module Array3 = Kuiper.Array3
open Kuiper.Bijection
open Kuiper.Injection
module SZ = Kuiper.SizeT

let layout (d0 d1 d2 : nat) : layout d0 d1 d2 = {
  ulen = d0 * (d1 * d2);
  imap = inj_bij (bij_tuple_nest `bij_comp` bij_prod (bij_self _) bij_nat_prod `bij_comp` bij_nat_prod); // row major
}

inline_for_extraction noextract
instance blah (d0 : SZ.t{SZ.fits d0}) (d1 : SZ.t{SZ.fits d1}) (d2 : SZ.t{SZ.fits d2 /\ SZ.fits (SZ.v d1 * SZ.v d2) /\ SZ.fits (SZ.v d0 * (SZ.v d1 * SZ.v d2))}) : clayout (layout d0 d1 d2) =
  {
    culen = d0 `SZ.mul` (d1 `SZ.mul` d2);
    all_fit = ();
    cimap = (fun i j k -> i `SZ.mul` (d1 `SZ.mul` d2) `SZ.add` (j `SZ.mul` d2 `SZ.add` k));
  }

fn test0 (m : array3 u32 (layout 3 5 4))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : clayout (layout 10 10 10) = blah 10sz 10sz 10sz

fn test1 (m : array3 u32 (layout 10 10 10))
  preserves m |-> 's
  returns u32
{
  let v = Array3.read m 1sz 2sz 3sz;
  v
}

fn test2 (m : array3 u32 (layout 10 10 10))
  requires m |-> 's
  ensures  m |-> Kuiper.EMatrix3.mupd 's 1 2 3 42ul
{
  Array3.write m 1sz 2sz 3sz 42ul;
  ()
}
