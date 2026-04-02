module Kuiper.Example.Array4

#lang-pulse
open Kuiper
open Kuiper.Array4
module Array4 = Kuiper.Array4
open Kuiper.Bijection
open Kuiper.Injection
module SZ = Kuiper.SizeT
open Kuiper.TensorLayout
open Kuiper.Index

let from_tlayout
  (#d0 #d1 #d2 #d3 : nat)
  (t : tlayout (d0 @| d1 @| d2 @| d3 @| INil))
: layout d0 d1 d2 d3 =
  {
    ulen = t.ulen;
    imap =
      mk_injection (fun (i, j, k, l) -> t.imap.f (i, (j, (k, (l, ())))))
        ez;
  }

let layout (d0 d1 d2 d3 : nat) : layout d0 d1 d2 d3 =
  from_tlayout <|
  pack <|
  g_grouped_by 0 d0 <|
  g_grouped_by 0 d1 <|
  g_grouped_by 0 d2 <|
  g_grouped_by 0 d3 <|
  lunit

#restart-solver
#push-options "--ifuel 4"
inline_for_extraction noextract
instance blah
  (d0 : SZ.t{SZ.fits d0})
  (d1 : SZ.t{SZ.fits d1})
  (d2 : SZ.t{SZ.fits d2})
  (d3 : SZ.t{SZ.fits d3})
  (#_ : squash (SZ.fits (d0 * d1 * d2 * d3) /\ SZ.fits (d1 * d2 * d3) /\ SZ.fits (d2 * d3)))
  : clayout (layout d0 d1 d2 d3) =
  {
    culen = d0 `SZ.mul` (d1 `SZ.mul` (d2 `SZ.mul` d3));
    all_fit = ();
    cimap = (fun i j k m ->
      assume False;
      i `SZ.mul` (d1 `SZ.mul` (d2 `SZ.mul` d3)) `SZ.add`
      (j `SZ.mul` (d2 `SZ.mul` d3)) `SZ.add`
      (k `SZ.mul` d3) `SZ.add`
      m);
  }
#pop-options

fn test0 (m : array4 u32 (layout 3 5 4 2))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : clayout (layout 10 10 10 10) = blah 10sz 10sz 10sz 10sz

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
