module Kuiper.Example.Tensor

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Injection
module SZ = Kuiper.SizeT

inline_for_extraction noextract
type snat = x:nat{SizeT.fits x}

let desc m n : idesc 2 =
  ICons m (ICons n INil)

let m2_rm_inj (m n : snat) : (abs (desc m n) @~> natlt (m*n)) =
  mk_injection #(abs (desc m n)) #(natlt (m*n))
    (fun (i, (j, ())) -> i*n + j)
    ez

let m2_rm_layout (m n : snat) : tlayout #2 (ICons m (ICons n INil)) = {
  ulen = m*n;
  imap = m2_rm_inj m n;
}

inline_for_extraction noextract
let cimap (m n : SZ.t{SZ.fits (m*n)}) (i : conc (desc m n)) :
  r : SZ.t {SZ.v r == (m2_rm_inj m n).f (up i)}
=
  (* Cannot use let (i, (j, ())) = i ... or the match remains in the .krml
  with a cast to Top. *)
  match i with
  | (i, (j, ())) -> i *^ n +^ j

inline_for_extraction noextract
instance blah (m n : SZ.t{SZ.fits (m*n)}) : ctlayout (m2_rm_layout m n) =
  let open Kuiper.SizeT in
  {
    culen = m *^ n;
    cimap = (fun i -> cimap m n i);
  }

fn test0 (m : tensor u32 (m2_rm_layout 10 20))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (m2_rm_layout 10 20) = blah 10sz 20sz

fn test1 (m : tensor u32 (m2_rm_layout 10 20))
  preserves m |-> 's
  returns u32
{
  let v = tensor_read m ((1sz <: szlt 10), ((2sz <: szlt 20), ()));
  v
}

fn test2 (m : tensor u32 (m2_rm_layout 10 20))
  requires m |-> 's
  ensures m |-> Kuiper.Chest.upd 's ((1 <: natlt 10), ((2 <: natlt 20), ())) 42ul
{
  tensor_write m ((1sz <: szlt 10), ((2sz <: szlt 20), ())) 42ul;
  ()
}
