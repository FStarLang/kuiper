module Kuiper.Example.Tensor

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Injection
open Kuiper.Tensor.Layout.Alg
module SZ = Kuiper.SizeT

inline_for_extraction noextract
type snat = x:nat{SizeT.fits x}

let desc m n : idesc 2 =
  m @| n @| INil


fn test0 (m : tensor u32 (l2_row_major 10 20))
{
  ()
}

// Should not be needed, maybe Kuiper.concrete can solve this eventually
inline_for_extraction noextract
instance _crutch : ctlayout (l2_row_major 10 20) = c_l2_row_major 10 20sz

fn test1 (m : tensor u32 (l2_row_major 10 20))
  preserves m |-> 's
  returns u32
{
  let v = tensor_read m ((1sz <: szlt 10), ((2sz <: szlt 20), ()));
  v
}

fn test2 (m : tensor u32 (l2_row_major 10 20))
  requires m |-> 's
  ensures m |-> Kuiper.Chest.upd 's ((1 <: natlt 10), ((2 <: natlt 20), ())) 42ul
{
  tensor_write m ((1sz <: szlt 10), ((2sz <: szlt 20), ())) 42ul;
  ()
}
