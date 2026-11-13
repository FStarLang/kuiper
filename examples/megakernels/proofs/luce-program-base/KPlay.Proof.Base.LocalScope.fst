module KPlay.Proof.Base.LocalScope

#lang-pulse

open Kuiper
open Pulse.Lib.Array
module A = Pulse.Lib.Array
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base

inline_for_extraction noextract
fn first_write ()
  preserves gpu
  returns value:FP.t
  ensures pure (value == F.zero #FP.t)
{
  let mut partial : A.array FP.t = [| 4sz |];
  A.mask_write partial 0sz (F.zero #FP.t);
  A.mask_read partial 0sz
}
