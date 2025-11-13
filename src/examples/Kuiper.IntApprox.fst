module Kuiper.IntApprox

open Kuiper
open Kuiper.Approximates
open Kuiper.Approximates.U32
module U32 = FStar.UInt32

(* To expose the definitions of to_real and v_approximates. *)
friend Kuiper.Approximates.U32

(* If we have a u32 z approximating x+y, it must be that z is *exactly*
   (x+y)%2^32, and that x+y is an integer. *)

let lem (z : u32) (x y : real)
  : Lemma (requires z %~ (x +. y))
          (ensures  exists (z' : int). x+.y == FStar.Real.of_int z' /\ z' % (pow2 32) == U32.v z)
  = ()

(* And if x and y were obtained from to_real, then we can specialize a bit more. *)
let lem' (z x y : u32)
  : Lemma (requires z %~ (to_real x +. to_real y))
          (ensures  FStar.UInt32.(z == x +%^ y))
  = ()
