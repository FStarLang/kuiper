module Kuiper.ArrayCoreAssumptions
#lang-pulse
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array

(* We assume every array has a base address, in bytes. This could
be exposed from inside Pulse. *)
val core_base_address (x:A.array 'a) : GTot (n:nat { n > 0 })
