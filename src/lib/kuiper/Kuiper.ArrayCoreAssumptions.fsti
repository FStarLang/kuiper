module Kuiper.ArrayCoreAssumptions
#lang-pulse
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array

//we could expose this from core_pcm_ref
//assuming that every allocation is at least 128-aligned
val core_base_address (x:A.array 'a) : GTot (n:nat { n > 0 /\ n%128==0 })