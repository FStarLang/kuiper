module TestEager

#lang-pulse
open Pulse.Nolib

assume val foo : int -> int -> slprop

[@@pulse_unfold]
let blah (v1 : int) = exists* (v:int). foo v1 v

ghost
fn test (v1 v2 : int)
  requires pure (v1 == v2) ** blah v1
  ensures  blah v2
{
  rewrite each foo v1 as foo v2;
}
