module Test.FnType
#lang-pulse
open Pulse

fn test
  (frame : slprop)
  (fn f (x : int) requires frame ensures frame)
  requires frame
  ensures frame
{ f 0; }
