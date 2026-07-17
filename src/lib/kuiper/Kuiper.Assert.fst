module Kuiper.Assert

#lang-pulse

open Pulse.Lib.Pervasives

// Dummy implementations to show soundness of the interface.
// These are replaced during extraction.

fn dassert (b:bool)
  requires pure b
  ensures  pure b
{}

(* Extracted primitively. A possible model is to loop
   if ~b, but that's not total. *)
fn dguard (b:bool)
  requires emp
  ensures  pure b
{
  admit()
}
