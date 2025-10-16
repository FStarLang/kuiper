module Kuiper.Assert

#lang-pulse

open Pulse.Lib.Pervasives

// Dummy implementations to show soundness of the interface.
// These are replaced during extraction.

fn dassert (b:bool)
  requires pure b
  ensures  pure b
{}

fn rec dguard (b:bool)
  requires emp
  ensures  pure b
{
  if (not b) {
    dguard b
  }
}
