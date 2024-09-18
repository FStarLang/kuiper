module Kuiper.Assert

#lang-pulse

open Pulse.Lib.Pervasives

fn dassert (b:bool)
  requires pure b
  ensures  pure b

fn dguard (b:bool)
  requires emp
  ensures  pure b
