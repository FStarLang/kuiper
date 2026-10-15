module Kuiper.Example.UnusedLocal

(* This is a regression test after a bad karamel patch.  A bad inlining pass
generated expressions like KRML_HOST_IGNORE(&0.0), instead of `float local = 0.0;
KRML_HOST_IGNORE(&local);`. This test ensures that we don't regress. *)

#lang-pulse

open Kuiper

fn unused_zero ()
  preserves cpu
{
  let mut unused : f32 = zero;
  ()
}

fn unused_negative_infinity ()
  preserves cpu
{
  let mut unused : f32 = neg infinity;
  ()
}
