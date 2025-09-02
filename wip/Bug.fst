module Bug

#lang-pulse
open Kuiper

class foo (a:Type) = {
  fooprop : a -> slprop;
}

instance all_foo (a:Type) : foo a = {
  fooprop = (fun _ -> emp);
}

(* - Could not solve typeclass constraint `foo (seq (*?u9*) _)`
why? Checking the pre should force ?u9 to be u64. *)

[@@expect_failure] // should work
ghost
fn test (a : vec u64) (vv : seq _)
  requires a |-> vv
  returns  x : seq u64
  ensures  a |-> vv ** fooprop vv
