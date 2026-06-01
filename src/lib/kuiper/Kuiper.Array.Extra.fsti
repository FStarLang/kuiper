module Kuiper.Array.Extra

#lang-pulse

(* Extra functions over Pulse's normal arrays. *)
open Pulse
open Pulse.Lib.Array
open FStar.Seq
open Kuiper.ForEvery
open Kuiper.Common

ghost
fn array_share
  (#t:Type0)
  (a : array t)
  (#s : seq t)
  (#f : perm)
  (n : pos)
  requires
    a |-> Frac f s
  ensures
    forall+ (_ : natlt n).
      a |-> Frac (f /. Real.of_int n) s

ghost
fn array_gather
  (#t:Type0)
  (a : array t)
  (#s : seq t)
  (#f : perm)
  (n : pos)
  requires
    forall+ (_ : natlt n).
      a |-> Frac (f /. Real.of_int n) s
  ensures
    a |-> Frac f s
