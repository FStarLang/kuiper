module Kuiper.View.Layout4

#lang-pulse
open Kuiper

(* flaky stuff *)
#push-options "--retry 5"

let inv1 (c:cfg) (i : idxt1 c)
  : Lemma (f2 c (f1 c i) == i)
  = ()

let inv2 (c:cfg) (i : idxt2 c)
  : Lemma (f1 c (f2 c i) == i)
  = ()

#pop-options
