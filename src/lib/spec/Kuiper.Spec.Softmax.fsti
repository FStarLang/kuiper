module Kuiper.Spec.Softmax

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Seq.Common

(* Real-value golden specification for softmax. *)
let softmax_real (s:Seq.seq real) =
  let exps = seq_map rexp s in
  let summ : real = rsum exps in
  seq_mapi s (fun x i -> rexp x /. summ)
