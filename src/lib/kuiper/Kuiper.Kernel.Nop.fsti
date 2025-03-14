module Kuiper.Kernel.Nop
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.IntAliases
open Kuiper.Array
open Kuiper.Base
module SZ = FStar.SizeT
open Kuiper.ForEvery

open Kuiper.Kernel.Desc
open Kuiper.Kernel.Casts

fn nop (#f: slprop) ()
  requires f
  ensures f
{ () }

inline_for_extraction noextract
let nop_desc_11 : kernel_desc_1_1 emp emp = {
  f = nop;
}
