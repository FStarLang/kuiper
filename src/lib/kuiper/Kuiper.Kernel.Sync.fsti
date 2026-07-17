module Kuiper.Kernel.Sync
#lang-pulse
open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.SizeT
include Kuiper.Kernel.Base
include Kuiper.Kernel.Desc

(* A model for launch_kernel_sync. The model implementation
   is marked divergent just for some technical reasons (Pulse's
   par combinator is divergent). If that is fixed, it should
   be possible to prove it terminating. *)

noextract
divergent
fn launch_kernel_full_sync
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    on gpu_loc full_pre
  ensures
    cpu **
    on gpu_loc full_post
