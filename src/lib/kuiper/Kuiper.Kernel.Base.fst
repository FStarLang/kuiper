module Kuiper.Kernel.Base
#lang-pulse

open Kuiper.Common
open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.Base
open Kuiper.Array
open Kuiper.Epoch
open Pulse.Lib.Pledge
open Kuiper.Kernel.Desc

noextract
fn launch_kernel_full
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    on gpu_loc full_pre
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e') (on gpu_loc full_post) **
    pure (e' >= e)
{
  admit()
}