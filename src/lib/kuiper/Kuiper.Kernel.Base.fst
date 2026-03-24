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
  preserves cpu
  preserves epoch_live e
  requires
    on gpu_loc full_pre
  ensures
    pledge0 (epoch_done e) (on gpu_loc full_post)
{
  admit(); // Intentional, model
}

noextract
fn sync_device () (#e:epoch_t)
  requires
    epoch_live e
  returns
    e' : epoch_t
  ensures
    epoch_done e **
    epoch_live e' **
    pure (e' >= e)
{
  admit(); // Intentional model
}
