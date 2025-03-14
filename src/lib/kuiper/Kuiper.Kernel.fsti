module Kuiper.Kernel
inline_for_extraction noextract let x = ()
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Epoch
include Kuiper.Kernel.Base
include Kuiper.Kernel.Desc
include Kuiper.Kernel.Casts
open FStar.Tactics.Typeclasses
open Pulse.Lib.Pledge

inline_for_extraction noextract
fn launch_kernel_full_sync
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    full_pre
  ensures
    cpu **
    full_post

(* A helper for very simple kernels, mostly for unit tests. *)
inline_for_extraction noextract
fn launch_kernel_1
  (#pre : slprop)
  (#post : slprop)
  (k : unit -> stt unit (gpu ** pre) (fun _ -> gpu ** post))
  requires
    cpu **
    pre
  ensures
    cpu **
    post

(* NOTE: commented-out is how to define these functions using a typeclass
of launchable things instead of making the kernel casts coercions. But this
hurts inference. If we have a function of type `r:ref a -> #v:erased a -> kernel_desc ..`
and try to launch it, F* will not instantiate the implicit (which makes sense,
there is no reason to), and will then fail to find an instance. *)

// (* A class for different configurations that can be launched. *)
// [@@fundeps [1;2]]
// class launchable (t : Type) (full_pre full_post : slprop) = { 
//   [@@@no_method] cast : t -> kernel_desc full_pre full_post;
// }

inline_for_extraction noextract
val launch
  // (#t:Type)
  (#full_pre #full_post : slprop)
  // {| d : launchable t full_pre full_post |}
  // (x : t)
  (k : kernel_desc full_pre full_post)
  (#e : epoch_t)
  : stt epoch_t
      (cpu **
       epoch_live e **
       full_pre)
      (fun e' ->
        cpu **
        epoch_live e' **
        pledge0 (epoch_done e') full_post **
        pure (e' >= e))

inline_for_extraction noextract
val launch_sync
  // (#t:Type)
  (#full_pre #full_post : slprop)
  // {| d : launchable t full_pre full_post |}
  // (x : t)
  (_ : kernel_desc full_pre full_post)
  : stt unit
      (cpu ** full_pre)
      (fun _ ->
        cpu **
        full_post)

// inline_for_extraction noextract
// instance val launchable_self
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc full_pre full_post) full_pre full_post

// inline_for_extraction noextract
// instance val launchable_m_n
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_m_n full_pre full_post) full_pre full_post

// inline_for_extraction noextract
// instance val launchable_m_1
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_m_1 full_pre full_post) full_pre full_post

// inline_for_extraction noextract
// instance val launchable_1_n
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_1_n full_pre full_post) full_pre full_post

// inline_for_extraction noextract
// instance val launchable_1_1
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_1_1 full_pre full_post) full_pre full_post
