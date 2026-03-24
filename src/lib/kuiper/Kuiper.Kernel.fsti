module Kuiper.Kernel
inline_for_extraction noextract let _ = ()
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Epoch
include Kuiper.Kernel.Base
include Kuiper.Kernel.Desc
include Kuiper.Kernel.Casts
open Pulse.Lib.Pledge

inline_for_extraction noextract
fn launch_kernel_full_sync
  (#full_pre #full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    on gpu_loc full_pre
  ensures
    cpu **
    on gpu_loc full_post

(* A helper for very simple kernels, mostly for unit tests. *)
inline_for_extraction noextract
fn launch_kernel_1
  (#pre : slprop)
  (#post : slprop)
  {| is_send_across gpu_of pre, is_send_across gpu_of post |}
  (k : fn () requires gpu ** pre
            ensures gpu ** post)
  requires
    cpu **
    on gpu_loc pre
  ensures
    cpu **
    on gpu_loc post

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
