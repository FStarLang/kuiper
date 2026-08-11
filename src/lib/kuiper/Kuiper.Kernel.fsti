module Kuiper.Kernel
inline_for_extraction noextract let _ = ()
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Epoch
open Kuiper.Kernel.Stream
include Kuiper.Kernel.Base
include Kuiper.Kernel.Desc
include Kuiper.Kernel.Casts
open Pulse.Lib.Pledge

(* Move a stream-ordered pledge later in the queue. Sound because
[epoch_flushed] is downward closed: arriving by a later position is a weaker
claim than arriving by an earlier one. Useful to bring two pledges obtained at
different positions to a common position so they can be joined and handed to a
single launch. *)
ghost
fn pledge_flushed_advance
  (s: stream_t)
  (e e' : epoch_t)
  (#p : slprop)
  requires pledge0 (epoch_flushed s e) p
  requires pure (e <= e')
  ensures  pledge0 (epoch_flushed s e') p

(* Weaken a stream-ordered pledge into an ordinary host-observable one, to be
redeemed after [sync_stream] or [sync_device]. This is the direction that
always holds; there is no way back, which is what keeps non-stream-ordered
effects from being fed to a later launch. *)
ghost
fn pledge_flushed_done
  (s: stream_t)
  (e : epoch_t)
  (#p : slprop)
  requires pledge0 (epoch_flushed s e) p
  ensures  pledge0 (epoch_done s e) p

(* Launch with a precondition owned outright on the host, rather than pledged.
This is the way to start a chain of dependent launches; subsequent ones in the
chain use [launch_kernel_full] directly and consume the pledge produced here. *)
inline_for_extraction noextract
fn launch_kernel_full_owned
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  (s: stream_t)
  (#e : epoch_t)
  preserves cpu ** stream_live s
  requires
    epoch_live s e **
    on gpu_loc full_pre
  ensures
    epoch_live s (epoch_next e) **
    pledge0 (epoch_flushed s (epoch_next e)) (on gpu_loc full_post)

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
