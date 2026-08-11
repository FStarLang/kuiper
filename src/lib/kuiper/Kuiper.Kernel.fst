module Kuiper.Kernel
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Kernel.Base
open Kuiper.Epoch
open Kuiper.Kernel.Stream
open Pulse.Lib.Pledge
open FStar.Tactics.Typeclasses { solve }

ghost
fn pledge_flushed_advance
  (s: stream_t)
  (e e' : epoch_t)
  (#p : slprop)
  requires pledge0 (epoch_flushed s e) p
  requires pure (e <= e')
  ensures  pledge0 (epoch_flushed s e') p
{
  make_pledge emp_inames (epoch_flushed s e') p (pledge0 (epoch_flushed s e) p)
    fn _ {
      flushed_lower s e' e;
      redeem_pledge emp_inames (epoch_flushed s e) p;
      drop_ (epoch_flushed s e);
    };
}

ghost
fn pledge_flushed_done
  (s: stream_t)
  (e : epoch_t)
  (#p : slprop)
  requires pledge0 (epoch_flushed s e) p
  ensures  pledge0 (epoch_done s e) p
{
  make_pledge emp_inames (epoch_done s e) p (pledge0 (epoch_flushed s e) p)
    fn _ {
      done_flushed s e;
      redeem_pledge emp_inames (epoch_flushed s e) p;
      drop_ (epoch_flushed s e);
    };
}

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
{
  return_pledge (epoch_flushed s e) (on gpu_loc full_pre) #solve;
  launch_kernel_full k s;
}

inline_for_extraction noextract
fn launch_kernel_full_sync
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    on gpu_loc full_pre
  ensures
    cpu **
    on gpu_loc full_post
{
  let s = fresh_stream ();
  init_epoch s ();
  launch_kernel_full_owned k s;
  sync_stream s;
  done_flushed s _;
  redeem_pledge emp_inames (epoch_flushed s _) (on gpu_loc full_post);
  drop_ (epoch_flushed s _);
  drop_ (epoch_done s _);
  drop_ (epoch_live s _);
  destroy_stream s;
}

// inline_for_extraction noextract
// instance launchable_self
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc full_pre full_post) full_pre full_post
//   = { cast = id; }

// inline_for_extraction noextract
// instance launchable_m_n
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_m_n full_pre full_post) full_pre full_post
//   = { cast = kmn_as_kfull; }

// inline_for_extraction noextract
// instance launchable_m_1
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_m_1 full_pre full_post) full_pre full_post
//   = { cast = (fun k -> k |> km1_as_kmn |> kmn_as_kfull); }

// inline_for_extraction noextract
// instance launchable_1_n
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_1_n full_pre full_post) full_pre full_post
//   = { cast = (fun k -> k |> k1n_as_kmn |> kmn_as_kfull); }

// inline_for_extraction noextract
// instance launchable_1_1
//   (#full_pre #full_post : slprop)
//   : launchable (kernel_desc_1_1 full_pre full_post) full_pre full_post
//   = { cast = (fun k -> k |> k11_as_k1n |> k1n_as_kmn |> kmn_as_kfull); }

inline_for_extraction noextract
fn launch_kernel_1
  (#pre : slprop)
  (#post : slprop)
  {| is_send_across gpu_of pre, is_send_across gpu_of post |}
  (k : fn () requires gpu ** pre ensures gpu ** post)
  requires
    cpu **
    on gpu_loc pre
  ensures
    cpu **
    on gpu_loc post
{
  launch_kernel_full_sync ({ f = k; full_pre_sendable=solve; full_post_sendable=solve } <: kernel_desc_1_1 _ _);
}
