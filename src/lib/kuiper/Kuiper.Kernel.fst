module Kuiper.Kernel
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.Kernel.Base
open Kuiper.Epoch
open Pulse.Lib.Pledge

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
  get_epoch ();
  launch_kernel_full k;
  sync_device ();
  redeem_pledge emp_inames (epoch_done _) (on gpu_loc full_post);
  drop_ (epoch_done _);
  drop_ (epoch_live _);
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
