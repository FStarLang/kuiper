module Kuiper.Kernel.Casts
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc

[@@coercion]
inline_for_extraction noextract
let kmn_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
     : kernel_desc     full_pre full_post
= let open k <: kernel_desc_m_n full_pre full_post in
  {
  nblk;
  nthr;

  shmem_type = FStar.UInt8.t;
  shmem_type_is_sized = solve;
  shmem_sz = 0sz;

  frame = frame;

  block_pre   = (magic ());
  block_post  = (magic ());
  block_frame = (magic ());

  setup = (magic ());
  teardown = (magic ());

  kpre  = (fun _ar -> kpre);
  kpost = (fun _ar -> kpost);

  block_setup = magic();
  block_teardown = magic();

  f = (fun _ear -> f);
}

inline_for_extraction noextract
let km1_as_kmn (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
     : kernel_desc_m_n full_pre full_post
= let open k <: kernel_desc_m_1 full_pre full_post in
  {
  nblk;
  nthr = 1sz;

  frame = frame;

  block_pre   = (magic ());
  block_post  = (magic ());
  block_frame = (magic ());

  setup = (magic ());
  teardown = (magic ());

  kpre  = (fun bid _tid -> kpre bid);
  kpost = (fun bid _tid -> kpost bid);

  block_setup = magic();
  block_teardown = magic();

  f = (fun bid _tid -> admit(); f bid);
}

inline_for_extraction noextract
let k1n_as_kmn (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
     : kernel_desc_m_n full_pre full_post
= let open k <: kernel_desc_1_n full_pre full_post in
  {
  nblk = 1sz;
  nthr;

  frame = frame;

  block_pre   = (magic ());
  block_post  = (magic ());
  block_frame = (magic ());

  setup = (magic ());
  teardown = (magic ());

  kpre  = (fun _bid -> kpre);
  kpost = (fun _bid -> kpost);

  block_setup = magic();
  block_teardown = magic();

  f = (fun _bid tid () -> Kuiper.Frame.frame_right1 _ (f tid));
}

inline_for_extraction noextract
let k11_as_k1n (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
     : kernel_desc_1_n full_pre full_post
= let open k <: kernel_desc_1_1 full_pre full_post in
  {
  nthr = 1sz;

  frame = emp;

  block_setup = (magic ());
  block_teardown = (magic ());

  kpre  = (fun _tid -> full_pre);
  kpost = (fun _tid -> full_post);

  f = (fun _tid () -> Kuiper.Frame.frame_right1 _ f);
}

[@@coercion]
inline_for_extraction noextract
let km1_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
     : kernel_desc     full_pre full_post
  = k |> km1_as_kmn |> kmn_as_kfull

[@@coercion]
inline_for_extraction noextract
let k1n_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
     : kernel_desc     full_pre full_post
  = k |> k1n_as_kmn |> kmn_as_kfull

[@@coercion]
inline_for_extraction noextract
let k11_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
     : kernel_desc     full_pre full_post
  = k |> k11_as_k1n |> k1n_as_kmn |> kmn_as_kfull
