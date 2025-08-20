module Kuiper.Kernel.Casts
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.Array
open Kuiper.SHMem

ghost
fn kmn_as_kfull_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
  (sh : c_shmems [])
  (bid: szlt k.nblk)
  ()
requires
  block_setup_tok k.nthr **
  live_c_shmems sh **
  k.block_pre bid
ensures
  block_setup_tok k.nthr **
  (forall+ (i : szlt k.nthr). k.kpre bid i) **
  (k.block_frame bid **
    live_c_shmems sh)
{
  let f = k.block_setup;
  f bid;
}

ghost
fn kmn_as_kfull_block_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
  (sh : c_shmems [])
  (bid: szlt k.nblk)
  ()
requires
  (forall+ (i : szlt k.nthr). k.kpost bid i) **
  (k.block_frame bid ** live_c_shmems sh)
ensures
  live_c_shmems sh **
  k.block_post bid
{
  let f = k.block_teardown;
  f bid;
}

[@@coercion]
inline_for_extraction noextract
let kmn_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
     : kernel_desc     full_pre full_post
= let open k <: kernel_desc_m_n full_pre full_post in
  {
  nblk=k.nblk;
  nthr=k.nthr;

  shmems_desc = [];

  frame = k.frame;

  block_pre   = k.block_pre;
  block_post  = k.block_post;
  block_frame = (fun sh bid -> k.block_frame bid ** live_c_shmems sh);

  setup = k.setup;
  teardown = k.teardown;

  kpre  = (fun _ar -> k.kpre);
  kpost = (fun _ar -> k.kpost);

  block_setup = kmn_as_kfull_block_setup k;
  block_teardown = kmn_as_kfull_block_teardown k;

  f = (fun _ear -> f);
}

ghost
fn km1_as_kmn_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
  (bid: szlt k.nblk)
requires
  block_setup_tok 1sz **
  k.kpre bid
ensures
  block_setup_tok 1sz **
  (forall+ (i : szlt 1sz). k.kpre bid) **
  emp
{
  forevery_singleton_intro #(szlt 1sz) (fun _ -> k.kpre bid);
}

ghost
fn km1_as_kmn_block_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
  (bid: szlt k.nblk)
requires
  (forall+ (i : szlt 1sz). k.kpost bid) **
  emp
ensures
  k.kpost bid
{
  forevery_singleton_elim #(szlt 1sz) _;
}

inline_for_extraction noextract
fn frame_2
  (e #p0 #p1 #q0 #q1 #r0 #r1 : slprop)
  (f : unit -> stt unit (requires p0 ** q0 ** r0) (fun _ -> p1 ** q1 ** r1))
  ()
requires p0 ** q0 ** e ** r0
ensures  p1 ** q1 ** e ** r1
{
  f ()
}

inline_for_extraction noextract
let km1_as_kmn (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
     : kernel_desc_m_n full_pre full_post
= let open k <: kernel_desc_m_1 full_pre full_post in
  {
  nblk=k.nblk;
  nthr = 1sz;

  frame = k.frame;

  block_pre   = k.kpre;
  block_post  = k.kpost;
  block_frame = (fun _ -> emp);

  setup = k.setup;
  teardown = k.teardown;

  kpre  = (fun bid _tid -> k.kpre bid);
  kpost = (fun bid _tid -> k.kpost bid);

  block_setup = km1_as_kmn_block_setup k;
  block_teardown = km1_as_kmn_block_teardown k;

  f = (fun bid _tid -> frame_2 (thread_id 1sz _tid) (k.f bid));
}

ghost
fn k1n_as_kmn_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
  ()
requires
  full_pre
ensures
  (forall+ (bid: szlt 1sz). full_pre) **
  emp
{
  forevery_singleton_intro #(szlt 1sz) (fun _ -> full_pre);
}


ghost
fn k1n_as_kmn_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
  ()
requires
  (forall+ (bid: szlt 1sz). full_post) **
  emp
ensures
  full_post
{
  forevery_singleton_elim #(szlt 1sz) (fun _ -> full_post);
}

inline_for_extraction noextract
let k1n_as_kmn (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
     : kernel_desc_m_n full_pre full_post
= let open k <: kernel_desc_1_n full_pre full_post in
  {
  nblk = 1sz;
  nthr = k.nthr;

  frame = emp;

  block_pre   = (fun _ -> full_pre);
  block_post  = (fun _ -> full_post);
  block_frame = (fun _ -> k.frame);

  setup = k1n_as_kmn_setup k;
  teardown = k1n_as_kmn_teardown k;

  kpre  = (fun _bid -> kpre);
  kpost = (fun _bid -> kpost);

  block_setup = (fun _ -> k.block_setup ());
  block_teardown = (fun _ -> k.block_teardown ());

  f = (fun _bid tid () -> Kuiper.Frame.frame_3left _ (f tid));
}

ghost
fn k11_as_k1n_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
  ()
requires
  block_setup_tok 1sz **
  full_pre
ensures
  block_setup_tok 1sz **
  (forall+ (bid: szlt 1sz). full_pre) **
  emp
{
  forevery_singleton_intro #(szlt 1sz) (fun _ -> full_pre);
}

ghost
fn k11_as_k1n_block_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
  ()
requires
  (forall+ (bid: szlt 1sz). full_post) **
  emp
ensures
  full_post
{
  forevery_singleton_elim #(szlt 1sz) (fun _ -> full_post);
}

inline_for_extraction noextract
let k11_as_k1n (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
     : kernel_desc_1_n full_pre full_post
= let open k <: kernel_desc_1_1 full_pre full_post in
  {
  nthr = 1sz;

  frame = emp;

  block_setup = k11_as_k1n_block_setup k;
  block_teardown = k11_as_k1n_block_teardown k;

  kpre  = (fun _tid -> full_pre);
  kpost = (fun _tid -> full_post);

  f = (fun _tid () -> Kuiper.Frame.frame_2left _ f);
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
