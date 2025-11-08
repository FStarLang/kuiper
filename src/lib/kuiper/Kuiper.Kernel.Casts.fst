module Kuiper.Kernel.Casts
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.Array
open Kuiper.SHMem
open FStar.Ghost

ghost
fn kmn_as_kfull_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
  (sh : c_shmems [])
  (bid: natlt k.nblk)
  ()
  norewrite
  requires
    can_create_barrier k.nthr **
    live_c_shmems sh **
    k.block_pre bid
  ensures
    consumed_can_create_barrier **
    (forall+ (i : natlt k.nthr). k.kpre bid i) **
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
  (bid: natlt k.nblk)
  ()
  norewrite
  requires
    (forall+ (i : natlt k.nthr). k.kpost bid i) **
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

inline_for_extraction noextract
fn adapt_kn_as_kmn
  (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
  (#_ : squash (SZ.fits (k.nthr + 1024)))
  (bid : szlt (sdivup k.nthr 1024sz))
  (tid : szlt 1024sz)
  ()
  requires
    gpu **
    pad_f (sdivup k.nthr 1024sz * 1024) k.kpre (1024 * bid + tid) **
    thread_id 1024sz bid tid **
    block_id (sdivup k.nthr 1024sz) bid
  ensures
    gpu **
    pad_f (sdivup k.nthr 1024sz * 1024) k.kpost (1024 * bid + tid) **
    thread_id 1024sz bid tid **
    block_id (sdivup k.nthr 1024sz) bid
{
  open FStar.SizeT;
  let gid = 1024sz *^ bid +^ tid;
  if (gid <^ k.nthr) {
    rewrite
      (if 1024 * bid + tid < k.nthr then k.kpre (1024 * bid + tid) else emp)
    as
      k.kpre gid;
    let f = k.f;
    f gid ();
    rewrite
      k.kpost gid
    as
      (if 1024 * bid + tid < k.nthr then k.kpost (1024 * bid + tid) else emp);
    ()
  } else {
    rewrite
      (if 1024 * bid + tid < k.nthr then k.kpre (1024 * bid + tid) else emp)
    as
      emp;
    rewrite
      emp
    as
      (if 1024 * bid + tid < k.nthr then k.kpost (1024 * bid + tid) else emp);
    ()
  }
}

let pad_kn
  (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
  (p : natlt k.nthr -> slprop)
  (bid : natlt (sdivup k.nthr 1024sz))
  (tid : natlt 1024sz)
  : slprop
= pad_f (sdivup k.nthr 1024sz * 1024) p (1024 * bid + tid)

ghost
fn pad_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
  ()
  norewrite
  requires
    full_pre
  ensures
    (forall+ (bid : natlt (sdivup k.nthr 1024sz)).
      forall+ (tid : natlt 1024sz).
        pad_kn k k.kpre bid tid) **
    k.frame
{
  let setup = k.setup;
  setup ();

  (* We now factor that forall+ into chunks of blocksz.
  But first, we gotta pad it with empties. *)

  forevery_pad k.nthr (SZ.v (sdivup k.nthr 1024sz) * 1024) _;
  forevery_factor
    ((sdivup k.nthr 1024sz) * 1024)
    (sdivup k.nthr 1024sz)
    1024
    (pad_f ((sdivup k.nthr 1024sz) * 1024) k.kpre);

  (* Convince Z3 *)
  forevery_ext_2
    #(natlt (sdivup k.nthr 1024sz))
    #(natlt (1024sz))
    _
    (fun bid tid ->
        pad_kn k k.kpre bid tid);
}

ghost
fn pad_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (sdivup k.nthr 1024sz)).
      forall+ (tid : natlt 1024sz).
        pad_kn k k.kpost bid tid) **
    k.frame
  ensures
    full_post
{
  forevery_ext_2
    #(natlt (sdivup k.nthr 1024sz))
    #(natlt (1024sz))
    _
    (fun bid tid ->
        pad_f ((sdivup k.nthr 1024sz) * 1024) k.kpost (bid * 1024 + tid));
  forevery_unfactor
    ((sdivup k.nthr 1024sz) * 1024)
    (sdivup k.nthr 1024sz)
    1024
    (pad_f ((sdivup k.nthr 1024sz) * 1024) k.kpost);

  forevery_unpad k.nthr ((sdivup k.nthr 1024sz) * 1024) _;

  let teardown = k.teardown;
  teardown ();
}

ghost
fn kn_as_kmn_block_setup (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
  (bid : natlt (sdivup k.nthr 1024sz))
  norewrite
  requires
    can_create_barrier 1024 **
    (forall+ (tid : natlt 1024sz). pad_kn k k.kpre bid tid)
  ensures
    consumed_can_create_barrier **
    (forall+ (tid : natlt 1024sz). pad_kn k k.kpre bid tid) **
    emp
{
  no_mk_barrier ();
}

inline_for_extraction noextract
let kn_as_kmn (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
     : kernel_desc_m_n full_pre full_post
= //let open k <: kernel_desc_n full_pre full_post in; this causes pad_setup to fail!
  {
  nblk = sdivup k.nthr 1024sz;
  nthr = 1024sz;

  frame = k.frame;

  block_pre   = (fun bid -> forall+ (tid : natlt 1024sz). pad_kn k k.kpre bid tid);
  block_post  = (fun bid -> forall+ (tid : natlt 1024sz). pad_kn k k.kpost bid tid);
  block_frame = (fun bid -> emp);

  setup    = pad_setup k;
  teardown = pad_teardown k;

  kpre  = pad_kn k k.kpre;
  kpost = pad_kn k k.kpost;

  block_setup    = kn_as_kmn_block_setup k;
  block_teardown = (fun bid -> Kuiper.Frame.emp_elim_r ());

  f = adapt_kn_as_kmn k #();
}

inline_for_extraction noextract
let kn_as_kfull (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
     : kernel_desc full_pre full_post
  = k |> kn_as_kmn |> kmn_as_kfull

ghost
fn km1_as_kmn_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
  (bid: natlt k.nblk)
  norewrite
  requires
    can_create_barrier 1sz **
    k.kpre bid
  ensures
    consumed_can_create_barrier **
    (forall+ (i : natlt 1sz). k.kpre bid) **
    emp
{
  no_mk_barrier ();
  forevery_singleton_intro #(natlt 1sz) (fun _ -> k.kpre bid);
}

ghost
fn km1_as_kmn_block_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
  (bid: natlt k.nblk)
  norewrite
  requires
    (forall+ (i : natlt 1sz). k.kpost bid) **
    emp
  ensures
    k.kpost bid
{
  forevery_singleton_elim #(natlt 1sz) _;
}

inline_for_extraction noextract
fn frame_2
  (e #p0 #p1 #q0 #q1 #r0 #r1 : slprop)
  (f : unit -> stt unit (requires p0 ** q0 ** r0) (fun _ -> p1 ** q1 ** r1))
  ()
  norewrite
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

  f = (fun bid _tid -> frame_2 (thread_id 1sz bid _tid) (k.f bid));
}

ghost
fn k1n_as_kmn_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
  ()
  norewrite
  requires
    full_pre
  ensures
    (forall+ (bid: natlt 1sz). full_pre) **
    emp
{
  forevery_singleton_intro #(natlt 1sz) (fun _ -> full_pre);
}


ghost
fn k1n_as_kmn_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
  ()
  norewrite
  requires
    (forall+ (bid: natlt 1sz). full_post) **
    emp
  ensures
    full_post
{
  forevery_singleton_elim #(natlt 1sz) (fun _ -> full_post);
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

  f = (fun _bid tid () -> 
        Kuiper.Frame.frame_3left (block_id 1 0) (f tid));
}

ghost
fn k11_as_k1n_block_setup
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
  ()
  norewrite
  requires
    can_create_barrier 1sz **
    full_pre
  ensures
    consumed_can_create_barrier **
    (forall+ (bid: natlt 1sz). full_pre) **
    emp
{
  no_mk_barrier ();
  forevery_singleton_intro #(natlt 1sz) (fun _ -> full_pre);
}

ghost
fn k11_as_k1n_block_teardown
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
  ()
  norewrite
  requires
    (forall+ (bid: natlt 1sz). full_post) **
    emp
  ensures
    full_post
{
  forevery_singleton_elim #(natlt 1sz) (fun _ -> full_post);
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
