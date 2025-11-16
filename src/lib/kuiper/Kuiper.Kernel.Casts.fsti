module Kuiper.Kernel.Casts
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.SizeT
module SZ = Kuiper.SizeT

(* PLEASE NOTE: the types here are very order sensitive. Make
sure to keep uniformity if you change anything. For instance,
all the frames are on the right most component of a star
so they can be easily intro/elim'd when empty. *)

(* MxN, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_m_n (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { x <= max_blocks });
  nthr : (x : SZ.t { x <= max_threads });

  frame : slprop;

  block_pre  : (bid : natlt nblk) -> slprop;
  block_post : (bid : natlt nblk) -> slprop;

  block_frame : (bid : natlt nblk) -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires full_pre)
      (ensures fun _ ->
        (forall+ (bid : natlt nblk). block_pre bid) **
        frame)
  );
  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (bid : natlt nblk). block_post bid) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  kpre  : (bid : natlt nblk) -> (tid : natlt nthr) -> slprop;
  kpost : (bid : natlt nblk) -> (tid : natlt nthr) -> slprop;

  block_setup : (
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        can_create_barrier nthr **
        block_pre bid)
      (ensures fun _ ->
        consumed_can_create_barrier **
        (forall+ (i : natlt nthr). kpre bid i) **
        block_frame bid)
  );

  block_teardown : (
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpost bid i) **
        block_frame bid)
      (ensures fun _ ->
        block_post bid)
  );

  f : (
    bid : szlt nblk ->
    tid : szlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre bid tid **
         thread_id nthr tid **
         block_id nblk bid)
      (ensures fun _ ->
         gpu **
         kpost bid tid **
         thread_id nthr tid **
         block_id nblk bid)
  );

  block_pre_sendable: (i:natlt nblk -> is_send_across gpu_of (block_pre i));

  block_post_sendable: (i:natlt nblk -> is_send_across gpu_of (block_post i));

  kpre_sendable: (i:natlt nblk -> j:natlt nthr -> is_send_across block_of (kpre i j));

  kpost_sendable: (i:natlt nblk -> j:natlt nthr -> is_send_across block_of (kpost i j));

}

(* N independent jobs, no shared memory, to be broken up
into blocks/threads as needed. *)
noeq
inline_for_extraction noextract
type kernel_desc_n (full_pre : slprop) (full_post : slprop) = {
  nthr : (x : SZ.t { x <= max_blocks * max_threads });

  frame : slprop;

  kpre  : (tid : natlt nthr) -> slprop;
  kpost : (tid : natlt nthr) -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        full_pre)
      (ensures fun _ ->
        (forall+ (tid : natlt nthr). kpre tid) **
        frame)
  );

  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (tid : natlt nthr). kpost tid) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  f : (
    tid : szlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre tid)
      (ensures fun _ ->
         gpu **
         kpost tid)
  );

  kpre_sendable: (j:natlt nthr -> is_send_across gpu_of (kpre j));
  kpost_sendable: (j:natlt nthr -> is_send_across gpu_of (kpost j));

}


(* 1xN, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_1_n (full_pre : slprop) (full_post : slprop) = {
  nthr : (x : SZ.t { x <= max_threads });

  frame : slprop;

  kpre  : (tid : natlt nthr) -> slprop;
  kpost : (tid : natlt nthr) -> slprop;

  block_setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        can_create_barrier nthr **
        full_pre)
      (ensures fun _ ->
        consumed_can_create_barrier **
        (forall+ (tid : natlt nthr). kpre tid) **
        frame)
  );

  block_teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (tid : natlt nthr). kpost tid) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  f : (
    tid : szlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre tid **
         thread_id nthr tid)
      (ensures fun _ ->
         gpu **
         kpost tid **
         thread_id nthr tid)
  );

  full_pre_sendable: is_send_across gpu_of full_pre;
  full_post_sendable: is_send_across gpu_of full_post;
  kpre_sendable: (j:natlt nthr -> is_send_across block_of (kpre j));
  kpost_sendable: (j:natlt nthr -> is_send_across block_of (kpost j));
}

(* Mx1, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_m_1 (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { x <= max_blocks });

  frame : slprop;

  kpre  : (bid : natlt nblk) -> slprop;
  kpost : (bid : natlt nblk) -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        full_pre)
      (ensures fun _ ->
        (forall+ (bid : natlt nblk). kpre bid) **
        frame)
  );

  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (bid : natlt nblk). kpost bid) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  f : (
    bid : szlt nblk ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre bid **
         block_id nblk bid)
      (ensures fun _ ->
         gpu **
         kpost bid **
         block_id nblk bid)
  );

  kpre_sendable: (j:natlt nblk -> is_send_across gpu_of (kpre j));
  kpost_sendable: (j:natlt nblk -> is_send_across gpu_of (kpost j));
}

(* 1x1, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_1_1 (full_pre : slprop) (full_post : slprop) = {
  f : (
    unit ->
    stt unit
      (requires
         gpu **
         full_pre)
      (ensures fun _ ->
         gpu **
         full_post)
  );

  full_pre_sendable: is_send_across gpu_of full_pre;
  full_post_sendable: is_send_across gpu_of full_post;
}

[@@coercion]
inline_for_extraction noextract
val kmn_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
     : kernel_desc full_pre full_post

[@@coercion]
inline_for_extraction noextract
val kn_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_n full_pre full_post)
     : kernel_desc full_pre full_post

inline_for_extraction noextract
val km1_as_kmn
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
     : kernel_desc_m_n full_pre full_post

inline_for_extraction noextract
val k1n_as_kmn
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
     : kernel_desc_m_n full_pre full_post

inline_for_extraction noextract
val k11_as_k1n
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
     : kernel_desc_1_n full_pre full_post

[@@coercion]
inline_for_extraction noextract
val km1_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_1 full_pre full_post)
     : kernel_desc     full_pre full_post

[@@coercion]
inline_for_extraction noextract
val k1n_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_n full_pre full_post)
     : kernel_desc     full_pre full_post

[@@coercion]
inline_for_extraction noextract
val k11_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_1_1 full_pre full_post)
     : kernel_desc     full_pre full_post
