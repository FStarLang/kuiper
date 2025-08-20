module Kuiper.Kernel.Casts
#lang-pulse

open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc
open Kuiper.SizeT
module SZ = FStar.SizeT

(* PLEASE NOTE: the types here are very order sensitive. Make
sure to keep uniformity if you change anything. For instance,
all the frames are on the right most component of a star
so they can be easily intro/elim'd when empty. *)

(* MxN, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_m_n (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  frame : slprop;

  block_pre  : (bid : szlt nblk) -> slprop;
  block_post : (bid : szlt nblk) -> slprop;

  block_frame : (bid : szlt nblk) -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires full_pre)
      (ensures fun _ ->
        (forall+ (bid : szlt nblk). block_pre bid) **
        frame)
  );
  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (bid : szlt nblk). block_post bid) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  kpre  : (bid : szlt nblk) -> (tid : szlt nthr) -> slprop;
  kpost : (bid : szlt nblk) -> (tid : szlt nthr) -> slprop;

  block_setup : (
    (bid: szlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        block_pre bid)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (i : szlt nthr). kpre bid i) **
        block_frame bid)
  );

  block_teardown : (
    (bid: szlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : szlt nthr). kpost bid i) **
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
}

(* 1xN, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_1_n (full_pre : slprop) (full_post : slprop) = {
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  frame : slprop;

  kpre  : (tid : szlt nthr) -> slprop;
  kpost : (tid : szlt nthr) -> slprop;

  block_setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        full_pre)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (tid : szlt nthr). kpre tid) **
        frame)
  );

  block_teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (tid : szlt nthr). kpost tid) **
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
}

(* Mx1, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_m_1 (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });

  frame : slprop;

  kpre  : (bid : szlt nblk) -> slprop;
  kpost : (bid : szlt nblk) -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        full_pre)
      (ensures fun _ ->
        (forall+ (bid : szlt nblk). kpre bid) **
        frame)
  );

  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (bid : szlt nblk). kpost bid) **
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
}

[@@coercion]
inline_for_extraction noextract
val kmn_as_kfull
  (#full_pre #full_post : slprop)
  (k : kernel_desc_m_n full_pre full_post)
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
