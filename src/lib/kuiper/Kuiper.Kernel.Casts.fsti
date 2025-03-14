module Kuiper.Kernel.Casts
#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Kuiper.Common
open Kuiper.ForEvery
open Kuiper.Base
open Kuiper.Kernel.Desc
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

  block_pre  : natlt nblk -> slprop;
  block_post : natlt nblk -> slprop;

  block_frame : natlt nblk -> slprop;

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

  kpre  : natlt nblk -> natlt nthr -> slprop;
  kpost : natlt nblk -> natlt nthr -> slprop;

  block_setup : (
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        block_pre bid)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (i : natlt nthr). kpre bid i) **
        block_frame bid)
  );

  block_teardown : (
    (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpre bid i) **
        block_frame bid)
      (ensures fun _ ->
        block_pre bid)
  );

  f : (
    ebid : enatlt nblk ->
    etid : enatlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre ebid etid **
         thread_id nthr etid **
         block_id nblk ebid)
      (ensures fun _ ->
         gpu **
         kpost ebid etid **
         thread_id nthr etid **
         block_id nblk ebid)
  );
}

(* 1xN, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_1_n (full_pre : slprop) (full_post : slprop) = {
  nthr : (x : SZ.t { 0 < x /\ x <= max_threads });

  frame : slprop;

  kpre  : natlt nthr -> slprop;
  kpost : natlt nthr -> slprop;

  block_setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        block_setup_tok nthr **
        full_pre)
      (ensures fun _ ->
        block_setup_tok nthr **
        (forall+ (i : natlt nthr). kpre i) **
        frame)
  );

  block_teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nthr). kpost i) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  f : (
    etid : enatlt nthr ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre etid **
         thread_id nthr etid)
      (ensures fun _ ->
         gpu **
         kpost etid **
         thread_id nthr etid)
  );
}

(* Mx1, no shared memory *)
noeq
inline_for_extraction noextract
type kernel_desc_m_1 (full_pre : slprop) (full_post : slprop) = {
  nblk : (x : SZ.t { 0 < x /\ x <= max_blocks });

  frame : slprop;

  kpre  : natlt nblk -> slprop;
  kpost : natlt nblk -> slprop;

  setup : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        full_pre)
      (ensures fun _ ->
        (forall+ (i : natlt nblk). kpre i) **
        frame)
  );

  teardown : (
    unit ->
    stt_ghost unit emp_inames
      (requires
        (forall+ (i : natlt nblk). kpost i) **
        frame)
      (ensures fun _ ->
        full_post)
  );

  f : (
    ebid : enatlt nblk ->
    unit ->
    stt unit
      (requires
         gpu **
         kpre ebid **
         block_id nblk ebid)
      (ensures fun _ ->
         gpu **
         kpost ebid **
         block_id nblk ebid)
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
