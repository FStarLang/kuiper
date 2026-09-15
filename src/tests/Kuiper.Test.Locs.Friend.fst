module Kuiper.Test.Locs.Friend

friend Kuiper.Locs

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.SizeT
module L = Kuiper.Locs.Base

(* These successful checks exercise the definitions through the friend
   boundary, so the rejection tests below cannot pass just due to opacity. *)
ghost
fn inspect_thread_count (nthr tid:int)
  preserves thread_id nthr tid
  ensures pure (0 <= nthr)
{
  unfold thread_id nthr tid;
  with l. _;
  assert pure (L.block_dim_of l == nthr);
  fold thread_id nthr tid;
}

ghost
fn inspect_block_count (nblk bid:int)
  preserves block_id nblk bid
  ensures pure (0 <= nblk)
{
  unfold block_id nblk bid;
  with l. _;
  assert pure (L.grid_dim_of l == nblk);
  fold block_id nblk bid;
}

let different_launch_dimensions () : Lemma
  (let ctx32 : L.launch_context = { launch_gpu = 0; launch_nblk = 1sz; launch_nthr = 32sz } in
   let ctx64 : L.launch_context = { launch_gpu = 0; launch_nblk = 1sz; launch_nthr = 64sz } in
   L.block_id_loc ctx32 0 =!= L.block_id_loc ctx64 0 /\
   L.thread_id_loc ctx32 0 0 =!= L.thread_id_loc ctx64 0 0)
  = ()

[@@expect_failure [19]]
ghost
fn cannot_change_thread_count ()
  requires thread_id 32 0
  ensures thread_id 64 0
{
  rewrite thread_id 32 0 as thread_id 64 0;
}

[@@expect_failure [19]]
ghost
fn cannot_change_block_count ()
  requires block_id 1 0
  ensures block_id 2 0
{
  rewrite block_id 1 0 as block_id 2 0;
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn cannot_derive_false_from_bdim ()
  preserves thread_id 32 0
  ensures pure False
{
  rewrite thread_id 32 0 as thread_id (-1) 0;
  let n = get_bdim ();
  assert pure (v n == -1);
}

[@@expect_failure [19]]
inline_for_extraction noextract
fn cannot_derive_false_from_gdim ()
  preserves block_id 1 0
  ensures pure False
{
  rewrite block_id 1 0 as block_id (-1) 0;
  let n = get_gdim ();
  assert pure (v n == -1);
}
