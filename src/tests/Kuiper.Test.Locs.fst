module Kuiper.Test.Locs

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Base
open Kuiper.SizeT
module L = Kuiper.Locs.Base

(* The abstract tokens still justify the hardware dimension getters. *)
inline_for_extraction noextract
fn read_dimensions ()
  preserves block_id 1 0 ** thread_id 32 0
  returns dim_sum : sz
  ensures pure (v dim_sum == 33)
{
  let nblk = get_gdim ();
  let nthr = get_bdim ();
  nblk +^ nthr
}

ghost
fn equal_dimensions (n m tid:int)
  requires thread_id n tid ** pure (n == m)
  ensures thread_id m tid
{
  rewrite thread_id n tid as thread_id m tid;
}

ghost
fn dimensions_agree (nblk bid nblk' bid' nthr tid nthr' tid':int)
  preserves block_id nblk bid ** block_id nblk' bid'
  preserves thread_id nthr tid ** thread_id nthr' tid'
  ensures pure (nblk == nblk' /\ bid == bid' /\ nthr == nthr' /\ tid == tid')
{
  block_id_agree nblk bid nblk' bid';
  thread_id_agree nthr tid nthr' tid';
}

(* Even a client that imports the location primitives cannot unfold tokens. *)
[@@expect_failure [19]]
let thread_representation_is_hidden () : Lemma
  (thread_id 32 0 ==
    (exists* (l:loc_id). loc l **
      pure (L.block_dim_of l == 32 /\ thread_id_of l == 0)))
  = ()

[@@expect_failure [19]]
let block_representation_is_hidden () : Lemma
  (block_id 1 0 ==
    (exists* (l:loc_id). loc l **
      pure (L.grid_dim_of l == 1 /\
            block_of l == L.block_id_loc (L.launch_of l) 0 /\
            block_id_of l == 0)))
  = ()

(* These rewrites previously succeeded because the dimensions were ignored.
   Check positive dimensions too: a nonnegativity check alone is insufficient. *)
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
