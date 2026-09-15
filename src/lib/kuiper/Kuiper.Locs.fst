module Kuiper.Locs

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Locs.Base

[@@no_mkeys]
let block_id (nblk bid : int) : slprop =
  exists* (l:loc_id). loc l **
    pure (grid_dim_of l == nblk /\
          block_of l == block_id_loc (launch_of l) bid /\
          block_id_of l == bid)

[@@no_mkeys]
let thread_id (nthr tid : int) : slprop =
  exists* (l:loc_id). loc l **
    pure (block_dim_of l == nthr /\ thread_id_of l == tid)

ghost
fn block_id_agree (nblk bid nblk' bid' : int)
  preserves block_id nblk bid ** block_id nblk' bid'
  ensures pure (nblk == nblk' /\ bid == bid')
{
  unfold block_id nblk bid;
  with l. _;
  unfold block_id nblk' bid';
  with l'. _;
  loc_gather l #l';
  rewrite each l' as l;
  fold block_id nblk bid;
  fold block_id nblk' bid';
}

ghost
fn thread_id_agree (nthr tid nthr' tid' : int)
  preserves thread_id nthr tid ** thread_id nthr' tid'
  ensures pure (nthr == nthr' /\ tid == tid')
{
  unfold thread_id nthr tid;
  with l. _;
  unfold thread_id nthr' tid';
  with l'. _;
  loc_gather l #l';
  rewrite each l' as l;
  fold thread_id nthr tid;
  fold thread_id nthr' tid';
}
