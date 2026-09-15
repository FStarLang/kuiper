module Kuiper.Locs.Base

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Array.Core { visibility }
module SZ = FStar.SizeT

(* Location primitives for the GPU model. A launch's dimensions are part of
   its locations, so even code that unfolds the public tokens cannot change
   the dimensions of the current location. *)
[@@erasable]
noeq
type launch_context = {
  launch_gpu : int;
  launch_nblk : SZ.t;
  launch_nthr : SZ.t;
}

val launch_of : loc_id -> GTot launch_context

let grid_dim_of (l:loc_id) : GTot nat = SZ.v (launch_of l).launch_nblk
let block_dim_of (l:loc_id) : GTot nat = SZ.v (launch_of l).launch_nthr

val gpu_of : visibility
// Visibility maps are projections to a representative, not involutions.
val gpu_of_idem (l:loc_id) : Lemma (gpu_of (gpu_of l) == gpu_of l)
val gpu_id_of : loc_id -> GTot int

val block_of : visibility
val block_of_idem (l:loc_id) : Lemma (block_of (block_of l) == block_of l)
val block_id_of : loc_id -> GTot int
val thread_id_of : loc_id -> GTot int

val gpu_id_loc (gpu_id:int) : l:loc_id { gpu_of l == l }
val gpu_id_loc_lemma (gpu_id:int) : Lemma
  (gpu_id_of (gpu_id_loc gpu_id) == gpu_id)
let gpu_loc = gpu_id_loc 0

val block_id_loc (ctx:launch_context) (bid:int)
: l:loc_id { gpu_of l == gpu_id_loc ctx.launch_gpu /\ launch_of l == ctx }
val block_id_loc_lemma (ctx:launch_context) (bid:int) : Lemma
  (let l = block_id_loc ctx bid in
    block_id_of l == bid /\ block_of l == l /\ gpu_id_of l == ctx.launch_gpu)

val thread_id_loc (ctx:launch_context) (bid tid:int)
: l:loc_id {
    block_of l == block_id_loc ctx bid /\
    gpu_of l == gpu_id_loc ctx.launch_gpu /\
    launch_of l == ctx
  }
val thread_id_loc_lemma (ctx:launch_context) (bid tid:int) : Lemma
  (let l = thread_id_loc ctx bid tid in
    thread_id_of l == tid /\ block_id_of l == bid /\ gpu_id_of l == ctx.launch_gpu)

// Locations that agree on their blocks are on the same GPU.
val block_of_same_gpu (l0 l1:_{block_of l0 == block_of l1})
: Lemma (gpu_of l0 == gpu_of l1)

val is_cpu_loc (l:loc_id) : prop
val is_cpu_loc_single_process (l0 l1:loc_id)
: Lemma (is_cpu_loc l0 /\ is_cpu_loc l1 ==> process_of l0 == process_of l1)
