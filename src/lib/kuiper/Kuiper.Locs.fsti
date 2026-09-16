module Kuiper.Locs

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Lib.Send
module T = FStar.Tactics.V2
include Kuiper.Locs.Base {
  gpu_of, gpu_of_idem, gpu_id_of, gpu_id_loc, gpu_id_loc_lemma, gpu_loc,
  block_of, block_of_idem, block_id_of, thread_id_of, block_of_same_gpu,
  is_cpu_loc, is_cpu_loc_single_process
}

inline_for_extraction let () = ()

instance send_across_if_send_across_gpu (p:slprop) (sp:is_send_across gpu_of p)
: is_send_across block_of p
= fun l0 l1 ->
    block_of_same_gpu l0 l1;
    sp l0 l1

instance cond_sendable (b:bool) (p q:slprop)
      (vis:loc_id -> 'a)
      (f:is_send_across vis p)
      (g:is_send_across vis q)
: is_send_across vis (Pulse.Lib.Primitives.cond b p q)
= fun l0 l1 -> if b then f l0 l1 else g l0 l1

(* Token for being in GPU code *)
[@@no_mkeys]
let gpu (#[T.exact (`0)] gpu_id:int) : slprop =
  exists* (l:loc_id). loc l ** pure (gpu_of l == gpu_id_loc gpu_id /\ gpu_id_of l == gpu_id)

(* Tokens for the current block/thread, including the actual launch
   dimensions. Keep the arguments as int for inference; the implementation
   ties them to the size_t dimensions of the current location. *)
[@@no_mkeys]
val block_id (nblk bid : int) : slprop

[@@no_mkeys]
val thread_id (nthr tid : int) : slprop

ghost
fn block_id_agree (nblk bid nblk' bid' : int)
  preserves block_id nblk bid ** block_id nblk' bid'
  ensures pure (nblk == nblk' /\ bid == bid')

ghost
fn thread_id_agree (nthr tid nthr' tid' : int)
  preserves thread_id nthr tid ** thread_id nthr' tid'
  ensures pure (nthr == nthr' /\ tid == tid')

(* Token for being in CPU code *)
let cpu : slprop = exists* l. loc l ** pure (is_cpu_loc l)

ghost
fn map_loc (loc:loc_id) (#p #q:slprop) (f : ghost fn () requires p ensures q)
  requires on loc p
  ensures  on loc q
{
  ghost_impersonate loc (on loc p) (on loc q) fn () {
    on_elim p;
    f();
    on_intro q;
  }
}

ghost
fn reduce_with_steps (p:slprop) (steps:_)
requires p
ensures norm steps p
{
  norm_spec steps p;
  rewrite p as (norm steps p);
}

ghost
fn elim_gpu (p : slprop) {| sendable: is_send_across gpu_of p |} ()
  preserves gpu
  requires on gpu_loc p
  ensures p
{
  unfold gpu;
  with l. assert (loc l);
  is_send_across_elim gpu_of p #sendable #gpu_loc l;
  on_elim p;
  fold gpu;
}

ghost
fn intro_gpu (p : slprop) {| sendable: is_send_across gpu_of p |} ()
  preserves gpu
  requires p
  ensures on gpu_loc p
{
  unfold gpu;
  with l. assert (loc l);
  on_intro p;
  is_send_across_elim gpu_of p #sendable #l gpu_loc;
  fold gpu;
}
