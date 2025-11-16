module Kuiper.Base

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
include Pulse.Lib.Send
open Pulse.Main
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2
open Pulse.Lib.Array.Core { visibility }

val is_thread_loc (l:loc_id) : prop
let thread_loc = l:loc_id { is_thread_loc l }

val gpu_of: visibility
val gpu_of_idem (l:loc_id) : Lemma (gpu_of (gpu_of l) == l)
val gpu_id_of : loc_id -> GTot int

val block_of : visibility
val block_of_idem (l:loc_id) : Lemma (block_of (block_of l) == l)
val block_id_of : loc_id -> GTot int

val gpu_id_loc (gpu_id:int) : l:loc_id { gpu_of l == l }
val gpu_id_loc_lemma (gpu_id:int) : Lemma
  (let l = gpu_id_loc gpu_id in
    gpu_id_of l == gpu_id
  )
let gpu_loc = gpu_id_loc 0

val block_id_loc (#[T.exact (`0)]gpu_id:int) (bid:int)
: l:loc_id { gpu_of l == gpu_id_loc gpu_id }
val block_id_loc_lemma (#[T.exact (`0)]gpu_id:int) (bid:int) : Lemma
  (let l = block_id_loc #gpu_id bid in
    block_id_of l == bid /\ block_of l == l
  )

val thread_id_loc (#[T.exact (`0)]gpu_id:int) (bid tid:int)
: l:loc_id { block_of l == block_id_loc #gpu_id bid /\ gpu_of l == gpu_id_loc gpu_id }
val thread_id_of (l:loc_id) : GTot int
val thread_id_loc_lemma (#[T.exact (`0)]gpu_id:int) (bid tid:int) : Lemma
  (let l = thread_id_loc #gpu_id bid tid in
    thread_id_of l == tid /\ block_id_of l == bid /\ gpu_id_of l == gpu_id
  )

//locations that agree on their blocks are on the same gpu
val block_of_same_gpu (l0 l1:_{block_of l0 == block_of l1})
: Lemma (gpu_of l0 == gpu_of l1)

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

(* Token given to a particular block within a grid. Both here
and in thread_id, the first argument is always positive
when this resource is actually live, but not placing that refinement
here helps with inference in some places. *)
[@@no_mkeys]
let block_id (nblk : int) (bid : int) : slprop =
  exists* (l:loc_id). loc l ** pure (block_of l == block_id_loc bid /\ block_id_of l == bid)

(* Token given to a particular thread within a block *)
[@@no_mkeys]
let thread_id (nthr : int) (tid : int) : slprop =
  exists* (l:loc_id). loc l ** pure (thread_id_of l == tid)

val is_cpu_loc (l:loc_id) : prop

val is_cpu_loc_single_process (l0 l1:loc_id)
: Lemma (is_cpu_loc l0 /\ is_cpu_loc l1 ==> process_of l0 == process_of l1)

(* Token for being in CPU code *)
let cpu : slprop = exists* l. loc l ** pure (is_cpu_loc l)

(* Token allowing to create a barrier for n threads. Only
   available while in the block_setup of a kernel. *)
[@@no_mkeys]
val can_create_barrier (nthr : nat) : slprop

(* Token marking we have in fact created a barrier, or ditched the
   token.  This makes sure the token is not stashed away somewhere. *)
[@@no_mkeys]
val consumed_can_create_barrier : slprop

(* A function that can be called to consume the token
   without actually creating a barrier. We could also
   define consumed_can_create_barrier as a trivial
   barrier token. *)
ghost
fn no_mk_barrier (#n:nat) ()
  requires can_create_barrier n
  ensures  consumed_can_create_barrier

(* This should be 2^31-1, or 2^30. We constrain this more than normal due to our
hack about interpreting size_t as uint32_t in karamel (see Kuiper.SizeT). When
that is gone, this should be increased. *)
unfold
let max_blocks : erased int = 2097152 // 2^21

(* Help F* *)
let max_blocks_explicit : squash (reveal max_blocks == 2097152 /\ reveal max_blocks == pow2 21) =
  assert_norm (reveal max_blocks == 2097152);
  assert_norm (reveal max_blocks == pow2 21)

(* Hard CUDA limit *)
unfold
let max_threads : erased int = 1024

inline_for_extraction noextract
unfold let warp_sz = 32sz
inline_for_extraction noextract
unfold let warp_size = 32


(* Get a concrete value for the number of blocks (~ gridDim.x) *)
fn get_gdim ()
  preserves block_id 'nblk 'bid
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nblk)

(* Get a concrete value for the number of threads (~ blockDim.x) *)
fn get_bdim ()
  preserves thread_id 'nthr 'tid
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nthr)

ghost
fn map_loc (loc:loc_id) (#p #q:slprop) (f:unit -> stt_ghost unit emp_inames p (fun _ -> q))
requires on loc p
ensures on loc q
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