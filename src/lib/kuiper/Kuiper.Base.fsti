module Kuiper.Base

#lang-pulse

open FStar.Ghost
open Pulse.Lib.Core
open Pulse.Main
module SZ = Kuiper.SizeT
module T = FStar.Tactics.V2
val is_thread_loc (l:loc_id) : prop
let thread_loc = l:loc_id { is_thread_loc l }

val gpu_of: loc_id -> loc_id
val gpu_of_idem (l:loc_id) : Lemma (gpu_of (gpu_of l) == l)

val block_of : loc_id -> loc_id
val block_of_idem (l:loc_id) : Lemma (block_of (block_of l) == l)

val gpu_id_loc (gpu_id:int) : loc_id

val block_id_loc (#[T.exact (`0)]gpu_id:int) (bid:int)
: l:loc_id { gpu_of l == gpu_id_loc gpu_id }

val thread_id_loc (#[T.exact (`0)]gpu_id:int) (bid tid:int)
: l:loc_id { block_of l == block_id_loc #gpu_id bid /\ gpu_of l == gpu_id_loc gpu_id } 

(* Token for being in GPU code *)
[@@no_mkeys]
let gpu (#[T.exact (`0)] gpu_id:int) : slprop =
  exists* (l:loc_id). loc l ** pure (gpu_of l == gpu_id_loc gpu_id)

(* Token given to a particular block within a grid. Both here
and in thread_id, the first argument is always positive
when this resource is actually live, but not placing that refinement
here helps with inference in some places. *)
[@@no_mkeys]
let block_id (#[T.exact (`0)]gpu_id:int) (nblk : int) (bid : int) : slprop =
  exists* (l:loc_id). loc l ** pure (block_of l == block_id_loc #gpu_id bid)

(* Token given to a particular thread within a block *)
[@@no_mkeys]
let thread_id (nthr : int) (#[T.exact (`0)]gpu_id:int) (bid tid : int) : slprop =
  loc (thread_id_loc #gpu_id bid tid)

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
  preserves thread_id 'nthr 'bid 'tid
  returns   x : SZ.t
  ensures   pure (SZ.v x == 'nthr)
