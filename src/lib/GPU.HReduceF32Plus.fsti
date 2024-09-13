module GPU.HReduceF32Plus

(* This module is specialized to U64 and addition.

The only admits are a boring fact about associativity of add_mod (unsure why
it's not already trivial in F* ) and lack of overflow of the iteration counter.
This last thing should fall out from the fact that any the size of an array must
fit in a sizet, and the log of that size even more so. *)

#lang-pulse

open GPU
open GPU.Barrier.RPM
open GPU.Math
open GPU.Seq.Common

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module F32 = GPU.Float32

let size : sz = 1024sz

(* no polymorphism, but at least keep the definitions here *)
let ety = f32
inline_for_extraction noextract let op = F32.add
inline_for_extraction noextract let neu = F32.zero

(* using seq_fold_left op neu directly in pulse code blows up
in many colorful ways. Probably the refinment of op? Anyway, 
specialize it here. *)
let sum (s : seq ety) : GTot ety =
  seq_fold_left op neu s
  
(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
let gpu_pts_to_slice_sum_inner
  (#sz:nat)
  (r : gpu_array ety sz)
  (i j :nat)
  (v : seq ety)
  (s : seq ety)
: slprop
= gpu_pts_to_array_slice r i j s
  ** pure (i < j /\ j <= sz /\
           Seq.length v = sz /\
           Seq.length s = j - i /\
           Seq.index s 0 == sum (Seq.slice v i j))

let gpu_pts_to_slice_sum
  (#sz:nat)
  ([@@@equate_strict] r: gpu_array ety sz)
  (i j:nat)
  (v: seq ety)
: slprop
= if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s)

// Barrier

val barrier_matrix (nth: nat) (r : gpu_array ety nth) (v: seq ety) (it from to: nat)
  : slprop

[@@pulse_unfold]
let kpre (nth: nat) (a : gpu_array ety nth) (s : erased (seq ety))
  (#_: squash (Seq.length s == nth)) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_array_slice a tid (tid+1) seq![Seq.index s tid]

[@@pulse_unfold]
let kpost (nth: nat) (a : gpu_array ety nth) (s : erased (seq ety))
  (#_: squash (Seq.length s == nth)) (tid:nat{tid < nth})
  : slprop =
    if_ (tid = 0) (gpu_pts_to_slice_sum a 0 nth s)

[@@ CPrologue "__device__"]
inline_for_extraction
fn reduce
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth a s) 0 (tidx_x etid) **
    kpre nth a s (thread_index etid)
  ensures 
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth a s) it (tidx_x etid)) **
    kpost nth a s (thread_index etid)

[@@ CPrologue "__global__"]
inline_for_extraction
fn k_reduce
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (Seq.length s == nth))
  (etid : erased tid_t { (gdim_x etid <: nat) == 1ul /\ (bdim_x etid <: nat) == SZ.sizet_to_uint32 nth })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nth (barrier_matrix nth a s) 0 (tidx_x etid) **
    kpre nth a s (thread_index etid)
  ensures
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nth (barrier_matrix nth a s) it (tidx_x etid)) **
    kpost nth a s (thread_index etid)
