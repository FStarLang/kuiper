module GPU.HReduceF32Plus

#lang-pulse

open GPU
open GPU.Barrier.RPM
open GPU.Math
open GPU.Seq.Common
open GPU.IsReduction

module A = Pulse.Lib.Array
module SZ = FStar.SizeT
module F32 = GPU.Float32

let size : sz = 1024sz

(* no polymorphism, but at least keep the definitions here *)
let ety = f32
inline_for_extraction noextract let op  : ety -> ety -> ety = F32.add
inline_for_extraction noextract let neu : ety = F32.zero

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
           squash (is_reduction neu op (Seq.slice v i j) (Seq.index s 0))) // SQUASH VERY IMPORTANT!!

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
