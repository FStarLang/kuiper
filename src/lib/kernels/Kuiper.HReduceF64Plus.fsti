module Kuiper.HReduceF64Plus

#lang-pulse

open Kuiper
open Kuiper.Barrier.RPM
open Kuiper.Math
open Kuiper.Seq.Common
open Kuiper.IsReduction

module SZ = FStar.SizeT
module F64 = Kuiper.Float64

let size : sz = 1024sz

(* no polymorphism, but at least keep the definitions here *)
let ety = f64
inline_for_extraction noextract let op  : ety -> ety -> ety = F64.add
inline_for_extraction noextract let neu : ety = F64.zero

(* Ownership of array r between i and j. The first value of that slice
is the reduction of all the values in the (original) slice v. *)
let gpu_pts_to_slice_sum_inner
  (#sz:nat)
  (r : gpu_array ety sz)
  (i j :nat)
  (v : seq ety)
  (s : seq ety)
: slprop
= gpu_pts_to_slice r i j s
  ** pure (i < j /\ j <= sz /\
           len v = sz /\
           len s = j - i /\
           squash (is_reduction neu op (Seq.slice v i j) (Seq.index s 0))) // SQUASH VERY IMPORTANT!!

let gpu_pts_to_slice_sum
  (#sz:nat)
  ([@@@mkey] r: gpu_array ety sz)
  ([@@@mkey] i : nat)
  (j:nat)
  (v: seq ety)
: slprop
= if_ (i < j && j <= sz) (exists* s. gpu_pts_to_slice_sum_inner #sz r i j v s)

// Barrier

val barrier_matrix (nth: nat) (r : gpu_array ety nth) (v: seq ety) (it from to: nat)
  : slprop

unfold
let kpre (nth: nat) (a : gpu_array ety nth) (s : erased (seq ety))
  (#_: squash (len s == nth)) (tid:nat{tid < nth})
  : slprop =
    gpu_pts_to_slice a tid (tid+1) seq![Seq.index s tid]

unfold
let kpost (nth: nat) (a : gpu_array ety nth) (s : erased (seq ety))
  (#_: squash (len s == nth)) (tid:nat{tid < nth})
  : slprop =
    if_ (tid = 0) (gpu_pts_to_slice_sum a 0 nth s)

[@@ CPrologue "__device__"]
inline_for_extraction
fn reduce
  (nth : sz { 0 < SZ.v nth /\ SZ.v nth <= 1024 })
  (a : gpu_array ety nth)
  (#s :  erased (seq ety))
  (#_: squash (len s == nth))
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
  (#_: squash (len s == nth))
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
