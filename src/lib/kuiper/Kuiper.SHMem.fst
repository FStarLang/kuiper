module Kuiper.SHMem

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Array
module SZ = Kuiper.SizeT
module T = FStar.Tactics

(* Description of one shared memory array "request" *)
noeq
inline_for_extraction
type shmem_desc =
  | SHArray :
    (ty : Type0) ->
    {| sized : Kuiper.Sized.sized ty |} ->
    len    : SZ.t ->
    shmem_desc

inline_for_extraction unfold
let c_shmem (d : shmem_desc) (#[T.exact (`0)]gpu_id:int) (bid:int) : Type0 =
  match d with
  | SHArray ty len -> gpu_shmem_array #gpu_id bid ty len

inline_for_extraction
let rec c_shmems (d : list shmem_desc) (#[T.exact (`0)]gpu_id:int) (bid:int) : Type0 =
  match d with
  | [] -> int // This could (and should) be unit, but karamel extraction gets confused with it
  | d :: ds ->
    c_shmem d #gpu_id bid & c_shmems ds #gpu_id bid

let live_c_shmem #d #gpu_id #bid (c : c_shmem d #gpu_id bid) : slprop =
  match d with
  | SHArray ty len -> exists* v. gpu_pts_to_array #ty #len c #1.0R v

let rec live_c_shmems #ds #gpu_id #bid (c : c_shmems ds #gpu_id bid) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d #gpu_id bid & c_shmems ds #gpu_id bid = c in (* coerce *)
    live_c_shmem #d #gpu_id #bid (fst c) ** live_c_shmems #ds #gpu_id #bid (snd c)