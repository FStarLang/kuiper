module Kuiper.SHMem

#lang-pulse

open Pulse.Lib.Core
open Kuiper.Array
module SZ = Kuiper.SizeT

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
let c_shmem (d : shmem_desc) : Type0 =
  match d with
  | SHArray ty len -> gpu_array ty len

inline_for_extraction
let rec c_shmems (d : list shmem_desc) : Type0 =
  match d with
  | [] -> int // This could (and should) be unit, but karamel extraction gets confused with it
  | d :: ds ->
    c_shmem d & c_shmems ds

let live_c_shmem #d (c : c_shmem d) : slprop =
  match d with
  | SHArray ty len -> exists* v. gpu_pts_to_array #ty #len c #1.0R v

let rec live_c_shmems #ds (c : c_shmems ds) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    live_c_shmem #d (fst c) ** live_c_shmems #ds (snd c)
