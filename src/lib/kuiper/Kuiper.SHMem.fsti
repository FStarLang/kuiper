module Kuiper.SHMem

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Array
open FStar.Tactics.V2
open Kuiper.ForEvery
open Kuiper.Common
module SZ = Kuiper.SizeT
module T = FStar.Tactics
module A = Pulse.Lib.Array

(* Description of one shared memory array "request" *)
// TODO: Does the length really need to be nonzero?
noeq
inline_for_extraction
type shmem_desc =
  | SHArray :
    (ty : Type0) ->
    {| sized : Kuiper.Sized.sized ty |} ->
    len    : SZ.t { len > 0 } ->
    shmem_desc

let is_block_array #ty (g : array ty)
  = visibility_of g == block_of

val is_send_across_block_array
  (#et:Type0)
  (a : array et { is_block_array a })
  (#f:perm) (#s:_)
: is_send_across block_of (pts_to a #f s)

inline_for_extraction unfold
let c_shmem (d : shmem_desc) : Type0 =
  match d with
  | SHArray ty len -> larray ty len
  //would be nice to just add as is_block_array refinement here, but it messes with typeclass resolution

let rec c_shmems (d : list shmem_desc) : Type0 =
  match d with
  | [] -> int // This could (and should) be unit, but karamel extraction gets confused with it
  | d :: ds ->
    c_shmem d & c_shmems ds

let c_shmem_inv (#d : shmem_desc) (c:c_shmem d) : prop =
  match d with
  | SHArray ty len -> is_block_array #ty c

let rec c_shmems_inv (#ds : list shmem_desc) (c:c_shmems ds) : prop =
  match ds with
  | [] -> True
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    c_shmem_inv #d (fst c) /\
    c_shmems_inv #ds (snd c)

let live_c_shmem #d (c : c_shmem d) (#[T.exact (`1.0R)]f:_) : slprop =
  match d with
  | SHArray ty len ->
    exists* (v : Seq.seq ty).
      A.pts_to c #f v

instance val is_send_across_live_c_shmem #d (c:c_shmem d) #f (_:squash (c_shmem_inv c))
: is_send_across block_of (live_c_shmem #d c #f)

let rec live_c_shmems #ds (c : c_shmems ds) (#[T.exact (`1.0R)]f:_) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f

ghost
fn unfold_live_c_shmems_nil (c : c_shmems []) (#[T.exact (`1.0R)]f:_)
  requires live_c_shmems c #f
  ensures emp

ghost
fn fold_live_c_shmems_nil (c : c_shmems []) (#[T.exact (`1.0R)]f:_)
  ensures live_c_shmems c #f

ghost
fn unfold_live_c_shmems_cons #d #ds (c : c_shmems (d::ds)) (#[T.exact (`1.0R)]f:_)
  requires live_c_shmems c #f
  ensures live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f

ghost
fn fold_live_c_shmems_cons #d #ds (c : c_shmems (d::ds)) (#[T.exact (`1.0R)]f:_)
  requires live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f
  ensures live_c_shmems c #f

instance val is_send_across_live_c_shmems #ds (c:c_shmems ds) #f (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c #f)

ghost
fn unfold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
  requires live_c_shmems c #f
  ensures FStar.Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)

ghost
fn fold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
  requires FStar.Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)
  ensures live_c_shmems c #f

let d_ty d : Type0 = let SHArray ty _ = d in ty
let d_ty_sized d : Kuiper.Sized.sized (d_ty d) = let SHArray _ #s _ = d in s
let d_len d : SZ.t = let SHArray _ len = d in len

ghost
fn unfold_live_c_shmem #d (c:c_shmem d) #f
  requires live_c_shmem c #f
  ensures
    // This should be pts_to_slice....
    exists* (s:Seq.seq (d_ty d)).
      pts_to (c <: larray (d_ty d) (d_len d)) #f s

ghost
fn fold_live_c_shmem #d (c:c_shmem d) #f
  requires
    // Idem
    exists* (s:Seq.seq (d_ty d)).
      pts_to (c <: larray (d_ty d) (d_len d)) #f s
  ensures live_c_shmem c #f

ghost
fn gpu_live_c_shmem_share_underspec
    (#d:_) (c:c_shmem d) (#f:_) (#k:nat { k > 0 })
  requires
    live_c_shmem c #f
  ensures
    forall+ (_ : natlt k). live_c_shmem c #(f /. Real.of_int k)

ghost
fn gpu_live_c_shmems_share_underspec
  (#ds:_) (c:c_shmems ds) (#f:_) (#k:nat { k > 0 })
  requires
    live_c_shmems c #f
  ensures
    forall+ (_ : natlt k). live_c_shmems c #(f /. Real.of_int k)

ghost
fn gpu_live_c_shmem_gather_underspec
  (#d:_) (c:c_shmem d) (#f:perm) (#k:nat { k > 0 })
  requires
    forall+ (_ : natlt k). live_c_shmem c #(f /. Real.of_int k)
  ensures
    live_c_shmem c #f

ghost
fn gpu_live_c_shmems_gather_underspec
  (#ds:_) (c:c_shmems ds) (#f:perm) (#k:nat { k > 0 })
  requires
    forall+ (_ : natlt k). live_c_shmems c #(f /. Real.of_int k)
  ensures
    live_c_shmems c #f
