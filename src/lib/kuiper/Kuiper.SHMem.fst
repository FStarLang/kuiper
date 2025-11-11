module Kuiper.SHMem

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.SendSync
open Kuiper.Base
open Kuiper.Array
open FStar.Tactics.V2
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

let is_block_array #ty #len (g:gpu_array ty len) 
  = visibility_of g == block_of

//don't mark this an instance, to avoid clashing with other instances
//for visibility_of, gpu_of
let is_send_across_block_array
  (#et:Type0) (#sz:_) 
  (a:gpu_array et sz { is_block_array a })
  (#i #j #f #s:_)
: is_send_across block_of (gpu_pts_to_slice a #f i j s)
= let i : is_send_across (visibility_of a) (gpu_pts_to_slice a #f i j s)
   = FStar.Tactics.Typeclasses.solve in
  i

inline_for_extraction unfold
let c_shmem (d : shmem_desc) : Type0 =
  match d with
  | SHArray ty len -> gpu_array ty len 
   //would be nice to just add as is_block_array refinement here, but it messes with typeclass resolution

inline_for_extraction
let rec c_shmems (d : list shmem_desc) : Type0 =
  match d with
  | [] -> int // This could (and should) be unit, but karamel extraction gets confused with it
  | d :: ds ->
    c_shmem d & c_shmems ds

let c_shmem_inv (#d : shmem_desc) (c:c_shmem d) : prop =
  match d with
  | SHArray ty len -> is_block_array #ty #len c 

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
    exists* v. gpu_pts_to_array #ty #len c #f v

instance is_send_across_live_c_shmem #d (c:c_shmem d) #f (_:squash (c_shmem_inv c))
: is_send_across block_of (live_c_shmem #d c #f)
= match d with
  | SHArray ty len ->
    let ff (v:_) : is_send_across block_of  (gpu_pts_to_array #ty #len c #f v) =
      is_send_across_block_array c
    in
    let ff : is_send_across block_of (exists* v. gpu_pts_to_array #ty #len c #f v) =
      is_send_across_exists _ #ff
    in 
    let ff : is_send_across block_of (live_c_shmem #(SHArray ty len) c #f)
      = ff
    in
    ff

let rec live_c_shmems #ds (c : c_shmems ds) (#[T.exact (`1.0R)]f:_) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f

let rec is_send_across_live_c_shmems_ #ds (c:c_shmems ds) #f (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c #f)
= let open FStar.Tactics.Typeclasses in
  match ds with
  | [] -> solve #(is_send_across block_of emp)
  | d::ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    let s = is_send_across_live_c_shmems_ #ds (snd c) #f () in
    let f : is_send_across block_of (live_c_shmem #d (fst c) #f) = solve in
    is_send_across_star _ _ #f #s

instance is_send_across_live_c_shmems #ds (c:c_shmems ds) #f (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c #f)
= is_send_across_live_c_shmems_ #ds c #f pf

ghost
fn unfold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
requires live_c_shmems c #f
ensures FStar.Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)
{
  reduce_with_steps (live_c_shmems c #f) [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]];
}


ghost
fn fold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
requires FStar.Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)
ensures live_c_shmems c #f
{
  norm_spec [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f);
  rewrite (FStar.Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f))
  as (live_c_shmems c #f);
}

(* 1xn, shared memory *)


