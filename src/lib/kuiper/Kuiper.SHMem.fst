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

instance is_send_across_block_array
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


// inline_for_extraction
// let first_ty (d:list shmem_desc { not (Nil? d) }) : Type0 =
//   match d with
//   | SHArray ty len::_ -> ty

// instance first_ty_sized (d:list shmem_desc) (_:squash(not (Nil? d))) 
// : Kuiper.Sized.sized (first_ty d)
// = match d with
//   | SHArray ty #sized len::_ -> sized

// inline_for_extraction
// let first_len (d:list shmem_desc { not (Nil? d) }) : SZ.t =
//   match d with
//   | SHArray ty len::_ -> len

// let first (#d:list shmem_desc { not (Nil? d) }) (c:c_shmems d)
// : gpu_array (first_ty d) (first_len d)
//  = match d with
//   | d::ds -> 
//     let c : c_shmem d & c_shmems ds = c in
//     fst c

let live_c_shmem #d (c : c_shmem d) : slprop =
  match d with
  | SHArray ty len -> 
    exists* v. gpu_pts_to_array #ty #len c #1.0R v

instance is_send_across_live_c_shmem #d (c:c_shmem d) (_:squash (c_shmem_inv c))
: is_send_across block_of (live_c_shmem #d c)
= match d with
  | SHArray ty len ->
    let ff (v:_) : is_send_across block_of  (gpu_pts_to_array #ty #len c #1.0R v) =
      FStar.Tactics.Typeclasses.solve
    in
    let ff : is_send_across block_of (exists* v. gpu_pts_to_array #ty #len c #1.0R v) =
      is_send_across_exists _ #ff
    in 
    let ff : is_send_across block_of (live_c_shmem #(SHArray ty len) c)
      = ff
    in
    ff

let rec live_c_shmems #ds (c : c_shmems ds) : slprop =
  match ds with
  | [] -> emp
  | d :: ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    live_c_shmem #d (fst c) ** live_c_shmems #ds (snd c)

let rec is_send_across_live_c_shmems_ #ds (c:c_shmems ds) (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c)
= let open FStar.Tactics.Typeclasses in
  match ds with
  | [] -> solve #(is_send_across block_of emp)
  | d::ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    let s = is_send_across_live_c_shmems_ #ds (snd c) () in
    let f : is_send_across block_of (live_c_shmem #d (fst c)) = solve in
    is_send_across_star _ _ #f #s

instance is_send_across_live_c_shmems #ds (c:c_shmems ds) (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c)
= is_send_across_live_c_shmems_ #ds c pf

(* 1xn, shared memory *)


