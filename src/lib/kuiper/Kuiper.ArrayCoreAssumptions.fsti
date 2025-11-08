module Kuiper.ArrayCoreAssumptions
#lang-pulse
open Pulse.Lib.Pervasives
module A = Pulse.Lib.Array
module SZ = FStar.SizeT

//we could expose this from core_pcm_ref
//assuming that every allocation is at least 128-aligned
val core_base_address (x:A.array 'a) : GTot (n:nat { n > 0 /\ n%128==0 })

let visibility = loc_id -> loc_id

val loc_id_of_array (#a:Type u#a) (x:A.array a) : loc_id

val visibility_of_array (#a:Type u#a) (x:A.array a) : visibility

let array_visible_at (#a:Type) (x:A.array a) (l:loc_id) =
  visibility_of_array x l ==
  visibility_of_array x (loc_id_of_array x)

fn alloc_array_with_vis u#a (#elt: Type u#a) {| small_type u#a |} 
      (x: elt) (n: SZ.t) (l:loc_id)
      (vis:visibility)
  requires loc l
  returns a: array elt
  ensures loc l
  ensures pts_to_mask a (Seq.create (SZ.v n) x) (fun _ -> True)
  ensures pure (
    A.length a == SZ.v n /\
    A.is_full_array a /\
    visibility_of_array a == vis /\
    loc_id_of_array a == l)

val is_send_across_pts_to_mask (#a: Type u#a) (x:A.array a) (f:perm) (s:Seq.seq a) (mask:nat -> prop)
: is_send_across (visibility_of_array x) (pts_to_mask x #f s mask)
