module Kuiper.Ref.Base

friend Pulse.Lib.Array.PtsTo

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base

module A = Pulse.Lib.Array

instance is_send_across_array_mask (#a:Type0) (arr:A.array a) (f:perm)
  (s:Seq.seq (option a)) (mask:nat -> prop)
  : is_send_across (A.visibility_of_array arr) (A.pts_to_mask arr #f s mask)
  = A.is_send_across_pts_to_mask arr f s mask

inline_for_extraction noextract
let ref (a:Type0) = A.larray a 1

inline_for_extraction noextract
let to_array #a (r:ref a) = r

inline_for_extraction noextract
let of_array #a (arr:A.larray a 1) = arr

let is_send_across_pts_to #a r #p v vis =
  let i : is_send_across (A.visibility_of_array (to_array r)) (pts_to r #p v) =
    FStar.Tactics.Effect.synth_by_tactic (fun () -> FStar.Tactics.V2.norm [delta_only [`%pts_to; `%A.pts_to]];
        FStar.Tactics.Typeclasses.tcresolve ()) in
  i

let is_send_pts_to #a r #p v = A.is_send_pts_to (to_array r) #p seq![v]
