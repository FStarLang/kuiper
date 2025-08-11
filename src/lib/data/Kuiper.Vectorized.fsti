module Kuiper.Vectorized

#lang-pulse

open Pulse.Lib.Vec
open Pulse
open Pulse.Lib.BigStar
open FStar.Tactics.V2
open FStar.Seq

open Kuiper
open Kuiper.IView
open Kuiper.IArray { iarray }
// open Kuiper.Array
// open Kuiper.Base
// open Kuiper.Sized
// open Kuiper.SizeT
// open Kuiper.Seq.Common

module SZ = FStar.SizeT

// class has_vec_t (et : Type0) = {
//   vec_lens : seq nat;
//   vec_ts : seq Type0;
//   _eq_len : squash (length vec_lens == length vec_ts)
// }

new
val float4 : Type0

new
val make_float4 (x y z w : float) : float4

new
val getx (v : float4) : float

new
val gety (v : float4) : float

new
val getz (v : float4) : float

new
val getw (v : float4) : float

new
val lemma_getx_projects_x (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getx (make_float4 x y z w) == x)
  [SMTPat (getx (make_float4 x y z w))]

new
val lemma_gety_projects_y (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures gety (make_float4 x y z w) == y)
  [SMTPat (gety (make_float4 x y z w))]

new
val lemma_getz_projects_z (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getz (make_float4 x y z w) == z)
  [SMTPat (getz (make_float4 x y z w))]

new
val lemma_getw_projects_w (x : float) (y : float) (z : float) (w : float)
  : Lemma (ensures getw (make_float4 x y z w) == w)
  [SMTPat (getw (make_float4 x y z w))]

new
val lemma_make_float4_from_float4 (v : float4)
  : Lemma (ensures make_float4 (getx v) (gety v) (getz v) (getw v) == v)

// let vec_t (et : Type0) (vec_sz : erased nat)
//   = match (et, reveal vec_sz) with
//     // | (f32, 1) -> Some(vf32_1)
//     // | (f32, 2) -> Some(vf32_2)
//     // | (f32, 3) -> Some(vf32_3)
//     | (f32, 4) -> Some(vf32_4)
//     // | (f64, 1) -> Some(vf32_1)
//     // | (f64, 2) -> Some(vf32_2)
//     // | (f64, 3) -> Some(vf32_3)
//     // | (f64, 4) -> Some(v4_64)
//     | _ -> None

[@@noextract_to "krml"]
atomic
fn gpu_array_read_vec4
  // (#et : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat{i <= j /\ j <= sz})
  // (#vec_sz : erased nat)
  // (vect : vec_t et vec_sz)
  (a:gpu_array float sz)
  // (vec_sz : erased nat)
  // {| hvt : has_vec_t et |}
  (#f:perm)
  (idx : SZ.t)
  (#s : erased (seq float))
  preserves gpu
  preserves gpu_pts_to_slice #float #sz a #f i j s
  requires pure (i <= SZ.v idx /\ SZ.v idx + 3 < j)
  // requires pure (contains hvt.vec_lens (reveal vec_sz)) 
  returns  x: float4
  ensures  pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                 i <= SZ.v idx /\ SZ.v idx + 3 < j /\
                 x == make_float4
                        (Seq.index s (SZ.v idx - i))
                        (Seq.index s (SZ.v idx + 1 - i))
                        (Seq.index s (SZ.v idx + 2 - i))
                        (Seq.index s (SZ.v idx + 3 - i)))

let upd_seq_vec4 (s : seq float) (idx : nat{idx+3 < Seq.length s}) (v : float4) : seq float //s':seq float{length s' == lenght s}
  = Seq.upd (Seq.upd (Seq.upd (Seq.upd s idx (getx v)) (idx + 1) (gety v)) (idx + 2) (getz v)) (idx + 3) (getw v)

// #push-options "--debug SMTFail --split_queries always"
[@@noextract_to "krml"]
atomic
fn gpu_array_write_vec4
  // (#et : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat{i <= j /\ j <= sz})
  // (#vec_sz : erased nat)
  // (vect : vec_t et vec_sz)
  (a:gpu_array float sz)
  // (vec_sz : erased nat)
  // {| hvt : has_vec_t et |}
  (idx : SZ.t)
  (v : float4)
  (#s : erased (seq float))
  preserves gpu
  requires pure (i <= SZ.v idx /\ SZ.v idx + 3 < j)
  requires gpu_pts_to_slice #float #sz a #1.0R i j s
  ensures (exists* (s':seq float). gpu_pts_to_slice #float #sz a #1.0R i j s' **
          pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                i <= SZ.v idx /\ SZ.v idx + 3 < j /\
                s' == upd_seq_vec4 s (idx - i) v))

// #push-options "--debug SMTFail --split_queries always"
// fn iarray_read_vec4
//   // (#et:Type0)
//   (#len : erased nat)
//   (#vw : aiview len) {| cw : ciview vw |}
//   // (a : iarray et vw)
//   (a : iarray float vw)
//   // (ci : cw.sch.cit)
//   (ci : SZ.t)
//   (#f : perm)
//   // (#v : (vw.sch.ait -> GTot et))
//   (#v : (vw.sch.ait -> GTot float))
//   preserves
//     gpu **
//     (a |-> Frac f v)
//   returns
//     e : float4
//   ensures
//     pure (e == make_float4
//                  (v (ci_to_ai vw ci))
//                  (v (ci_to_ai vw (ci +^ 1sz)))
//                  (v (ci_to_ai vw (ci +^ 2sz)))
//                  (v (ci_to_ai vw (ci +^ 3sz))))
// {
//   admit();
// }
