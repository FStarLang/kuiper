module Kuiper.Array.Vectorized

#lang-pulse

open FStar.Seq

open Kuiper
open Kuiper.IView
open Kuiper.VectorType

module SZ = FStar.SizeT

[@@noextract_to "krml"]
atomic
fn gpu_array_vec4_read
  // (#et : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
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

[@@noextract_to "krml"]
atomic
fn gpu_array_vec4_write
  // (#et : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat)
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

