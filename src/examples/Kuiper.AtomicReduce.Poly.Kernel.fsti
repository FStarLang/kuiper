module Kuiper.AtomicReduce.Poly.Kernel

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics

module SZ = FStar.SizeT

val contributions
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
  : Tot prop

val inv_p
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (a: gpu_array et nn)
  (v_a: seq et)
  (r: gpu_ref et)
  (done: seq (gref bool))
  : Tot slprop

unfold
let kpre
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : nat{0 <= tid /\ tid < nn})
=
  gref_pts_to (done @! tid) #0.5R false  **
  inv i (inv_p nn a v_a r done)

unfold
let kpost
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : nat{0 <= tid /\ tid < nn})
=
  gref_pts_to (done @! tid) #0.5R true **
  inv i (inv_p nn a v_a r done)

unfold
type kernel_ty (et : Type0) {| scalar et |} {| d : has_atomic_add et |} =
  (n: erased SZ.t) ->
  (a : gpu_array et (SZ.v n)) ->
  (r : gpu_ref et) ->
  (done : erased (seq (gref bool)){len done == reveal (SZ.v n)}) ->
  (i : iname) ->
  (v_a : erased (seq et)) ->
  (etid : tid_t { gdim_x etid == SZ.v n /\ bdim_x etid == 1 }) ->
  stt unit
  (requires
    gpu **
    thread_id etid **
    kpre  (SZ.v n) a v_a r done i (thread_index etid))
  (ensures fun _ ->
    gpu **
    thread_id etid **
    kpost (SZ.v n) a v_a r done i (thread_index etid))

inline_for_extraction noextract
val kernel
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  : kernel_ty et

ghost
fn done_lemma
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: erased nat)
  (a : gpu_array et nn)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == reveal nn})
  (i : iname)
  (v_a : erased (seq et))
  (etid : tid_t { gdim_x etid == 1 /\ bdim_x etid == reveal nn})
  requires
    gpu **
    bigstar 0 nn (fun tid -> kpost nn a v_a r done i tid)
  ensures
    gpu **
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a) **
    (a |-> v_a)
