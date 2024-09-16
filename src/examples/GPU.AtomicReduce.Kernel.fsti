module GPU.AtomicReduce.Kernel

(* Reducing an array with some given operation. *)

#lang-pulse

open GPU

module SZ = FStar.SizeT

val contributions
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq u64{Seq.length v_done >= Seq.length v_a})
  (v_r : u64) (acc : u64)
: Tot prop

val inv_p
      (nn: nat)
      (a: gpu_array u64 nn)
      (v_a: seq u64)
      (r: gpu_ref u64)
      (done: seq (gref bool))
: Tot slprop

[@@pulse_unfold]
let kpre
  (nn: nat)
  (a : gpu_array u64 nn)
  (v_a : seq u64)
  (r : gpu_ref u64)
  (done : seq (gref bool){Seq.length done == Ghost.reveal nn})
  (i:iname)
  (tid : nat{0 <= tid /\ tid < nn})
=
  gref_pts_to (done @! tid) #0.5R false  **
  inv i (inv_p nn a v_a r done)

[@@pulse_unfold]
let kpost
  (nn: nat)
  (a : gpu_array u64 nn)
  (v_a : seq u64)
  (r : gpu_ref u64)
  (done : seq (gref bool){Seq.length done == Ghost.reveal nn})
  (i:iname)
  (tid : nat{0 <= tid /\ tid < nn})
=
  gref_pts_to (done @! tid) #0.5R true **
  inv i (inv_p nn a v_a r done)

[@@CPrologue "__global__"]
inline_for_extraction
fn kernel
  (n: erased SZ.t)
  (a : gpu_array u64 (SZ.v n))
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){Seq.length done == reveal n})
  (i : iname)
  (v_a : erased (seq u64))
  (etid : tid_t { gdim_x etid == n /\ SZ.v (bdim_x etid) == 1sz })
  requires gpu ** thread_id etid ** kpre  (SZ.v n) a v_a r done i (thread_index etid)
  ensures  gpu ** thread_id etid ** kpost (SZ.v n) a v_a r done i (thread_index etid)

ghost
fn done_lemma
  (nn: erased nat)
  (a : gpu_array u64 nn)
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){Seq.length done == reveal nn})
  (i : iname)
  (v_a : erased (seq u64))
  (etid : tid_t { gdim_x etid == 1sz /\ SZ.v (bdim_x etid) == reveal nn})
  requires gpu ** bigstar 0 nn (fun tid -> kpost nn a v_a r done i tid)
  ensures  
    gpu **
    GPU.Ref.gpu_pts_to r (GPU.Seq.Common.seq_fold_left (fun x y -> UInt64.add_mod x y) 0uL v_a) ** // FIXME: eta needed
    gpu_pts_to_array a v_a
