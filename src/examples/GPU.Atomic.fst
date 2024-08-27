module GPU.Atomic

#set-options "--debug __unrefine"

(* Reducing an array with some given operation. *)

#lang-pulse

open GPU
open Pulse.Lib.GhostReference { ref as gref, pts_to as gref_pts_to }

unfold let ( @! ) (#a:Type) (s : seq a) (i : nat { i < Seq.length s }) : a = Seq.index #a s i

module SZ = FStar.SizeT

val n : (n : sz { 0 < SZ.v n /\ SZ.v n <= 1024 })
let n = 123sz

let nn : erased nat = SZ.v n

let rec contributions
  (v_done : seq bool)
  (v_a : seq u64{Seq.length v_done >= Seq.length v_a})
  (v_r : u64) (acc : u64)
: Tot prop (decreases Seq.length v_a)
=
  if Seq.length v_a = 0 then
    v_r == acc
  else
    let hd = Seq.head v_a in
    let tl = Seq.tail v_a in
    let hd_done = Seq.head v_done in
    let tl_done = Seq.tail v_done in
    if hd_done then
      contributions tl_done tl v_r (UInt64.add_mod hd acc)
    else
      contributions tl_done tl v_r acc

let inv_p (a : gpu_array u64 n) (v_a : seq u64) (r : gpu_ref u64) (done : seq (gref bool)) =
  // pure (Seq.length done == nn) **
  exists* (v_done : seq bool {Seq.length v_done >= Seq.length done /\ Seq.length v_done >= Seq.length v_a}) v_r.
    gpu_pts_to_array a v_a **
    gpu_pts_to r v_r **
    bigstar 0 (Seq.length done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    pure (contributions v_done v_a v_r 0uL)

[@@pulse_unfold]
let kpre
  (a : gpu_array u64 n) (v_a : erased (seq u64))
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){Seq.length done == Ghost.reveal nn})
  (i:iname) (tid : nat{tid < nn})
=
  gref_pts_to (done @! tid) #0.5R false  **
  inv i (inv_p a v_a r done)

[@@pulse_unfold]
let kpost
  (a : gpu_array u64 n) (v_a : erased (seq u64))
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){Seq.length done == Ghost.reveal nn})
  (i:iname) (tid : nat{tid < nn})
=
  gref_pts_to (done @! tid) #0.5R true **
  inv i (inv_p a v_a r done)
  

[@@noextract_to "krml"]
atomic
fn atomic_gpu_array_read
  (#a : Type u#0)
  (#sz : erased nat)
  (#i  : erased nat)
  (#j  : erased nat{i <= j /\ j <= sz})
  (r:gpu_array a sz)
  (#f:perm)
  (idx : SZ.t {i <= SZ.v idx /\ SZ.v idx < j})
  (#s : erased (seq a))
  requires gpu ** gpu_pts_to_array_slice #a #sz r #f i j s
  returns  x:a
  ensures  gpu ** gpu_pts_to_array_slice #a #sz r #f i j s **
            pure (i <= j /\ j <= sz /\ Seq.length s == (j-i) /\
                  x == Seq.index s (SZ.v idx - i))
{
  admit();
}

atomic
fn gpu_faa_u64
  (r : gpu_ref u64)
  (v : u64)
  (#v0 : erased u64)
  requires gpu ** gpu_pts_to r #1.0R v0
  ensures  gpu ** gpu_pts_to r #1.0R (FStar.UInt64.add_mod v v0)
{
  admit();
}

ghost
fn bigstar_ghost_upd_lemma
    (done : seq (gref bool))
    (v_done : seq bool{Seq.length v_done >= Seq.length done})
    (tid : nat{tid < Seq.length done})
  requires
    bigstar 0 (Seq.length done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    gref_pts_to (done @! tid) #0.5R false
  // returns
  //   v_done' : (v_done' : seq bool{Seq.length v_done' >= Seq.length done})
  ensures
    bigstar 0 (Seq.length done) (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) **
    gref_pts_to (done @! tid) #0.5R true
{
  admit();
}

// #set-options "--debug SMTFail --split_queries always"

assume
val contributions_lemma
  (v_done : seq bool)
  (v_a : seq u64{Seq.length v_done >= Seq.length v_a})
  (v_r : u64) (acc : u64)
  (tid : nat{tid < Seq.length v_a})
  : Lemma (requires contributions v_done v_a v_r acc /\ v_done @! tid == false)
          (ensures  contributions (Seq.upd v_done tid true) v_a (UInt64.add_mod v_r (v_a @! tid)) acc)
          [SMTPat (contributions (Seq.upd v_done tid true) v_a (UInt64.add_mod v_r (v_a @! tid)) acc)]

[@@ CPrologue "__global__"]
fn kernel
  (a : gpu_array u64 n)
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){Seq.length done == nn})
  (i : iname)
  (v_a : erased (seq u64))
  (etid : erased tid_t { gdim_x etid == 1sz /\ bdim_x etid == n })
  requires gpu ** thread_id etid ** kpre  a v_a r done i (thread_index etid)
  ensures  gpu ** thread_id etid ** kpost a v_a r done i (thread_index etid)
{
  assume_ (pure (Seq.length v_a == nn));
  let tid = thread_idx_x ();
  let tid : sz = SZ.uint32_to_sizet tid;
  rewrite each thread_index etid as SZ.v tid;
  (* Read array at idx *)
  let v =
    with_invariants i
      returns v : u64
      ensures
        gpu **
        thread_id etid **
        gref_pts_to (done @! tid) #0.5R false **
        inv_p a v_a r done **
        pure (v == v_a @! SZ.v tid)
    {
      unfold inv_p;
      unfold gpu_pts_to_array a v_a;
      let rr = atomic_gpu_array_read #u64 #nn #0 #nn a tid;
      fold (gpu_pts_to_array a v_a);
      fold (inv_p a v_a r done);
      rr
    };
  (* Fetch and add into result cell. *)
  with_invariants i
  {
    unfold inv_p;
    gpu_faa_u64 r v;
    bigstar_ghost_upd_lemma done _ _ ;
    assume_ (pure False);
    rewrite each SZ.v tid as thread_index etid;
    fold (inv_p a v_a r done);
  }
}
