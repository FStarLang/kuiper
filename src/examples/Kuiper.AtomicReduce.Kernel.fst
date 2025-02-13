module Kuiper.AtomicReduce.Kernel

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper

module SZ = FStar.SizeT

let rec contributions
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq u64{len v_done >= len v_a})
  (v_r : u64) (acc : u64)
: Tot prop (decreases len v_a)
=
  if len v_a = 0 then
    v_r == acc
  else
    let hd = Seq.head v_a in
    let tl = Seq.tail v_a in
    let hd_done = Seq.head v_done in
    let tl_done = Seq.tail v_done in
    if hd_done then
      contributions nn tl_done tl v_r (UInt64.add_mod hd acc)
    else
      contributions nn tl_done tl v_r acc

let inv_p
      (nn: nat)
      (a: gpu_array u64 nn)
      (v_a: seq u64)
      (r: gpu_ref u64)
      (done: seq (gref bool))
     =
  // pure (len done == nn) **
  exists* (v_done:
    seq bool {len v_done >= len done /\ len v_done >= len v_a})
    v_r.
    ((a |-> v_a) ** (r |-> v_r) **
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i))) **
    pure (contributions nn v_done v_a v_r 0uL)


ghost
fn bigstar_ghost_upd_lemma
  (done : seq (gref bool))
  (v_done : seq bool{len v_done >= len done})
  (tid : nat{tid < len done})
  requires
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    gref_pts_to (done @! tid) #0.5R false
  // returns
  //   v_done' : (v_done' : seq bool{len v_done' >= len done})
  ensures
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) **
    gref_pts_to (done @! tid) #0.5R true
{
  admit();
}

assume
val contributions_lemma
  (nn: nat)
  (v_done : seq bool)
  (v_a : seq u64{len v_done >= len v_a})
  (v_r : u64) (acc : u64)
  (tid : nat{tid < len v_a})
  : Lemma (requires contributions nn v_done v_a v_r acc /\ v_done @! tid == false)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (UInt64.add_mod v_r (v_a @! tid)) acc)
          [SMTPat (contributions nn (Seq.upd v_done tid true) v_a (UInt64.add_mod v_r (v_a @! tid)) acc)]

[@@ CPrologue "__global__"]
fn kernel
  (nn: erased SZ.t)
  (a : gpu_array u64 (SZ.v nn))
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){len done == reveal nn})
  (i : iname)
  (v_a : erased (seq u64))
  (etid : tid_t { gdim_x etid == reveal nn /\ bdim_x etid == 1})
  requires gpu ** thread_id etid ** kpre  (SZ.v nn) a v_a r done i (thread_index etid)
  ensures  gpu ** thread_id etid ** kpost (SZ.v nn) a v_a r done i (thread_index etid)
{
  assume (pure (len v_a == reveal nn));
  let tid = thread_idx_all ();
  rewrite each thread_index etid as SZ.v tid;
  later_credit_buy 1;
  later_credit_buy 1;
  (* Read array at idx *)
  let v =
    with_invariants i
      returns v : u64
      ensures
        gpu **
        thread_id etid **
        gref_pts_to (done @! tid) #0.5R false **
        later (inv_p (SZ.v nn) a v_a r done) **
        pure (v == v_a @! SZ.v tid) **
        later_credit 1
    {
      later_elim _;
      unfold inv_p;
      let rr = gpu_array_read #u64 #(SZ.v nn) #0 #(SZ.v nn) a tid;
      fold inv_p;
      later_intro (inv_p (SZ.v nn) a v_a r done);
      rr
    };
  (* Fetch and add into result cell. *)
  with_invariants i
  {
    later_elim _;
    unfold inv_p;
    let _ = gpu_faa_u64 r v;
    bigstar_ghost_upd_lemma done _ _ ;
    assume (pure False); (* FIXME *)
    rewrite each SZ.v tid as thread_index etid;
    fold inv_p;
    later_intro (inv_p (SZ.v nn) a v_a r done);
  }
}

ghost
fn done_lemma
  (nn: erased nat)
  (a : gpu_array u64 nn)
  (r : gpu_ref u64)
  (done : erased (seq (gref bool)){len done == reveal nn})
  (i : iname)
  (v_a : erased (seq u64))
  (etid : tid_t { gdim_x etid == 1 /\ bdim_x etid == reveal nn})
  requires gpu ** bigstar 0 nn (fun tid -> kpost  nn a v_a r done i tid)
  ensures  
    gpu **
    (r |-> Kuiper.Seq.Common.seq_fold_left (fun x y -> UInt64.add_mod x y) 0uL v_a) ** // FIXME: eta needed
    (a |-> v_a)

{
  admit();
}
