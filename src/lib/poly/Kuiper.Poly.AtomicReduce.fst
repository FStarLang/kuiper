module Kuiper.Poly.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics

module SZ = FStar.SizeT
module W = Pulse.Lib.WithPure

let rec contributions
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
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
      contributions nn tl_done tl v_r (d.pure_op hd acc)
    else
      contributions nn tl_done tl v_r acc

let inv_p
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (a: gpu_array et nn)
  (v_a: seq et)
  (r: gpu_ref et)
  (done: seq (gref bool))
=
  // pure (len done == nn) **
  exists* (v_done:
    seq bool {len v_done >= len done /\ len v_done >= len v_a})
    v_r.
    ((a |-> v_a) ** (r |-> v_r) **
    bigstar 0 (len done) (fun i -> (done @! i) |-> Frac 0.5R (v_done @! i))) **
    pure (contributions nn v_done v_a v_r zero)

unfold
let kpre
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : natlt nn)
=
  (done @! tid) |-> Frac 0.5R false **
  inv i (inv_p nn a v_a r done)

unfold
let kpost
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : natlt nn)
=
  (done @! tid) |-> Frac 0.5R true **
  inv i (inv_p nn a v_a r done)

ghost
fn bigstar_ghost_upd_lemma
  (done : seq (gref bool))
  (v_done : seq bool{len v_done >= len done})
  (tid : nat{tid < len done})
  preserves
    bigstar 0 (len done) (fun i -> (done @! i) |-> Frac 0.5R (v_done @! i))
  requires
    (done @! tid) |-> Frac 0.5R false
  // returns
  //   v_done' : (v_done' : seq bool{len v_done' >= len done})
  ensures
    (done @! tid) |-> Frac 0.5R true
{
  admit();
}

assume
val contributions_lemma
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn: nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
  (tid : nat{tid < len v_a})
  : Lemma (requires contributions nn v_done v_a v_r acc /\ v_done @! tid == false)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op v_r (v_a @! tid)) acc)
          [SMTPat (contributions nn (Seq.upd v_done tid true) v_a (d.pure_op v_r (v_a @! tid)) acc)]

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#nn: SZ.t)
  (a : gpu_array et (SZ.v nn))
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == SZ.v nn})
  (i : iname)
  (bid : szlt (SZ.v nn))
  ()
  requires
    gpu **
    kpre (SZ.v nn) a v_a r done i bid **
    block_id (SZ.v nn) bid
  ensures
    gpu **
    kpost (SZ.v nn) a v_a r done i bid **
    block_id (SZ.v nn) bid
{
  assume (pure (len v_a == SZ.v nn));
  later_credit_buy 1;
  later_credit_buy 1;
  (* Read array at idx *)
  let v =
    with_invariants i
      returns v : et
      ensures
        gpu **
        block_id (SZ.v nn) bid **
        (done @! bid) |-> Frac 0.5R false **
        later (inv_p (SZ.v nn) a v_a r done) **
        pure (v == v_a @! SZ.v bid) **
        later_credit 1
    {
      later_elim _;
      unfold (inv_p (SZ.v nn) a v_a r done);
      let rr = gpu_array_read a bid;
      fold (inv_p (SZ.v nn) a v_a r done);
      later_intro (inv_p (SZ.v nn) a v_a r done);
      rr
    };
  (* Fetch and add into result cell. *)
  with_invariants i
  {
    later_elim _;
    unfold (inv_p (SZ.v nn) a v_a r done);
    let _ = atomic_add r v;
    bigstar_ghost_upd_lemma done _ _ ;
    assume (pure False); (* FIXME *)
    fold (inv_p (SZ.v nn) a v_a r done);
    later_intro (inv_p (SZ.v nn) a v_a r done);
  }
}

ghost
fn done_lemma
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn: erased nat)
  (a : gpu_array et nn)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == reveal nn})
  (i : iname)
  (v_a : erased (seq et))
  requires
    gpu **
    bigstar 0 nn (fun tid -> kpost nn a v_a r done i tid)
  ensures
    gpu **
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a) **
    (a |-> v_a)
{
  admit();
}


ghost
fn setup
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (n : sz)
  (a : gpu_array et n)
  (#f : perm)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  requires
    cpu **
    (a |-> Frac f v_a) **
    (r |-> zero #et) **
    pure (SZ.v n <= 1024)
  returns
    i_done : iname & erased (seq (gref bool))
  ensures
    (match i_done with | (i, done) ->
    cpu
    ** W.with_pure (len done == SZ.v n) (fun _ ->
       bigstar 0 (SZ.v n) (fun tid ->
        (done @! tid) |-> Frac 0.5R false **
        inv i (inv_p (SZ.v n) a v_a r done))
    ))
{
  admit();
}

ghost
fn teardown
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (n : sz)
  (a : gpu_array et n)
  (#f : perm)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (i : iname)
  (done : lseq (gref bool) (SZ.v n))
  // returns
  //   i_done : erased (iname & erased (seq (gref bool)))
  requires
    pure (len done == SZ.v n) **
    bigstar 0 (SZ.v n) (fun tid ->
     (done @! tid) |-> Frac 0.5R true **
     inv i (inv_p (SZ.v n) a v_a r done))
  ensures
    (a |-> Frac f v_a) **
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a) **
    pure (SZ.v n <= 1024)
{
  admit();
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (n : szp{n < max_blocks})
  (a : gpu_array et n)
  (#f : perm)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (#r0 : erased et)
  (* FIXME: these two arguments should be created by the setup
     and passed to each block. We should extend the kernel_desc
     to allow for that. *)
  (done : erased (seq (gref bool)){len done == SZ.v n})
  (i : iname)
: kernel_desc
    ((a |-> Frac f v_a) **
      (r |-> r0))
    ((a |-> Frac f v_a) **
      (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a))
 = {
  nblk = n;
  f = kf a #v_a r done i;
  setup = magic();
  teardown = magic();
  kpre  = kpre  n a v_a r done i;
  kpost = kpost n a v_a r done i;
  frame = emp;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn reduce
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (n : szp {n < max_blocks})
  (a : gpu_array et n)
  (#f : perm)
  (#v_a : erased (seq et))
  requires
    cpu **
    pure (f == 1.0R) **
    (a |-> Frac f v_a) **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024)
  returns
    r : et
  ensures
    cpu **
    (a |-> Frac f v_a) **
    pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a)
{
  let mut r : et = zero;
  let gr = gpu_alloc0 #et ();
  Kuiper.Ref.gpu_memcpy_host_to_device #et #_ gr r;

  with v. assert (pts_to r v);
  assert (pure (v == zero));

  assert (pure (n < max_blocks));

  // let i_done = setup n a gr;
  // let i = (i_done)._1;
  // let done : erased (seq (gref bool)) = hide (reveal (i_done._2));
  // rewrite each i_done as (i, done) by (tadmit());
  // New fancy syntax, does not extract
  // let Mktuple2 i done = setup n a gr;
  // The problem is essentially https://github.com/FStarLang/pulse/issues/93.
  // The match is over a (Pulse) non-informative type, a pair of iname and erased (seq (gref bool)).
  // It gets erased to unit, but the patterns do not, so the match is ill-typed and krml complains.

  // W.elim_with_pure (len done == SZ.v n) _;

  let done = magic #(erased (seq (gref bool))) ();
  let i    = magic #iname ();
  assume (pure (len done == SZ.v n));
  // (done : erased (seq (gref bool)){len done == SZ.v n})

  launch_sync (kdesc n a #f #v_a gr #(zero #et) done i);

  // launch_kernel_n_blocks n
  //   #(kpre  (SZ.v n) a v_a gr done i)
  //   #(kpost (SZ.v n) a v_a gr done i)
  //   (fun tid -> k (hide n) a gr done i v_a tid);

  // forevery_tostar #(natlt (SZ.v n))
  //   (kpost (SZ.v n) a v_a gr done i);

  // teardown n a #f #v_a gr i done;

  Kuiper.Ref.gpu_memcpy_device_to_host r gr #_ #_ #_;

  Kuiper.Ref.gpu_free gr;

  let v = !r;
  v
}


(*
1. Let-binding a tuple, or any pattern really
2. Easily returning multiple things, writing specs over the exploded components
3-  refinements have to match exactly
4- err locations
4.1-  in particular when function type does not match fsti
5- jonas' coerce_eq bug
*)
