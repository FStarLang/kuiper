module Kuiper.Kernel.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Tensor.Layout.Alg

module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module W = Pulse.Lib.WithPure
module CInv = Pulse.Lib.CancellableInvariant

(* These should go somewhere in the library *)
ghost
fn bring (p : slprop)
  preserves gpu
  requires  on gpu_loc p
  ensures   p
{
  unfold gpu;
  with l. assert (loc l);
  gpu_of_idem l;
  rewrite (on gpu_loc p) as (on l p);
  on_elim p;
  fold gpu;
}

ghost
fn putback (p : slprop)
  preserves gpu
  requires  p
  ensures   on gpu_loc p
{
  unfold gpu;
  with l. assert (loc l);
  gpu_of_idem l;
  on_intro p;
  rewrite (on l p) as (on gpu_loc p);
  fold gpu;
}

(* Relating a sequence of erased bools to v_r. Essentially, v_r containts
exaclty the contributions of the indices i where Seq.index v_done i is true. *)
let rec contributions'
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn : nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  (v_r : et)
  (acc : et)
  (i : natle nn)
: Tot prop (decreases i)
=
  if i = 0 then
    v_r == acc
  else
    let hd = Kuiper.Chest.acc v_a (i-1, ()) in
    let hd_done = Seq.index v_done (i-1) in
    if hd_done then
      contributions' nn v_done v_a v_r (d.pure_op hd acc) (i-1)
    else
      contributions' nn v_done v_a v_r acc (i-1)

let contributions
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (nn : nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  (v_r : et)
: Tot prop
= contributions' nn v_done v_a v_r zero nn

let inv_p'
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (v_a: chest1 et nn)
  (r: gpu_ref et)
  (done: seq (gref bool) {len done == nn})
  (v_done: seq bool {len v_done == nn})
  (v_r : et)
  : slprop
=
  on gpu_loc (r |-> v_r) **
  (forall+ (i : natlt nn). (Seq.index done i) |-> Frac 0.5R (Seq.index v_done i)) **
  pure (contributions nn v_done v_a v_r)

(* Invariant. The reference r (on GPU) contains a value that is
exactly the contributions expected. *)
let inv_p
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (v_a : chest1 et nn)
  (r: gpu_ref et)
  (done: seq (gref bool))
  (#_ : squash (len done == nn))
=
  exists*
    (v_done: seq bool {len v_done == nn})
    (v_r : et).
    inv_p' nn v_a r done v_done v_r

(* Permission for thread i out of n threads.
   Split by recursive halving: thread 0 gets p/2,
   remaining n-1 threads share the other p/2. *)
let rec tperm (n : nat{n > 0}) (i : nat{i < n}) (p : perm)
  : Tot perm (decreases n)
=
  if n = 1 then p
  else if i = 0 then p /. 2.0R
  else tperm (n-1) (i-1) (p /. 2.0R)

unfold
let kpre
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (nn: nat)
  (a: array1 et (repr nn))
  (v_a : chest1 et nn)
  (r: gpu_ref et)
  (done: seq (gref bool) {len done == nn})
  (c: CInv.cinv)
  (tid : natlt nn)
=
  (Seq.index done tid) |-> Frac 0.5R false **
  (a |-> Frac (1.0R /. nn) v_a) **
  inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p nn v_a r done)) **
  CInv.active c (tperm nn tid 1.0R)

unfold
let kpost
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (nn: nat)
  (a : array1 et (repr nn))
  (v_a : chest1 et nn)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (c: CInv.cinv)
  (tid : natlt nn)
=
  (Seq.index done tid) |-> Frac 0.5R true **
  (a |-> Frac (1.0R /. nn) v_a) **
  CInv.active c (tperm nn tid 1.0R)

ghost
fn forevery_ghost_upd_lemma
  (nn : nat)
  (done : seq (gref bool) {len done == nn})
  (v_done : seq bool {len v_done == nn})
  (tid : nat{tid < nn})
  requires
    (forall+ (i : natlt nn). (Seq.index done i) |-> Frac 0.5R (Seq.index v_done i)) **
    (Seq.index done tid) |-> Frac 0.5R false
  ensures
    (forall+ (i : natlt nn). (Seq.index done i) |-> Frac 0.5R (Seq.index (Seq.upd v_done tid true) i)) **
    (Seq.index done tid) |-> Frac 0.5R true **
    pure (Seq.index v_done tid == false)
{
  forevery_extract'
    tid
    (fun (i : natlt nn) -> (Seq.index done i) |-> Frac 0.5R (Seq.index v_done i));

  Pulse.Lib.GhostReference.gather (Seq.index done tid);
  Pulse.Lib.GhostReference.write (Seq.index done tid) true;
  Pulse.Lib.GhostReference.share (Seq.index done tid);

  Pulse.Lib.Forall.elim_forall
    (fun (i : natlt nn) -> (Seq.index done i) |-> Frac 0.5R (Seq.index (Seq.upd v_done tid true) i));

  rewrite ((Seq.index done tid) |-> Frac 0.5R true)
      as  ((Seq.index done tid) |-> Frac 0.5R (Seq.index (Seq.upd v_done tid true) tid));

  Pulse.Lib.Trade.elim_trade _ _;
}

let tail_upd_0 (#a:Type) (s:seq a{Seq.length s > 0}) (v:a)
  : Lemma (Seq.tail (Seq.upd s 0 v) == Seq.tail s)
= let s' = Seq.upd s 0 v in
  let t1 = Seq.tail s' in
  let t2 = Seq.tail s in
  assert (Seq.length t1 == Seq.length t2);
  let aux (i:nat{i < Seq.length t2})
    : Lemma (Seq.index t1 i == Seq.index t2 i)
  = Seq.lemma_index_upd2 s 0 v (i + 1);
    FStar.Seq.Properties.index_tail s' i;
    FStar.Seq.Properties.index_tail s i
  in
  FStar.Classical.forall_intro aux;
  Seq.lemma_eq_intro t1 t2

let tail_upd_succ (#a:Type) (s:seq a{Seq.length s > 0}) (n:nat{n > 0 /\ n < Seq.length s}) (v:a)
  : Lemma (Seq.tail (Seq.upd s n v) == Seq.upd (Seq.tail s) (n - 1) v)
= let s' = Seq.upd s n v in
  let t1 = Seq.tail s' in
  let t2 = Seq.upd (Seq.tail s) (n - 1) v in
  assert (Seq.length t1 == Seq.length t2);
  let aux (i:nat{i < Seq.length t2})
    : Lemma (Seq.index t1 i == Seq.index t2 i)
  = FStar.Seq.Properties.index_tail s' i;
    FStar.Seq.Properties.index_tail s i;
    if i = n - 1 then
      Seq.lemma_index_upd1 s n v
    else begin
      Seq.lemma_index_upd2 s n v (i + 1);
      Seq.lemma_index_upd2 (Seq.tail s) (n - 1) v i
    end
  in
  FStar.Classical.forall_intro aux;
  Seq.lemma_eq_intro t1 t2

(* Adding x to the accumulator is equivalent to adding x to v_r. *)
let rec contributions_shift
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn : nat) (v_done : lseq bool nn) (v_a : chest1 et nn)
  (v_r : et) (acc : et) (x : et)
  (i : natle nn)
  : Lemma (requires contributions' nn v_done v_a v_r acc i)
          (ensures  contributions' nn v_done v_a (d.pure_op x v_r) (d.pure_op x acc) i)
          (decreases i)
= if i = 0 then ()
  else
    let hd = Kuiper.Chest.acc v_a (i-1, ()) in
    let hd_done = Seq.index v_done (i-1) in
    if hd_done then begin
      contributions_shift ac nn v_done v_a v_r (d.pure_op hd acc) x (i-1);
      ac.assoc x hd acc;
      ac.comm x hd;
      ac.assoc hd x acc
    end else
      contributions_shift ac nn v_done v_a v_r acc x (i-1)

let rec contributions_lemma
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn: nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  (v_r : et)
  (tid : natlt nn)
  : Lemma (requires contributions nn v_done v_a v_r /\ Seq.index v_done tid == false)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (Kuiper.Chest.acc v_a (tid, ())) v_r))
          (decreases tid)
= admit()
(*
if tid = 0 then begin
    tail_upd_0 v_done true;
    contributions_shift ac nn (Seq.tail v_done) (Seq.tail v_a) v_r acc (Seq.head v_a)
  end else begin
    tail_upd_succ v_done tid true;
    FStar.Seq.Properties.index_tail v_a (tid - 1);
    FStar.Seq.Properties.index_tail v_done (tid - 1);
    if Seq.head v_done then
      contributions_lemma ac nn (Seq.tail v_done) (Seq.tail v_a) v_r (d.pure_op (Seq.head v_a) acc) (tid - 1)
    else
      contributions_lemma ac nn (Seq.tail v_done) (Seq.tail v_a) v_r acc (tid - 1)
  end
  *)

let is_ac_from_ac_w (#t:Type) (#f: t -> t -> t) (ac : is_ac_w f)
  : Lemma (is_ac f)
= FStar.Classical.forall_intro_2 (fun x y -> ac.comm x y);
  FStar.Classical.forall_intro_3 (fun x y z -> ac.assoc x y z)

let contributions_lemma_smt
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  (v_r : et)
  (tid : natlt nn)
  : Lemma (requires contributions nn v_done v_a v_r /\ Seq.index v_done tid == false /\ is_ac d.pure_op)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (Kuiper.Chest.acc v_a (tid, ())) v_r))
          [SMTPat (contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (Kuiper.Chest.acc v_a (tid, ())) v_r))]
= contributions_lemma { comm = (fun x y -> ()); assoc = (fun x y z -> ()) } nn v_done v_a v_r tid

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (ac : is_ac_w d.pure_op)
  (#nn: SZ.t)
  (a : array1 et (repr nn))
  (#v_a: chest1 et nn)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == SZ.v nn})
  (c : CInv.cinv)
  (bid : szlt (SZ.v nn))
  ()
  requires
    gpu **
    kpre (SZ.v nn) a v_a r done c bid **
    block_id (SZ.v nn) bid
  ensures
    gpu **
    kpost (SZ.v nn) a v_a r done c bid **
    block_id (SZ.v nn) bid
{
  (* Read our value *)
  let v = tensor_read a (bid, ());

  (* Atomically add it to result, marking our contribution as done. *)
  with_invariants unit emp_inames (CInv.iname_of c)
    (CInv.cinv_vp c (inv_p (SZ.v nn) v_a r done))
    (gpu **
     (Seq.index done bid) |-> Frac 0.5R false **
     a |-> Frac (1.0R /. SZ.v nn) v_a **
     CInv.active c (tperm (SZ.v nn) bid 1.0R))
    (fun _ ->
     gpu **
     (Seq.index done bid) |-> Frac 0.5R true **
     a |-> Frac (1.0R /. SZ.v nn) v_a **
     CInv.active c (tperm (SZ.v nn) bid 1.0R))
  fn _
  {
    CInv.unpack_cinv_vp c;
    is_ac_from_ac_w ac;
    unfold inv_p nn v_a r done;
    unfold inv_p' nn v_a r done;
    with v_r. assert on gpu_loc (r |-> v_r);
    bring (r |-> v_r);
    let _ = atomic_add r v;
    putback (r |-> (d.pure_op v v_r));
    forevery_ghost_upd_lemma nn done _ _;
    fold inv_p' nn v_a r done _ _;
    fold inv_p nn v_a r done;
    CInv.pack_cinv_vp c;
  }
}

let rec contributions_all_done
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn : nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  (v_r : et)
  : Lemma (requires contributions nn v_done v_a v_r
                /\ (forall (i : nat{i < nn}). Seq.index v_done i == true))
          (ensures v_r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero (chest1_to_seq v_a))
          (decreases nn)
= admit()
// if nn = 0 then ()
//   else begin
//     assert (Seq.index v_done 0 == true);
//     let aux (i:nat{i < len (Seq.tail v_a)})
//       : Lemma (Seq.index (Seq.tail v_done) i == true)
//     = FStar.Seq.Properties.index_tail v_done i
//     in
//     FStar.Classical.forall_intro aux;
//     ac.comm (Seq.head v_a) acc;
//     contributions_all_done ac nn (Seq.tail v_done) (Seq.tail v_a) v_r (d.pure_op (Seq.head v_a) acc)
//   end

let rec contributions_init
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat)
  (v_done : lseq bool nn)
  (v_a : chest1 et nn)
  : Lemma (requires forall (i : nat{i < nn}). Seq.index v_done i == false)
          (ensures contributions nn v_done v_a zero)
          // (decreases i)
= admit()
(*if len v_a = 0 then ()
  else begin
    assert (Seq.index v_done 0 == false);
    let aux (i:nat{i < len (Seq.tail v_a)})
      : Lemma (Seq.index (Seq.tail v_done) i == false)
    = FStar.Seq.Properties.index_tail v_done i
    in
    FStar.Classical.forall_intro aux;
    contributions_init nn (Seq.tail v_done) (Seq.tail v_a)
  end *)

ghost
fn rec allocate_ref_seq (n : nat)
  returns s : erased (lseq (gref bool) n)
  ensures (forall+ (i : natlt n). (Seq.index s i) |-> false)
  decreases n
{
  if (n = 0) {
    let s = Seq.empty #(gref bool);
    forevery_intro_empty (fun (i : natlt n) -> (Seq.index s i) |-> false);
    s
  } else {
    let tail = allocate_ref_seq (n - 1);
    let r = Pulse.Lib.GhostReference.alloc false;
    let s = Seq.cons r tail;
    rewrite r |-> false as (Seq.index s 0) |-> false;
    forevery_map
      (fun (i : natlt (n-1)) -> (Seq.index tail i) |-> false)
      (fun (i : natlt (n-1)) -> (Seq.index s (i+1)) |-> false)
      fn i {
        rewrite each (Seq.index tail i) as (Seq.index s (i+1));
        ();
      };
    assert (Seq.index s 0) |-> false;
    assert forall+ (i : natlt (n-1)). (Seq.index s (i+1)) |-> false;
    forevery_natlt_push_shift n (fun (i : natlt n) -> (Seq.index s i) |-> false);
    s
  }
}

ghost
fn allocate_ref_seq' (n : nat)
  returns s : erased (seq (gref bool))
  ensures W.with_pure (len s == n)
            (fun _ -> forall+ (i : natlt n). (Seq.index s i) |-> false)
{
  let s = allocate_ref_seq n;
  Ghost.hide (Ghost.reveal s)
}

(* Split active c p into n per-thread fractions via recursive halving. *)
ghost
fn rec share_active_n (#p:perm) (c : CInv.cinv) (n : nat{n > 0})
  requires CInv.active c p
  ensures forall+ (i : natlt n). CInv.active c (tperm n i p)
  decreases n
{
  if (n = 1) {
    rewrite CInv.active c p
        as  CInv.active c (tperm n 0 p);
    forevery_singleton_intro'
      (fun (i : natlt n) -> CInv.active c (tperm n i p)) (0 <: natlt n);
  } else {
    CInv.share c;
    share_active_n c (n-1);
    forevery_map
      (fun (i : natlt (n-1)) -> CInv.active c (tperm (n-1) i (p /. 2.0R)))
      (fun (i : natlt (n-1)) -> CInv.active c (tperm n (i+1) p))
      fn i {
        rewrite CInv.active c (tperm (n-1) i (p /. 2.0R))
            as  CInv.active c (tperm n (i+1) p);
      };
    rewrite CInv.active c (p /. 2.0R)
        as  CInv.active c (tperm n 0 p);
    forevery_natlt_push_shift n
      (fun (i : natlt n) -> CInv.active c (tperm n i p));
  }
}

(* Gather n per-thread active fractions back into a single active c p. *)
ghost
fn rec gather_active_n (#p:perm) (c : CInv.cinv) (n : nat{n > 0})
  requires forall+ (i : natlt n). CInv.active c (tperm n i p)
  ensures CInv.active c p
  decreases n
{
  if (n = 1) {
    forevery_singleton_elim'
      (fun (i : natlt n) -> CInv.active c (tperm n i p)) (0 <: natlt n);
    rewrite CInv.active c (tperm n 0 p)
        as  CInv.active c p;
  } else {
    forevery_natlt_pop_shift n
      (fun (i : natlt n) -> CInv.active c (tperm n i p));
    rewrite CInv.active c (tperm n 0 p)
        as  CInv.active c (p /. 2.0R);
    forevery_map
      (fun (i : natlt (n-1)) -> CInv.active c (tperm n (i+1) p))
      (fun (i : natlt (n-1)) -> CInv.active c (tperm (n-1) i (p /. 2.0R)))
      fn i {
        rewrite CInv.active c (tperm n (i+1) p)
            as  CInv.active c (tperm (n-1) i (p /. 2.0R));
      };
    gather_active_n c (n-1);
    CInv.gather c;
    rewrite CInv.active c (p /. 2.0R +. p /. 2.0R)
        as  CInv.active c p;
  }
}

(* Setup: receives ghost ref halves, cancellable invariant, and active permission
   (allocated on CPU), splits array and active ownership, and distributes into
   per-block preconditions. *)
ghost
fn setup
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n <= max_blocks})
  (a : array1 et (repr n))
  (#v_a : chest1 et n)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == SZ.v n})
  (c : CInv.cinv)
  ()
  norewrite
  requires
    (a |-> v_a) **
    (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R false) **
    inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)) **
    CInv.active c 1.0R
  ensures
    (forall+ (bid : natlt n). kpre (SZ.v n) a v_a r done c bid) **
    inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done))
{
  (* Share the array into n fractions *)
  tensor_share_n a n;

  (* Split active permission into n per-thread fractions *)
  share_active_n c (SZ.v n);

  (* Zip ghost halves with array fractions *)
  forevery_zip
    (fun (tid:natlt (SZ.v n)) -> (Seq.index done tid) |-> Frac 0.5R false)
    (fun (tid:natlt (SZ.v n)) -> a |-> Frac (1.0R /. SZ.v n) v_a);

  (* Zip (ghost+array) with active fractions *)
  forevery_zip
    (fun (tid:natlt (SZ.v n)) ->
      (Seq.index done tid) |-> Frac 0.5R false **
      (a |-> Frac (1.0R /. SZ.v n) v_a))
    (fun (tid:natlt (SZ.v n)) -> CInv.active c (tperm (SZ.v n) tid 1.0R));

  (* Duplicate invariant into each forall+ element *)
  forevery_map_extra
    (inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)))
    (fun (tid:natlt (SZ.v n)) ->
      (((Seq.index done tid) |-> Frac 0.5R false **
       (a |-> Frac (1.0R /. SZ.v n) v_a)) **
       CInv.active c (tperm (SZ.v n) tid 1.0R)))
    (fun (tid:natlt (SZ.v n)) ->
      (Seq.index done tid) |-> Frac 0.5R false **
      (a |-> Frac (1.0R /. SZ.v n) v_a) **
      inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)) **
      CInv.active c (tperm (SZ.v n) tid 1.0R))
    fn tid {
      dup_inv (CInv.iname_of c)
              (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done))
    };

  ();
}

(* Teardown: gathers array fractions, active fractions, and collects ghost ref
   halves (now true). Returns full active permission. *)
ghost
fn teardown
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n <= max_blocks})
  (a : array1 et (repr n))
  (#v_a : chest1 et n)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == SZ.v n})
  (c : CInv.cinv)
  ()
  norewrite
  requires
    (forall+ (bid : natlt n). kpost (SZ.v n) a v_a r done c bid) **
    inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done))
  ensures
    (a |-> v_a) **
    (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R true) **
    inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)) **
    CInv.active c 1.0R
{
  (* Unzip done from (array ** active) *)
  forevery_unzip
    (fun (bid:natlt (SZ.v n)) -> (Seq.index done bid) |-> Frac 0.5R true)
    (fun (bid:natlt (SZ.v n)) ->
      (a |-> Frac (1.0R /. SZ.v n) v_a) **
      CInv.active c (tperm (SZ.v n) bid 1.0R));

  (* Unzip array from active *)
  forevery_unzip
    (fun (bid:natlt (SZ.v n)) -> a |-> Frac (1.0R /. SZ.v n) v_a)
    (fun (bid:natlt (SZ.v n)) -> CInv.active c (tperm (SZ.v n) bid 1.0R));

  (* Gather array fractions *)
  tensor_gather_n a n;

  (* Gather active fractions *)
  gather_active_n c (SZ.v n);
}

(* Kernel descriptor: the kernel receives the cancellable invariant, ghost ref
   halves, and active permission. Each thread flips its ghost ref to true via
   atomic_add under the invariant. The full_post returns everything needed
   to cancel the invariant and recover the result. *)
inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#repr : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (repr l)) |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n <= max_blocks})
  (a : array1 et (repr n) { is_global a })
  (#v_a : chest1 et n)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)))
  (#_ : squash (len done == SZ.v n))
  (c : CInv.cinv)
: kernel_desc
    ((a |-> v_a) **
      (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R false) **
      inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)) **
      CInv.active c 1.0R)
    ((a |-> v_a) **
      (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R true) **
      inv (CInv.iname_of c) (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done)) **
      CInv.active c 1.0R)
 = {
  nblk     = n;
  f        = kf ac a #v_a r done c;
  setup    = setup ac n a #v_a r done c;
  teardown = teardown ac n a #v_a r done c;
  kpre     = kpre  (SZ.v n) a v_a r done c;
  kpost    = kpost (SZ.v n) a v_a r done c;
  frame    = inv (CInv.iname_of c)
                 (CInv.cinv_vp c (inv_p (SZ.v n) v_a r done));
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn reduce
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (#r : (l:nat -> layout1 l)) {| (l:sz -> ctlayout (r l)) |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n <= max_blocks})
  (a : array1 et (r n) { is_global a })
  (#v_a : chest1 et n)
  norewrite (* needed to match spec in fsti... they do not get elaborated *)
  preserves
    cpu ** on gpu_loc (a |-> v_a)
  returns
    r : et
  ensures
    pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero (chest1_to_seq v_a))
{
  let mut r : et = zero;
  let gr = gpu_alloc0 #et ();
  Kuiper.Ref.gpu_memcpy_host_to_device #et #_ gr r;

  (* --- CPU-side invariant setup --- *)

  (* Allocate n ghost refs, all false *)
  let done = allocate_ref_seq' (SZ.v n);

  let falses : erased (seq bool) = Seq.init_ghost (SZ.v n) (fun _ -> false);

  (* Share each ghost ref into two halves *)
  forevery_map
    #(natlt (SZ.v n))
    (fun i -> (Seq.index done i) |-> false)
    (fun i -> (Seq.index done i) |-> Frac 0.5R false ** (Seq.index done i) |-> Frac 0.5R (Seq.index falses i))
    fn i {
      Pulse.Lib.GhostReference.share (Seq.index done i);
      ();
    };
  forevery_unzip _ _;

  (* Establish the initial contributions predicate *)
  contributions_init (SZ.v n) falses (reveal v_a);

  fold inv_p' n v_a gr done falses zero;
  fold inv_p n v_a gr done;

  (* Create a cancellable invariant instead of a plain one *)
  let c = CInv.new_cancellable_invariant (inv_p (SZ.v n) v_a gr done);

  (* Move resources to GPU *)
  placeless_on_intro
    (inv (CInv.iname_of c)
         (CInv.cinv_vp c (inv_p (SZ.v n) v_a gr done #())))
    gpu_loc;

  placeless_on_intro
    (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R false)
    gpu_loc;

  placeless_on_intro
    (CInv.active c 1.0R)
    gpu_loc;

  (* --- Launch kernel --- *)
  launch_sync (kdesc ac n a #v_a gr done c);

  (* Bring back from GPU *)
  placeless_on_elim
    (inv (CInv.iname_of c)
         (CInv.cinv_vp c (inv_p (SZ.v n) v_a gr done #())))
    gpu_loc;

  placeless_on_elim
    (forall+ (tid : natlt n). (Seq.index done tid) |-> Frac 0.5R true)
    gpu_loc;

  placeless_on_elim
    (CInv.active c 1.0R)
    gpu_loc;

  (* Cancel the invariant to recover the protected value *)
  later_credit_buy 1;
  CInv.cancel c;

  (* At this point, we have inv_p, which means 'done' (with perm 0.5) points to
  a sequence of values related to the contributions into v_r. But we know these
  values are all true via the kernel post. *)
  unfold inv_p (SZ.v n) v_a gr done;
  with v_done v_r.
    unfold inv_p' (SZ.v n) v_a gr done v_done v_r;
  assert
    (forall+ (i : natlt (SZ.v n)). (Seq.index done i) |-> Frac 0.5R true) **
    (forall+ (i : natlt (SZ.v n)). (Seq.index done i) |-> Frac 0.5R (Seq.index v_done i));

  forevery_zip
    (fun (i : natlt (SZ.v n)) -> (Seq.index done i) |-> Frac 0.5R true)
    (fun (i : natlt (SZ.v n)) -> (Seq.index done i) |-> Frac 0.5R (Seq.index v_done i));
  forevery_map
    (fun (i : natlt (SZ.v n)) -> ((Seq.index done i) |-> Frac 0.5R true) **
                               ((Seq.index done i) |-> Frac 0.5R (Seq.index v_done i)))
    (fun (i : natlt (SZ.v n)) -> ((Seq.index done i) |-> true) ** pure (Seq.index v_done i == true))
    fn i {
      Pulse.Lib.GhostReference.gather (Seq.index done i);
    };
  forevery_extract_pure
    (fun (i : natlt (SZ.v n)) -> (Seq.index done i) |-> true ** pure (Seq.index v_done i == true))
    (fun (i : natlt (SZ.v n)) -> (Seq.index v_done i) == true)
    fn _ {};

  assert pure (contributions n v_done v_a v_r);
  contributions_all_done ac n v_done v_a v_r;
  assert pure (v_r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero (chest1_to_seq v_a));

  (* Drop ghost state *)
  drop_ (forall+ (i : natlt (SZ.v n)). (Seq.index done i) |-> true ** pure (Seq.index v_done i == true));

  Kuiper.Ref.gpu_memcpy_device_to_host r gr #_ #_ #_;

  Kuiper.Ref.gpu_free gr;

  let v = !r;
  v
}
