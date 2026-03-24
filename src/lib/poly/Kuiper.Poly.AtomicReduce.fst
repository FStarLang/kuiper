module Kuiper.Poly.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics

module SZ = Kuiper.SizeT
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
    ((r |-> v_r) **
    (forall+ (i : natlt (len done)). (done @! i) |-> Frac 0.5R (v_done @! i))) **
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
  (a |-> Frac (1.0R /. nn) v_a) **
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
  (a |-> Frac (1.0R /. nn) v_a)

ghost
fn forevery_ghost_upd_lemma
  (done : seq (gref bool))
  (v_done : seq bool{len v_done >= len done})
  (tid : nat{tid < len done})
  requires
    (forall+ (i : natlt (len done)). (done @! i) |-> Frac 0.5R (v_done @! i)) **
    (done @! tid) |-> Frac 0.5R false
  ensures
    (forall+ (i : natlt (len done)). (done @! i) |-> Frac 0.5R (Seq.upd v_done tid true @! i)) **
    (done @! tid) |-> Frac 0.5R true **
    pure (v_done @! tid == false)
{
  forevery_extract'
    tid
    (fun (i : natlt (len done)) -> (done @! i) |-> Frac 0.5R (v_done @! i));

  Pulse.Lib.GhostReference.gather (done @! tid);
  Pulse.Lib.GhostReference.write (done @! tid) true;
  Pulse.Lib.GhostReference.share (done @! tid);

  Pulse.Lib.Forall.elim_forall
    (fun (i : natlt (len done)) -> (done @! i) |-> Frac 0.5R (Seq.upd v_done tid true @! i));

  rewrite ((done @! tid) |-> Frac 0.5R true)
      as  ((done @! tid) |-> Frac 0.5R (Seq.upd v_done tid true @! tid));

  Pulse.Lib.Trade.elim_trade _ _;
}

private
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

private
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

private
let rec contributions_shift
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn : nat) (v_done : seq bool) (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et) (x : et)
  : Lemma (requires contributions nn v_done v_a v_r acc)
          (ensures contributions nn v_done v_a (d.pure_op x v_r) (d.pure_op x acc))
          (decreases len v_a)
= if len v_a = 0 then ()
  else
    let hd = Seq.head v_a in
    let tl = Seq.tail v_a in
    let hd_done = Seq.head v_done in
    let tl_done = Seq.tail v_done in
    if hd_done then begin
      contributions_shift ac nn tl_done tl v_r (d.pure_op hd acc) x;
      ac.assoc x hd acc;
      ac.comm x hd;
      ac.assoc hd x acc
    end else
      contributions_shift ac nn tl_done tl v_r acc x

let rec contributions_lemma
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn: nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
  (tid : nat{tid < len v_a})
  : Lemma (requires contributions nn v_done v_a v_r acc /\ v_done @! tid == false)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc)
          (decreases len v_a)
= if tid = 0 then begin
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

private
let is_ac_from_ac_w (#t:Type) (#f: t -> t -> t) (ac : is_ac_w f)
  : Lemma (is_ac f)
= FStar.Classical.forall_intro_2 (fun x y -> ac.comm x y);
  FStar.Classical.forall_intro_3 (fun x y z -> ac.assoc x y z)

private
let contributions_lemma_smt
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
  (tid : nat{tid < len v_a})
  : Lemma (requires contributions nn v_done v_a v_r acc /\ v_done @! tid == false /\ is_ac d.pure_op)
          (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc)
          [SMTPat (contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc)]
= contributions_lemma { comm = (fun x y -> ()); assoc = (fun x y z -> ()) } nn v_done v_a v_r acc tid

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (#nn: SZ.t)
  (a : gpu_global_array et (SZ.v nn))
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
  // later_credit_buy 1;
  (* Read array at idx *)
  let v = gpu_array_read a bid;
  (* Fetch and add into result cell. *)
  //  admit();
  with_invariants unit emp_inames i (inv_p (SZ.v nn) a v_a r done)
    (gpu_pts_to_slice a #(1.0R /. SZ.v nn) 0 (SZ.v nn) v_a **
     block_id (SZ.v nn) (SZ.v bid) **
     gpu **
     Pulse.Lib.GhostReference.pts_to (Seq.Base.index done (SZ.v bid)) #0.5R false)
    (fun _ ->
      gpu **
      (done @! bid) |-> Frac 0.5R true **
       gpu_pts_to_slice a #(1.0R /. SZ.v nn) 0 (SZ.v nn) v_a **
      block_id (SZ.v nn) bid)
  fn _
  {
    // later_elim _;
    is_ac_from_ac_w ac;
    unfold (inv_p (SZ.v nn) a v_a r done);
    let _ = atomic_add r v;
    forevery_ghost_upd_lemma done _ _;
    fold (inv_p (SZ.v nn) a v_a r done);
  }
}

private
let rec contributions_all_done
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn : nat) (v_done : seq bool) (v_a : seq et{len v_done >= len v_a})
  (v_r : et) (acc : et)
  : Lemma (requires contributions nn v_done v_a v_r acc
                /\ (forall (i:nat{i < len v_a}). v_done @! i == true))
          (ensures v_r == Kuiper.Seq.Common.seq_fold_left d.pure_op acc v_a)
          (decreases len v_a)
= if len v_a = 0 then ()
  else begin
    assert (v_done @! 0 == true);
    let aux (i:nat{i < len (Seq.tail v_a)})
      : Lemma (Seq.tail v_done @! i == true)
    = FStar.Seq.Properties.index_tail v_done i
    in
    FStar.Classical.forall_intro aux;
    ac.comm (Seq.head v_a) acc;
    contributions_all_done ac nn (Seq.tail v_done) (Seq.tail v_a) v_r (d.pure_op (Seq.head v_a) acc)
  end

ghost
fn done_lemma
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (nn: szp)
  (a : gpu_array et nn)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == SZ.v nn})
  (i : iname)
  (#v_a : erased (seq et))
  requires
    forall+ (tid : natlt nn). kpost nn a v_a r done i tid
  ensures
    (a |-> v_a) **
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a)
{
  forevery_unzip _ _;
  gpu_slice_gather a 0 (SZ.v nn) (SZ.v nn) #1.0R #v_a;
  (* Blocked: invariant cancellation not yet in Pulse — can't permanently
     extract r |-> v_r from inv i (inv_p ...). *)
  admit();
}

private
let rec contributions_init
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat) (v_done : seq bool) (v_a : seq et{len v_done >= len v_a})
  : Lemma (requires forall (i:nat{i < len v_a}). v_done @! i == false)
          (ensures contributions nn v_done v_a zero zero)
          (decreases len v_a)
= if len v_a = 0 then ()
  else begin
    assert (v_done @! 0 == false);
    let aux (i:nat{i < len (Seq.tail v_a)})
      : Lemma (Seq.tail v_done @! i == false)
    = FStar.Seq.Properties.index_tail v_done i
    in
    FStar.Classical.forall_intro aux;
    contributions_init nn (Seq.tail v_done) (Seq.tail v_a)
  end

ghost
fn setup0
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (n : sz{SZ.v n > 0})
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  requires
    (a |-> v_a) **
    (r |-> zero #et) **
    pure (SZ.v n <= 1024 /\ len v_a == SZ.v n)
  returns
    i_done : iname & erased (seq (gref bool))
  ensures
    (match i_done with | (i, done) ->
      W.with_pure (len done == SZ.v n) (fun _ ->
       (forall+ (tid : natlt n).
        kpre (SZ.v n) a v_a r done i tid)
    ))
{
  (* Allocate n ghost refs, all false *)
  let done : erased (seq (gref bool)) =
    admit(); (* ghost ref allocation — need Seq.init_ghost of GhostReference.alloc *)

  assume (len done == SZ.v n);

  (* Share each ghost ref into two halves:
     forall+ tid. (done @! tid) |-> false
       ==> forall+ tid. (done @! tid) |-> Frac 0.5R false
        ** forall+ tid. (done @! tid) |-> Frac 0.5R false *)
  assume (forall+ (tid : natlt (SZ.v n)). (done @! tid) |-> Frac 0.5R false);
  assume (forall+ (i0 : natlt (len done)). (done @! i0) |-> Frac 0.5R false);

  (* Build the initial inv_p content and create the invariant *)
  contributions_init (SZ.v n) (Seq.init_ghost (SZ.v n) (fun _ -> false)) (reveal v_a);
  assume (inv_p (SZ.v n) a v_a r done);
  let i = new_invariant (inv_p (SZ.v n) a v_a r done);

  (* Share the array into n fractions *)
  gpu_slice_share a 0 (SZ.v n) (SZ.v n) #1.0R;

  (* Duplicate the invariant into each branch of the forall+ *)
  forevery_zip _ _;
  forevery_map_extra
    (inv i (inv_p (SZ.v n) a v_a r done))
    _
    (fun (tid:natlt (SZ.v n)) ->
      (done @! tid) |-> Frac 0.5R false **
      (a |-> Frac (1.0R /. SZ.v n) v_a) **
      inv i (inv_p (SZ.v n) a v_a r done))
    fn tid { dup_inv i (inv_p (SZ.v n) a v_a r done) };

  W.intro_with_pure (len done == SZ.v n) _ ();
  (i, done)
}

ghost
fn setup
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (n : sz{SZ.v n > 0})
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (done : seq (gref bool){len done == SZ.v n})
  (i : iname)
  ()
  norewrite
  requires
    a |-> v_a **
    inv i (inv_p (SZ.v n) a v_a r done)
  ensures
    (forall+ (bid : natlt n). kpre n a v_a r done i bid) **
    inv i (inv_p (SZ.v n) a v_a r done)
{
  (* Ghost ref allocation/sharing and invariant creation — assumed.
     Real proof: alloc ghost refs, share into halves, fold inv_p
     (consuming r |-> zero + ghost halves + contributions_init pure),
     then new_invariant. *)
  assume (forall+ (tid : natlt (SZ.v n)). (done @! tid) |-> Frac 0.5R false);
  assume (inv i (inv_p (SZ.v n) a v_a r done));

  (* Share the array into n fractions — proved *)
  gpu_slice_share a 0 (SZ.v n) (SZ.v n) #1.0R;

  (* Zip ghost halves with array fractions — proved *)
  forevery_zip
    (fun (tid:natlt (SZ.v n)) -> (done @! tid) |-> Frac 0.5R false)
    (fun (tid:natlt (SZ.v n)) -> a |-> Frac (1.0R /. SZ.v n) v_a);

  (* Duplicate invariant into each forall+ element — proved *)
  forevery_map_extra
    (inv i (inv_p (SZ.v n) a v_a r done))
    (fun (tid:natlt (SZ.v n)) ->
      (done @! tid) |-> Frac 0.5R false **
      (a |-> Frac (1.0R /. SZ.v n) v_a))
    (fun (tid:natlt (SZ.v n)) ->
      (done @! tid) |-> Frac 0.5R false **
      (a |-> Frac (1.0R /. SZ.v n) v_a) **
      inv i (inv_p (SZ.v n) a v_a r done))
    fn tid { dup_inv i (inv_p (SZ.v n) a v_a r done) };
}

ghost
fn teardown
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n < max_blocks})
  (a : gpu_array et n { is_global_array a })
  (#v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == SZ.v n})
  (i : iname)
  ()
  norewrite
  requires
    (forall+ (bid : natlt n). kpost n a v_a r done i bid) **
    inv i (inv_p (SZ.v n) a v_a r done)
  ensures
    a |-> v_a **
    r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a
{
  done_lemma ac n a r done i #v_a;
  ()
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (n : szp{n < max_blocks})
  (a : gpu_array et n { is_global_array a })
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (#r0 : erased et)
  (done : erased (seq (gref bool)){len done == SZ.v n})
  (i : iname)
: kernel_desc
    ((a |-> v_a) **
      inv i (inv_p (SZ.v n) a v_a r done))
    ((a |-> v_a) **
      (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a))
 = {
  nblk     = n;
  f        = kf ac a #v_a r done i;
  setup    = setup    ac n a #v_a r done i;
  teardown = teardown ac n a #v_a r done i;
  kpre     = kpre  n a v_a r done i;
  kpost    = kpost n a v_a r done i;
  frame    = inv i (inv_p (SZ.v n) a v_a r done);
  kpre_sendable  = solve;
  kpost_sendable = solve;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn reduce
  (#et : Type0) {| scalar et, d : has_atomic_add et |}
  (ac : is_ac_w d.pure_op)
  (n : szp {n < max_blocks})
  (a : gpu_array et n { is_global_array a })
  (#v_a : erased (seq et))
  requires
    cpu **
    on gpu_loc (a |-> v_a) **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024)
  returns
    r : et
  ensures
    cpu **
    on gpu_loc (a |-> v_a) **
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
  assume (on gpu_loc (inv i (inv_p (SZ.v n) a v_a gr done)));
  drop_ (on gpu_loc (gr |-> zero #et)); // SHOULD NOT BE HERE, only here until we solve ambig problem from having duplicated permissions to r
  launch_sync (kdesc ac n a #v_a gr #(zero #et) done i);

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
