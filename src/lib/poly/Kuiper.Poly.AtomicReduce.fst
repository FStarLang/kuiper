module Kuiper.Poly.AtomicReduce

(* Reducing an array with some given operation. *)

#lang-pulse

open Kuiper
open Kuiper.Atomics

module SZ = FStar.SizeT
module W = Pulse.Lib.WithPure

let rec contributions
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq et{len v_a == len v_done})
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

let rec contributions_zero 
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq et{len v_a == len v_done})
: Lemma
  (requires v_done `Seq.equal` Seq.create (len v_done) false)
  (ensures  contributions nn v_done v_a zero zero)
  (decreases len v_a)
= if len v_a = 0 then
    ()
  else
    let tl = Seq.tail v_a in
    let tl_done = Seq.tail v_done in
    contributions_zero nn tl_done tl

// let rec contributions_split
//   (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
//   (nn : nat)
//   (v_done_0 v_done_1 : seq bool)
//   (v_a_0 v_a_1 : seq et{len v_a == len v_done})
//   (v_r : et) (acc : et)

let contrib
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn : nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done == len v_a})
  (v_r : et) (acc : et)
: slprop
= pure (contributions nn v_done v_a v_r acc)

let inv_p
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (f:perm)
  (nn: nat)
  (a: gpu_array et nn)
  (v_a: seq et)
  (r: gpu_ref et)
  (done: seq (gref bool))
=
  // pure (len done == nn) **
  exists* (v_done:
    seq bool {len v_done == len done /\ len v_done == len v_a})
    v_r.
    (gpu_pts_to_array a #f v_a) ** (r |-> v_r) **
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    contrib nn v_done v_a v_r zero

unfold
let kpre
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (f:perm)
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : natlt nn)
=
  gref_pts_to (done @! tid) #0.5R false **
  inv i (inv_p f nn a v_a r done)

unfold
let kpost
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (f:perm)
  (nn: nat)
  (a : gpu_array et nn)
  (v_a : seq et)
  (r : gpu_ref et)
  (done : seq (gref bool){len done == Ghost.reveal nn})
  (i:iname)
  (tid : natlt nn)
=
  gref_pts_to (done @! tid) #0.5R true **
  inv i (inv_p f nn a v_a r done)

ghost
fn bigstar_ghost_upd_lemma
  (done : seq (gref bool))
  (v_done : seq bool{len v_done >= len done})
  (tid : nat{tid < len done})
  requires
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    gref_pts_to (done @! tid) #0.5R false
  ensures
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) **
    gref_pts_to (done @! tid) #0.5R true
{
  open Pulse.Lib.GhostReference;
  bigstar_extract 0 (len done) _ tid;
  gather (done @! tid);
  (done @! tid) := true;
  share (done @! tid);
  bigstar_rw_congr 0 tid 
    (fun i -> gref_pts_to (done @! i) #0.5R (v_done  @! i))
    (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) (fun _ -> ());
  bigstar_rw_congr (tid + 1) (len done)
    (fun i -> gref_pts_to (done @! i) #0.5R (v_done  @! i))
    (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) (fun _ -> ());
  bigstar_compose 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) tid;
}

let contributions_lemma
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (nn: nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done == len v_a})
  (v_r : et) (acc : et)
  (tid : nat{tid < len v_a})
: Lemma
  (requires contributions nn v_done v_a v_r acc /\ v_done @! tid == false)
  (ensures  contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc)
  // [SMTPat (contributions nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc)]
= admit()

ghost
fn vdone_bid_false
  (#v_done : seq bool)
  (tid : nat)
  (done : seq (gref bool) {
     len v_done >= len done /\ len done > tid 
    })
requires
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    gref_pts_to (done @! tid) #0.5R false
ensures
    bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
    gref_pts_to (done @! tid) #0.5R false **
    pure (v_done @! tid == false)
{
  bigstar_extract 0 (len done) _ tid;
  Pulse.Lib.GhostReference.pts_to_injective_eq (done @! tid);
  bigstar_compose 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) tid;
}

ghost
fn upd_contrib 
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (#nn: nat)
  (v_done : seq bool)
  (v_a : seq et{len v_done == len v_a})
  (#v_r : et) 
  (#acc : et)
  (tid : nat{tid < len v_a})
  (done : seq (gref bool) {
     len v_done == len done /\ len done > tid 
    })
requires
  contrib nn v_done v_a v_r acc **
  bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R (v_done @! i)) **
  gref_pts_to (done @! tid) #0.5R false
ensures 
  contrib nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc **
  bigstar 0 (len done) (fun i -> gref_pts_to (done @! i) #0.5R ((Seq.upd v_done tid true) @! i)) **
  gref_pts_to (done @! tid) #0.5R true
{
  vdone_bid_false #v_done tid done;
  unfold (contrib nn v_done v_a v_r acc);
  contributions_lemma nn v_done v_a v_r acc tid;
  fold (contrib nn (Seq.upd v_done tid true) v_a (d.pure_op (v_a @! tid) v_r) acc);
  bigstar_ghost_upd_lemma done v_done tid;
}


inline_for_extraction noextract
fn kf
  (#et : Type0) {| sc_et : scalar et |} {| d : has_atomic_add et |}
  (#f:perm)
  (#nn: SZ.t)
  (a : gpu_array et (SZ.v nn))
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == reveal nn /\ len v_a == len done})
  (i : iname)
  (bid : szlt (SZ.v nn))
  ()
  requires
    gpu **
    kpre f (SZ.v nn) a v_a r done i bid **
    block_id (SZ.v nn) bid
  ensures
    gpu **
    kpost f (SZ.v nn) a v_a r done i bid **
    block_id (SZ.v nn) bid
{
  later_credit_buy 1;
  later_credit_buy 1;
  (* Read array at idx *)
  let v =
    with_invariants i
      returns v : et
      ensures
        gpu **
        block_id (SZ.v nn) bid **
        gref_pts_to (done @! bid) #0.5R false **
        later (inv_p f (SZ.v nn) a v_a r done) **
        pure (v == v_a @! SZ.v bid) **
        later_credit 1
    {
      later_elim _;
      unfold (inv_p f (SZ.v nn) a v_a r done);
      let rr = gpu_array_read #et #(SZ.v nn) #0 #(SZ.v nn) a bid;
      fold (inv_p f (SZ.v nn) a v_a r done);
      later_intro (inv_p f (SZ.v nn) a v_a r done);
      rr
    };
  (* Fetch and add into result cell. *)
  with_invariants i
  {
    later_elim _;
    unfold (inv_p f (SZ.v nn) a v_a r done);
    with nn' v_done v_a' v_r acc.
      assert (contrib #et #sc_et #d nn' v_done v_a' v_r acc);
    let _ = atomic_add r v;
    upd_contrib v_done v_a bid done; //need to explicitly instantate v_done and v_a
    fold (inv_p f (SZ.v nn) a v_a r done);
    later_intro (inv_p f (SZ.v nn) a v_a r done);
  }
}

ghost
fn done_lemma
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (f:perm)
  (nn: erased nat)
  (a : gpu_array et nn)
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == reveal nn})
  (i : iname)
  (v_a : erased (seq et))
  requires
    bigstar 0 nn (fun tid -> kpost f nn a v_a r done i tid)
  ensures
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a) **
    (gpu_pts_to_array a #f v_a)
{
  admit();
}

[@@erasable]
noeq
type i_done (n:nat) = {
  i: iname;
  done: seq (gref bool);
  pf: squash (len done == n)
}
instance non_informative_i_done (n:nat)
: Pulse.Lib.NonInformative.non_informative (i_done n) = {
  reveal = (fun r -> Ghost.reveal r) <: Pulse.Lib.NonInformative.revealer (i_done n);
}

let iname_of #n (x: i_done n) : iname = x.i
let refs_of #n (x: i_done n) : erased (seq (gref bool)) = x.done

ghost
fn rec bigstar_share
    (l0 l1: int)
    (m :nat)
    (n :nat {m <= n})
    (f g h: (i:nat {m <= i /\ i < n} -> slprop))
    (share_i : 
      (i:nat {m <= i /\ i < n} -> stt_ghost unit emp_inames (requires f i) (ensures fun _ -> g i ** h i)))
requires bigstar m n f
ensures bigstar #l0 m n g ** bigstar #l1 m n h
decreases (n - m)
{
  if (m = n)
  {
    rewrite bigstar m n f as bigstar m m f;
    bigstar_zs_elim #_ #_ #f;
    bigstar_zs_intro m g;
    bigstar_zs_intro m h;
    bigstar_ext 0 l0 m n h h;
    bigstar_ext 0 l1 m n g g;
    rewrite bigstar m m h as bigstar #l0 m n h;
    rewrite bigstar m m g as bigstar #l1 m n g;
  }
  else
  {
    ghost
    fn share_i' (i:nat {m + 1 <= i /\ i < n})
    requires f i 
    ensures g i ** h i
    {
      share_i i;
    };
    bigstar_split #0 m n f (m + 1);
    rewrite bigstar m n f 
    as bigstar m (m + 1) f ** bigstar (m + 1) n f;
    bigstar_share l0 l1 (m + 1) n f g h share_i';
    bigstar_single_elim #0 #m #f;
    share_i m;
    bigstar_single_intro #l0 m g;
    bigstar_single_intro #l1 m h;
    bigstar_paste #l0 #m #n (m + 1) #g;
    bigstar_paste #l1 #m #n (m + 1) #h;
  }
}


ghost
fn rec alloc_refs (n:nat)
requires emp
returns refs : (s : seq (gref bool) { len s == n })
ensures bigstar 0 n (fun i -> gref_pts_to (refs @! i) false)
decreases n
{
  open Pulse.Lib.BigStar;
  if (n=0)
  {
    let refs = Seq.empty #(gref bool);
    bigstar_zs_intro 0 (fun i -> gref_pts_to (refs @! i) false);
    rewrite 
       bigstar 0 0 (fun i -> gref_pts_to (refs @! i) false)
    as bigstar 0 n (fun i -> gref_pts_to (refs @! i) false);
    refs
  }
  else
  {
    let refs = alloc_refs (n-1);
    let r = Pulse.Lib.GhostReference.alloc false;
    let refs' = Seq.snoc refs r;
    bigstar_rw_congr #0 0 (n - 1)
      (fun i -> gref_pts_to (refs @! i) false) 
      (fun i -> gref_pts_to (refs' @! i) false) //NS: needed to be very explicit
      (fun i -> () <: squash (gref_pts_to (refs @! i) false == gref_pts_to (refs' @! i) false)); 
    rewrite gref_pts_to r false as gref_pts_to (refs' @! (n-1)) false;
    //NS: needed to be very careful about the bound var type of the lambda
    bigstar_single_intro (n - 1) (fun (i:nat { 0 <= i /\ i < n }) -> gref_pts_to (refs' @! i) false);
    //NS: so that i can use it in paste
    bigstar_paste #_ #0 #(n - 1 + 1) (n - 1);
    rewrite each (n - 1 + 1) as n;
    refs'
  }
}

ghost
fn intro_contrib_zero #et {| scalar et |} {| d : has_atomic_add et |}
      (nn: nat) (v_a: seq et { nn == len v_a }) //(v_r: et { v_r == zero })
requires emp
returns v_done : (s:seq bool { len s == nn /\ s == Seq.create nn false })
ensures contrib nn v_done v_a zero zero
{
  let v_done = Seq.create nn false;
  contributions_zero nn v_done v_a;
  fold (contrib nn v_done v_a zero zero);
  v_done
}

ghost
fn relabel_bigstar #l0 #l1 (m:nat) (n:nat {m <= n}) (f: (i:nat {m <= i /\ i < n} -> slprop))
requires bigstar #l0 m n f
ensures bigstar #l1 m n f
{
  bigstar_extensionality_lem l0 l1 m n f f (fun _ -> slprop_equiv_refl _);
  rewrite bigstar #l0 m n f
       as bigstar #l1 m n f;
}

ghost
fn pre_setup
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (#f:perm)
  (n : sz)
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  requires
    cpu **
    gpu_pts_to_array a #f v_a **
    (r |-> zero #et) **
    pure (SZ.v n <= 1024 /\ len v_a == SZ.v n)
  returns
    i_done : i_done (SZ.v n)
  ensures
    cpu **
    bigstar 0 (SZ.v n) (fun tid -> gref_pts_to (refs_of i_done @! tid) #0.5R false) **
    inv (iname_of i_done) (inv_p f (SZ.v n) a v_a r (refs_of i_done))
{
  let refs = alloc_refs (SZ.v n);
  ghost
  fn share_ref_i (i:nat { 0 <= i /\ i < SZ.v n })
  requires gref_pts_to (refs @! i) false
  ensures gref_pts_to (refs @! i) #0.5R false ** gref_pts_to (refs @! i) #0.5R false
  {
    Pulse.Lib.GhostReference.share (refs @! i);
  };
  bigstar_share 0 1 0 (SZ.v n) 
    (fun i -> gref_pts_to (refs @! i) false) 
    (fun i -> gref_pts_to (refs @! i) #0.5R false) 
    (fun i -> gref_pts_to (refs @! i) #0.5R false)
    share_ref_i;
  let v_done = intro_contrib_zero n v_a;
  rewrite each (SZ.v n) as (len refs);
  bigstar_rw_congr #0 0 (len refs) 
    (fun i -> gref_pts_to (refs @! i) #0.5R false)
    (fun i -> gref_pts_to (refs @! i) #0.5R (v_done @! i))
    (fun i -> ()); //<: squash (gref_pts_to (refs @! i) #0.5R false == gref_pts_to (refs @! i) #0.5R (v_done @! i)));
  fold (inv_p f (SZ.v n) a v_a r refs);
  let i = new_invariant (inv_p f (SZ.v n) a v_a r refs);
  relabel_bigstar #1 #0 0 (len refs) _;
  let result : i_done (SZ.v n) = { i = i; done = refs; pf = () };
  rewrite each (len refs) as (SZ.v n);
  rewrite each refs as (refs_of result);
  rewrite each i as (iname_of result);
  result
}

ghost
fn rec replicate 
    (p:slprop)
    (m:nat)
    (n:nat { m <= n })
    (dup: (unit -> stt_ghost unit emp_inames (requires p) (ensures fun _ -> p ** p)))
requires p
ensures p ** bigstar m n (fun i -> p)
decreases (n - m)
{
  open Pulse.Lib.BigStar;
  if (m = n)
  {
    bigstar_zs_intro m (fun i -> p);
    rewrite bigstar m m (fun i -> p)
         as bigstar m n (fun i -> p);
  }
  else
  {
    dup (); 
    replicate p (m + 1) n dup;
    bigstar_single_intro m 
      (fun (i:nat { m <= i /\ i < n }) -> p);
    bigstar_paste #_ #m #n (m + 1) 
      #(fun (i:nat { m <= i /\ i < n }) -> p);
  }
}

ghost
fn setup 
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (#f:perm)
  (n : szp{n < max_blocks})
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (done : erased (seq (gref bool)){len done == SZ.v n})
  (i : iname)
  ()
requires
  bigstar 0 (SZ.v n) (fun tid -> gref_pts_to (done @! tid) #0.5R false) **
  inv i (inv_p f (SZ.v n) a v_a r done)
ensures
  (forall+ (bid: natlt n). kpre f n a v_a r done i bid)  **
  emp
{
  ghost fn duplicate ()
  requires
    inv i (inv_p f (SZ.v n) a v_a r done)
  ensures
    inv i (inv_p f (SZ.v n) a v_a r done) **
    inv i (inv_p f (SZ.v n) a v_a r done)
  {
    dup_inv _ _;
  };
  replicate _ 0 (SZ.v n) duplicate;
  bigstar_zip 0 (SZ.v n)
      (fun i -> gref_pts_to (done @! i) #0.5R false)
      (fun _ -> inv i (inv_p f (SZ.v n) a v_a r done));
  rewrite 
    bigstar 0 (SZ.v n) (fun tid -> gref_pts_to (done @! tid) #0.5R false **
      inv i (inv_p f (SZ.v n) a v_a r done))
  as bigstar 0 (SZ.v n) (fun tid -> kpre f n a v_a r done i tid);
  forevery_fromstar #(natlt n) (fun tid -> kpre f n a v_a r done i tid);
  drop_ (inv i _);
}

ghost
fn teardown
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (f:perm)
  (n : sz)
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (done : lseq (gref bool) (SZ.v n))
  (i : iname)
  ()
  requires
    (forall+ (bid: natlt n). kpost f n a v_a r done i bid) **
    emp
  ensures
    gpu_pts_to_array a #f v_a **
    (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a) 
{
  forevery_tonat (SZ.v n) (fun tid -> kpost f n a v_a r done i tid);
  done_lemma f (SZ.v n) a r done i v_a;
}

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (#f:perm)
  (n : szp{n < max_blocks})
  (a : gpu_array et n)
  (#v_a : erased (seq et))
  (r : gpu_ref et)
  (#r0 : erased et)
  (done : erased (seq (gref bool)){len done == SZ.v n /\ len v_a == len done})
  (i : iname)
: kernel_desc
    (bigstar 0 (SZ.v n) (fun tid -> gref_pts_to (done @! tid) #0.5R false) **
    inv i (inv_p f (SZ.v n) a v_a r done))
    (gpu_pts_to_array a #f v_a **
      (r |-> Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a))
 = {
  nblk = n;
  f = kf a #v_a r done i;
  setup=setup n a r done i;
  teardown = teardown f n a #v_a r done i;
  kpre  = kpre f n a v_a r done i;
  kpost = kpost f n a v_a r done i;
  frame = emp;
} <: kernel_desc_m_1 _ _

inline_for_extraction noextract
fn reduce
  (#et : Type0) {| scalar et |} {| d : has_atomic_add et |}
  (n : szp {n < max_blocks})
  (a : gpu_array et n)
  (#f : perm)
  (#v_a : erased (seq et))
  requires
    cpu **
    pure (f == 1.0R) **
    gpu_pts_to_array a #f v_a **
    pure (SZ.v n > 0 /\ SZ.v n <= 1024)
  returns
    r : et
  ensures
    cpu **
    gpu_pts_to_array a #f v_a **
    pure (r == Kuiper.Seq.Common.seq_fold_left d.pure_op zero v_a)
{
  gpu_pts_to_ref a;
  let mut r = zero #et #_;
  let gr = gpu_alloc0 #et ();
  Kuiper.Ref.gpu_memcpy_host_to_device #et #_ gr r;

  with v. assert (pts_to r v);
  assert (pure (v == zero));

  assert (pure (n < max_blocks));

  let i_done = pre_setup n a gr;
  launch_sync (kdesc n a #v_a gr #(zero #et) (refs_of i_done) (iname_of i_done));

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
