module Klas.Amax

(* cuBLAS I<t>amax: index of the element of largest absolute value (first one
   on ties). cuBLAS is 1-based; we return the 0-based index.

   This is a one-pass argmax reduction. Its correctness rests on the absolute
   value being a total preorder, in particular transitivity of [lte] on
   non-NaN floats (Kuiper.Floating.Base.lte_trans). We therefore require the
   input to contain no NaNs.

   The pure spec [is_amax] here matches (the amax instance of) the generic
   relational reduction in Klas.Reduce.Argmax: [is_amax s i] iff [i] is a
   correct argmax of [s]. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module U64 = FStar.UInt64

(* ----------------------------------------------------------------------- *)
(* Internal lemmas (the pure spec lives in Klas.Amax.fsti)                    *)
(* ----------------------------------------------------------------------- *)

(* abs preserves kind (in particular, abs of a non-NaN is non-NaN). Follows
   from neg_kind on the negative branch of [abs]. *)
#push-options "--fuel 2 --ifuel 1"
let abs_not_nan (#et:Type0) {| floating et |} (x : et)
  : Lemma (requires ~(NaN? (kind x))) (ensures ~(NaN? (kind (abs x))))
          [SMTPat (kind (abs x))]
  = ()
#pop-options

(* Module-local transitivity trigger. lte_trans carries no SMTPat globally (to
   avoid a codebase-wide transitivity loop); here we give it a scoped
   multi-pattern so the argmax proof discharges automatically. *)
let lte_trans_pat (#et:Type0) {| floating et |} (x y z : et)
  : Lemma (requires ~(NaN? (kind x)) /\ ~(NaN? (kind y)) /\ ~(NaN? (kind z)) /\
                    lte x y /\ lte y z)
          (ensures lte x z)
          [SMTPat (lte x y); SMTPat (lte y z)]
  = lte_trans x y z

(* The one-pass argmax is correct, by induction. The strict-vs-non-strict and
   transitivity reasoning is left to the SMT solver via lte_trans_pat /
   negate_lt_is_lte / lte_is_lt_or_eq. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
let rec amax_pre_correct (#et:Type0) {| floating et |}
  (s : Seq.seq et) (k:nat{1 <= k /\ k <= Seq.length s})
  : Lemma (requires all_not_nan s)
          (ensures is_amax_pre s k (amax_pre s k))
          (decreases k)
  = if k = 1 then ()
    else amax_pre_correct s (k-1)
#pop-options

(* u64 encoding of the argmax index (the value written by the kernel). *)
let amax_u64 (#et:Type0) {| floating et |}
  (s : Seq.seq et{Seq.length s >= 1}) : u64 =
  FStar.UInt64.uint_to_t ((amax_pre s (Seq.length s)) % pow2 64)

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: one sequential pass computing the argmax index.    *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 4 --ifuel 2 --z3rlimit 150"
inline_for_extraction noextract
fn amax_kf
  (#et:Type0) {| floating et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  (#vo : erased u64)
  ()
  norewrite
  requires
    gpu ** a |-> va ** (out |-> vo)
  ensures
    gpu ** a |-> va ** (out |-> amax_u64 va)
{
  let mut bi : sz = 0sz;
  let mut bv : et = Array1.(a.(0sz));
  let mut k : szle lena = 1sz;

  while (!k <^ lena)
    invariant live k
    invariant exists* (vbi:sz) (vbv:et).
      bi |-> vbi ** bv |-> vbv **
      pure (1 <= SZ.v !k /\ SZ.v !k <= SZ.v lena /\
            SZ.v vbi == amax_pre va (SZ.v !k) /\
            SZ.v vbi < SZ.v !k /\
            vbv == Seq.index va (SZ.v vbi))
    decreases (SZ.v lena - SZ.v !k)
  {
    let vk = !k;
    let x = Array1.(a.(vk));
    let cur = !bv;
    let vksz : sz = vk;
    let cond = lt (abs cur) (abs x);
    (* amax_pre va (vk+1) unfolds to: if lt |va[amax_pre va vk]| |va[vk]| then vk
       else amax_pre va vk; with cur == va[amax_pre va vk] and x == va[vk] this
       is exactly the assignment below. *)
    assert (pure (amax_pre va (SZ.v vk + 1)
                  == (if cond then SZ.v vk else amax_pre va (SZ.v vk))));
    if cond {
      bv := x;
      bi := vksz;
    };
    let nb = !bi;
    let nbv = !bv;
    assert (pure (SZ.v nb == amax_pre va (SZ.v vk + 1)));
    assert (pure (nbv == Seq.index va (SZ.v nb)));
    k := !k +^ 1sz;
  };

  let r = FStar.SizeT.sizet_to_uint64 !bi;
  assert (pure (r == amax_u64 va));
  gpu_write out r;
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper: allocate the output, launch, copy back.                 *)
(* ----------------------------------------------------------------------- *)

(* Single-thread kernel descriptor (nthr = 1): thread 0 gets the whole array
   plus the output cell and runs the sequential scan. Built as a kernel_desc_n
   (the shape the extraction plugin understands), not via the kernel_desc_1_1
   coercion chain (which the plugin cannot read as a record). *)

ghost
fn amax_setup
  (#et:Type0) {| floating et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  (#vo : erased u64)
  ()
  norewrite
  requires
    (a |-> va) ** (out |-> vo)
  ensures
    (forall+ (i : natlt 1sz). (a |-> va) ** (out |-> vo)) ** emp
{
  forevery_singleton_intro #(natlt 1sz) (fun (_:natlt 1sz) -> (a |-> va) ** (out |-> vo));
}

ghost
fn amax_teardown
  (#et:Type0) {| floating et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  ()
  norewrite
  requires
    (forall+ (i : natlt 1sz). (a |-> va) ** (out |-> amax_u64 va)) ** emp
  ensures
    (a |-> va) ** (out |-> amax_u64 va)
{
  forevery_singleton_elim #(natlt 1sz) (fun (_:natlt 1sz) -> (a |-> va) ** (out |-> amax_u64 va));
}

inline_for_extraction noextract
let kamax
  (#et:Type0) {| floating et |}
  (lena : szp)
  (a : array1 et (l1_forward lena))
  (#_ : squash (Array1.is_global a))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  (#vo : erased u64)
  : kernel_desc
      (requires (a |-> va) ** (out |-> vo))
      (ensures  (a |-> va) ** (out |-> amax_u64 va))
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> amax_kf a out #va #vo);

    frame    = emp;
    teardown = amax_teardown a out;
    setup    = amax_setup a out;
    kpre  = (fun (_i : natlt 1sz) -> (a |-> va) ** (out |-> vo));
    kpost = (fun (_i : natlt 1sz) -> (a |-> va) ** (out |-> amax_u64 va));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn amax_gen
  (#et:Type0) {| floating et |}
  (lena : szp)
  (a : array1 et (l1_forward lena) { Array1.is_global a })
  (#va : erased (lseq et lena))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (all_not_nan va)
  returns
    res : u64
  ensures
    pure (is_amax va (U64.v res))
{
  let out = Kuiper.Ref.gpu_alloc0 #u64 ();
  with vo. assert (on gpu_loc (out |-> vo));

  on_star_eq gpu_loc (a |-> va) (out |-> vo);
  rewrite (on gpu_loc (a |-> va) ** on gpu_loc (out |-> vo))
       as (on gpu_loc ((a |-> va) ** (out |-> vo)));

  launch_sync (kamax lena a out #va #vo);

  on_star_eq gpu_loc (a |-> va) (out |-> amax_u64 va);
  rewrite (on gpu_loc ((a |-> va) ** (out |-> amax_u64 va)))
       as (on gpu_loc (a |-> va) ** on gpu_loc (out |-> amax_u64 va));

  let mut hout : u64 = 0uL;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  amax_pre_correct va (SZ.v lena);
  !hout
}

let amax_f32 = amax_gen #f32
let amax_f64 = amax_gen #f64

(* ----------------------------------------------------------------------- *)
(* amin: same scan with the comparison reversed (switch on a strict          *)
(* decrease). Shares lte_trans_pat / abs_not_nan and the value-agnostic      *)
(* amax_setup; only the teardown (which mentions the result value) differs.  *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
let rec amin_pre_correct (#et:Type0) {| floating et |}
  (s : Seq.seq et) (k:nat{1 <= k /\ k <= Seq.length s})
  : Lemma (requires all_not_nan s)
          (ensures is_amin_pre s k (amin_pre s k))
          (decreases k)
  = if k = 1 then ()
    else amin_pre_correct s (k-1)
#pop-options

let amin_u64 (#et:Type0) {| floating et |}
  (s : Seq.seq et{Seq.length s >= 1}) : u64 =
  FStar.UInt64.uint_to_t ((amin_pre s (Seq.length s)) % pow2 64)

#push-options "--fuel 4 --ifuel 2 --z3rlimit 150"
inline_for_extraction noextract
fn amin_kf
  (#et:Type0) {| floating et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  (#vo : erased u64)
  ()
  norewrite
  requires
    gpu ** a |-> va ** (out |-> vo)
  ensures
    gpu ** a |-> va ** (out |-> amin_u64 va)
{
  let mut bi : sz = 0sz;
  let mut bv : et = Array1.(a.(0sz));
  let mut k : szle lena = 1sz;

  while (!k <^ lena)
    invariant live k
    invariant exists* (vbi:sz) (vbv:et).
      bi |-> vbi ** bv |-> vbv **
      pure (1 <= SZ.v !k /\ SZ.v !k <= SZ.v lena /\
            SZ.v vbi == amin_pre va (SZ.v !k) /\
            SZ.v vbi < SZ.v !k /\
            vbv == Seq.index va (SZ.v vbi))
    decreases (SZ.v lena - SZ.v !k)
  {
    let vk = !k;
    let x = Array1.(a.(vk));
    let cur = !bv;
    let vksz : sz = vk;
    let cond = lt (abs x) (abs cur);
    assert (pure (amin_pre va (SZ.v vk + 1)
                  == (if cond then SZ.v vk else amin_pre va (SZ.v vk))));
    if cond {
      bv := x;
      bi := vksz;
    };
    let nb = !bi;
    let nbv = !bv;
    assert (pure (SZ.v nb == amin_pre va (SZ.v vk + 1)));
    assert (pure (nbv == Seq.index va (SZ.v nb)));
    k := !k +^ 1sz;
  };

  let r = FStar.SizeT.sizet_to_uint64 !bi;
  assert (pure (r == amin_u64 va));
  gpu_write out r;
}
#pop-options

ghost
fn amin_teardown
  (#et:Type0) {| floating et |}
  (#lena : szp)
  (a : array1 et (l1_forward lena))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  ()
  norewrite
  requires
    (forall+ (i : natlt 1sz). (a |-> va) ** (out |-> amin_u64 va)) ** emp
  ensures
    (a |-> va) ** (out |-> amin_u64 va)
{
  forevery_singleton_elim #(natlt 1sz) (fun (_:natlt 1sz) -> (a |-> va) ** (out |-> amin_u64 va));
}

inline_for_extraction noextract
let kamin
  (#et:Type0) {| floating et |}
  (lena : szp)
  (a : array1 et (l1_forward lena))
  (#_ : squash (Array1.is_global a))
  (out : gpu_ref u64)
  (#va : erased (lseq et lena))
  (#vo : erased u64)
  : kernel_desc
      (requires (a |-> va) ** (out |-> vo))
      (ensures  (a |-> va) ** (out |-> amin_u64 va))
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> amin_kf a out #va #vo);

    frame    = emp;
    teardown = amin_teardown a out;
    setup    = amax_setup a out;
    kpre  = (fun (_i : natlt 1sz) -> (a |-> va) ** (out |-> vo));
    kpost = (fun (_i : natlt 1sz) -> (a |-> va) ** (out |-> amin_u64 va));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn amin_gen
  (#et:Type0) {| floating et |}
  (lena : szp)
  (a : array1 et (l1_forward lena) { Array1.is_global a })
  (#va : erased (lseq et lena))
  norewrite
  preserves
    cpu **
    on gpu_loc (a |-> va)
  requires
    pure (all_not_nan va)
  returns
    res : u64
  ensures
    pure (is_amin va (U64.v res))
{
  let out = Kuiper.Ref.gpu_alloc0 #u64 ();
  with vo. assert (on gpu_loc (out |-> vo));

  on_star_eq gpu_loc (a |-> va) (out |-> vo);
  rewrite (on gpu_loc (a |-> va) ** on gpu_loc (out |-> vo))
       as (on gpu_loc ((a |-> va) ** (out |-> vo)));

  launch_sync (kamin lena a out #va #vo);

  on_star_eq gpu_loc (a |-> va) (out |-> amin_u64 va);
  rewrite (on gpu_loc ((a |-> va) ** (out |-> amin_u64 va)))
       as (on gpu_loc (a |-> va) ** on gpu_loc (out |-> amin_u64 va));

  let mut hout : u64 = 0uL;
  Kuiper.Ref.gpu_memcpy_device_to_host hout out;
  Kuiper.Ref.gpu_free out;

  amin_pre_correct va (SZ.v lena);
  !hout
}

let amin_f32 = amin_gen #f32
let amin_f64 = amin_gen #f64
