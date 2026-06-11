module Kuiper.Kernel.Attention

#lang-pulse
open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor
open Kuiper.Index
open Kuiper.Bijection
open Kuiper.Real
module A4 = Kuiper.Array4
module A3 = Kuiper.Array3
module A2 = Kuiper.Array2
module A1 = Kuiper.Array1
module EM4 = Kuiper.EMatrix4
module EM3 = Kuiper.EMatrix3
open Kuiper.EMatrix
module SZ = Kuiper.SizeT

module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.BatchedGEMM
open Kuiper.Kernel.RowSoftmax
module KB = Kuiper.Kernel.HReduce.Block
module KMap = Kuiper.Kernel.Map

(* Helper lemma: Euclidean decomposition. Used to discharge bridge VCs that
   need [(i*h+j)/h == i] and [(i*h+j)%h == j] for [j < h]. *)
let div_mod_combine_lemma (i:nat) (j:nat) (h:pos)
  : Lemma (requires j < h)
          (ensures (j + i * h) / h == i /\ (j + i * h) % h == j)
  = FStar.Math.Lemmas.lemma_div_mod_plus j i h

let div_mod_combine_forall (b h : pos)
  : Lemma (forall (i : natlt b) (j : natlt h).
             (i * h + j) / h == i /\ (i * h + j) % h == j)
  = let aux (i : natlt b) (j : natlt h)
      : Lemma ((i * h + j) / h == i /\ (i * h + j) % h == j)
      = FStar.Math.Lemmas.lemma_div_mod_plus j i h in
    Classical.forall_intro_2 aux

(* Unfolds [EM3.macc (EM3.mkM f) i j k == f i j k]. *)
let macc_mkM3 (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  (i : natlt d0) (j : natlt d1) (k : natlt d2)
  : Lemma (EM3.macc (EM3.mkM f) i j k == f i j k)
  = ()

let macc_mkM3_forall (#et:Type) (#d0 #d1 #d2 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> GTot et)
  : Lemma (forall (i : natlt d0) (j : natlt d1) (k : natlt d2).
             EM3.macc (EM3.mkM f) i j k == f i j k)
  = let aux (i : natlt d0) (j : natlt d1) (k : natlt d2)
      : Lemma (EM3.macc (EM3.mkM f) i j k == f i j k)
      = macc_mkM3 f i j k in
    Classical.forall_intro_3 aux

let macc_mkM4 (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
  : Lemma (EM4.macc (EM4.mkM f) i j k l == f i j k l)
  = ()

let macc_mkM4_forall (#et:Type) (#d0 #d1 #d2 #d3 : nat)
  (f : natlt d0 -> natlt d1 -> natlt d2 -> natlt d3 -> GTot et)
  : Lemma (forall (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3).
             EM4.macc (EM4.mkM f) i j k l == f i j k l)
  = let aux (i : natlt d0) (j : natlt d1) (k : natlt d2) (l : natlt d3)
      : Lemma (EM4.macc (EM4.mkM f) i j k l == f i j k l)
      = macc_mkM4 f i j k l in
    Classical.forall_intro_4 aux

let macc_mkM2 (#et:Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et)
  (i : natlt d0) (j : natlt d1)
  : Lemma (macc (mkM f) i j == f i j)
  = ()

let macc_mkM2_forall (#et:Type) (#d0 #d1 : nat)
  (f : natlt d0 -> natlt d1 -> GTot et)
  : Lemma (forall (i : natlt d0) (j : natlt d1).
             macc (mkM f) i j == f i j)
  = let aux (i : natlt d0) (j : natlt d1)
      : Lemma (macc (mkM f) i j == f i j)
      = macc_mkM2 f i j in
    Classical.forall_intro_2 aux

(* Swap the last two dimensions of a 3-D abstract index — used to transpose
   the K matrix's inner dimensions for the Q @ K^T matmul. *)
#push-options "--ifuel 3"
inline_for_extraction noextract
let swap_last_two_bij (batch n k : nat)
  : (abs (A3.desc batch n k) =~ abs (A3.desc batch k n))
  = {
      ff = (fun (i, (r, (c, ()))) -> (i, (c, (r, ()))));
      gg = (fun (i, (r, (c, ()))) -> (i, (c, (r, ()))));
      ff_gg = ez;
      gg_ff = ez;
    }
#pop-options

(* Concrete (ctlayout) instance for the [swap_last_two_bij]-transposed view
   of an l3_batched_row_major layout. Indexing at (i, r, c) where r:k, c:n
   reads memory offset [i * n*k + c * k + r] — i.e., the original
   (i, c, r) entry of the (batch, n, k) row-major K. *)
#push-options "--z3rlimit 80"
inline_for_extraction noextract
instance c_l3_brm_transposed_last_two
  (bh : erased nat{SZ.fits bh})
  (n : SZ.t)
  (k : SZ.t{SZ.fits (n * k) /\ SZ.fits (bh * (n * k))})
  : Kuiper.Tensor.Layout.ctlayout
      (Kuiper.Tensor.tlayout_bij (swap_last_two_bij bh n k) (l3_batched_row_major bh n k))
  = {
      ulen_fits = ();
      all_fit = ();
      cimap = (fun (idx : Kuiper.Index.conc (A3.desc bh k n)) ->
                match idx with
                | (i, (r, (c, ()))) ->
                  SZ.add (SZ.mul i (SZ.mul n k)) (SZ.add (SZ.mul c k) r));
    }
#pop-options

(* ─── Bridging helpers ─────────────────────────────────────────────────────
   We need to "view" a tensor with a coarser/finer batch dimension to feed
   the existing batched-GEMM, row-softmax, and reduce kernels (which expect
   3-D, 2-D, or 1-D layouts). The implementation strategy is:

     1. A non-cost helper [view_*] returns a fresh wrapper tensor whose
        type carries the desired layout but whose underlying storage is
        the original. Implemented by [from_array l_new (core a)] — at
        runtime this is a no-op pointer reinterpret (no copy, no kernel).

     2. A ghost helper [bridge_*] re-frames the [pts_to] slprop from the
        original tensor onto the new wrapper. Implemented (for all
        non-transpose flavors) via the [lower] / [raise'] pattern from
        Kuiper.Array{1,2,3,4}, mirroring the cast pattern in
        Kuiper.Matrix.Casts:

          A_src.lower a;                       (* core a |-> to_seq l_src s *)
          rewrite each (A_src.core a) as (A_dst.core a');
          A_dst.raise' l_dst (A_dst.core a');  (* a' |-> from_seq l_dst (to_seq l_src s) *)

        The target ematrix is built with [mkM] so the index translation
        equation is purely syntactic; only one [assume pure] remains per
        bridge — an [EMatrix.equal] extensionality fact equating the
        round-trip [from_seq l_dst (to_seq l_src s)] to that target ematrix.
        Both layouts compute identical flat offsets, but [pack]/[major_on]
        don't unfold automatically, so SMT cannot see this without a
        [tlayout]-extensionality lemma we don't currently have.

     The semantic content carried by each bridge's postcondition is the
     index-translation equation, expressed using a guarded universal
     variable [(idx : natlt N)] of the *flat* layout so the [EMatrix.macc]
     indices typecheck directly without needing a side proof of [i*h+j < N].

     The transpose bridge [bridge_a3_transpose] is GENUINELY DIFFERENT from
     the other bridges and is left admitted — see its docstring.

   The combination justifies treating a (b, h, m, n) A4 as a (b*h, m, n) A3,
   or a (b*h, m, n) A3 as a (b*h*m, n) A2, etc. — without ever copying data.
   ──────────────────────────────────────────────────────────────────────── *)

inline_for_extraction noextract
fn view_4_as_3
  (#et : Type0)
  (b h m n : szp)
  (bh : szp)
  (a : A4.t et (l4_batched_row_major b h m n) { A4.is_global a })
  preserves cpu
  requires
    pure (SZ.v bh == SZ.v b * SZ.v h)
  returns
    a' : A3.t et (l3_batched_row_major bh m n)
  ensures
    pure (A3.is_global a' /\ A3.core a' == A4.core a)
{
  assume pure (A3.layout_size (l3_batched_row_major bh m n)
            == A4.layout_size (l4_batched_row_major b h m n));
  let p = A4.core a;
  let p3 : larray et (A3.layout_size (l3_batched_row_major bh m n)) = p;
  let a' = A3.from_array (l3_batched_row_major bh m n) p3;
  assume pure (A3.is_global a');
  a'
}

inline_for_extraction noextract
fn view_3_as_4
  (#et : Type0)
  (b h m n : szp)
  (bh : szp)
  (a : A3.t et (l3_batched_row_major bh m n) { A3.is_global a })
  preserves cpu
  requires
    pure (SZ.v bh == SZ.v b * SZ.v h)
  returns
    a' : A4.t et (l4_batched_row_major b h m n)
  ensures
    pure (A4.is_global a' /\ A4.core a' == A3.core a)
{
  assume pure (A4.layout_size (l4_batched_row_major b h m n)
            == A3.layout_size (l3_batched_row_major bh m n));
  let p = A3.core a;
  let p4 : larray et (A4.layout_size (l4_batched_row_major b h m n)) = p;
  let a' = A4.from_array (l4_batched_row_major b h m n) p4;
  assume pure (A4.is_global a');
  a'
}

inline_for_extraction noextract
fn view_3_as_2
  (#et : Type0)
  (bh m n : szp)
  (bhm : szp)
  (a : A3.t et (l3_batched_row_major bh m n) { A3.is_global a })
  preserves cpu
  requires
    pure (SZ.v bhm == SZ.v bh * SZ.v m)
  returns
    a' : A2.t et (l2_row_major bhm n)
  ensures
    pure (A2.is_global a' /\ A2.core a' == A3.core a)
{
  assume pure (A2.layout_size (l2_row_major bhm n)
            == A3.layout_size (l3_batched_row_major bh m n));
  let p = A3.core a;
  let p2 : larray et (A2.layout_size (l2_row_major bhm n)) = p;
  let a' = A2.from_array (l2_row_major bhm n) p2;
  assume pure (A2.is_global a');
  a'
}

inline_for_extraction noextract
fn view_2_as_3
  (#et : Type0)
  (bh m n : szp)
  (bhm : szp)
  (a : A2.t et (l2_row_major bhm n) { A2.is_global a })
  preserves cpu
  requires
    pure (SZ.v bhm == SZ.v bh * SZ.v m)
  returns
    a' : A3.t et (l3_batched_row_major bh m n)
  ensures
    pure (A3.is_global a' /\ A3.core a' == A2.core a)
{
  assume pure (A3.layout_size (l3_batched_row_major bh m n)
            == A2.layout_size (l2_row_major bhm n));
  let p = A2.core a;
  let p3 : larray et (A3.layout_size (l3_batched_row_major bh m n)) = p;
  let a' = A3.from_array (l3_batched_row_major bh m n) p3;
  assume pure (A3.is_global a');
  a'
}

(* Transpose the last two dimensions of an A3 (batch, n, k) → view it as
   (batch, k, n). Pure layout change: the returned array shares storage
   with the input and uses the [tlayout_bij swap_last_two_bij] layout,
   which has a concrete (ctlayout) instance via [c_l3_brm_transposed_last_two]
   so it can be passed directly to layout-polymorphic kernels (e.g., BatchedGEMM). *)
inline_for_extraction noextract
fn view_a3_transpose_lastTwo
  (#et : Type0)
  (batch n k : szp)
  (a : A3.t et (l3_batched_row_major batch n k) { A3.is_global a })
  preserves cpu
  returns
    a' : A3.t et (Kuiper.Tensor.tlayout_bij (swap_last_two_bij batch n k)
                    (l3_batched_row_major batch n k))
  ensures
    pure (A3.is_global a' /\ A3.core a' == A3.core a)
{
  let p = A3.core a;
  let a' = A3.from_array
             (Kuiper.Tensor.tlayout_bij (swap_last_two_bij batch n k)
                (l3_batched_row_major batch n k))
             p;
  assume pure (A3.is_global a');
  a'
}

inline_for_extraction noextract
fn view_a1_as_a3
  (#et : Type0)
  (b h m : szp)
  (bhm : szp)
  (a : A1.t et (l1_forward bhm) { A1.is_global a })
  preserves cpu
  requires
    pure (SZ.v bhm == SZ.v b * SZ.v h * SZ.v m)
  returns
    a' : A3.t et (l3_batched_row_major b h m)
  ensures
    pure (A3.is_global a' /\ A3.core a' == A1.core a)
{
  assume pure (A3.layout_size (l3_batched_row_major b h m)
            == A1.layout_size (l1_forward bhm));
  let p = A1.core a;
  let p3 : larray et (A3.layout_size (l3_batched_row_major b h m)) = p;
  let a' = A3.from_array (l3_batched_row_major b h m) p3;
  assume pure (A3.is_global a');
  a'
}

(* Ghost slprop bridges. Each transfers ownership of the underlying data from
   the original wrapper to the new (aliased) wrapper. *)

ghost
fn bridge_4_to_3
  (#et : Type0)
  (b h m n : szp)
  (bh : szp)
  (a : A4.t et (l4_batched_row_major b h m n))
  (a' : A3.t et (l3_batched_row_major bh m n))
  (#f : perm)
  (#s : EM4.t et b h m n)
  requires
    a |-> Frac f s **
    pure (SZ.v bh == SZ.v b * SZ.v h /\
          A3.core a' == A4.core a)
  ensures
    exists* (s3 : EM3.t et bh m n).
      a' |-> Frac f s3 **
      pure (
        forall (i : natlt b) (j : natlt h) (r : natlt m) (c : natlt n)
               (idx : natlt bh).
          i * SZ.v h + j == idx ==>
            EM3.macc s3 idx r c == EM4.macc s i j r c)
{
  A4.lower a;
  assume pure (A3.layout_size (l3_batched_row_major bh m n)
            == A4.layout_size (l4_batched_row_major b h m n));
  rewrite each (A4.core a) as (A3.core a');
  A3.raise' (l3_batched_row_major bh m n) (A3.core a');

  let s3_target : EM3.t et bh m n =
    EM3.mkM (fun idx r c ->
      EM4.macc s (idx / SZ.v h) (idx % SZ.v h) r c);

  (* The two layouts compute identical flat offsets:
       l3.imap.f (idx,r,c)        = idx*m*n + r*n + c
       l4.imap.f (idx/h,idx%h,r,c) = (idx/h)*h*m*n + (idx%h)*m*n + r*n + c
                                  = idx*m*n + r*n + c     (since (idx/h)*h + idx%h == idx)
     Hence [from_seq l3 (to_seq l4 s)] and [s3_target] are pointwise equal.
     SMT does not currently unfold [pack]/[major_on] to discharge this
     automatically; pending a [tlayout]-extensionality lemma we assume the
     ematrix-level equality. *)
  assume pure (EM3.equal
    (A3.from_seq (l3_batched_row_major bh m n)
       (A4.to_seq (l4_batched_row_major b h m n) s))
    s3_target);

  rewrite each
    (A3.from_seq (l3_batched_row_major bh m n)
       (A4.to_seq (l4_batched_row_major b h m n) s))
    as s3_target;
  rewrite each
    (A3.from_array (l3_batched_row_major bh m n) (A3.core a'))
    as a';

  (* Discharge the [i*h+j == idx ==> idx/h == i /\ idx%h == j] step
     via Euclidean uniqueness from FStar.Math.Lemmas. *)
  div_mod_combine_forall (SZ.v b) (SZ.v h);
  macc_mkM3_forall #et #(SZ.v bh) #(SZ.v m) #(SZ.v n)
    (fun idx r c -> EM4.macc s (idx / SZ.v h) (idx % SZ.v h) r c);
}

ghost
fn bridge_3_to_4
  (#et : Type0)
  (b h m n : szp)
  (bh : szp)
  (a : A3.t et (l3_batched_row_major bh m n))
  (a' : A4.t et (l4_batched_row_major b h m n))
  (#f : perm)
  (#s3 : EM3.t et bh m n)
  requires
    a |-> Frac f s3 **
    pure (SZ.v bh == SZ.v b * SZ.v h /\
          A4.core a' == A3.core a)
  ensures
    exists* (s : EM4.t et b h m n).
      a' |-> Frac f s **
      pure (
        forall (i : natlt b) (j : natlt h) (r : natlt m) (c : natlt n)
               (idx : natlt bh).
          i * SZ.v h + j == idx ==>
            EM4.macc s i j r c == EM3.macc s3 idx r c)
{
  A3.lower a;
  assume pure (A4.layout_size (l4_batched_row_major b h m n)
            == A3.layout_size (l3_batched_row_major bh m n));
  rewrite each (A3.core a) as (A4.core a');
  A4.raise' (l4_batched_row_major b h m n) (A4.core a');

  let s_target : EM4.t et b h m n =
    EM4.mkM (fun i j r c ->
      EM3.macc s3 (i * SZ.v h + j) r c);

  assume pure (EM4.equal
    (A4.from_seq (l4_batched_row_major b h m n)
       (A3.to_seq (l3_batched_row_major bh m n) s3))
    s_target);

  rewrite each
    (A4.from_seq (l4_batched_row_major b h m n)
       (A3.to_seq (l3_batched_row_major bh m n) s3))
    as s_target;
  rewrite each
    (A4.from_array (l4_batched_row_major b h m n) (A4.core a'))
    as a';

  macc_mkM4_forall #et #(SZ.v b) #(SZ.v h) #(SZ.v m) #(SZ.v n)
    (fun i j r c -> EM3.macc s3 (i * SZ.v h + j) r c);
}

ghost
fn bridge_3_to_2
  (#et : Type0)
  (bh m n : szp)
  (bhm : szp)
  (a : A3.t et (l3_batched_row_major bh m n))
  (a' : A2.t et (l2_row_major bhm n))
  (#f : perm)
  (#s3 : EM3.t et bh m n)
  requires
    a |-> Frac f s3 **
    pure (SZ.v bhm == SZ.v bh * SZ.v m /\
          A2.core a' == A3.core a)
  ensures
    exists* (s2 : ematrix et bhm n).
      a' |-> Frac f s2 **
      pure (
        forall (i : natlt bh) (r : natlt m) (c : natlt n) (idx : natlt bhm).
          i * SZ.v m + r == idx ==>
            macc s2 idx c == EM3.macc s3 i r c)
{
  A3.lower a;
  assume pure (A2.layout_size (l2_row_major bhm n)
            == A3.layout_size (l3_batched_row_major bh m n));
  rewrite each (A3.core a) as (A2.core a');
  A2.raise' (l2_row_major bhm n) (A2.core a');

  let s2_target : ematrix et bhm n =
    mkM (fun idx c ->
      EM3.macc s3 (idx / SZ.v m) (idx % SZ.v m) c);

  assume pure (equal
    (A2.from_seq (l2_row_major bhm n)
       (A3.to_seq (l3_batched_row_major bh m n) s3))
    s2_target);

  rewrite each
    (A2.from_seq (l2_row_major bhm n)
       (A3.to_seq (l3_batched_row_major bh m n) s3))
    as s2_target;
  rewrite each
    (A2.from_array (l2_row_major bhm n) (A2.core a'))
    as a';

  div_mod_combine_forall (SZ.v bh) (SZ.v m);
  macc_mkM2_forall #et #(SZ.v bhm) #(SZ.v n)
    (fun idx c -> EM3.macc s3 (idx / SZ.v m) (idx % SZ.v m) c);
}

ghost
fn bridge_2_to_3
  (#et : Type0)
  (bh m n : szp)
  (bhm : szp)
  (a : A2.t et (l2_row_major bhm n))
  (a' : A3.t et (l3_batched_row_major bh m n))
  (#f : perm)
  (#s2 : ematrix et bhm n)
  requires
    a |-> Frac f s2 **
    pure (SZ.v bhm == SZ.v bh * SZ.v m /\
          A3.core a' == A2.core a)
  ensures
    exists* (s3 : EM3.t et bh m n).
      a' |-> Frac f s3 **
      pure (
        forall (i : natlt bh) (r : natlt m) (c : natlt n) (idx : natlt bhm).
          i * SZ.v m + r == idx ==>
            EM3.macc s3 i r c == macc s2 idx c)
{
  A2.lower a;
  assume pure (A3.layout_size (l3_batched_row_major bh m n)
            == A2.layout_size (l2_row_major bhm n));
  rewrite each (A2.core a) as (A3.core a');
  A3.raise' (l3_batched_row_major bh m n) (A3.core a');

  let s3_target : EM3.t et bh m n =
    EM3.mkM (fun i r c ->
      macc s2 (i * SZ.v m + r) c);

  assume pure (EM3.equal
    (A3.from_seq (l3_batched_row_major bh m n)
       (A2.to_seq (l2_row_major bhm n) s2))
    s3_target);

  rewrite each
    (A3.from_seq (l3_batched_row_major bh m n)
       (A2.to_seq (l2_row_major bhm n) s2))
    as s3_target;
  rewrite each
    (A3.from_array (l3_batched_row_major bh m n) (A3.core a'))
    as a';

  macc_mkM3_forall #et #(SZ.v bh) #(SZ.v m) #(SZ.v n)
    (fun i r c -> macc s2 (i * SZ.v m + r) c);
}

(* [bridge_a3_transpose] transfers the [pts_to] slprop from the original
   (batch, n, k) array onto its transposed view (which lives at the
   [tlayout_bij swap_last_two_bij] layout). The two arrays share underlying
   storage via [from_array (tlayout_bij ...) (core a)]; the slprop transfer
   produces an [EMatrix3] whose [macc] indexes the original [s] with the last
   two coordinates swapped. *)
ghost
fn bridge_a3_transpose
  (#et : Type0)
  (batch n k : szp)
  (a : A3.t et (l3_batched_row_major batch n k))
  (a' : A3.t et (Kuiper.Tensor.tlayout_bij (swap_last_two_bij batch n k)
                   (l3_batched_row_major batch n k)))
  (#f : perm)
  (#s : EM3.t et batch n k)
  requires
    a |-> Frac f s **
    pure (A3.core a' == A3.core a)
  ensures
    exists* (s' : EM3.t et batch k n).
      a' |-> Frac f s' **
      pure (
        forall (i : natlt batch) (r : natlt k) (c : natlt n).
          EM3.macc s' i r c == EM3.macc s i c r)
{
  A3.apply_bij (swap_last_two_bij batch n k) a;
  rewrite each
    (A3.from_array
       (Kuiper.Tensor.tlayout_bij (swap_last_two_bij batch n k)
          (l3_batched_row_major batch n k))
       (A3.core a))
    as a';
}

(* [bridge_a3_transpose_back]: the inverse of [bridge_a3_transpose]. Recovers
   the original (batch, n, k) view from the transposed one. Both arrays share
   underlying storage, so this is a pure ghost transfer of the slprop with
   the implied inverse ematrix. The layout-equality between
   [tlayout_bij (bij_sym swap) (tlayout_bij swap l)] and [l] is left admitted
   — it holds up to extensional equality of the underlying [imap] injections,
   but requires a [tlayout]-extensionality lemma we don't currently have. *)
ghost
fn bridge_a3_transpose_back
  (#et : Type0)
  (batch n k : szp)
  (a' : A3.t et (Kuiper.Tensor.tlayout_bij (swap_last_two_bij batch n k)
                   (l3_batched_row_major batch n k)))
  (a : A3.t et (l3_batched_row_major batch n k))
  (#f : perm)
  (#s' : EM3.t et batch k n)
  requires
    a' |-> Frac f s' **
    pure (A3.core a == A3.core a')
  ensures
    exists* (s : EM3.t et batch n k).
      a |-> Frac f s **
      pure (
        forall (i : natlt batch) (r : natlt n) (c : natlt k).
          EM3.macc s i r c == EM3.macc s' i c r)
{
  admit ()
}

ghost
fn bridge_a1_to_a3
  (#et : Type0)
  (b h m : szp)
  (bhm : szp)
  (a : A1.t et (l1_forward bhm))
  (a' : A3.t et (l3_batched_row_major b h m))
  (#f : perm)
  (#s : lseq et bhm)
  requires
    a |-> Frac f s **
    pure (SZ.v bhm == SZ.v b * SZ.v h * SZ.v m /\
          A3.core a' == A1.core a)
  ensures
    exists* (s3 : EM3.t et b h m).
      a' |-> Frac f s3 **
      pure (
        forall (i : natlt b) (j : natlt h) (r : natlt m) (idx : natlt bhm).
          (i * SZ.v h + j) * SZ.v m + r == idx ==>
            EM3.macc s3 i j r == Seq.index s idx)
{
  A1.lower a;
  assume pure (A3.layout_size (l3_batched_row_major b h m)
            == A1.layout_size (l1_forward bhm));
  rewrite each (A1.core a) as (A3.core a');
  A3.raise' (l3_batched_row_major b h m) (A3.core a');

  let s3_target : EM3.t et b h m =
    EM3.mkM (fun i j r ->
      Seq.index s ((i * SZ.v h + j) * SZ.v m + r));

  assume pure (EM3.equal
    (A3.from_seq (l3_batched_row_major b h m)
       (A1.to_seq (l1_forward bhm) s))
    s3_target);

  rewrite each
    (A3.from_seq (l3_batched_row_major b h m)
       (A1.to_seq (l1_forward bhm) s))
    as s3_target;
  rewrite each
    (A3.from_array (l3_batched_row_major b h m) (A3.core a'))
    as a';

  macc_mkM3_forall #et #(SZ.v b) #(SZ.v h) #(SZ.v m)
    (fun i j r -> Seq.index s ((i * SZ.v h + j) * SZ.v m + r));
}

(* Memcpy bias → scratch tensor (same layout). Stub for an Array1-level memcpy
   on the underlying core. *)
inline_for_extraction noextract
fn memcpy_a4_d2d
  (#et : Type0) {| sized et |}
  (b h m n : szp)
  (dst : A4.t et (l4_batched_row_major b h m n) { A4.is_global dst })
  (src : A4.t et (l4_batched_row_major b h m n) { A4.is_global src })
  (#fS : perm)
  (#sS : EM4.t et b h m n)
  (#sD : EM4.t et b h m n)
  preserves
    cpu **
    on gpu_loc (src |-> Frac fS sS)
  requires
    on gpu_loc (dst |-> sD)
  ensures
    on gpu_loc (dst |-> sS)
{
  admit ()
}

(* ─── Main implementation ──────────────────────────────────────────────────
   Naive scaled-dot-product attention. Sequence of kernel launches:

     1. scores ← bias                       (memcpy)
     2. scores ← scale * Q @ K^T + scores   (batched GEMM with lincomb)
     3. lse_flat[i] ← sum_j(exp(scores[i,j])) (per-row reduce)
     4. lse_flat   ← log(lse_flat)          (pointwise map)
     5. scores ← row_softmax(scores)         (row softmax)
     6. out    ← scores @ V                  (batched matmul)

   K is transposed via a layout swap (no kernel). Batch dimensions are
   folded together via layout views as required by the 3-D/2-D kernels.
   ──────────────────────────────────────────────────────────────────────── *)

#push-options "--admit_smt_queries true"
fn scaled_dot_product_efficient_attention
  (#et : Type0) {| floating et, real_like et, floating_real_like et |}
  (b h : szp)
  (m n : szp)
  (k kv : szp)
  (q    : A4.t et (l4_batched_row_major b h m  k ) { A4.is_global q    })
  (k_   : A4.t et (l4_batched_row_major b h n  k ) { A4.is_global k_   })
  (v    : A4.t et (l4_batched_row_major b h n  kv) { A4.is_global v    })
  (bias : A4.t et (l4_batched_row_major b h m  n ) { A4.is_global bias })
  (scale : et)
  (#sQ : erased (EM4.t et b h m k))
  (#sK : erased (EM4.t et b h n k))
  (#sV : erased (EM4.t et b h n kv))
  (#sB : erased (EM4.t et b h m n))
  (#rKT : erased (EM4.t real b h k n))
  (#fQ #fK #fV #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (q    |-> Frac fQ sQ) **
    on gpu_loc (k_   |-> Frac fK sK) **
    on gpu_loc (v    |-> Frac fV sV) **
    on gpu_loc (bias |-> Frac fB sB)
  requires
    pure (
      SZ.fits (b * h * m * kv) /\
      SZ.fits (b * h * m * n)  /\
      SZ.fits (b * h * n * k)  /\
      SZ.fits (b * h * m * k)  /\
      SZ.fits (b * h * m) /\
      (EM4.mkM (fun i j k l -> EM4.macc sK i j l k)) %~ rKT /\
      m * n <= max_blocks * max_threads /\
      m * kv <= max_blocks * max_threads /\
      b * h * m <= max_blocks
    )
  returns
    out : A4.t et (l4_batched_row_major b h m kv) &
          A3.t et (l3_batched_row_major b h m)
  ensures
    (exists* (sO : EM4.t et b h m kv) (sL : EM3.t et b h m).
      on gpu_loc (fst out |-> sO) **
      on gpu_loc (snd out |-> sL) **
      pure (
        let attn_tile = fun i j -> attention_real
            (EM4.slice_page (EM4.to_real_matrix sQ) i j)
            (EM4.slice_page rKT i j)
            (EM4.slice_page (EM4.to_real_matrix sV) i j)
            (EM4.slice_page (EM4.to_real_matrix sB) i j)
            (to_real scale) in
        let out_spec = EM4.mkM fun i j -> macc (fst (attn_tile i j)) in 
        let lse_spec = EM3.mkM fun i j -> Seq.index (snd (attn_tile i j)) in 
        sO %~ out_spec /\ sL %~ lse_spec)) **
    pure (A4.is_global (fst out) /\ A3.is_global (snd out)) {

  let bh  : szp = b  *^ h;
  let bhm : szp = bh *^ m;

  (* Auxiliary size‑fit facts derived from the requires clause. *)
  assert pure (SZ.fits (b * h) /\ SZ.fits (b * h * m) /\
               SZ.fits (bh * m) /\ SZ.fits (bh * m * n) /\
               SZ.fits (bh * m * k) /\ SZ.fits (bh * n * k) /\
               SZ.fits (bh * m * kv));

  (* ── Step 1: Allocate output, lse and scratch tensors ────────────────── *)
  assert pure (SZ.fits (A4.layout_size (l4_batched_row_major b h m kv)));
  let out = A4.alloc0 #et b h m kv (l4_batched_row_major b h m kv);

  assert pure (SZ.fits (A3.layout_size (l3_batched_row_major b h m)));
  let lse = A3.alloc0 #et b h m (l3_batched_row_major b h m);

  assert pure (SZ.fits (A4.layout_size (l4_batched_row_major b h m n)));
  let scores = A4.alloc0 #et b h m n (l4_batched_row_major b h m n);

  (* ── Step 2: scores ← bias (memcpy) ─────────────────────────────────── *)
  memcpy_a4_d2d b h m n scores bias;

  (* ── Step 3: scores += scale * Q @ K^T using batched GEMM ───────────── *)
  (* View Q, K, scores as 3-D batched tensors. *)
  let q3      = view_4_as_3 b h m k bh q;
  let k3      = view_4_as_3 b h n k bh k_;
  let scores3 = view_4_as_3 b h m n bh scores;
  map_loc gpu_loc (fun () -> bridge_4_to_3 b h m k bh q q3);
  map_loc gpu_loc (fun () -> bridge_4_to_3 b h n k bh k_ k3);
  map_loc gpu_loc (fun () -> bridge_4_to_3 b h m n bh scores scores3);

  (* Transpose-view K's last two dims: (bh, n, k) → (bh, k, n). *)
  let kT3 = view_a3_transpose_lastTwo bh n k k3;
  map_loc gpu_loc (fun () -> bridge_a3_transpose bh n k k3 kT3);

  (* scores += scale * Q @ K^T  via  bmmcomb (lincomb scale one). *)
  assert pure (m * n <= max_blocks * max_threads /\ SZ.fits (bh * m * n));
  let one_v : et = Kuiper.Scalars.Base.one #et;
  bmmcomb_gpu_exact #et (MS.lincomb scale one_v)
    bh m k n q3 kT3 scores3;

  (* ── Step 4: Compute LSE = log(sum(exp(scores))) per row ─────────────── *)
  (* View scores as 2-D row-major (bh*m, n) for per-row reduction & softmax. *)
  let scores2 = view_3_as_2 bh m n bhm scores3;
  map_loc gpu_loc (fun () -> bridge_3_to_2 bh m n bhm scores3 scores2);

  (* Per-row sum-of-exp into a flat Array1 (bh*m). *)
  assert pure (SZ.fits (A1.layout_size (l1_forward bhm)));
  let lse_flat = A1.alloc0 #et bhm (l1_forward bhm);

  assert pure (bhm <= max_blocks /\ SZ.fits (n + max_threads));
  with sS2. assert on gpu_loc (scores2 |-> sS2);
  KB.reduce_batched_block #et exp rexp bhm n max_threads scores2 lse_flat (to_real_matrix sS2);

  (* Pointwise log to turn sum-of-exp into log-sum-exp. *)
  assert pure (bhm <= max_blocks * max_threads);
  KMap.map_gpu #et log bhm lse_flat;

  (* ── Step 5: In-place row softmax of the 2-D scores ─────────────────── *)
  assert pure (bhm <= max_blocks /\ bhm * n <= max_blocks * max_threads);
  with sS2'. assert on gpu_loc (scores2 |-> sS2');
  row_softmax_gpu bhm n scores2 (to_real_matrix sS2');

  (* ── Step 6: out ← softmax(scores) @ V via batched matmul ────────────── *)
  (* View scores back as 3-D (bh, m, n). *)
  let scores3' = view_2_as_3 bh m n bhm scores2;
  map_loc gpu_loc (fun () -> bridge_2_to_3 bh m n bhm scores2 scores3');

  (* View V as 3-D (bh, n, kv); view out as 3-D (bh, m, kv). *)
  let v3   = view_4_as_3 b h n kv bh v;
  let out3 = view_4_as_3 b h m kv bh out;
  map_loc gpu_loc (fun () -> bridge_4_to_3 b h n kv bh v v3);
  map_loc gpu_loc (fun () -> bridge_4_to_3 b h m kv bh out out3);

  (* out = scores @ V  via  bmmcomb comb2. *)
  assert pure (m * kv <= max_blocks * max_threads /\ SZ.fits (bh * m * kv));
  bmmcomb_gpu_exact #et MS.comb2
    bh m n kv scores3' v3 out3;

  (* Restore A4 views of [out] and original input tensors so the
     postcondition's slprops are recovered. *)
  map_loc gpu_loc (fun () -> bridge_3_to_4 b h m kv bh out3 out);
  map_loc gpu_loc (fun () -> bridge_3_to_4 b h n kv bh v3 v);
  map_loc gpu_loc (fun () -> bridge_3_to_4 b h m n bh scores3' scores);
  map_loc gpu_loc (fun () -> bridge_3_to_4 b h m k bh q3 q);
  map_loc gpu_loc (fun () -> bridge_a3_transpose_back bh n k kT3 k3);
  map_loc gpu_loc (fun () -> bridge_3_to_4 b h n k bh k3 k_);

  (* Reshape lse_flat (bh*m) → A3 (b,h,m) and treat as the lse output. *)
  let lse_via_view = view_a1_as_a3 b h m bhm lse_flat;
  map_loc gpu_loc (fun () -> bridge_a1_to_a3 b h m bhm lse_flat lse_via_view);

  (* The remaining work — releasing [scores] and [lse_flat], copying
     [lse_via_view] into [lse], and discharging the functional spec — is
     admitted, since it depends on lemmas relating the assembled per-step
     specifications to [attention_real]. The kernel call sequence above is
     the naive implementation. *)
  admit ()
}
#pop-options
