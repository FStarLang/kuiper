module Klas.Spr

(* cuBLAS spr (lower) : symmetric packed rank-1 update (in place)

       AP := AP + alpha * x * x^T,

   with AP the lower triangle of an n x n symmetric matrix in PACKED row-major
   storage (entry (i,j), j<=i, at offset off(i)+j; AP length np = n*(n+1)/2) and
   x length n. Each stored entry (i,j) becomes AP[off(i)+j] + alpha*x[i]*x[j].
   Using the inverse packed index (prow/pcol from Klas.Trttp):

       AP_result[p] = AP_old[p] + alpha * x[prow p] * x[pcol p].

   The packed analog of Klas.Syr (lower triangle only). In place, single thread,
   each packed position written exactly once. The invariant is stated over the
   packed index p via the inverse (processed positions hold the new value,
   unprocessed hold the original). *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tpmv { off, off_mono, poff_bound }
open Klas.Trttp { off_ge, scan_row, prow, pcol, pcol_le, prow_off, pidx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Spec: the updated packed cell (via the inverse) and the whole array.      *)
(* ----------------------------------------------------------------------- *)

let fspr_at (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx : lseq et n)
  (p:nat{p < np}) : et
  = pidx_bound n p;
    (Seq.index sAP0 p) `add` (alpha `mul` ((Seq.index sx (prow p)) `mul` (Seq.index sx (pcol p))))

noextract
let fspr (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx : lseq et n) : lseq et np
  = Seq.init np (fun (p:nat{p < np}) -> fspr_at n np alpha sAP0 sx p)

let fspr_ext (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx : lseq et n) (sAP : lseq et np)
  : Lemma (requires (forall (p:nat). p < np /\ prow p < n ==> Seq.index sAP p == fspr_at n np alpha sAP0 sx p))
          (ensures sAP == fspr n np alpha sAP0 sx)
  = introduce forall (p:nat). p < np ==> Seq.index sAP p == fspr_at n np alpha sAP0 sx p
    with introduce _ ==> _
    with _. pidx_bound n p;
    Seq.lemma_eq_intro sAP (fspr n np alpha sAP0 sx)

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: update the lower triangle in place.                 *)
(* ----------------------------------------------------------------------- *)

(* Writing packed position pij (the unique p with (prow,pcol)=(i,j)) moves it
   from the "original" set to the "done" set. Two clauses over the packed index
   p: done positions hold the new value, the rest hold the original. *)
#push-options "--fuel 2 --ifuel 2 --z3rlimit 250"
let spr_upd (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx : lseq et n) (sAP : lseq et np)
  (i:nat{i < n}) (j:nat{j <= i}) (pij:nat{pij < np /\ prow pij == i /\ pcol pij == j}) (v : et)
  : Lemma
      (requires v == fspr_at n np alpha sAP0 sx pij /\
        (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j)) ==>
           Seq.index sAP p == fspr_at n np alpha sAP0 sx p) /\
        (forall (p:nat). p < np /\ ~(prow p < i \/ (prow p == i /\ pcol p < j)) ==>
           Seq.index sAP p == Seq.index sAP0 p))
      (ensures
        (let sAP' = Seq.upd sAP pij v in
         (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
            Seq.index sAP' p == fspr_at n np alpha sAP0 sx p) /\
         (forall (p:nat). p < np /\ ~(prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
            Seq.index sAP' p == Seq.index sAP0 p)))
  = let sAP' = Seq.upd sAP pij v in
    introduce forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
                Seq.index sAP' p == fspr_at n np alpha sAP0 sx p
    with introduce _ ==> _
    with _. (if p = pij then Seq.lemma_index_upd1 sAP pij v
             else (Seq.lemma_index_upd2 sAP pij v p; pcol_le p; pcol_le pij));
    introduce forall (p:nat). p < np /\ ~(prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
                Seq.index sAP' p == Seq.index sAP0 p
    with introduce _ ==> _
    with _. (Seq.lemma_index_upd2 sAP pij v p; pcol_le p; pcol_le pij)
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 250"
inline_for_extraction noextract
fn spr_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fx : perm)
  ()
  norewrite
  requires
    gpu ** gAP |-> sAP0 ** gx |-> Frac fx sx
  ensures
    gpu ** gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sAP : lseq et (SZ.v np)). gAP |-> sAP **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (p:nat). p < SZ.v np /\ prow p < SZ.v !i ==>
               Seq.index sAP p == fspr_at (SZ.v n) (SZ.v np) alpha sAP0 sx p) /\
            (forall (p:nat). p < SZ.v np /\ ~(prow p < SZ.v !i) ==>
               Seq.index sAP p == Seq.index sAP0 p))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let mut j : sz = 0sz;

    while (!j <=^ vi)
      invariant live j
      invariant exists* (sAP : lseq et (SZ.v np)). gAP |-> sAP **
        pure (SZ.v !j <= SZ.v vi + 1 /\
              (forall (p:nat). p < SZ.v np /\
                 (prow p < SZ.v vi \/ (prow p == SZ.v vi /\ pcol p < SZ.v !j)) ==>
                 Seq.index sAP p == fspr_at (SZ.v n) (SZ.v np) alpha sAP0 sx p) /\
              (forall (p:nat). p < SZ.v np /\
                 ~(prow p < SZ.v vi \/ (prow p == SZ.v vi /\ pcol p < SZ.v !j)) ==>
                 Seq.index sAP p == Seq.index sAP0 p))
      decreases (SZ.v vi + 1 - SZ.v !j)
    {
      let vj = !j;
      poff_bound (SZ.v n) (SZ.v vi) (SZ.v vj);
      prow_off (SZ.v vi) (SZ.v vj);
      pidx_bound (SZ.v n) (SZ.v (vo +^ vj));
      let aij = Array1.(gAP.(vo +^ vj));
      let xi = Array1.(gx.(vi));
      let xj = Array1.(gx.(vj));
      let v = aij `add` (alpha `mul` (xi `mul` xj));
      with sappre. assert (gAP |-> sappre);
      Array1.(gAP.(vo +^ vj) <- v);
      spr_upd (SZ.v n) (SZ.v np) alpha sAP0 sx sappre (SZ.v vi) (SZ.v vj) (SZ.v (vo +^ vj)) v;
      j := !j +^ 1sz;
    };
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with sapf. assert (gAP |-> sapf);
  fspr_ext (SZ.v n) (SZ.v np) alpha sAP0 sx sapf;
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n).                           *)
(* ----------------------------------------------------------------------- *)

ghost
fn spr_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fx : perm)
  ()
  norewrite
  requires gAP |-> sAP0 ** gx |-> Frac fx sx
  ensures (forall+ (tid : natlt 1sz). gAP |-> sAP0 ** gx |-> Frac fx sx) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gAP |-> sAP0 ** gx |-> Frac fx sx);
}

ghost
fn spr_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fx : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx) ** emp
  ensures
    gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx);
}

inline_for_extraction noextract
let kamspr
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gAP /\ Array1.is_global gx))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fx : perm)
  : kernel_desc
      (requires gAP |-> sAP0 ** gx |-> Frac fx sx)
      (ensures  gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> spr_kf alpha gAP gx #sAP0 #sx #fx);
    frame    = emp;
    teardown = spr_teardown alpha gAP gx;
    setup    = spr_setup gAP gx;
    kpre  = (fun (_i : natlt 1sz) -> gAP |-> sAP0 ** gx |-> Frac fx sx);
    kpost = (fun (_i : natlt 1sz) -> gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx ** gx |-> Frac fx sx);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn spr_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np) { Array1.is_global gAP })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fx : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx)
  requires
    on gpu_loc (gAP |-> sAP0)
  ensures
    on gpu_loc (gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx)
{
  on_star_eq gpu_loc (gx |-> Frac fx sx) (gAP |-> sAP0);
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gAP |-> sAP0))
       as (on gpu_loc ((gx |-> Frac fx sx) ** (gAP |-> sAP0)));

  launch_sync (kamspr n np alpha gAP gx #() #sAP0 #sx #fx);

  on_star_eq gpu_loc (gx |-> Frac fx sx) (gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx);
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** (gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx)))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc (gAP |-> fspr (SZ.v n) (SZ.v np) alpha sAP0 sx));
  ()
}

let spr_f32 = spr_gen #f32
let spr_f64 = spr_gen #f64
