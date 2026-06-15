module Klas.Spr2

(* cuBLAS spr2 (lower) : symmetric packed rank-2 update (in place)

       AP := AP + alpha * (x*y^T + y*x^T),

   with AP the lower triangle of an n x n symmetric matrix in PACKED row-major
   storage (entry (i,j), j<=i, at offset off(i)+j; length np = n*(n+1)/2) and
   x, y length n. Using the inverse packed index (prow/pcol from Klas.Trttp):

       AP_result[p] = AP_old[p] + alpha*(x[prow p]*y[pcol p] + y[prow p]*x[pcol p]).

   Packed analog of Klas.Syr2 (lower triangle only); like Klas.Spr with two
   vectors. In place, single thread. *)

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

let fspr2_at (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx sy : lseq et n)
  (p:nat{p < np}) : et
  = pidx_bound n p;
    (Seq.index sAP0 p) `add`
    (alpha `mul` (((Seq.index sx (prow p)) `mul` (Seq.index sy (pcol p))) `add`
                  ((Seq.index sy (prow p)) `mul` (Seq.index sx (pcol p)))))

noextract
let fspr2 (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx sy : lseq et n) : lseq et np
  = Seq.init np (fun (p:nat{p < np}) -> fspr2_at n np alpha sAP0 sx sy p)

let fspr2_ext (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx sy : lseq et n) (sAP : lseq et np)
  : Lemma (requires (forall (p:nat). p < np /\ prow p < n ==> Seq.index sAP p == fspr2_at n np alpha sAP0 sx sy p))
          (ensures sAP == fspr2 n np alpha sAP0 sx sy)
  = introduce forall (p:nat). p < np ==> Seq.index sAP p == fspr2_at n np alpha sAP0 sx sy p
    with introduce _ ==> _
    with _. pidx_bound n p;
    Seq.lemma_eq_intro sAP (fspr2 n np alpha sAP0 sx sy)

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: update the lower triangle in place.                 *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 250"
let spr2_upd (#et:Type0) {| floating et |}
  (n:pos) (np:nat{np == off n}) (alpha : et) (sAP0 : lseq et np) (sx sy : lseq et n) (sAP : lseq et np)
  (i:nat{i < n}) (j:nat{j <= i}) (pij:nat{pij < np /\ prow pij == i /\ pcol pij == j}) (v : et)
  : Lemma
      (requires v == fspr2_at n np alpha sAP0 sx sy pij /\
        (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j)) ==>
           Seq.index sAP p == fspr2_at n np alpha sAP0 sx sy p) /\
        (forall (p:nat). p < np /\ ~(prow p < i \/ (prow p == i /\ pcol p < j)) ==>
           Seq.index sAP p == Seq.index sAP0 p))
      (ensures
        (let sAP' = Seq.upd sAP pij v in
         (forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
            Seq.index sAP' p == fspr2_at n np alpha sAP0 sx sy p) /\
         (forall (p:nat). p < np /\ ~(prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
            Seq.index sAP' p == Seq.index sAP0 p)))
  = let sAP' = Seq.upd sAP pij v in
    introduce forall (p:nat). p < np /\ (prow p < i \/ (prow p == i /\ pcol p < j + 1)) ==>
                Seq.index sAP' p == fspr2_at n np alpha sAP0 sx sy p
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
fn spr2_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fx #fy : perm)
  ()
  norewrite
  requires
    gpu ** gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy
  ensures
    gpu ** gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
    gx |-> Frac fx sx ** gy |-> Frac fy sy
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sAP : lseq et (SZ.v np)). gAP |-> sAP **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (p:nat). p < SZ.v np /\ prow p < SZ.v !i ==>
               Seq.index sAP p == fspr2_at (SZ.v n) (SZ.v np) alpha sAP0 sx sy p) /\
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
                 Seq.index sAP p == fspr2_at (SZ.v n) (SZ.v np) alpha sAP0 sx sy p) /\
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
      let yi = Array1.(gy.(vi));
      let yj = Array1.(gy.(vj));
      let v = aij `add` (alpha `mul` ((xi `mul` yj) `add` (yi `mul` xj)));
      with sappre. assert (gAP |-> sappre);
      Array1.(gAP.(vo +^ vj) <- v);
      spr2_upd (SZ.v n) (SZ.v np) alpha sAP0 sx sy sappre (SZ.v vi) (SZ.v vj) (SZ.v (vo +^ vj)) v;
      j := !j +^ 1sz;
    };
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with sapf. assert (gAP |-> sapf);
  fspr2_ext (SZ.v n) (SZ.v np) alpha sAP0 sx sy sapf;
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n).                           *)
(* ----------------------------------------------------------------------- *)

ghost
fn spr2_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fx #fy : perm)
  ()
  norewrite
  requires gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy
  ensures (forall+ (tid : natlt 1sz). gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy);
}

ghost
fn spr2_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fx #fy : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
       gx |-> Frac fx sx ** gy |-> Frac fy sy) ** emp
  ensures
    gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
    gx |-> Frac fx sx ** gy |-> Frac fy sy
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
       gx |-> Frac fx sx ** gy |-> Frac fy sy);
}

inline_for_extraction noextract
let kamspr2
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (gy : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gAP /\ Array1.is_global gx /\ Array1.is_global gy))
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fx #fy : perm)
  : kernel_desc
      (requires gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy)
      (ensures  gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
                gx |-> Frac fx sx ** gy |-> Frac fy sy)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> spr2_kf alpha gAP gx gy #sAP0 #sx #sy #fx #fy);
    frame    = emp;
    teardown = spr2_teardown alpha gAP gx gy;
    setup    = spr2_setup gAP gx gy;
    kpre  = (fun (_i : natlt 1sz) -> gAP |-> sAP0 ** gx |-> Frac fx sx ** gy |-> Frac fy sy);
    kpost = (fun (_i : natlt 1sz) -> gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy **
                                     gx |-> Frac fx sx ** gy |-> Frac fy sy);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn spr2_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (alpha : et)
  (gAP : array1 et (l1_forward np) { Array1.is_global gAP })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (gy : array1 et (l1_forward n) { Array1.is_global gy })
  (#sAP0 : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#sy : erased (lseq et (SZ.v n)))
  (#fx #fy : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gx |-> Frac fx sx) **
    on gpu_loc (gy |-> Frac fy sy)
  requires
    on gpu_loc (gAP |-> sAP0)
  ensures
    on gpu_loc (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy)
{
  on_star_eq gpu_loc (gy |-> Frac fy sy) (gAP |-> sAP0);
  rewrite (on gpu_loc (gy |-> Frac fy sy) ** on gpu_loc (gAP |-> sAP0))
       as (on gpu_loc ((gy |-> Frac fy sy) ** (gAP |-> sAP0)));
  on_star_eq gpu_loc (gx |-> Frac fx sx) ((gy |-> Frac fy sy) ** (gAP |-> sAP0));
  rewrite (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc ((gy |-> Frac fy sy) ** (gAP |-> sAP0)))
       as (on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gAP |-> sAP0))));

  launch_sync (kamspr2 n np alpha gAP gx gy #() #sAP0 #sx #sy #fx #fy);

  on_star_eq gpu_loc (gx |-> Frac fx sx)
             ((gy |-> Frac fy sy) ** (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy));
  rewrite (on gpu_loc ((gx |-> Frac fx sx) ** ((gy |-> Frac fy sy) ** (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy))))
       as (on gpu_loc (gx |-> Frac fx sx) ** on gpu_loc ((gy |-> Frac fy sy) ** (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy)));
  on_star_eq gpu_loc (gy |-> Frac fy sy) (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy);
  rewrite (on gpu_loc ((gy |-> Frac fy sy) ** (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy)))
       as (on gpu_loc (gy |-> Frac fy sy) ** on gpu_loc (gAP |-> fspr2 (SZ.v n) (SZ.v np) alpha sAP0 sx sy));
  ()
}

let spr2_f32 = spr2_gen #f32
let spr2_f64 = spr2_gen #f64
