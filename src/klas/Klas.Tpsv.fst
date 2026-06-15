module Klas.Tpsv

(* cuBLAS tpsv (lower, non-unit, no transpose) : triangular packed solve

       solve A * y = x  for y,

   with A an n x n lower-triangular matrix in PACKED row-major storage (entry
   (i,j) with j<=i at offset off(i)+j, off(i)=i*(i+1)/2; AP has length
   np = n*(n+1)/2). Forward substitution:

       y[i] = (x[i] - sum_{j<i} AP[off(i)+j]*y[j]) / AP[off(i)+i].

   The packed analog of Klas.Trsv: same mutually-recursive float spec, but
   indexing the packed array via the recursive offset off (reused from
   Klas.Tpmv) with a running offset carried in the kernel. Float division is
   total, so the spec is total (a nonzero diagonal is needed for the result to
   approximate the real solution, but that is the caller's concern). *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tpmv { off, off_mono, poff_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Float specification: the value the kernel computes (total).               *)
(* ----------------------------------------------------------------------- *)

let rec ftpsv_at (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sb : lseq et n) (i:nat{i < n})
  : Tot et (decreases %[i; 1; 0])
  = poff_bound n i i;
    div (psrow n np sA sb i i) (Seq.index sA (off i + i))

(* [psrow i m] is the running accumulator b[i] - sum_{j<m} A[i][j]*y[j]. *)
and psrow (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sb : lseq et n) (i:nat{i < n}) (m:nat{m <= i})
  : Tot et (decreases %[i; 0; m])
  = if m = 0 then Seq.index sb i
    else begin
      poff_bound n i (m - 1);
      (psrow n np sA sb i (m - 1))
      `sub` ((Seq.index sA (off i + (m - 1))) `mul` (ftpsv_at n np sA sb (m - 1)))
    end

noextract
let ftpsv (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sb : lseq et n)
  : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftpsv_at n np sA sb i)

let ftpsv_index (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sb : lseq et n) (k:nat{k < n})
  : Lemma (ensures Seq.index (ftpsv n np sA sb) k == ftpsv_at n np sA sb k)
          [SMTPat (Seq.index (ftpsv n np sA sb) k)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row forward substitution (offset oi == off i carried in).             *)
(* ----------------------------------------------------------------------- *)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 80"
inline_for_extraction noextract
fn tpsv_row
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (oi : sz { SZ.v oi == off (SZ.v i) })
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fb #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> Frac fx sx
  requires
    pure (forall (k:nat). k < SZ.v i ==>
            Seq.index sx k == ftpsv_at (SZ.v n) (SZ.v np) sA sb k)
  returns
    xi : et
  ensures
    pure (xi == ftpsv_at (SZ.v n) (SZ.v np) sA sb (SZ.v i))
{
  let mut s : et = Array1.(gb.(i));
  let mut j : szle i = 0sz;

  while (!j <^ i)
    invariant live s
    invariant live j
    invariant pure (SZ.v !j <= SZ.v i /\
                    !s == psrow (SZ.v n) (SZ.v np) sA sb (SZ.v i) (SZ.v !j))
    decreases (SZ.v i - SZ.v !j)
  {
    let vj = !j;
    poff_bound (SZ.v n) (SZ.v i) (SZ.v vj);
    let aij = Array1.(gA.(oi +^ vj));
    let xj = Array1.(gx.(vj));
    s := !s `sub` (aij `mul` xj);
    j := !j +^ 1sz;
  };

  poff_bound (SZ.v n) (SZ.v i) (SZ.v i);
  let aii = Array1.(gA.(oi +^ i));
  !s `div` aii
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: forward substitution row by row.                    *)
(* ----------------------------------------------------------------------- *)

let upd_preserves_eq
  (#et:Type0) {| floating et |}
  (n:nat) (np:nat{np == off n}) (sA : lseq et np) (sb : lseq et n)
  (sxpre : lseq et n) (vi : nat{vi < n}) (xi : et)
  : Lemma (requires (forall (k:nat). k < vi ==> Seq.index sxpre k == ftpsv_at n np sA sb k) /\
                    xi == ftpsv_at n np sA sb vi)
          (ensures (forall (k:nat). k < vi + 1 ==>
                      Seq.index (Seq.upd sxpre vi xi) k == ftpsv_at n np sA sb k))
  = introduce forall (k:nat). k < vi + 1 ==>
                Seq.index (Seq.upd sxpre vi xi) k == ftpsv_at n np sA sb k
    with introduce _ ==> _
    with _. (if k < vi
             then Seq.lemma_index_upd2 sxpre vi xi k
             else Seq.lemma_index_upd1 sxpre vi xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn tpsv_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0
  ensures
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb **
    gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb
{
  let mut i : szle n = 0sz;
  let mut o : sz = 0sz;

  while (!i <^ n)
    invariant live i
    invariant live o
    invariant exists* (sx : lseq et (SZ.v n)). gx |-> sx **
      pure (SZ.v !i <= SZ.v n /\ SZ.v !o == off (SZ.v !i) /\
            (forall (k:nat). k < SZ.v !i ==>
               Seq.index sx k == ftpsv_at (SZ.v n) (SZ.v np) sA sb k))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let vo = !o;
    let xi = tpsv_row gA gb gx vi vo;
    with sxpre. assert (gx |-> sxpre);
    Array1.(gx.(vi) <- xi);
    upd_preserves_eq (SZ.v n) (SZ.v np) sA sb sxpre (SZ.v vi) xi;
    off_mono (SZ.v vi + 1) (SZ.v n);
    o := !o +^ (vi +^ 1sz);
    i := !i +^ 1sz;
  };

  with sxf. assert (gx |-> sxf);
  Seq.lemma_eq_intro sxf (ftpsv (SZ.v n) (SZ.v np) sA sb);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Trsv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn tpsv_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0
  ensures (forall+ (tid : natlt 1sz).
             gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0) ** emp
{
  forevery_singleton_intro #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0);
}

ghost
fn tpsv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (#np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  ()
  norewrite
  requires
    (forall+ (tid : natlt 1sz).
       gA |-> Frac fa sA ** gb |-> Frac fb sb **
       gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb) ** emp
  ensures
    gA |-> Frac fa sA ** gb |-> Frac fb sb **
    gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
       gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb);
}

inline_for_extraction noextract
let kamtpsv
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (#_ : squash (Array1.is_global gA /\ Array1.is_global gb /\ Array1.is_global gx))
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  : kernel_desc
      (requires gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0)
      (ensures  gA |-> Frac fa sA ** gb |-> Frac fb sb **
                gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> tpsv_kf gA gb gx #sA #sb #sx0 #fa #fb);
    frame    = emp;
    teardown = tpsv_teardown gA gb gx;
    setup    = tpsv_setup gA gb gx;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
                                     gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tpsv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (np : szp { SZ.v np == off (SZ.v n) })
  (gA : array1 et (l1_forward np) { Array1.is_global gA })
  (gb : array1 et (l1_forward n) { Array1.is_global gb })
  (gx : array1 et (l1_forward n) { Array1.is_global gx })
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx0 : erased (lseq et (SZ.v n)))
  (#fa #fb : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fa sA) **
    on gpu_loc (gb |-> Frac fb sb)
  requires
    on gpu_loc (gx |-> sx0)
  ensures
    on gpu_loc (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb)
{
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> sx0);
  rewrite (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> sx0))
       as (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gb |-> Frac fb sb) ** (gx |-> sx0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> sx0))));

  launch_sync (kamtpsv n np gA gb gx #() #sA #sb #sx0 #fa #fb);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gb |-> Frac fb sb) ** (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb)));
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb);
  rewrite (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb)))
       as (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> ftpsv (SZ.v n) (SZ.v np) sA sb));
  ()
}

let tpsv_f32 = tpsv_gen #f32
let tpsv_f64 = tpsv_gen #f64
