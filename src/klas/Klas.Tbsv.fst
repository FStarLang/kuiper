module Klas.Tbsv

(* cuBLAS tbsv (lower, non-unit, no transpose) : triangular banded solve

       solve A * y = b  for y,

   with A an n x n lower-triangular BAND matrix with k sub-diagonals in the
   cuBLAS column-major band layout (A(i,j), j<=i<=j+k, at AB[i + j*k]; AB length
   np = (k+1)*n). Forward substitution:

       y[i] = (b[i] - sum_{i-k<=j<i} AB[i+j*k]*y[j]) / AB[i + i*k].

   The diagonal A(i,i) is at band index i + i*k, reached exactly when the running
   band index finishes the sub-diagonal loop (j: 0..i-1). Band analog of
   Klas.Tpsv / Klas.Trsv; float division is total. Reuses bidx_bound from
   Klas.Tbmv. *)

#lang-pulse
open Kuiper
open Kuiper.Array1
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Klas.Tbmv { bidx_bound }
module Array1 = Kuiper.Array1
module SZ = Kuiper.SizeT
module ML = FStar.Math.Lemmas

(* ----------------------------------------------------------------------- *)
(* Float specification (total): forward substitution over the band.          *)
(* ----------------------------------------------------------------------- *)

let rec ftbsv_at (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sb : lseq et n) (i:nat{i < n})
  : Tot et (decreases %[i; 1; 0])
  = bidx_bound n k i i;
    div (bsrow n k np sA sb i i) (Seq.index sA (i + i * k))

(* [bsrow i m] is b[i] - sum_{j<m, in band} A(i,j)*y[j]. *)
and bsrow (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sb : lseq et n) (i:nat{i < n}) (m:nat{m <= i})
  : Tot et (decreases %[i; 0; m])
  = if m = 0 then Seq.index sb i
    else (let j = m - 1 in
          let term : et =
            if i - j <= k then (bidx_bound n k i j; (Seq.index sA (i + j * k)) `mul` (ftbsv_at n k np sA sb j))
            else zero in
          (bsrow n k np sA sb i (m - 1)) `sub` term)

noextract
let ftbsv (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sb : lseq et n)
  : lseq et n
  = Seq.init n (fun (i:nat{i < n}) -> ftbsv_at n k np sA sb i)

let ftbsv_index (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sb : lseq et n) (l:nat{l < n})
  : Lemma (ensures Seq.index (ftbsv n k np sA sb) l == ftbsv_at n k np sA sb l)
          [SMTPat (Seq.index (ftbsv n k np sA sb) l)]
  = ()

(* ----------------------------------------------------------------------- *)
(* Per-row banded forward substitution (running band index bi == i + j*k).   *)
(* ----------------------------------------------------------------------- *)

(* The j-th subtraction term, isolated so the bridge sx[j]==ftbsv_at j and the
   band coefficient discharge in a minimal context (it still inlines). *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 100"
inline_for_extraction noextract
fn tbsv_term
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
  (gA : array1 et (l1_forward np))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (j : szlt i)
  (bi : sz { SZ.v bi == SZ.v i + SZ.v j * SZ.v k })
  (sb : erased (lseq et (SZ.v n)))
  (#sA : erased (lseq et (SZ.v np)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gx |-> Frac fx sx
  requires
    pure (forall (l:nat). l < SZ.v i ==>
            Seq.index sx l == ftbsv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sb l)
  returns
    term : et
  ensures
    pure (term == (if SZ.v i - SZ.v j <= SZ.v k
                   then (bidx_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v j);
                         (Seq.index sA (SZ.v i + SZ.v j * SZ.v k))
                         `mul` (ftbsv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sb (SZ.v j)))
                   else zero))
{
  let z : et = zero;
  bidx_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v j);
  let a = Array1.(gA.(bi));
  let xj = Array1.(gx.(j));
  if ((i -^ j) <=^ k) { a `mul` xj } else { z }
}
#pop-options

#push-options "--fuel 2 --ifuel 1 --z3rlimit 150"
inline_for_extraction noextract
fn tbsv_row
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
  (gA : array1 et (l1_forward np))
  (gb : array1 et (l1_forward n))
  (gx : array1 et (l1_forward n))
  (i : szlt n)
  (#sA : erased (lseq et (SZ.v np)))
  (#sb : erased (lseq et (SZ.v n)))
  (#sx : erased (lseq et (SZ.v n)))
  (#fa #fb #fx : perm)
  preserves
    gpu ** gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> Frac fx sx
  requires
    pure (forall (l:nat). l < SZ.v i ==>
            Seq.index sx l == ftbsv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sb l)
  returns
    xi : et
  ensures
    pure (xi == ftbsv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sb (SZ.v i))
{
  let mut s : et = Array1.(gb.(i));
  let mut j : sz = 0sz;
  let mut bi : sz = i;

  while (!j <^ i)
    invariant live s
    invariant live j
    invariant live bi
    invariant pure (SZ.v !j <= SZ.v i /\
                    SZ.v !bi == SZ.v i + SZ.v !j * SZ.v k /\
                    !s == bsrow (SZ.v n) (SZ.v k) (SZ.v np) sA sb (SZ.v i) (SZ.v !j))
    decreases (SZ.v i - SZ.v !j)
  {
    let vj = !j;
    let term = tbsv_term k np gA gx i vj (!bi) sb;
    s := !s `sub` term;
    bidx_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v vj + 1);
    bi := !bi +^ k;
    j := !j +^ 1sz;
  };

  bidx_bound (SZ.v n) (SZ.v k) (SZ.v i) (SZ.v i);
  let aii = Array1.(gA.(!bi));
  !s `div` aii
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* Single-thread kernel: forward substitution row by row.                    *)
(* ----------------------------------------------------------------------- *)

let upd_preserves_eq
  (#et:Type0) {| floating et |}
  (n:nat) (k:nat) (np:nat{np == (k + 1) * n}) (sA : lseq et np) (sb : lseq et n)
  (sxpre : lseq et n) (vi : nat{vi < n}) (xi : et)
  : Lemma (requires (forall (l:nat). l < vi ==> Seq.index sxpre l == ftbsv_at n k np sA sb l) /\
                    xi == ftbsv_at n k np sA sb vi)
          (ensures (forall (l:nat). l < vi + 1 ==>
                      Seq.index (Seq.upd sxpre vi xi) l == ftbsv_at n k np sA sb l))
  = introduce forall (l:nat). l < vi + 1 ==>
                Seq.index (Seq.upd sxpre vi xi) l == ftbsv_at n k np sA sb l
    with introduce _ ==> _
    with _. (if l < vi then Seq.lemma_index_upd2 sxpre vi xi l
             else Seq.lemma_index_upd1 sxpre vi xi)

#push-options "--fuel 2 --ifuel 1 --z3rlimit 120"
inline_for_extraction noextract
fn tbsv_kf
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb
{
  let mut i : szle n = 0sz;

  while (!i <^ n)
    invariant live i
    invariant exists* (sx : lseq et (SZ.v n)). gx |-> sx **
      pure (SZ.v !i <= SZ.v n /\
            (forall (l:nat). l < SZ.v !i ==>
               Seq.index sx l == ftbsv_at (SZ.v n) (SZ.v k) (SZ.v np) sA sb l))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let xi = tbsv_row k np gA gb gx vi;
    with sxpre. assert (gx |-> sxpre);
    Array1.(gx.(vi) <- xi);
    upd_preserves_eq (SZ.v n) (SZ.v k) (SZ.v np) sA sb sxpre (SZ.v vi) xi;
    i := !i +^ 1sz;
  };

  with sxf. assert (gx |-> sxf);
  Seq.lemma_eq_intro sxf (ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb);
  ()
}
#pop-options

(* ----------------------------------------------------------------------- *)
(* CPU-side wrapper (single-thread kernel_desc_n, like Klas.Tpsv).           *)
(* ----------------------------------------------------------------------- *)

ghost
fn tbsv_setup
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
fn tbsv_teardown
  (#et:Type0) {| floating et |}
  (#n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
       gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb) ** emp
  ensures
    gA |-> Frac fa sA ** gb |-> Frac fb sb **
    gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb
{
  forevery_singleton_elim #(natlt 1sz)
    (fun (_:natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
       gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb);
}

inline_for_extraction noextract
let kamtbsv
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
                gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb)
= {
    nthr = 1sz;
    f = (fun (_tid : szlt 1sz) -> tbsv_kf k np gA gb gx #sA #sb #sx0 #fa #fb);
    frame    = emp;
    teardown = tbsv_teardown k np gA gb gx;
    setup    = tbsv_setup k np gA gb gx;
    kpre  = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb ** gx |-> sx0);
    kpost = (fun (_i : natlt 1sz) -> gA |-> Frac fa sA ** gb |-> Frac fb sb **
                                     gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb);
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn tbsv_gen
  (#et:Type0) {| floating et |}
  (n : szp)
  (k : sz)
  (np : szp { SZ.v np == (SZ.v k + 1) * SZ.v n })
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
    on gpu_loc (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb)
{
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> sx0);
  rewrite (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> sx0))
       as (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)));
  on_star_eq gpu_loc (gA |-> Frac fa sA) ((gb |-> Frac fb sb) ** (gx |-> sx0));
  rewrite (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> sx0)))
       as (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> sx0))));

  launch_sync (kamtbsv n k np gA gb gx #() #sA #sb #sx0 #fa #fb);

  on_star_eq gpu_loc (gA |-> Frac fa sA)
             ((gb |-> Frac fb sb) ** (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb));
  rewrite (on gpu_loc ((gA |-> Frac fa sA) ** ((gb |-> Frac fb sb) ** (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb))))
       as (on gpu_loc (gA |-> Frac fa sA) ** on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb)));
  on_star_eq gpu_loc (gb |-> Frac fb sb) (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb);
  rewrite (on gpu_loc ((gb |-> Frac fb sb) ** (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb)))
       as (on gpu_loc (gb |-> Frac fb sb) ** on gpu_loc (gx |-> ftbsv (SZ.v n) (SZ.v k) (SZ.v np) sA sb));
  ()
}

let tbsv_f32 = tbsv_gen #f32
let tbsv_f64 = tbsv_gen #f64
