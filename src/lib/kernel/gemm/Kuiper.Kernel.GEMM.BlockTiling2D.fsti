module Kuiper.Kernel.GEMM.BlockTiling2D

#lang-pulse

open Kuiper
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Array2.Strided.Slice
open Kuiper.Chest
open Kuiper.EMatrix
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM
module MU = Kuiper.Kernel.GEMM.Util

(* Note: BlockTiling2D is the only tiled GEMM that has an exact spec
   (mmcomb_gpu_exact). This is so because it
   iterates through the shared dimension in the same left-to-right order
   as the pure mathematical product.

   The other tiled kernels (Tiled, SHMem, BlockTiling1D) accumulate
   partial results differently (e.g. via tiles that are added together)
   subproduct_cols), which introduces a different association order.

   We should probably rewrite the previous kernels to also
   attain an exact spec, though it is not a problem for now. *)

(* ─── batched (rank-3) entries ───────────────────────────────────────────── *)
(* These are the primitive entries: the kernel launches a single batched
   [kernel_desc].  The rank-2 entries below are derived from these at
   [batch = 1] (via the page-0 layout bridge + nat rank-2⇄rank-3 casts). *)

(* The batched kernel itself, as a [kernel_desc]: clients that need to launch
asynchronously on their own stream (e.g. under CUDA graph capture, where a
device sync is illegal) can [launch] this directly instead of going through the
[*_gpu_*] entries below, which sync.  All the divisibility/alignment/tiling
side-conditions are the same as for [gbmmcomb_gpu_exact]. *)
inline_for_extraction noextract
val bmk_kernel
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#fA : perm)
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#fB : perm)
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (nblk_v : szp{SZ.v nblk_v == batch * (m/bm * (n/bn))})
  (nthr_v : szp{SZ.v nthr_v == bm/tm * (bn/tn)})
  (#_ : squash (batch * (m/bm * (n/bn)) <= max_blocks
               /\ (bm/tm * (bn/tn)) <= max_threads))
  (#_ : squash (aligned 16 (core gA) /\ aligned 16 (core gB)))
  ()
  : kernel_desc
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> eC)
      (gA |-> Frac fA eA ** gB |-> Frac fB eB ** gC |-> MS.gbmmcomb mapA mapB comb eC eA eB)

inline_for_extraction noextract
fn gbmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gbmmcomb mapA mapB comb eC eA eB)

(* Scalar batched wrapper: [bmmcomb] is [gbmmcomb] at the identity pre-maps. *)
inline_for_extraction noextract
fn bmmcomb_gpu_exact
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk et /?+ s3A.rstride3 /\ chunk et /?+ s3A.offset3 /\ chunk et /?+ s3A.pstride3))
  (#_ : squash (chunk et /?+ s3B.rstride3 /\ chunk et /?+ s3B.offset3 /\ chunk et /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 et lA { is_global gA })
  (#eA : chest3 et batch m k)
  (gB : array3 et lB { is_global gB })
  (#eB : chest3 et batch k n)
  (gC : array3 et lC { is_global gC })
  (#eC : chest3 et batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.bmmcomb comb eC eA eB)

inline_for_extraction noextract
fn gbmmcomb_gpu_approx
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc,
     has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { approx2 comb comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk ta /?+ s3A.rstride3 /\ chunk ta /?+ s3A.offset3 /\ chunk ta /?+ s3A.pstride3))
  (#_ : squash (chunk tb /?+ s3B.rstride3 /\ chunk tb /?+ s3B.offset3 /\ chunk tb /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 ta lA { is_global gA })
  (#eA : chest3 ta batch m k)
  (gB : array3 tb lB { is_global gB })
  (#eB : chest3 tb batch k n)
  (gC : array3 tc lC { is_global gC })
  (#eC : chest3 tc batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest3 tc batch m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB)

inline_for_extraction noextract
fn bmmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#batch #m #n #k : szp)
  (#lA : layout3 batch m k)
  (#lB : layout3 batch k n)
  (#lC : layout3 batch m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| s3A : strided_row_major_3 lA, s3B : strided_row_major_3 lB |}
  (#_ : squash (chunk et /?+ s3A.rstride3 /\ chunk et /?+ s3A.offset3 /\ chunk et /?+ s3A.pstride3))
  (#_ : squash (chunk et /?+ s3B.rstride3 /\ chunk et /?+ s3B.offset3 /\ chunk et /?+ s3B.pstride3))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3A.offset3 + s3A.pstride3 * p)))
  (#_ : squash (forall (p:natlt batch). SZ.fits (s3B.offset3 + s3B.pstride3 * p)))
  (gA : array3 et lA { is_global gA })
  (#eA : chest3 et batch m k)
  (gB : array3 et lB { is_global gB })
  (#eB : chest3 et batch k n)
  (gC : array3 et lC { is_global gC })
  (#eC : chest3 et batch m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (#sqf : squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (sq1 : squash (SZ.v m == (SZ.v m/SZ.v bm) * ((SZ.v bm/SZ.v tm) * SZ.v tm)))
  (sq2 : squash (SZ.v n == (SZ.v n/SZ.v bn) * ((SZ.v bn/SZ.v tn) * SZ.v tn)))
  (#_ : squash (SZ.fits (m * n)))
  (#fA #fB : perm)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (batch * (m/bm * (n/bn)) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest3 et batch m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.bmmcomb comb_r rC rA rB)

(* ─── rank-2 entries (derived at [batch = 1]) ──────────────────────────────── *)

inline_for_extraction noextract
fn gmmcomb_gpu_exact
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, scalar tb, scalar tc, scalar tacc, has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk ta) str_A))
  (#_ : squash (aligned_strided_row_major (chunk tb) str_B))
  (gA : array2 ta lA { is_global gA })
  (#eA : chest2 ta m k)
  (gB : array2 tb lB { is_global gB })
  (#eB : chest2 tb k n)
  (gC : array2 tc lC { is_global gC })
  (#eC : chest2 tc m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.gmmcomb mapA mapB comb eC eA eB)

inline_for_extraction noextract
fn gmmcomb_gpu_approx
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc,
     has_vec_cpy ta, has_vec_cpy tb |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk ta) str_A))
  (#_ : squash (aligned_strided_row_major (chunk tb) str_B))
  (gA : array2 ta lA { is_global gA })
  (#eA : chest2 ta m k)
  (gB : array2 tb lB { is_global gB })
  (#eB : chest2 tb k n)
  (gC : array2 tc lC { is_global gC })
  (#eC : chest2 tc m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk tb /?+ bn))
  (#_ : squash (chunk ta /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk ta * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk tb * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (MU.approx1 mapA mapA_r /\ MU.approx1 mapB mapB_r) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 tc m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB)

inline_for_extraction noextract
fn mmcomb_gpu_exact
  (#et : Type0) {| scalar et, has_vec_cpy et |}
  (comb : binop et)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { is_global gA })
  (#eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (#eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    on gpu_loc (gC |-> eC)
  ensures
    on gpu_loc (gC |-> MS.mmcomb comb eC eA eB)

inline_for_extraction noextract
fn mmcomb_gpu_approx
  (#et : Type0) {| scalar et, has_vec_cpy et, real_like et |}
  (comb : binop et)
  (comb_r : binop real { approx2 comb comb_r })
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (#lC : layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  {| str_A : strided_row_major lA,
     str_B : strided_row_major lB |}
  (#_ : squash (aligned_strided_row_major (chunk et) str_A))
  (#_ : squash (aligned_strided_row_major (chunk et) str_B))
  (gA : array2 et lA { is_global gA })
  (#eA : chest2 et m k)
  (gB : array2 et lB { is_global gB })
  (#eB : chest2 et k n)
  (gC : array2 et lC { is_global gC })
  (#eC : chest2 et m n)
  (bm : szp{bm /?+ m})
  (bn : szp{bn /?+ n})
  (bk : szp{bk /?+ k})
  (#_ : squash (chunk et /?+ bn))
  (#_ : squash (chunk et /?+ bk))
  (tm : szp{tm /?+ bm})
  (tn : szp{tn /?+ bn})
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bm * bk)))
  (#_ : squash (chunk et * (bm/tm * (bn/tn)) /?+ (bk * bn)))
  (#_ : squash (SZ.fits (bm*bk + bm/tm*(bn/tn))))
  (#_ : squash (SZ.fits (bk*bn + bm/tm*(bn/tn))))
  (#_: squash (SZ.fits (bm * bk) /\ SZ.fits (bk * bn)))
  (slA : full_layout2 bm bk)
  (slB : full_layout2 bk bn)
  {| T.ctlayout slA, T.ctlayout slB |}
  (#fA #fB : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  norewrite
  preserves
    cpu **
    on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (aligned 16 (core gA)) **
    pure (aligned 16 (core gB)) **
    pure (m/bm * (n/bn) <= max_blocks) **
    pure (bm/tm * (bn/tn) <= max_threads) **
    pure (eA %~ rA /\ eB %~ rB /\ eC %~ rC) **
    on gpu_loc (gC |-> eC)
  ensures
    exists* (eC' : chest2 et m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.mmcomb comb_r rC rA rB)
