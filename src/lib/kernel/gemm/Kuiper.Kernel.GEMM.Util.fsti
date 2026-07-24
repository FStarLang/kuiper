module Kuiper.Kernel.GEMM.Util

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module C = Kuiper.Matrix.Casts
open Kuiper.Chest
open Kuiper.Shape
open Kuiper.EMatrix

(* This is now only spec and lemmas. *)

(* Real-valued matrix for specification purposes *)
let ematrix_to_real (#et:Type) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (em : chest2 et rows cols)
  : GTot (chest2 real rows cols)
  = mk2 (fun i j -> to_real (acc2 em i j))

(* Real-valued matmul_single using real arithmetic (which IS associative) *)
let real_matmul_single
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = MS.__gmatmul_single 0.0R ( *. ) ( +. )
      (ematrix_to_real m1) (ematrix_to_real m2)
      row col shared

(* Real-valued gemm_single: combines initial value with real matmul using comb_r *)
let real_gemm_single
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared cols)
  (m0 : chest2 et rows cols)
  (row : natlt rows)
  (col : natlt cols)
  : GTot real
  = comb_r (to_real (acc2 m0 row col)) (real_matmul_single m1 m2 row col)

(* Real-valued GEMM matrix: each cell is real_gemm_single *)
let real_mmcomb
  (comb_r : binop real)
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m0 : chest2 et rows cols)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared cols)
  : GTot (chest2 real rows cols)
  = mk2 (fun i j -> real_gemm_single comb_r m1 m2 m0 i j)

(* Splitting partial sum over real matrices:
   sum(0 to base+n) = sum(0 to base) + sum over subtile elements *)
val __gmatmul_single_split
  (#rows #shared #cols : nat)
  (m1 : chest2 real rows shared)
  (m2 : chest2 real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (base : nat{base <= shared})
  (n : nat{base + n <= shared})
  (#sub_n : nat{n <= sub_n})
  (sub_m1 : chest2 real sub_n sub_n)
  (sub_m2 : chest2 real sub_n sub_n)
  (sub_row : natlt sub_n)
  (sub_col : natlt sub_n)
  : Lemma
    (requires
      (forall (k:nat). k < n ==>
        acc2 sub_m1 sub_row k == acc2 m1 row (base + k) /\
        acc2 sub_m2 k sub_col == acc2 m2 (base + k) col))
    (ensures
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col (base + n)
      ==
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col base +.
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n)

(* Approximation of partial matmul over external real matrices:
   if eA %~ rA and eB %~ rB then
   __gmatmul_single ... eA eB row col n %~ __gmatmul_single ... rA rB row col n *)
val __matmul_single_approx_real
  (#et:Type) {| scalar et |} {| real_like et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (n : nat{n <= shared})
  : Lemma
    (requires eA %~ rA /\ eB %~ rB)
    (ensures
      MS.__gmatmul_single zero mul add eA eB row col n
      %~
      MS.__gmatmul_single zero mul add rA rB row col n)

(* mmcomb approximation over external real matrices:
   If eA %~ rA, eB %~ rB, eC %~ rC, and approx2 comb comb_r,
   then mmcomb comb eC eA eB %~ mmcomb comb_r rC rA rB. *)
val mmcomb_approx_real
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : chest2 et rows cols)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (rC : chest2 real rows cols)
  : Lemma
    (requires approx2 comb comb_r /\
              eA %~ rA /\ eB %~ rB /\ eC %~ rC)
    (ensures MS.mmcomb comb eC eA eB %~ MS.mmcomb comb_r rC rA rB)

(* Batched (rank-3) analogue of [mmcomb_approx_real]: the reusable
   approximation lemma for the natively-batched GEMM spec. If eA %~ rA,
   eB %~ rB, eC %~ rC (per batch page) and approx2 comb comb_r, then the
   per-page batched combine [bmmcomb] approximates its real counterpart. *)
val bmmcomb_approx_real
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#batch #m #n #k : nat)
  (eA : chest3 et batch m k)
  (eB : chest3 et batch k n)
  (eC : chest3 et batch m n)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  : Lemma
    (requires approx2 comb comb_r /\
              eA %~ rA /\ eB %~ rB /\ eC %~ rC)
    (ensures MS.bmmcomb comb eC eA eB %~ MS.bmmcomb comb_r rC rA rB)

(* Approximation of a unary map: [f : a -> b] approximates the real map
   [g : real -> real] iff it sends approximating inputs to approximating
   outputs.  Unary analogue of [approx2], used for the fused input pre-maps. *)
let approx1
  (#a #b : Type0) {| scalar a, real_like a, scalar b, real_like b |}
  (f : a -> b)
  (g : real -> real)
  : prop
  = forall (x:a) (r:real). x %~ r ==> f x %~ g r

(* Mapping preserves approximation: if [approx1 mapE mapR] and [e %~ rr], then
   [chest_map mapE e %~ chest_map mapR rr]. *)
val chest_map_approx
  (#et1 #et2 : Type0) {| scalar et1, real_like et1, scalar et2, real_like et2 |}
  (mapE : et1 -> et2)
  (mapR : real -> real)
  (#rows #cols : nat)
  (e : chest2 et1 rows cols)
  (rr : chest2 real rows cols)
  : Lemma
    (requires approx1 mapE mapR /\ e %~ rr)
    (ensures Kuiper.Chest.chest_map mapE e %~ Kuiper.Chest.chest_map mapR rr)

(* General (fused-map, multi-type) analogue of [mmcomb_approx_real]: if each
   element pre-map approximates its real counterpart ([approx1 mapA mapA_r],
   [approx1 mapB mapB_r]), [approx2 comb comb_r], and the inputs approximate the
   real chests, then the element-level general combine [gmmcomb] approximates
   its real-level counterpart. *)
val gmmcomb_approx_real
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : chest2 tc rows cols)
  (eA : chest2 ta rows shared)
  (eB : chest2 tb shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (rC : chest2 real rows cols)
  : Lemma
    (requires approx1 mapA mapA_r /\ approx1 mapB mapB_r /\ approx2 comb comb_r /\
              eA %~ rA /\ eB %~ rB /\ eC %~ rC)
    (ensures MS.gmmcomb mapA mapB comb eC eA eB
             %~ MS.gmmcomb mapA_r mapB_r comb_r rC rA rB)

(* Batched (rank-3) analogue of [gmmcomb_approx_real]. *)
val gbmmcomb_approx_real
  (#ta #tb #tc #tacc : Type0)
  {| scalar ta, real_like ta, scalar tb, real_like tb,
     scalar tc, real_like tc, scalar tacc, real_like tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (mapA_r mapB_r : real -> real)
  (comb_r : binop real)
  (#batch #m #n #k : nat)
  (eA : chest3 ta batch m k)
  (eB : chest3 tb batch k n)
  (eC : chest3 tc batch m n)
  (rA : chest3 real batch m k)
  (rB : chest3 real batch k n)
  (rC : chest3 real batch m n)
  : Lemma
    (requires approx1 mapA mapA_r /\ approx1 mapB mapB_r /\ approx2 comb comb_r /\
              eA %~ rA /\ eB %~ rB /\ eC %~ rC)
    (ensures MS.gbmmcomb mapA mapB comb eC eA eB
             %~ MS.gbmmcomb mapA_r mapB_r comb_r rC rA rB)

(* ── Batch-one bridges between rank-2 and single-page rank-3 GEMM ────────── *)

(* [c2_to_c3n] preserves the approximation relation (cellwise reindex). *)
val c2_to_c3_approx
  (#et : Type0) {| scalar et, real_like et |}
  (a b : nat)
  (af : squash (all_fit (a @| b @| INil)))
  (e : chest2 et a b)
  (r : chest2 real a b)
  : Lemma (requires e %~ r)
          (ensures C.c2_to_c3n a b af e %~ C.c2_to_c3n a b af r)

(* [c3_to_c2n] preserves the approximation relation (cellwise reindex). *)
val c3_to_c2_approx
  (#et : Type0) {| scalar et, real_like et |}
  (a b : nat)
  (af : squash (all_fit (a @| b @| INil)))
  (e : chest3 et 1 a b)
  (r : chest3 real 1 a b)
  : Lemma (requires e %~ r)
          (ensures C.c3_to_c2n a b af e %~ C.c3_to_c2n a b af r)

(* Lowering a one-page batched gmmcomb yields the rank-2 gmmcomb. *)
val batch1_gmmcomb
  (#ta #tb #tc #tacc : Type0) {| scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (a1 a2 a3 : nat)
  (afC : squash (all_fit (a1 @| a3 @| INil)))
  (afA : squash (all_fit (a1 @| a2 @| INil)))
  (afB : squash (all_fit (a2 @| a3 @| INil)))
  (eC : chest2 tc a1 a3)
  (eA : chest2 ta a1 a2)
  (eB : chest2 tb a2 a3)
  : Lemma (
      C.c3_to_c2n a1 a3 afC
        (MS.gbmmcomb mapA mapB comb
          (C.c2_to_c3n a1 a3 afC eC)
          (C.c2_to_c3n a1 a2 afA eA)
          (C.c2_to_c3n a2 a3 afB eB))
      == MS.gmmcomb mapA mapB comb eC eA eB)
