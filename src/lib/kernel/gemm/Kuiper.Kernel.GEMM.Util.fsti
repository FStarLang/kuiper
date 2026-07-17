module Kuiper.Kernel.GEMM.Util

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
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
