module Kuiper.Kernel.GEMM.Util

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
module MS = Kuiper.Spec.GEMM
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type

(* Helper: for reals, sum(0 to base+n) = sum(0 to base) + sum over elements base..base+n-1 *)
let rec __gmatmul_single_split
  (#rows #shared #cols : nat)
  (m1 : ematrix real rows shared)
  (m2 : ematrix real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (base : nat{base <= shared})
  (n : nat{base + n <= shared})
  (#sub_n : nat{n <= sub_n})
  (sub_m1 : ematrix real sub_n sub_n)
  (sub_m2 : ematrix real sub_n sub_n)
  (sub_row : natlt sub_n)
  (sub_col : natlt sub_n)
  : Lemma
    (requires
      (forall (k:nat). k < n ==>
        macc sub_m1 sub_row k == macc m1 row (base + k) /\
        macc sub_m2 k sub_col == macc m2 (base + k) col))
    (ensures
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col (base + n)
      ==
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) m1 m2 row col base +.
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n)
    (decreases n)
  = if n = 0 then begin
      ()
    end
    else begin
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) m1 m2 row col (base + n);
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n;
      assert (macc sub_m1 sub_row (n-1) == macc m1 row (base + (n-1)));
      assert (macc sub_m2 (n-1) sub_col == macc m2 (base + (n-1)) col);
      __gmatmul_single_split m1 m2 row col base (n-1) #sub_n sub_m1 sub_m2 sub_row sub_col;
      ()
    end

let rec __matmul_single_approx_real
  (#et:Type) {| d1: scalar et |} {| d2: real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (n : nat{n <= shared})
  : Lemma
    (requires eA %~ rA /\ eB %~ rB)
    (ensures
      MS.__gmatmul_single zero mul add eA eB row col n
      %~
      MS.__gmatmul_single #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col n)
    (decreases n)
  = if n = 0 then ()
    else begin
      __matmul_single_approx_real eA eB rA rB row col (n - 1);
      let a = macc eA row (n-1) in
      let b = macc eB (n-1) col in
      let ra = macc rA row (n-1) in
      let rb = macc rB (n-1) col in
      let ps = MS.__gmatmul_single zero mul add eA eB row col (n-1) in
      let rps = MS.__gmatmul_single #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col (n-1) in
      MS.__gmatmul_single_lemma zero mul add eA eB row col n;
      MS.__gmatmul_single_lemma #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col n
    end

let mmcomb_approx_real
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : ematrix et rows cols)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (rC : ematrix real rows cols)
  = let aux (i : natlt rows) (j : natlt cols)
      : Lemma
        (requires eA %~ rA /\ eB %~ rB /\ eC %~ rC /\ approx2 comb comb_r)
        (ensures macc (MS.mmcomb comb eC eA eB) i j %~ macc (MS.mmcomb comb_r rC rA rB) i j)
      =
        __matmul_single_approx_real eA eB rA rB i j shared;
        (* eC[i,j] %~ rC[i,j] from eC %~ rC *)
        (* matmul_single eA eB i j %~ matmul_single rA rB i j from above *)
        (* approx2 comb comb_r gives: comb x y %~ comb_r r s when x %~ r /\ y %~ s *)
        assert (macc eC i j %~ macc rC i j);
        assert (MS.matmul_single eA eB i j %~ MS.matmul_single rA rB i j);
        ()
    in
    Classical.forall_intro_2 (fun i j ->
      Classical.move_requires (aux i) j)
