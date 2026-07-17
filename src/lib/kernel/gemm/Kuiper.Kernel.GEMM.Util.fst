module Kuiper.Kernel.GEMM.Util

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
module MS = Kuiper.Spec.GEMM
open Kuiper.EMatrix
module Chest = Kuiper.Chest

(* Helper: for reals, sum(0 to base+n) = sum(0 to base) + sum over elements base..base+n-1 *)
let rec __gmatmul_single_split
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
    (decreases n)
  = if n = 0 then begin
      ()
    end
    else begin
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) m1 m2 row col (base + n);
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n;
      assert (acc2 sub_m1 sub_row (n-1) == acc2 m1 row (base + (n-1)));
      assert (acc2 sub_m2 (n-1) sub_col == acc2 m2 (base + (n-1)) col);
      __gmatmul_single_split m1 m2 row col base (n-1) #sub_n sub_m1 sub_m2 sub_row sub_col;
      ()
    end

let rec __matmul_single_approx_real
  (#et:Type) {| d1: scalar et |} {| d2: real_like et |}
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
      MS.__gmatmul_single #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col n)
    (decreases n)
  = if n = 0 then ()
    else begin
      __matmul_single_approx_real eA eB rA rB row col (n - 1);
      let a = acc2 eA row (n-1) in
      let b = acc2 eB (n-1) col in
      let ra = acc2 rA row (n-1) in
      let rb = acc2 rB (n-1) col in
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
  (eC : chest2 et rows cols)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (rC : chest2 real rows cols)
  = let aux (idx : natlt rows & (natlt cols & unit))
      : Lemma
        (requires eA %~ rA /\ eB %~ rB /\ eC %~ rC /\ approx2 comb comb_r)
        (ensures acc2 (MS.mmcomb comb eC eA eB) idx._1 idx._2._1 %~ acc2 (MS.mmcomb comb_r rC rA rB) idx._1 idx._2._1)
      =
        let (i, (j, ())) = idx in
        __matmul_single_approx_real eA eB rA rB i j shared;
        (* eC[i,j] %~ rC[i,j] from eC %~ rC *)
        (* matmul_single eA eB i j %~ matmul_single rA rB i j from above *)
        (* approx2 comb comb_r gives: comb x y %~ comb_r r s when x %~ r /\ y %~ s *)
        assert (Chest.acc eC idx %~ Chest.acc rC idx);
        assert (MS.matmul_single eA eB i j %~ MS.matmul_single rA rB i j);
        ()
    in
    Classical.forall_intro (fun idx -> Classical.move_requires aux idx)

let chest3_slice_page_approx
  (#et:Type) {| scalar et, real_like et |}
  (#batch #rows #cols : nat)
  (e : chest3 et batch rows cols)
  (r : chest3 real batch rows cols)
  (page : natlt batch)
  = let aux (idx : natlt rows & (natlt cols & unit))
      : Lemma
        (requires e %~ r)
        (ensures acc2 (slice_page e page) idx._1 idx._2._1 %~ acc2 (slice_page r page) idx._1 idx._2._1)
      =
        let (i, (j, ())) = idx in
        assert (Chest.acc e (page, (i, (j, ()))) %~ Chest.acc r (page, (i, (j, ()))));
        ()
    in
    Classical.forall_intro (fun idx -> Classical.move_requires aux idx)

let bmmcomb_approx_real
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
  = let aux (idx : natlt batch & (natlt m & (natlt n & unit)))
      : Lemma
        (requires approx2 comb comb_r /\ eA %~ rA /\ eB %~ rB /\ eC %~ rC)
        (ensures
          acc3 (MS.bmmcomb comb eC eA eB) idx._1 idx._2._1 idx._2._2._1
          %~
          acc3 (MS.bmmcomb comb_r rC rA rB) idx._1 idx._2._1 idx._2._2._1)
      =
        let (page, (row, (col, ()))) = idx in
        chest3_slice_page_approx eA rA page;
        chest3_slice_page_approx eB rB page;
        chest3_slice_page_approx eC rC page;
        mmcomb_approx_real comb comb_r
          (slice_page eC page) (slice_page eA page) (slice_page eB page)
          (slice_page rA page) (slice_page rB page) (slice_page rC page);
        assert (acc2 (MS.mmcomb comb (slice_page eC page) (slice_page eA page) (slice_page eB page)) row col
                %~ acc2 (MS.mmcomb comb_r (slice_page rC page) (slice_page rA page) (slice_page rB page)) row col);
        ()
    in
    Classical.forall_intro (fun idx -> Classical.move_requires aux idx);
    ()
