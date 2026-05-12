module Kuiper.Kernel.GEMM.Util

#lang-pulse

open Kuiper
open Pulse.Lib.Trade
module MS = Kuiper.Spec.GEMM
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

(* Key lemma: stepping the tiled partial sum by one tile block.
   Shows that __real_matmul_single_tiled at ((bk+1)*tile) equals
   the value at (bk*tile) plus the subtile matmul_single.

   This is provable because real arithmetic is associative, so we can
   regroup the sum any way we like. *)

(* Helper: for reals, sum(0 to base+n) = sum(0 to base) + sum over elements base..base+n-1
   The second sum accesses elements at offset indices that match the sub_m matrices.
   Note: sub_m1 and sub_m2 have dimension sub_n, and we sum the first n elements (n <= sub_n). *)
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
      (* Elements match for indices 0..n-1 *)
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
      (* Base case: sum(0 to base+0) = sum(0 to base) + 0 *)
      ()
    end
    else begin
      (* Step the main sum *)
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) m1 m2 row col (base + n);

      (* Step the sub sum *)
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) sub_m1 sub_m2 sub_row sub_col n;

      (* The products are equal by the element matching hypothesis *)
      assert (macc sub_m1 sub_row (n-1) == macc m1 row (base + (n-1)));
      assert (macc sub_m2 (n-1) sub_col == macc m2 (base + (n-1)) col);

      (* Recurse for the (n-1) case *)
      __gmatmul_single_split m1 m2 row col base (n-1) #sub_n sub_m1 sub_m2 sub_row sub_col;

      (* Associativity of +. completes the proof *)
      ()
    end

(* The elements of ematrix_to_real(ematrix_subtile m ...) at (i,k) equal
   ematrix_to_real(m) at the global indices *)
let ematrix_to_real_subtile_index
  (#et:Type) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : ematrix et rows cols)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  (tr : natlt (rows/trows))
  (tc : natlt (cols/tcols))
  (i : natlt trows)
  (j : natlt tcols)
  : Lemma
    (ensures
      macc (ematrix_to_real (ematrix_subtile m trows tcols tr tc)) i j
      ==
      macc (ematrix_to_real m) (tr * trows + i) (tc * tcols + j))
  = ()

let __real_matmul_single_tiled_step
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (#tile : pos)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (bi : natlt rows) (bj : natlt cols) (bk : nat{bk < shared})
  (i : natlt tile) (j : natlt tile)
  : Lemma
    (requires True)
    (ensures (
      let row = bi * tile + i in
      let col = bj * tile + j in
      __real_matmul_single_tiled m1 m2 row col ((bk + 1) * tile)
      ==
      __real_matmul_single_tiled m1 m2 row col (bk * tile) +.
      real_matmul_single_subtile m1 m2 bi bj bk i j
    ))
  = let row = bi * tile + i in
    let col = bj * tile + j in
    let rm1 = ematrix_to_real m1 in
    let rm2 = ematrix_to_real m2 in
    let sub_rm1 = ematrix_to_real (ematrix_subtile m1 tile tile bi bk) in
    let sub_rm2 = ematrix_to_real (ematrix_subtile m2 tile tile bk bj) in

    (* (bk+1)*tile = bk*tile + tile *)
    assert ((bk + 1) * tile == bk * tile + tile);

    (* The subtile elements match: sub_rm1[i,k] = rm1[row, bk*tile+k] etc *)
    let aux (k:natlt tile) : Lemma
      (ensures
        macc sub_rm1 i k == macc rm1 row (bk * tile + k) /\
        macc sub_rm2 k j == macc rm2 (bk * tile + k) col)
      = ematrix_to_real_subtile_index m1 tile tile bi bk i k;
        ematrix_to_real_subtile_index m2 tile tile bk bj k j
    in
    Classical.forall_intro aux;

    (* Apply the split lemma *)
    __gmatmul_single_split rm1 rm2 row col (bk * tile) tile sub_rm1 sub_rm2 i j

(* Lemma: scalar matmul_single of subtile approximates the real version.
   This follows because:
   - Each scalar product a*b approximates to_real(a) *. to_real(b) by a_mul
   - The sum of approximations approximates the sum of reals by a_add (inductively) *)
let rec matmul_single_subtile_approx_aux
  (#et:Type) {| d1: scalar et |} {| d2: real_like et |}
  (#rows #cols : nat)
  (m1 : ematrix et rows cols)
  (m2 : ematrix et cols rows)
  (row : natlt rows)
  (col : natlt rows)
  (n : nat{n <= cols})
  : Lemma
    (ensures
      MS.__gmatmul_single zero mul add m1 m2 row col n
      %~
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col n)
    (decreases n)
  = if n = 0 then begin
      (* zero %~ 0.0R by a0 *)
      ()
    end
    else begin
      (* Recurse *)
      matmul_single_subtile_approx_aux m1 m2 row col (n - 1);

      (* Get the last product *)
      let a = macc m1 row (n-1) in
      let b = macc m2 (n-1) col in
      let ra = macc (ematrix_to_real m1) row (n-1) in
      let rb = macc (ematrix_to_real m2) (n-1) col in

      (* a %~ to_real a, b %~ to_real b *)
      to_real_ok a;
      to_real_ok b;

      (* a*b %~ ra *. rb by a_mul *)
      a_mul a b (to_real a) (to_real b);

      (* IH: partial_sum %~ real_partial_sum *)
      (* a*b %~ ra *. rb *)
      (* By a_add: partial_sum + a*b %~ real_partial_sum +. ra *. rb *)
      let ps = MS.__gmatmul_single zero mul add m1 m2 row col (n-1) in
      let rps = MS.__gmatmul_single 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col (n-1) in
      a_add ps (mul a b) rps (ra *. rb);

      (* Step both sums *)
      MS.__gmatmul_single_lemma zero mul add m1 m2 row col n;
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col n;

      ()
    end

let matmul_single_subtile_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (#tile : pos)
  (m1 : ematrix et (rows * tile) (shared * tile))
  (m2 : ematrix et (shared * tile) (cols * tile))
  (bi : natlt rows) (bj : natlt cols) (bk : natlt shared)
  (i : natlt tile) (j : natlt tile)
  : Lemma
    (ensures (
      MS.matmul_single (ematrix_subtile m1 tile tile bi bk)
                       (ematrix_subtile m2 tile tile bk bj)
                       i j
      %~ real_matmul_single_subtile m1 m2 bi bj bk i j
    ))
  = let sub_m1 = ematrix_subtile m1 tile tile bi bk in
    let sub_m2 = ematrix_subtile m2 tile tile bk bj in
    matmul_single_subtile_approx_aux sub_m1 sub_m2 i j tile

let rec __matmul_single_approx
  (#et:Type) {| d1: scalar et |} {| d2: real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  (n : nat{n <= shared})
  : Lemma
    (ensures
      MS.__gmatmul_single zero mul add m1 m2 row col n
      %~
      MS.__gmatmul_single 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col n)
    (decreases n)
  = if n = 0 then begin
      ()
    end
    else begin
      __matmul_single_approx m1 m2 row col (n - 1);

      let a = macc m1 row (n-1) in
      let b = macc m2 (n-1) col in
      let ra = macc (ematrix_to_real m1) row (n-1) in
      let rb = macc (ematrix_to_real m2) (n-1) col in

      to_real_ok a;
      to_real_ok b;
      a_mul a b (to_real a) (to_real b);

      let ps = MS.__gmatmul_single zero mul add m1 m2 row col (n-1) in
      let rps = MS.__gmatmul_single 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col (n-1) in
      a_add ps (mul a b) rps (ra *. rb);

      MS.__gmatmul_single_lemma zero mul add m1 m2 row col n;
      MS.__gmatmul_single_lemma 0.0R ( *. ) ( +. ) (ematrix_to_real m1) (ematrix_to_real m2) row col n;
      ()
    end

let matmul_single_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared cols)
  (row : natlt rows)
  (col : natlt cols)
  = __matmul_single_approx m1 m2 row col shared

let mmcomb_approx
  (#et:Type) {| scalar et, real_like et |}
  (comb : binop et)
  (comb_r : binop real)
  (#rows #shared #cols : nat)
  (eC : ematrix et rows cols)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  = let aux (i : natlt rows) (j : natlt cols)
      : Lemma (macc (MS.mmcomb comb eC eA eB) i j %~ macc (real_mmcomb comb_r eC eA eB) i j)
      =
        to_real_ok (macc eC i j);
        matmul_single_approx eA eB i j;
        ()
    in
    Classical.forall_intro_2 aux

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
      a_mul a b ra rb;
      let ps = MS.__gmatmul_single zero mul add eA eB row col (n-1) in
      let rps = MS.__gmatmul_single #real #real 0.0R Kuiper.Scalars.mul Kuiper.Scalars.add rA rB row col (n-1) in
      a_add ps (mul a b) rps (Kuiper.Scalars.mul ra rb);
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
