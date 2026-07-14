module KWitness2

(* Implementation of the imp2.cu witness (reverse K-accumulation). See
   KWitness2.fsti for the specification and the bit-equivalence rationale.

   The public entry point is `matmul_f32`. Everything else in this module
   (the reverse dot product `rdot`, its math lemmas, the partially-filled
   matrix invariant, and the single-thread kernel) is private implementation
   detail. *)

#lang-pulse
open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.Sum { sum, sum_pop_left, sum_pop_right }
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Tensor
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

(* -------------------------------------------------------------------------- *)
(* Phase A: pure specification of the reverse dot product and the math proof  *)
(* that, over the reals, it equals the (order-independent) matmul cell.       *)
(* -------------------------------------------------------------------------- *)

(* Reverse, left-associated partial dot product of row i of eA and column j of
   eB, accumulating indices from (shared-1) down to `from`. rdot(shared) = zero;
   rdot(from) = rdot(from+1) `op` (eA[i,from] * eB[from,j]). Thus rdot 0 is the
   full reverse-order sum that imp2.cu computes. *)
let rec rdot
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat{from <= shared})
  : GTot et (decreases (shared - from))
  = if from >= shared then zero
    else add (rdot eA eB i j (from + 1)) (mul (acc2 eA i from) (acc2 eB from j))

#push-options "--fuel 2 --ifuel 1"
let rdot_step
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat{from < shared})
  : Lemma (rdot eA eB i j from ==
           add (rdot eA eB i j (from + 1)) (mul (acc2 eA i from) (acc2 eB from j)))
  = ()
#pop-options

(* The (fixed-domain) product function whose sum is the dot product. Defined
   over [0, shared) so that the same function object is reused across all the
   `sum` calls below, avoiding domain-dependent rewriting. *)
let prodf
  (#rows #shared #cols : nat)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  : (natlt shared -> GTot real)
  = fun t -> mul (acc2 rA i t) (acc2 rB t j)

(* Over the reals, the reverse dot product equals the canonical (commutative)
   sum of the products. Reverse induction on `from`. *)
#push-options "--fuel 2 --ifuel 1"
let rec rdot_is_sum
  (#rows #shared #cols : nat)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat{from <= shared})
  : Lemma (ensures rdot rA rB i j from == sum from shared (prodf rA rB i j))
          (decreases (shared - from))
  = if from < shared then begin
      rdot_is_sum rA rB i j (from + 1);
      rdot_step rA rB i j from;
      sum_pop_left from shared (prodf rA rB i j)
    end else ()
#pop-options

(* Over the reals, the (forward) matmul cell also equals that same canonical
   sum. Forward induction on `to`, using the library's matmul recurrence. *)
#push-options "--fuel 2 --ifuel 1"
let rec matmul_is_sum
  (#rows #shared #cols : nat)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  (to : nat{to <= shared})
  : Lemma (ensures MS.__matmul_single rA rB i j to == sum 0 to (prodf rA rB i j))
          (decreases to)
  = if to > 0 then begin
      matmul_is_sum rA rB i j (to - 1);
      MS.matmul_single_lemma rA rB i j to;
      sum_pop_right 0 to (prodf rA rB i j)
    end else ()
#pop-options

(* Hence the reverse dot product equals the forward matmul cell, over reals. *)
let rdot_eq_matmul_single
  (#rows #shared #cols : nat)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (rdot rA rB i j 0 == MS.matmul_single rA rB i j)
  = rdot_is_sum rA rB i j 0;
    matmul_is_sum rA rB i j shared

(* The f32 reverse dot product approximates the real reverse dot product:
   induction using the add/mul approximation congruences. *)
#push-options "--fuel 2 --ifuel 1"
let rec rdot_approx
  (#et : Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat{from <= shared})
  : Lemma
      (requires
        (forall (a : natlt rows) (b : natlt shared). acc2 eA a b %~ acc2 rA a b) /\
        (forall (a : natlt shared) (b : natlt cols). acc2 eB a b %~ acc2 rB a b))
      (ensures rdot eA eB i j from %~ rdot rA rB i j from)
      (decreases (shared - from))
  = if from < shared then begin
      rdot_approx eA eB rA rB i j (from + 1);
      rdot_step eA eB i j from;
      rdot_step rA rB i j from;
      Kuiper.Approximates.Base.a_mul
        (acc2 eA i from) (acc2 eB from j) (acc2 rA i from) (acc2 rB from j);
      Kuiper.Approximates.Base.a_add
        (rdot eA eB i j (from + 1)) (mul (acc2 eA i from) (acc2 eB from j))
        (rdot rA rB i j (from + 1)) (mul (acc2 rA i from) (acc2 rB from j))
    end else ()
#pop-options

(* Per-cell: the f32 reverse dot product approximates the real matmul cell. *)
let cell_approx
  (#et : Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures rdot eA eB i j 0 %~ MS.matmul_single rA rB i j)
  = rdot_approx eA eB rA rB i j 0;
    rdot_eq_matmul_single rA rB i j

(* The full result matrix produced by the reverse-accumulation kernel. *)
let rev_result
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  : chest2 et rows cols
  = mk2 (fun i j -> rdot eA eB i j 0)

(* The full result matrix approximates the real matmul (elementwise). *)
let rev_result_approx
  (#et : Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (rA : chest2 real rows shared)
  (rB : chest2 real shared cols)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures rev_result eA eB %~ MS.matmul rA rB)
  = introduce forall (i : natlt rows) (j : natlt cols).
        acc2 (rev_result eA eB) i j %~ acc2 (MS.matmul rA rB) i j
    with (
      cell_approx eA eB rA rB i j
    )

(* -------------------------------------------------------------------------- *)
(* Phase B helpers: the partially-filled matrix invariant for the outer loop.  *)
(* `partial eC g` is the matrix where the first g cells (in flat row-major     *)
(* order) hold their reverse-dot-product result and the rest hold eC.          *)
(* -------------------------------------------------------------------------- *)

let partial
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  (g : nat)
  : chest2 et rows cols
  = mk2 (fun i j -> if i * cols + j < g then rdot eA eB i j 0 else acc2 eC i j)

let partial_zero
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  : Lemma (partial eA eB eC 0 == eC)
  = introduce forall (i : natlt rows) (j : natlt cols).
        acc2 (partial eA eB eC 0) i j == acc2 eC i j
    with ();
    assert (equal (partial eA eB eC 0) eC)

let partial_full
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  : Lemma (partial eA eB eC (rows * cols) == rev_result eA eB)
  = introduce forall (i : natlt rows) (j : natlt cols).
        acc2 (partial eA eB eC (rows * cols)) i j == acc2 (rev_result eA eB) i j
    with (
      Math.Lemmas.lemma_mult_le_right cols i (rows - 1)
    );
    assert (equal (partial eA eB eC (rows * cols)) (rev_result eA eB))

#push-options "--z3rlimit 40"
let partial_step
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : chest2 et rows shared)
  (eB : chest2 et shared cols)
  (eC : chest2 et rows cols)
  (g : nat{g < rows * cols})
  : Lemma
      (requires cols > 0 /\ g / cols < rows /\ g % cols < cols)
      (ensures
        upd2 (partial eA eB eC g) (g / cols) (g % cols)
             (rdot eA eB (g / cols) (g % cols) 0)
        == partial eA eB eC (g + 1))
  = let r : natlt rows = g / cols in
    let c : natlt cols = g % cols in
    Math.Lemmas.euclidean_division_definition g cols;
    introduce forall (i : natlt rows) (j : natlt cols).
        acc2 (upd2 (partial eA eB eC g) r c (rdot eA eB r c 0)) i j
        == acc2 (partial eA eB eC (g + 1)) i j
    with (
      if i = r && j = c then ()
      else begin
        (* (i,j) <> (r,c) and j < cols ==> i*cols+j <> g, so the < g and < g+1
           tests agree. *)
        Math.Lemmas.euclidean_division_definition g cols;
        assert (i * cols + j <> g)
      end
    );
    assert (equal (upd2 (partial eA eB eC g) r c (rdot eA eB r c 0))
                  (partial eA eB eC (g + 1)))
#pop-options

(* g / n < m, given g < m * n. *)
let div_bound (g m n : nat)
  : Lemma (requires n > 0 /\ g < m * n) (ensures g / n < m)
  = Math.Lemmas.euclidean_division_definition g n;
    if g / n >= m then Math.Lemmas.lemma_mult_le_right n m (g / n)

(* -------------------------------------------------------------------------- *)
(* Phase B: the single-block, single-thread kernel doing the whole matmul,    *)
(* accumulating each cell's dot product in reverse (high index to low).       *)
(* -------------------------------------------------------------------------- *)

inline_for_extraction noextract
fn rev_matmul_kf
  (#et : Type0) {| scalar et |}
  (m n k : szp)
  (#lA : M.layout2 m k)
  (#lB : M.layout2 k n)
  (#lC : M.layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (#eA : chest2 et _ _)
  (#eB : chest2 et _ _)
  (#eC : chest2 et _ _)
  (#fA #fB : perm)
  (#_ : squash (SZ.fits (v m * v n)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    gC |-> rev_result eA eB
{
  partial_zero eA eB eC;
  let mut g : szle (m *^ n) = 0sz;
  while (!g <^ m *^ n)
    invariant live g
    invariant gpu
    invariant gA |-> Frac fA eA
    invariant gB |-> Frac fB eB
    invariant gC |-> partial eA eB eC (v !g)
    decreases (v (m *^ n) - v !g)
  {
    let g0 = !g;
    div_bound (v g0) (v m) (v n);
    let row : sz = g0 /^ n; assert (rewrites_to row (g0 /^ n));
    let col : sz = g0 %^ n; assert (rewrites_to col (g0 %^ n));

    let mut idx : szle k = k;
    let mut s : et = zero;
    while (!idx >^ 0sz)
      invariant live idx
      invariant gpu
      invariant gA |-> Frac fA eA
      invariant gB |-> Frac fB eB
      invariant s |-> rdot eA eB (v row) (v col) (v !idx)
      decreases (v !idx)
    {
      let i0 : sz = !idx -^ 1sz; assert (rewrites_to i0 (!idx -^ 1sz));
      let av = M.tensor_read gA (M.cidx2 row i0);
      let bv = M.tensor_read gB (M.cidx2 i0 col);
      rdot_step eA eB (v row) (v col) (v i0);
      s := !s `add` (av `mul` bv);
      idx := !idx -^ 1sz;
    };

    let value = !s;
    M.tensor_write gC (M.cidx2 row col) value;
    partial_step eA eB eC (v g0);
    g := !g +^ 1sz;
  };
  assert (pure (v (m *^ n) == v m * v n));
  partial_full eA eB eC;
  rewrite (gC |-> partial eA eB eC (v !g)) as (gC |-> rev_result eA eB);
}

(* -------------------------------------------------------------------------- *)
(* Phase C: host wrapper, launching the single-thread kernel, and the         *)
(* concrete monomorphic f32 row-major entry point (the imp2.cu witness).      *)
(* -------------------------------------------------------------------------- *)

inline_for_extraction noextract
fn rev_matmul_gpu
  (#et : Type0) {| scalar et, real_like et |}
  (m n k : szp)
  (#lA : M.layout2 m k)
  (#lB : M.layout2 k n)
  (#lC : M.layout2 m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (rA : chest2 real _ _)
  (rB : chest2 real _ _)
  (#eA : chest2 et _ _)
  (#eB : chest2 et _ _)
  (#eC : chest2 et _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (v m * v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 et (v m) (v n)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
{
  M.tensor_pts_to_ref_located gA;
  M.tensor_pts_to_ref_located gB;
  M.tensor_pts_to_ref_located gC;

  launch_kernel_1 (fun _ -> rev_matmul_kf m n k gA gB gC #eA #eB #eC #fA #fB);

  rev_result_approx #et #_ #_ #(v m) #(v k) #(v n) eA eB rA rB;
  ()
}

(* imp2.cu is f32, row-major. This is the bit-equivalent witness. *)
fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major m n) { M.is_global gC })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (#eA : chest2 f32 m k)
  (#eB : chest2 f32 k n)
  (#eC : chest2 f32 m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (SZ.v m * SZ.v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : chest2 f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
{
  rev_matmul_gpu m n k gA gB gC rA rB;
}
