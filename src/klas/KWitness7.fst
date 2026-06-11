module KWitness7

(* Implementation of the imp7.cu witness (forward dot product, TRANSPOSED write).
   See KWitness7.fsti for the specification and the bit-equivalence rationale.

   The public entry point is `matmul_f32`. Everything else in this module
   (the forward-dot approximation lemma, the transposed result matrix, the
   partially-filled matrix invariant, and the single-thread kernel) is private
   implementation detail. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

(* -------------------------------------------------------------------------- *)
(* Phase A: the forward dot product is the library's matmul_single. We prove   *)
(* the f32 forward dot approximates the real matmul cell, then assemble the    *)
(* TRANSPOSED result matrix and show it approximates mtranspose (matmul).      *)
(* -------------------------------------------------------------------------- *)

(* The f32 forward partial dot product approximates the real one. Forward
   induction on `to`, using the library's matmul recurrence + the add/mul
   approximation congruences. (Same shape as KWitness2's rdot_approx, but both
   sides are the forward sum, so no reverse/forward bridge is needed.) *)
#push-options "--fuel 2 --ifuel 1"
let rec msingle_approx
  (#et : Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (to : nat{to <= shared})
  : Lemma
      (requires
        (forall (a : natlt rows) (b : natlt shared). macc eA a b %~ macc rA a b) /\
        (forall (a : natlt shared) (b : natlt cols). macc eB a b %~ macc rB a b))
      (ensures MS.__matmul_single eA eB i j to %~ MS.__matmul_single rA rB i j to)
      (decreases to)
  = if to > 0 then begin
      msingle_approx eA eB rA rB i j (to - 1);
      MS.matmul_single_lemma eA eB i j to;
      MS.matmul_single_lemma rA rB i j to;
      Kuiper.Approximates.Base.a_mul
        (macc eA i (to - 1)) (macc eB (to - 1) j) (macc rA i (to - 1)) (macc rB (to - 1) j);
      Kuiper.Approximates.Base.a_add
        (MS.__matmul_single eA eB i j (to - 1)) (mul (macc eA i (to - 1)) (macc eB (to - 1) j))
        (MS.__matmul_single rA rB i j (to - 1)) (mul (macc rA i (to - 1)) (macc rB (to - 1) j))
    end else ()
#pop-options

(* Per-cell: the f32 forward matmul cell approximates the real matmul cell. *)
let msingle_cell_approx
  (#et : Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures MS.matmul_single eA eB i j %~ MS.matmul_single rA rB i j)
  = msingle_approx eA eB rA rB i j shared

(* The TRANSPOSED result matrix produced by the kernel: A is m x k, B is k x n,
   so A*B is m x n and its transpose is n x m. Cell (i, j) of the output holds the
   forward dot product matmul_single eA eB j i (= (A*B)[j][i]). *)
let tpose_result
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  : ematrix et n m
  = mkM (fun i j -> MS.matmul_single eA eB j i)

(* The transposed result matrix approximates the transpose of the real matmul. *)
let tpose_result_approx
  (#et : Type) {| scalar et, real_like et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures tpose_result eA eB %~ mtranspose (MS.matmul rA rB))
  = introduce forall (i : natlt n) (j : natlt m).
        macc (tpose_result eA eB) i j %~ macc (mtranspose (MS.matmul rA rB)) i j
    with (
      msingle_cell_approx eA eB rA rB j i;
      MS.lemma_matmul_index rA rB j i
    )

(* -------------------------------------------------------------------------- *)
(* Phase B: the partially-filled matrix invariant for the outer cell loop.     *)
(* `tpartial eC g` is the n x m matrix whose first g cells (flat row-major)    *)
(* hold the transposed forward dot product and whose rest still hold eC.       *)
(* -------------------------------------------------------------------------- *)

let tpartial
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et n m)
  (g : nat)
  : ematrix et n m
  = mkM (fun i j -> if i * m + j < g then MS.matmul_single eA eB j i else macc eC i j)

let tpartial_zero
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et n m)
  : Lemma (tpartial eA eB eC 0 == eC)
  = introduce forall (i : natlt n) (j : natlt m).
        macc (tpartial eA eB eC 0) i j == macc eC i j
    with ();
    assert (equal (tpartial eA eB eC 0) eC)

let tpartial_full
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et n m)
  : Lemma (tpartial eA eB eC (n * m) == tpose_result eA eB)
  = introduce forall (i : natlt n) (j : natlt m).
        macc (tpartial eA eB eC (n * m)) i j == macc (tpose_result eA eB) i j
    with (
      Math.Lemmas.lemma_mult_le_right m i (n - 1)
    );
    assert (equal (tpartial eA eB eC (n * m)) (tpose_result eA eB))

#push-options "--z3rlimit 40"
let tpartial_step
  (#et : Type) {| scalar et |}
  (#m #n #k : nat)
  (eA : ematrix et m k)
  (eB : ematrix et k n)
  (eC : ematrix et n m)
  (g : nat{g < n * m})
  : Lemma
      (requires m > 0 /\ g / m < n /\ g % m < m)
      (ensures
        mupd (tpartial eA eB eC g) (g / m) (g % m)
             (MS.matmul_single eA eB (g % m) (g / m))
        == tpartial eA eB eC (g + 1))
  = let r : natlt n = g / m in
    let c : natlt m = g % m in
    Math.Lemmas.euclidean_division_definition g m;
    introduce forall (i : natlt n) (j : natlt m).
        macc (mupd (tpartial eA eB eC g) r c (MS.matmul_single eA eB c r)) i j
        == macc (tpartial eA eB eC (g + 1)) i j
    with (
      if i = r && j = c then ()
      else begin
        Math.Lemmas.euclidean_division_definition g m;
        assert (i * m + j <> g)
      end
    );
    assert (equal (mupd (tpartial eA eB eC g) r c (MS.matmul_single eA eB c r))
                  (tpartial eA eB eC (g + 1)))
#pop-options

(* g / d < q, given g < q * d. *)
let div_bound (g q d : nat)
  : Lemma (requires d > 0 /\ g < q * d) (ensures g / d < q)
  = Math.Lemmas.euclidean_division_definition g d;
    if g / d >= q then Math.Lemmas.lemma_mult_le_right d q (g / d)

(* cit_fits is a (non-unfold) let; this intro lemma exposes it from index bounds. *)
let cit_fits_intro (rows cols : nat) (a b : sz)
  : Lemma (requires v a < rows /\ v b < cols)
          (ensures M.cit_fits rows cols (a, b))
  = ()

(* -------------------------------------------------------------------------- *)
(* Phase C: the single-block, single-thread kernel doing the whole matmul,    *)
(* accumulating each cell's dot product forward and writing it transposed.     *)
(* The output gC is n x m (the transpose of the m x n product).               *)
(* -------------------------------------------------------------------------- *)

inline_for_extraction noextract
fn tpose_matmul_kf
  (#et : Type0) {| scalar et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout n m)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (#eA : ematrix et _ _)
  (#eB : ematrix et _ _)
  (#eC : ematrix et _ _)
  (#fA #fB : perm)
  (#_ : squash (SZ.fits (v n * v m)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    gC |-> tpose_result eA eB
{
  tpartial_zero eA eB eC;
  let mut g : szle (n *^ m) = 0sz;
  while (!g <^ n *^ m)
    invariant live g
    invariant gpu
    invariant gA |-> Frac fA eA
    invariant gB |-> Frac fB eB
    invariant gC |-> tpartial eA eB eC (v !g)
    decreases (v (n *^ m) - v !g)
  {
    let g0 = !g;
    div_bound (v g0) (v n) (v m);
    let row : sz = g0 /^ m; assert (rewrites_to row (g0 /^ m));
    let col : sz = g0 %^ m; assert (rewrites_to col (g0 %^ m));

    let mut idx : szle k = 0sz;
    let mut s : et = zero;
    MS.matmul_zero_lemma eA eB (v col) (v row);
    while (!idx <^ k)
      invariant live idx
      invariant gpu
      invariant gA |-> Frac fA eA
      invariant gB |-> Frac fB eB
      invariant
        (exists* (sv : et).
          (s |-> sv) **
          pure (sv == MS.__matmul_single eA eB (v col) (v row) (v !idx)))
      decreases (v k - v !idx)
    {
      let i0 : sz = !idx +^ 0sz; assert (rewrites_to i0 (!idx +^ 0sz));
      cit_fits_intro (v m) (v k) col i0;
      cit_fits_intro (v k) (v n) i0 row;
      let av = M.read gA (col, i0);
      let bv = M.read gB (i0, row);
      MS.matmul_single_lemma eA eB (v col) (v row) (v !idx + 1);
      s := !s `add` (av `mul` bv);
      idx := !idx +^ 1sz;
    };

    let value = !s;
    cit_fits_intro (v n) (v m) row col;
    M.write gC (row, col) value;
    tpartial_step eA eB eC (v g0);
    g := !g +^ 1sz;
  };
  assert (pure (v (n *^ m) == v n * v m));
  tpartial_full eA eB eC;
  rewrite (gC |-> tpartial eA eB eC (v !g)) as (gC |-> tpose_result eA eB);
}

(* -------------------------------------------------------------------------- *)
(* Phase D: host wrapper + the concrete f32 row-major entry point.             *)
(* -------------------------------------------------------------------------- *)

inline_for_extraction noextract
fn tpose_matmul_gpu
  (#et : Type0) {| scalar et, real_like et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout n m)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA { M.is_global gA })
  (gB : M.array2 et lB { M.is_global gB })
  (gC : M.array2 et lC { M.is_global gC })
  (rA : ematrix real _ _)
  (rB : ematrix real _ _)
  (#eA : ematrix et _ _)
  (#eB : ematrix et _ _)
  (#eC : ematrix et _ _)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (v n * v m)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (v n) (v m)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ mtranspose (MS.matmul rA rB)))
{
  M.pts_to_ref_located gA;
  M.pts_to_ref_located gB;
  M.pts_to_ref_located gC;

  launch_kernel_1 (fun _ -> tpose_matmul_kf m n k gA gB gC #eA #eB #eC #fA #fB);

  tpose_result_approx #et #_ #_ #(v m) #(v n) #(v k) eA eB rA rB;
  ()
}

fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major n m) { M.is_global gC })
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (#eA : ematrix f32 m k)
  (#eB : ematrix f32 k n)
  (#eC : ematrix f32 n m)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (SZ.v n * SZ.v m)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 n m).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ mtranspose (MS.matmul rA rB)))
{
  tpose_matmul_gpu m n k gA gB gC rA rB;
}
