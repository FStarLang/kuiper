module KWitness3

(* Implementation of the imp3.cu witness (tiled K-accumulation, tile = 16). See
   KWitness3.fsti for the specification and the bit-equivalence rationale.

   The public entry point is `matmul_f32`. Everything else (the segment sum
   `seg`, the tiled sum `tiled`, their math lemmas, the partial-matrix invariant,
   and the single-thread kernel) is private implementation detail. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Sum { sum, sum_pop_left, sum_pop_right, sum_split }
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT

noextract
let tile : pos = 16

noextract
let min_nat (a b : nat) : nat = if a <= b then a else b

#push-options "--ifuel 1"
let min_nat_ge (a b : nat)
  : Lemma (requires b <= a) (ensures min_nat a b == b)
  = ()
#pop-options

(* -------------------------------------------------------------------------- *)
(* Phase A: pure specification of the tiled dot product and the math proof     *)
(* that, over the reals, it equals the (order-independent) matmul cell.        *)
(* -------------------------------------------------------------------------- *)

(* Forward, left-associated sum of the products a[i,t]*b[t,j] for t in
   [from, to), accumulating from `zero`. This models one tile's inner loop
   (and, composed, a whole row of K). *)
let rec seg
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat) (to : nat{to <= shared})
  : GTot et (decreases (if to <= from then 0 else to - from))
  = if from >= to then zero
    else add (seg eA eB i j from (to - 1)) (mul (macc eA i (to - 1)) (macc eB (to - 1) j))

#push-options "--fuel 2 --ifuel 1"
let seg_step
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat) (to : nat{from < to /\ to <= shared})
  : Lemma (seg eA eB i j from to ==
           add (seg eA eB i j from (to - 1)) (mul (macc eA i (to - 1)) (macc eB (to - 1) j)))
  = ()
#pop-options

(* The (fixed-domain) product function whose sum is the dot product. *)
let prodf
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : (natlt shared -> GTot real)
  = fun t -> mul (macc rA i t) (macc rB t j)

(* Over the reals, a segment equals the canonical sum of products over it. *)
#push-options "--fuel 2 --ifuel 1"
let rec seg_is_sum
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat) (to : nat{from <= to /\ to <= shared})
  : Lemma (ensures seg rA rB i j from to == sum from to (prodf rA rB i j))
          (decreases (to - from))
  = if from < to then begin
      seg_is_sum rA rB i j from (to - 1);
      sum_pop_right from to (prodf rA rB i j)
    end else ()
#pop-options

(* Over the reals, the (forward) matmul cell equals the canonical sum over the
   full range [0, shared). Forward induction, using the library's recurrence. *)
#push-options "--fuel 2 --ifuel 1"
let rec matmul_is_sum
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
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

(* The tiled sum: process tiles 0, 1, ..., nt-1, each tile t covering indices
   [t*tile, (t+1)*tile) clamped to [0, shared). Within a tile we sum forward
   (seg); across tiles we sum forward. Clamping with min_nat makes this total
   for all nt (extra tiles past the data are empty segments contributing 0). *)
let rec tiled
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (nt : nat)
  : GTot et (decreases nt)
  = if nt = 0 then zero
    else add (tiled eA eB i j (nt - 1))
             (seg eA eB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared))

#push-options "--fuel 2 --ifuel 1"
let tiled_step
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (nt : nat{nt > 0})
  : Lemma (tiled eA eB i j nt ==
           add (tiled eA eB i j (nt - 1))
               (seg eA eB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared)))
  = ()

let tiled_zero
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (tiled eA eB i j 0 == zero)
  = ()

(* Same fact stated for a size_t whose value is 0, so the matcher gets the
   equality with the exact term `v z` (= v (uint_to_t 0)) appearing at the
   loop entry, with no unfolding/congruence required. *)
let tiled_zero_v
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (z : sz{v z == 0})
  : Lemma (tiled eA eB i j (v z) == zero)
  = ()
#pop-options

#push-options "--fuel 2 --ifuel 1"
let seg_zero_v
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (i : natlt rows) (j : natlt cols)
  (z : sz{v z <= shared})
  : Lemma (seg eA eB i j (v z) (v z) == zero)
  = ()
#pop-options

(* Number of tiles to cover [0, shared): ceil(shared / tile). *)
noextract
let ntiles (s : nat) : nat = (s + tile - 1) / tile

let ntiles_covers (s : nat)
  : Lemma (s <= ntiles s * tile /\ (s > 0 ==> (ntiles s - 1) * tile < s))
  = Math.Lemmas.euclidean_division_definition (s + tile - 1) tile

(* Over the reals, the tiled sum over the first nt tiles equals the canonical
   sum over [0, min(nt*tile, shared)). Induction on nt, splitting the sum at the
   tile boundary. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let rec tiled_is_sum
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (nt : nat)
  : Lemma (ensures tiled rA rB i j nt == sum 0 (min_nat (nt * tile) shared) (prodf rA rB i j))
          (decreases nt)
  = if nt > 0 then begin
      tiled_is_sum rA rB i j (nt - 1);
      seg_is_sum rA rB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared);
      sum_split 0 (min_nat (nt * tile) shared) (prodf rA rB i j)
               (min_nat ((nt - 1) * tile) shared)
    end else ()
#pop-options

(* Hence, over reals, the full tiled sum equals the matmul cell. *)
let tiled_eq_matmul_single
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (tiled rA rB i j (ntiles shared) == MS.matmul_single rA rB i j)
  = tiled_is_sum rA rB i j (ntiles shared);
    ntiles_covers shared;
    min_nat_ge (ntiles shared * tile) shared;
    matmul_is_sum rA rB i j shared

(* The f32 segment approximates the real segment (add/mul congruence). *)
#push-options "--fuel 2 --ifuel 1"
let rec seg_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (rA : ematrix real rows shared) (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (from : nat) (to : nat{from <= to /\ to <= shared})
  : Lemma
      (requires
        (forall (a : natlt rows) (b : natlt shared). macc eA a b %~ macc rA a b) /\
        (forall (a : natlt shared) (b : natlt cols). macc eB a b %~ macc rB a b))
      (ensures seg eA eB i j from to %~ seg rA rB i j from to)
      (decreases (to - from))
  = if from < to then begin
      seg_approx eA eB rA rB i j from (to - 1);
      seg_step eA eB i j from to;
      seg_step rA rB i j from to;
      Kuiper.Approximates.Base.a_mul
        (macc eA i (to - 1)) (macc eB (to - 1) j) (macc rA i (to - 1)) (macc rB (to - 1) j);
      Kuiper.Approximates.Base.a_add
        (seg eA eB i j from (to - 1)) (mul (macc eA i (to - 1)) (macc eB (to - 1) j))
        (seg rA rB i j from (to - 1)) (mul (macc rA i (to - 1)) (macc rB (to - 1) j))
    end else ()
#pop-options

(* The f32 tiled sum approximates the real tiled sum. *)
#push-options "--fuel 2 --ifuel 1"
let rec tiled_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (rA : ematrix real rows shared) (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (nt : nat)
  : Lemma
      (requires
        (forall (a : natlt rows) (b : natlt shared). macc eA a b %~ macc rA a b) /\
        (forall (a : natlt shared) (b : natlt cols). macc eB a b %~ macc rB a b))
      (ensures tiled eA eB i j nt %~ tiled rA rB i j nt)
      (decreases nt)
  = if nt > 0 then begin
      tiled_approx eA eB rA rB i j (nt - 1);
      seg_approx eA eB rA rB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared);
      tiled_step eA eB i j nt;
      tiled_step rA rB i j nt;
      Kuiper.Approximates.Base.a_add
        (tiled eA eB i j (nt - 1))
        (seg eA eB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared))
        (tiled rA rB i j (nt - 1))
        (seg rA rB i j (min_nat ((nt - 1) * tile) shared) (min_nat (nt * tile) shared))
    end else ()
#pop-options

(* Per-cell: the f32 tiled sum approximates the real matmul cell. *)
let cell_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (rA : ematrix real rows shared) (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures tiled eA eB i j (ntiles shared) %~ MS.matmul_single rA rB i j)
  = tiled_approx eA eB rA rB i j (ntiles shared);
    tiled_eq_matmul_single rA rB i j

(* The full result matrix produced by the tiled kernel, and its approximation. *)
let tiled_result
  (#et:Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  : ematrix et rows cols
  = mkM (fun i j -> tiled eA eB i j (ntiles shared))

let tiled_result_approx
  (#et:Type) {| scalar et, real_like et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared) (eB : ematrix et shared cols)
  (rA : ematrix real rows shared) (rB : ematrix real shared cols)
  : Lemma (requires eA %~ rA /\ eB %~ rB)
          (ensures tiled_result eA eB %~ MS.matmul rA rB)
  = introduce forall (i : natlt rows) (j : natlt cols).
        macc (tiled_result eA eB) i j %~ macc (MS.matmul rA rB) i j
    with (
      cell_approx eA eB rA rB i j
    )

(* -------------------------------------------------------------------------- *)
(* Phase B: the partially-filled matrix invariant for the outer cell loop.     *)
(* `partial eA eB eC g` is the matrix where the first g cells (in flat          *)
(* row-major order) hold their tiled result and the rest hold eC.              *)
(* -------------------------------------------------------------------------- *)

let partial
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (g : nat)
  : ematrix et rows cols
  = mkM (fun i j -> if i * cols + j < g then tiled eA eB i j (ntiles shared) else macc eC i j)

let partial_zero
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  : Lemma (partial eA eB eC 0 == eC)
  = introduce forall (i : natlt rows) (j : natlt cols).
        macc (partial eA eB eC 0) i j == macc eC i j
    with ();
    assert (equal (partial eA eB eC 0) eC)

let partial_full
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  : Lemma (partial eA eB eC (rows * cols) == tiled_result eA eB)
  = introduce forall (i : natlt rows) (j : natlt cols).
        macc (partial eA eB eC (rows * cols)) i j == macc (tiled_result eA eB) i j
    with (
      Math.Lemmas.lemma_mult_le_right cols i (rows - 1)
    );
    assert (equal (partial eA eB eC (rows * cols)) (tiled_result eA eB))

#push-options "--z3rlimit 40"
let partial_step
  (#et : Type) {| scalar et |}
  (#rows #shared #cols : nat)
  (eA : ematrix et rows shared)
  (eB : ematrix et shared cols)
  (eC : ematrix et rows cols)
  (g : nat{g < rows * cols})
  : Lemma
      (requires cols > 0 /\ g / cols < rows /\ g % cols < cols)
      (ensures
        mupd (partial eA eB eC g) (g / cols) (g % cols)
             (tiled eA eB (g / cols) (g % cols) (ntiles shared))
        == partial eA eB eC (g + 1))
  = let r : natlt rows = g / cols in
    let c : natlt cols = g % cols in
    Math.Lemmas.euclidean_division_definition g cols;
    introduce forall (i : natlt rows) (j : natlt cols).
        macc (mupd (partial eA eB eC g) r c (tiled eA eB r c (ntiles shared))) i j
        == macc (partial eA eB eC (g + 1)) i j
    with (
      if i = r && j = c then ()
      else begin
        Math.Lemmas.euclidean_division_definition g cols;
        assert (i * cols + j <> g)
      end
    );
    assert (equal (mupd (partial eA eB eC g) r c (tiled eA eB r c (ntiles shared)))
                  (partial eA eB eC (g + 1)))
#pop-options

(* -------------------------------------------------------------------------- *)
(* Phase C: the single-block, single-thread kernel computing the whole matmul, *)
(* accumulating each cell in tiles of width `tile`, forward within and across. *)
(* -------------------------------------------------------------------------- *)

(* g / n < m, given g < m * n. *)
let div_bound (g m n : nat)
  : Lemma (requires n > 0 /\ g < m * n) (ensures g / n < m)
  = Math.Lemmas.euclidean_division_definition g n;
    if g / n >= m then Math.Lemmas.lemma_mult_le_right n m (g / n)

(* cit_fits is a (non-unfold) let, so expose it from the index bounds. *)
let cit_fits_intro (rows cols : nat) (a b : sz)
  : Lemma (requires v a < rows /\ v b < cols)
          (ensures M.cit_fits rows cols (a, b))
  = ()

(* v (sdivup k 16sz) == ntiles (v k): both are (v k + 15) / 16. *)
let nt_eq (k : szp)
  : Lemma (v (SZ.sdivup k 16sz) == ntiles (v k))
  = ()

(* For a tile index t below the tile count, the tile base t*tile is < shared. *)
let tile_base_lt (k : szp) (t0 : sz)
  : Lemma (requires v t0 < ntiles (v k))
          (ensures v t0 * tile < v k)
  = ntiles_covers (v k);
    Math.Lemmas.lemma_mult_le_right tile (v t0) (ntiles (v k) - 1)

#push-options "--fuel 4 --ifuel 2 --z3rlimit 80"
inline_for_extraction noextract
fn tiled_cell
  (#et : Type0) {| scalar et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (row : szlt m)
  (col : szlt n)
  (#eA : ematrix et _ _)
  (#eB : ematrix et _ _)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns res : et
  ensures
    pure (res == tiled eA eB (v row) (v col) (ntiles (v k)))
{
  let ri : sz = row +^ 0sz; assert (rewrites_to ri (row +^ 0sz));
  let ci : sz = col +^ 0sz; assert (rewrites_to ci (col +^ 0sz));
  let nt = SZ.sdivup k 16sz;
  nt_eq k;
  let mut t : szle nt = 0sz;
  let mut sum : et = zero;
  tiled_zero_v eA eB (v row) (v col) 0sz;
  while (!t <^ nt)
    invariant live t
    invariant live sum
    invariant gpu
    invariant gA |-> Frac fA eA
    invariant gB |-> Frac fB eB
    invariant pure (!sum == tiled eA eB (v row) (v col) (v !t))
    decreases (v nt - v !t)
  {
    let t0 = !t;
    tile_base_lt k t0;
    let k0 : sz = t0 *^ 16sz; assert (rewrites_to k0 (t0 *^ 16sz));
    let hi : sz = (if (k -^ k0) <^ 16sz then k else k0 +^ 16sz);
    assert (pure (v k0 == v t0 * tile));
    assert (pure (v hi == min_nat ((v t0 + 1) * tile) (v k)));
    assert (pure (v k0 == min_nat (v t0 * tile) (v k)));
    assert (pure (v k0 <= v hi /\ v hi <= v k));

    let mut acc : et = zero;
    let mut jj : szle hi = k0;
    seg_zero_v eA eB (v row) (v col) k0;
    while (!jj <^ hi)
      invariant live jj
      invariant live acc
      invariant gpu
      invariant gA |-> Frac fA eA
      invariant gB |-> Frac fB eB
      invariant pure (v k0 <= v !jj /\ v hi <= v k)
      invariant pure (!acc == seg eA eB (v row) (v col) (v k0) (v !jj))
      decreases (v hi - v !jj)
    {
      let jc : sz = !jj +^ 0sz; assert (rewrites_to jc (!jj +^ 0sz));
      cit_fits_intro (v m) (v k) ri jc;
      cit_fits_intro (v k) (v n) jc ci;
      let av = M.read gA (ri, jc);
      let bv = M.read gB (jc, ci);
      seg_step eA eB (v row) (v col) (v k0) (v jc + 1);
      acc := !acc `add` (av `mul` bv);
      jj := !jj +^ 1sz;
    };

    tiled_step eA eB (v row) (v col) (v t0 + 1);
    sum := !sum `add` !acc;
    t := !t +^ 1sz;
  };
  !sum
}
#pop-options

inline_for_extraction noextract
fn tiled_matmul_kf
  (#et : Type0) {| scalar et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (gC : M.array2 et lC)
  (#eA : ematrix et _ _)
  (#eB : ematrix et _ _)
  (#eC : ematrix et _ _)
  (#fA #fB : perm)
  (#_ : squash (SZ.fits (v m * v n)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    gC |-> eC
  ensures
    gC |-> tiled_result eA eB
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
    let value = tiled_cell m n k gA gB row col;
    cit_fits_intro (v m) (v n) row col;
    M.write gC (row, col) value;
    partial_step eA eB eC (v g0);
    g := !g +^ 1sz;
  };
  assert (pure (v (m *^ n) == v m * v n));
  partial_full eA eB eC;
  rewrite (gC |-> partial eA eB eC (v !g)) as (gC |-> tiled_result eA eB);
}

(* -------------------------------------------------------------------------- *)
(* Phase D: host wrapper launching the single-thread kernel, and the concrete  *)
(* monomorphic f32 row-major entry point (the imp3.cu witness).                *)
(* -------------------------------------------------------------------------- *)

inline_for_extraction noextract
fn tiled_matmul_gpu
  (#et : Type0) {| scalar et, real_like et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
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
    pure (SZ.fits (v m * v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix et (v m) (v n)).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
{
  M.pts_to_ref_located gA;
  M.pts_to_ref_located gB;
  M.pts_to_ref_located gC;

  launch_kernel_1 (fun _ -> tiled_matmul_kf m n k gA gB gC #eA #eB #eC #fA #fB);

  tiled_result_approx #et #_ #_ #(v m) #(v k) #(v n) eA eB rA rB;
  ()
}

(* imp3.cu is f32, row-major. This is the bit-equivalent witness. *)
fn matmul_f32
  (m n k : szp)
  (gA : M.array2 f32 (Alg.l2_row_major m k) { M.is_global gA })
  (gB : M.array2 f32 (Alg.l2_row_major k n) { M.is_global gB })
  (gC : M.array2 f32 (Alg.l2_row_major m n) { M.is_global gC })
  (rA : ematrix real m k)
  (rB : ematrix real k n)
  (#eA : ematrix f32 m k)
  (#eB : ematrix f32 k n)
  (#eC : ematrix f32 m n)
  (#fA #fB : perm)
  preserves
    cpu ** on gpu_loc (gA |-> Frac fA eA ** gB |-> Frac fB eB)
  requires
    pure (SZ.fits (SZ.v m * SZ.v n)) **
    pure (eA %~ rA /\ eB %~ rB) **
    on gpu_loc (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 m n).
      on gpu_loc (gC |-> eC') **
      pure (eC' %~ MS.matmul rA rB))
{
  tiled_matmul_gpu m n k gA gB gC rA rB;
}
