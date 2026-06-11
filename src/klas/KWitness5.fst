module KWitness5

(* Implementation of the imp5.cu witness: a TILED (tile = 16) matmul whose
   cross-tile reduction uses KAHAN compensation. See KWitness5.fsti for the spec.

   Per output cell, each 16-wide tile is summed forward (naive) into a partial,
   and those tile partials are combined with a Kahan-compensated sum (the library
   combinator Kuiper.Kahan.kahan_sum). So it is "imp3 (tiled) but Kahan across
   tiles". The single thread does the whole matmul (no permission splitting).

   Everything except `matmul_f32` is private implementation detail. *)

#lang-pulse
open Kuiper
open Kuiper.EMatrix
open Kuiper.Sum { sum, sum_pop_right, sum_split }
module Alg = Kuiper.Tensor.Layout.Alg
module MS = Kuiper.Spec.GEMM
module M = Kuiper.Array2
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Kahan = Kuiper.Kahan

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
(* Phase A: forward segment sum (per tile, naive) and the real tiling proof.   *)
(* -------------------------------------------------------------------------- *)

(* Forward, left-associated sum of products a[i,t]*b[t,j] for t in [from, to). *)
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

noextract
let prodf
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : (natlt shared -> GTot real)
  = fun t -> mul (macc rA i t) (macc rB t j)

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

(* Tile geometry: tile t covers [tlo t, thi t) = [t*16, (t+1)*16) clamped to
   [0, shared). Number of tiles is ceil(shared/16). *)
noextract let tlo (shared t : nat) : nat = min_nat (t * tile) shared
noextract let thi (shared t : nat) : nat = min_nat ((t + 1) * tile) shared
noextract let ntiles (s : nat) : nat = (s + tile - 1) / tile

let ntiles_covers (s : nat)
  : Lemma (s <= ntiles s * tile /\ (s > 0 ==> (ntiles s - 1) * tile < s))
  = Math.Lemmas.euclidean_division_definition (s + tile - 1) tile

(* The real-valued sum over one tile (the "Kahan term" spec). Total over nat. *)
noextract
let vf_tile
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : (t:nat -> GTot real)
  = fun t -> sum (tlo shared t) (thi shared t) (prodf rA rB i j)

(* Over the reals, the sum of the first nt tile-sums equals the canonical sum
   over [0, min(nt*tile, shared)). Induction + sum_split. *)
#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
let rec tile_partition
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  (nt : nat)
  : Lemma (ensures sum 0 nt (vf_tile rA rB i j)
                   == sum 0 (min_nat (nt * tile) shared) (prodf rA rB i j))
          (decreases nt)
  = if nt > 0 then begin
      tile_partition rA rB i j (nt - 1);
      sum_pop_right 0 nt (vf_tile rA rB i j);
      sum_split 0 (min_nat (nt * tile) shared) (prodf rA rB i j)
               (min_nat ((nt - 1) * tile) shared)
    end else ()
#pop-options

(* Hence the full Kahan-term sum equals the matmul cell, over reals. *)
let cell_real
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma (sum 0 (ntiles shared) (vf_tile rA rB i j) == MS.matmul_single rA rB i j)
  = tile_partition rA rB i j (ntiles shared);
    ntiles_covers shared;
    min_nat_ge (ntiles shared * tile) shared;
    matmul_is_sum rA rB i j shared

(* A value approximating the Kahan-term sum (over v nt == ntiles tiles) also
   approximates the matmul cell. Isolated so the chain is discharged in a small
   context rather than amid the kernel. *)
let value_approx_cell
  (#rows #shared #cols : nat)
  (rA : ematrix real rows shared)
  (rB : ematrix real shared cols)
  (nt_val : nat)
  (i : natlt rows) (j : natlt cols)
  (value : f32)
  : Lemma (requires value %~ sum 0 nt_val (vf_tile rA rB i j) /\ nt_val == ntiles shared)
          (ensures value %~ macc (MS.matmul rA rB) i j)
  = cell_real rA rB i j;
    MS.lemma_matmul_index rA rB i j

(* -------------------------------------------------------------------------- *)
(* Phase B: the cell-fill invariant. Unlike the naive/reverse/tiled witnesses, *)
(* the Kahan cell value is not an exact function -- it only approximates the    *)
(* matmul cell. So the invariant tracks "first g cells approximate matmul, the  *)
(* rest still hold eC", with the result matrix existentially quantified.        *)
(* -------------------------------------------------------------------------- *)

noextract
let approx_partial
  (#m #shared #n : nat)
  (rA : ematrix real m shared)
  (rB : ematrix real shared n)
  (eC eCp : ematrix f32 m n)
  (g : nat)
  : prop
  = forall (i : natlt m) (j : natlt n).
      (i * n + j < g ==> macc eCp i j %~ macc (MS.matmul rA rB) i j) /\
      (i * n + j >= g ==> macc eCp i j == macc eC i j)

let approx_partial_zero
  (#m #shared #n : nat)
  (rA : ematrix real m shared)
  (rB : ematrix real shared n)
  (eC : ematrix f32 m n)
  : Lemma (approx_partial rA rB eC eC 0)
  = ()

#push-options "--z3rlimit 40"
let approx_partial_step
  (#m #shared #n : nat)
  (rA : ematrix real m shared)
  (rB : ematrix real shared n)
  (eC eCp : ematrix f32 m n)
  (g : nat{g < m * n})
  (value : f32)
  : Lemma
      (requires
        n > 0 /\ g / n < m /\ g % n < n /\
        approx_partial rA rB eC eCp g /\
        value %~ macc (MS.matmul rA rB) (g / n) (g % n))
      (ensures approx_partial rA rB eC (mupd eCp (g / n) (g % n) value) (g + 1))
  = let r : natlt m = g / n in
    let c : natlt n = g % n in
    Math.Lemmas.euclidean_division_definition g n;
    introduce forall (i : natlt m) (j : natlt n).
        (i * n + j < g + 1 ==> macc (mupd eCp r c value) i j %~ macc (MS.matmul rA rB) i j) /\
        (i * n + j >= g + 1 ==> macc (mupd eCp r c value) i j == macc eC i j)
    with (
      if i = r && j = c then ()
      else Math.Lemmas.euclidean_division_definition g n
    )
#pop-options

#push-options "--z3rlimit 40"
let approx_partial_full
  (#m #shared #n : nat)
  (rA : ematrix real m shared)
  (rB : ematrix real shared n)
  (eC eCp : ematrix f32 m n)
  : Lemma (requires approx_partial rA rB eC eCp (m * n))
          (ensures eCp %~ MS.matmul rA rB)
  = introduce forall (i : natlt m) (j : natlt n).
        macc eCp i j %~ macc (MS.matmul rA rB) i j
    with (
      Math.Lemmas.lemma_mult_le_right n i (m - 1)
    )
#pop-options

(* -------------------------------------------------------------------------- *)
(* Phase C: the single-thread kernel. Per cell, each tile is summed forward    *)
(* (tile_dot) and the tile partials are combined with Kahan (Kuiper.Kahan).    *)
(* -------------------------------------------------------------------------- *)

let div_bound (g m n : nat)
  : Lemma (requires n > 0 /\ g < m * n) (ensures g / n < m)
  = Math.Lemmas.euclidean_division_definition g n;
    if g / n >= m then Math.Lemmas.lemma_mult_le_right n m (g / n)

let cit_fits_intro (rows cols : nat) (a b : sz)
  : Lemma (requires v a < rows /\ v b < cols)
          (ensures M.cit_fits rows cols (a, b))
  = ()

let nt_eq (k : szp)
  : Lemma (v (SZ.sdivup k 16sz) == ntiles (v k))
  = ()

let tile_base_lt (k : szp) (t0 : sz)
  : Lemma (requires v t0 < ntiles (v k))
          (ensures v t0 * tile < v k)
  = ntiles_covers (v k);
    Math.Lemmas.lemma_mult_le_right tile (v t0) (ntiles (v k) - 1)

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

(* One tile's forward (naive) dot product, returning a value that approximates
   the real tile-sum vf_tile. This is the per-term function fed to kahan_sum. *)
inline_for_extraction noextract
fn tile_dot
  (#et : Type0) {| scalar et, real_like et |}
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : M.array2 et lA)
  (gB : M.array2 et lB)
  (row : szlt m)
  (col : szlt n)
  (rA : ematrix real (v m) (v k))
  (rB : ematrix real (v k) (v n))
  (ti : szlt (SZ.sdivup k 16sz))
  (#eA : ematrix et (v m) (v k))
  (#eB : ematrix et (v k) (v n))
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns r : et
  ensures
    pure (r %~ vf_tile rA rB (v row) (v col) (v ti))
{
  nt_eq k;
  tile_base_lt k ti;
  let k0 : sz = ti *^ 16sz; assert (rewrites_to k0 (ti *^ 16sz));
  let hi : sz = (if (k -^ k0) <^ 16sz then k else k0 +^ 16sz);
  assert (pure (v k0 == v ti * tile));
  assert (pure (v hi == min_nat ((v ti + 1) * tile) (v k)));
  assert (pure (v k0 == min_nat (v ti * tile) (v k)));
  assert (pure (v k0 <= v hi /\ v hi <= v k));
  let ri : sz = row +^ 0sz; assert (rewrites_to ri (row +^ 0sz));
  let ci : sz = col +^ 0sz; assert (rewrites_to ci (col +^ 0sz));

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
  seg_approx eA eB rA rB (v row) (v col) (v k0) (v hi);
  seg_is_sum rA rB (v row) (v col) (v k0) (v hi);
  !acc
}

(* One cell: Kahan-combine the tile partials, then map the real-level identity
   (sum of tile-sums == matmul cell) to an approximation of the matmul cell.
   Isolated as its own fn so value_approx_cell is discharged in a minimal
   context (just the kahan_sum result), away from the kernel's loop invariants. *)
inline_for_extraction noextract
fn cell_value
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : M.array2 f32 lA)
  (gB : M.array2 f32 lB)
  (row : szlt m)
  (col : szlt n)
  (rA : ematrix real (v m) (v k))
  (rB : ematrix real (v k) (v n))
  (#eA : ematrix f32 (v m) (v k))
  (#eB : ematrix f32 (v k) (v n))
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB)
  returns value : f32
  ensures
    pure (value %~ macc (MS.matmul rA rB) (v row) (v col))
{
  let nt = SZ.sdivup k 16sz;
  nt_eq k;
  let value =
    Kahan.kahan_sum #f32 nt
      (gpu ** gA |-> Frac fA eA ** gB |-> Frac fB eB)
      (vf_tile rA rB (v row) (v col))
      fn (ti : szlt nt) {
        tile_dot m n k gA gB row col rA rB ti;
      };
  value_approx_cell #(v m) #(v k) #(v n) rA rB (v nt) (v row) (v col) value;
  value
}

#push-options "--z3rlimit 30 --fuel 2 --ifuel 1"
inline_for_extraction noextract
fn kahan_tiled_matmul_kf
  (m n k : szp)
  (#lA : M.layout m k)
  (#lB : M.layout k n)
  (#lC : M.layout m n)
  {| T.ctlayout lA, T.ctlayout lB, T.ctlayout lC |}
  (gA : M.array2 f32 lA)
  (gB : M.array2 f32 lB)
  (gC : M.array2 f32 lC)
  (rA : ematrix real (v m) (v k))
  (rB : ematrix real (v k) (v n))
  (#eA : ematrix f32 (v m) (v k))
  (#eB : ematrix f32 (v k) (v n))
  (#eC : ematrix f32 (v m) (v n))
  (#fA #fB : perm)
  (#_ : squash (SZ.fits (v m * v n)))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    pure (eA %~ rA /\ eB %~ rB) **
    (gC |-> eC)
  ensures
    (exists* (eC' : ematrix f32 (v m) (v n)).
      (gC |-> eC') ** pure (eC' %~ MS.matmul rA rB))
{
  approx_partial_zero rA rB eC;
  let mut g : szle (m *^ n) = 0sz;
  while (!g <^ m *^ n)
    invariant live g
    invariant gpu
    invariant gA |-> Frac fA eA
    invariant gB |-> Frac fB eB
    invariant
      (exists* (eCp : ematrix f32 (v m) (v n)).
        (gC |-> eCp) ** pure (approx_partial rA rB eC eCp (v !g)))
    decreases (v (m *^ n) - v !g)
  {
    with eCp. assert (gC |-> eCp ** pure (approx_partial rA rB eC eCp (v !g)));
    let g0 = !g;
    div_bound (v g0) (v m) (v n);
    let row : sz = g0 /^ n; assert (rewrites_to row (g0 /^ n));
    let col : sz = g0 %^ n; assert (rewrites_to col (g0 %^ n));

    let value = cell_value m n k gA gB row col rA rB;

    cit_fits_intro (v m) (v n) row col;
    M.write gC (row, col) value;
    approx_partial_step rA rB eC eCp (v g0) value;
    g := !g +^ 1sz;
  };
  with eCp. assert (gC |-> eCp ** pure (approx_partial rA rB eC eCp (v !g)));
  assert (pure (v (m *^ n) == v m * v n));
  approx_partial_full rA rB eC eCp;
}
#pop-options

(* -------------------------------------------------------------------------- *)
(* Phase D: host wrapper + concrete f32 row-major entry point.                 *)
(* -------------------------------------------------------------------------- *)

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
  M.pts_to_ref_located gA;
  M.pts_to_ref_located gB;
  M.pts_to_ref_located gC;
  launch_kernel_1 (fun _ -> kahan_tiled_matmul_kf m n k gA gB gC rA rB #eA #eB #eC #fA #fB);
  ()
}
