module Kuiper.Poly.GEMM.Util

#lang-pulse

open Kuiper
open Kuiper.Approximates
open Pulse.Lib.Trade
module M  = Kuiper.Matrix
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module Tiling = Kuiper.Matrix.Tiling
open Kuiper.EMatrix
open Kuiper.Matrix.Reprs.Type
open Kuiper.Matrix.Tiling

inline_for_extraction noextract
fn matmul_dotprod
  (#et : Type0) {| scalar et |}
  (#rows #shared #cols : SZ.t)
  (#lA : mlayout rows shared)
  (#lB : mlayout shared cols)
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA : ematrix et rows shared)
  (#eB : ematrix et shared cols)
  (i : szlt rows)
  (j : szlt cols)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res == MS.matmul_single eA eB i j)
{
  let mut k : sz = 0sz;
  let mut sum : et = zero;

  while (SZ.(!k <^ shared))
    invariant
      exists* (vk : SZ.t{vk <= shared}).
        k |-> vk **
        sum |-> MS.__matmul_single eA eB i j vk
  {
    let v1 = M.gpu_matrix_read gA i !k;
    let v2 = M.gpu_matrix_read gB !k j;

    let vsum = !sum;
    sum := vsum `add` mul v1 v2;
    k := SZ.add !k 1sz;

    (**)MS.matmul_single_lemma eA eB i j !k;
    ();
  };
  !sum
}

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

inline_for_extraction noextract
fn matmul_tiled_dotprod'
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #shared #cols : sz)
  (#tile : szp)
  (#lA : mlayout (rows   * tile) (shared * tile))
  (#lB : mlayout (shared * tile) (cols   * tile))
  {| clayout lA, clayout lB |}
  (gA : M.gpu_matrix et lA)
  (gB : M.gpu_matrix et lB)
  (#eA #eB : ematrix _ _ _)
  (bi : szlt rows)
  (bj : szlt cols)
  (i : szlt tile)
  (j : szlt tile)
  (#fA #fB : perm)
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  returns
    res : et
  ensures
    pure (res %~ real_matmul_single eA eB (bi * tile + i) (bj * tile + j))
{
  let grow : erased (natlt (rows * tile)) = hide (bi * tile + i);
  let gcol : erased (natlt (cols * tile)) = hide (bj * tile + j);

  let mut sum : et = zero;
  let mut bk  : sz = 0sz;

  while (SZ.(!bk <^ shared))
    invariant
      exists* (vbk : SZ.t{vbk <= shared}) sumv.
        bk |-> vbk **
        sum |-> sumv **
        pure (v_approximates sumv (__real_matmul_single_tiled eA eB grow gcol (SZ.v vbk * tile)))
  {
    let vbk = !bk;
    assert (pure (bi  < (rows   * tile) / tile));
    assert (pure (vbk < (shared * tile) / tile));

    let tA = Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v bi) (SZ.v vbk);
    let tB = Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v bj);
    assert (rewrites_to tA (Tiling.gpu_matrix_subtile gA (SZ.v tile) (SZ.v tile) (SZ.v bi) (SZ.v vbk)));
    assert (rewrites_to tB (Tiling.gpu_matrix_subtile gB (SZ.v tile) (SZ.v tile) (SZ.v vbk) (SZ.v bj)));

    Tiling.gpu_matrix_extract_tile_ro gA tile tile bi vbk;
    Tiling.gpu_matrix_extract_tile_ro gB tile tile vbk bj;

    let s' = matmul_dotprod tA tB i j;
    (* s' == matmul_single (subtile_eA bi vbk) (subtile_eB vbk bj) i j *)

    (* Get the current sum value before mutation *)
    let s = !sum;

    sum := s `add` s';

    ambig_trade_elim ();
    ambig_trade_elim ();

    bk := !bk +^ 1sz;

    (* Use the lemmas to prove the invariant is maintained:
       1. s %~ __real_matmul_single_tiled ... (vbk * tile)  [from invariant]
       2. s' == matmul_single (subtile) i j  [from matmul_dotprod ensures]
       3. s' %~ real_matmul_single_subtile ...  [by matmul_single_subtile_approx]
       4. __real_matmul_single_tiled ... ((vbk+1)*tile)
            == __real_matmul_single_tiled ... (vbk*tile) +. real_matmul_single_subtile ...
          [by __real_matmul_single_tiled_step]
       5. s + s' %~ (__real_matmul_single_tiled ... (vbk*tile) +. real_matmul_single_subtile ...)
          [by a_add on 1 and 3]
       6. Combining 4 and 5: s + s' %~ __real_matmul_single_tiled ... ((vbk+1)*tile)
    *)
    matmul_single_subtile_approx eA eB bi bj vbk i j;
    __real_matmul_single_tiled_step eA eB bi bj (SZ.v vbk) i j;
    a_add s s'
      (__real_matmul_single_tiled eA eB grow gcol (SZ.v vbk * tile))
      (real_matmul_single_subtile eA eB bi bj (SZ.v vbk) i j);
    ()
  };

  !sum
}

(* Used by SHMEM, Blocktiling1D *)
inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 : mlayout tile tile) {| clayout l1 |}
  (#l2 : mlayout tile tile) {| clayout l2 |}
  (m1 : M.gpu_matrix et l1)
  (m2 : M.gpu_matrix et l2)
  (j : szlt tile)
  (#acc0 : erased (seq et))
  (#v1 #v2 : ematrix et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f v1 **
    m2 |-> Frac f v2
  requires
    pure (Seq.length acc0 == tile) **
    acc |-> acc0
  ensures
    exists* acc'.
      pure (Seq.length acc' == tile) **
      (acc |-> acc')
{
  pts_to_len acc;
  let mut sk : sz = 0sz;
  while (SZ.(!sk <^ tile))
    invariant live sk ** live acc
  {
    pts_to_len acc;
    let mut i = 0sz;
    (* We can read v2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let v2 = M.gpu_matrix_read m2 !sk j;
    while (SZ.(!i <^ tile))
      invariant live i ** live acc
    {
      let v1 = M.gpu_matrix_read m1 !i !sk;

      open Pulse.Lib.Array;
      pts_to_len acc;
      let sum0 = acc.(!i);
      let sum1 = sum0 `add` (v1 `mul` v2);
      acc.(!i) <- sum1;
      i := !i +^ 1sz;
    };
    pts_to_len acc;
    sk := !sk +^ 1sz;
  };
  pts_to_len acc;
}
