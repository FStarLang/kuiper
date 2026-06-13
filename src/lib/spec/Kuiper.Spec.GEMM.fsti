module Kuiper.Spec.GEMM

(* NOTE: this is for an "exact" matmul at the mathematical level. It does not
provide any weak approximate spec. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling

inline_for_extraction noextract
let comb2 (#et:Type) (x y : et) : et = y

inline_for_extraction noextract
let lincomb
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (x y : et)
  (* x is the old value, y is the new computed value *)
  : et
  = add (mul beta x) (mul alpha y)

let rlincomb
  (alpha beta : real)
  (x y : real)
  (* x is the old value, y is the new computed value *)
  : real
  = beta *. x +. alpha *. y

val lincomb_approx2
  (#et:Type) {| scalar et, real_like et |}
  (alpha beta : et) (alpha_r beta_r : real)
  : Lemma (requires alpha %~ alpha_r /\ beta %~ beta_r)
          (ensures approx2 (lincomb alpha beta) (rlincomb alpha_r beta_r))
          [SMTPat (approx2 (lincomb alpha beta) (rlincomb alpha_r beta_r))]

(* These functions defined a matmul over potentially
different types, which is useful to state a matmul
over a big matrix being a matmul over individual tiles.

In that case, we are multiplying something like
  ematrix (ematrix et tm tk) (rows/tm) (shared/tk)
with
  ematrix (ematrix et tk tn) (shared/tk) (cols/tn)
to get
  ematrix (ematrix et tm tn) (rows/tm) (cols/tn

Notably, the inner elements (ematrix et tm tk) and (ematrix et tk tn)
are not scalars. We therefore require a function to multiply them
into some other type (ematrix et tm tn), and a function to add
two such elements. *)

// computes
// sum_{i=0}{to} m1[row][i] * m2[i][col]
// when to=shared, it computes the (row,col) cell of m1*m2
// the sum  is associated to the left, i.e.
// ((zero + m1[row][0] * m2[0][col]) + m1[row][1] * m2[1][col]) + ...
val __gmatmul_single
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : ematrix t1 rows shared)
  (m2 : ematrix t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot t3

val __gmatmul_single_congr
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : ematrix t1 rows shared)
  (m2 : ematrix t2 shared columns)
  (#rows' #columns' : nat)
  (m1' : ematrix t1 rows' shared)
  (m2' : ematrix t2 shared columns')
  (row : nat{row < rows})
  (col : nat{col < columns})
  (row' : nat{row' < rows'})
  (col' : nat{col' < columns'})
  (to : nat{to <= shared})
  : Lemma (requires (forall k. 0 <= k /\ k < to ==>
                        macc m1 row k == macc m1' row' k /\
                        macc m2 k col == macc m2' k col'))
          (ensures (__gmatmul_single z mul add m1 m2 row col to
                    == __gmatmul_single z mul add m1' m2' row' col' to))

val __gmatmul_single_zero_lemma
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #cols #shared: nat)
  (m1 : ematrix t1 rows shared)
  (m2 : ematrix t2 shared cols)
  (i : natlt rows)
  (j : natlt cols)
: Lemma
  (ensures z == (__gmatmul_single z mul add m1 m2 i j 0))
  [SMTPat (__gmatmul_single z mul add m1 m2 i j 0)]

val __gmatmul_single_lemma
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : ematrix t1 rows shared)
  (m2 : ematrix t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : pos{to <= shared})
: Lemma
  (ensures
    __gmatmul_single z mul add m1 m2 row col to ==
    add
      (__gmatmul_single z mul add m1 m2 row col (to - 1))
      (mul (macc m1 row (to - 1))
           (macc m2 (to - 1) col)))

let gmatmul_single
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : ematrix t1 rows shared)
  (m2 : ematrix t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot t3
  = __gmatmul_single z mul add m1 m2 row col shared

(* For scalars, we specialize the above functions, using
the canonical multiplication and addition. *)

let __matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot et
  = __gmatmul_single zero mul add m1 m2 row col to

let matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = __matmul_single m1 m2 row col shared

let __matmul_up_to
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (to : nat{to <= shared})
: ematrix et rows columns
= mkM fun i j -> __matmul_single m1 m2 i j to

let gemm_single
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (m0 : ematrix et rows columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = comb
      (macc m0 row col)
      (matmul_single m1 m2 row col)

val matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  [SMTPat (__matmul_single m1 m2 row col 0)]

val matmul_single_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat)
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    __matmul_single m1 m2 row col to ==
    add
      (__matmul_single m1 m2 row col (to - 1))
      (mul (macc m1 row (to-1)) (macc m2 (to-1) col))
  ))
  // [SMTPat (matmul_single m1 m2 row col to)]

val matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns

let matplus
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
  : ematrix et rows columns
  = Kuiper.Chest.chest_comb add m1 m2

val lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j)
        [SMTPat (matmul_single m1 m2 i j)]

val __matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  (to : nat{to <= shared / tk})
  : GTot (ematrix et tm tn)

let matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  : GTot (ematrix et tm tn)
  = __matmul_single_tile tm tn tk m1 m2 trow tcol (shared/tk)

val matmul_single_tile_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
: Lemma
  (ensures (
    __matmul_single_tile tm tn tk m1 m2 trow tcol 0 == const_matrix zero
  ))

val matmul_single_tile_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  (to : nat{to <= shared / tk})
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    __matmul_single_tile tm tn tk m1 m2 trow tcol to ==
    matplus
      (__matmul_single_tile tm tn tk m1 m2 trow tcol (to-1))
      (matmul (ematrix_subtile m1 tm tk trow (to-1))
              (ematrix_subtile m2 tk tn (to-1) tcol)))
  )

let mmcomb
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= matrix_comb comb m0 (matmul m1 m2)

val matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]

let gemm
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= matrix_comb (lincomb alpha beta) m0 (matmul m1 m2)

(* If we take a full-width slice of A and a full-height slice of B, then
   the matmul of those slices is equal to the corresponding tile of the
   full matmul. *)
val matmul_decompose_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : pos)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trows : nat {trows /? rows})
  (tcolumns : nat {tcolumns /? columns})
  (i : natlt (rows / trows))
  (j : natlt (columns / tcolumns))
: Lemma
  (ensures
    matmul
      (ematrix_subtile m1 trows shared i 0)
      (ematrix_subtile m2 shared tcolumns 0 j)
    ==
    ematrix_subtile
      (matmul m1 m2)
      trows tcolumns
      i j)
  [SMTPat (
    matmul
      (ematrix_subtile m1 trows shared i 0)
      (ematrix_subtile m2 shared tcolumns 0 j))]

(* Tiling full-width and full-height slices of A and B and computing a
   matmul and add on the tiles is equal to computing a matmul and add on
   the slices withou tiling. *)
val matmul_tiles_lemma
  (#et : Type) {| scalar et |}
  (pf2 : (x:et -> squash (add x zero == x /\ add zero x == x)))  // zero is additive identity
  (pf3 : (x:et -> y:et -> z:et -> squash (add x (add y z) == add (add x y) z)))  // add is associative
  (#rows #columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (z : ematrix et trows tcols)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{i < rows/trows})
  (j : nat{j < columns/tcols})
: Lemma
  (ensures
    gmatmul_single z matmul matplus
      (ematrix_tiled m1 trows tshared)
      (ematrix_tiled m2 tshared tcols)
      i j
    ==
    matplus z (
      matmul
        (ematrix_subtile m1 trows shared i 0)
        (ematrix_subtile m2 shared tcols 0 j)
    ))
