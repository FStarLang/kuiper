module Kuiper.Spec.GEMM

let rec __gmatmul_single
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
  : GTot t3 (decreases to)
  =
  if reveal to = 0 then z
  else (
    add
      (__gmatmul_single z mul add m1 m2 row col (to - 1))
      (mul (macc m1 row (to - 1))
           (macc m2 (to - 1) col))
  )

let rec __gmatmul_single_congr
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
  = if reveal to = 0 then ()
    else (
     __gmatmul_single_congr z mul add m1 m2 m1' m2' row col row' col' (to - 1);
     ()
    )

let __gmatmul_single_zero_lemma
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
= ()

let __gmatmul_single_lemma
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
= ()

let matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  = ()

let matmul_single_lemma
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
  = ()

let matmul_single_at
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (idx : nat{idx < rows * columns})
  : GTot et
=
  let row = idx / columns in
  let col = idx % columns in
  matmul_single m1 m2 row col

let matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
: ematrix et rows columns
= mkM <| fun i j -> matmul_single m1 m2 i j

let matplus
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
: ematrix et rows columns
= mkM <| fun i j -> add (macc m1 i j) (macc m2 i j)

let lemma_matplus_index
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : ematrix et rows columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matplus m1 m2) i j == macc m1 i j `add` macc m2 i j)
        [SMTPat (macc (matplus m1 m2) i j)]
= ()

let lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j)
= ()

let rec __matmul_single_tile
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
  : GTot (ematrix et tm tn) (decreases to)
  =
  if reveal to = 0 then const_matrix zero
  else (
    matplus
      (__matmul_single_tile tm tn tk m1 m2 trow tcol (to-1))
      (matmul (ematrix_subtile m1 tm tk trow (to-1))
              (ematrix_subtile m2 tk tn (to-1) tcol)))

let matmul_single_tile_zero_lemma
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
  = ()

let matmul_single_tile_lemma
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
  = ()

let lemma_matmul_tile_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (macc (matmul m1 m2) i j == matmul_single m1 m2 i j)
= ()

let matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : ematrix et rows columns)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]
  = ematrix_ext (mmcomb comb2 m0 m1 m2) (matmul m1 m2)

(* If we take a full-width slice of A and a full-width slice of B, then
   the matmul of those slices is equal to the corresponding slice of the
   full matmul. *)
let __matmul_decompose_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : pos)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (trows : nat {trows /? rows})
  (tcolumns : nat {tcolumns /? columns})
  (i1 : natlt (rows / trows))
  (j1 : natlt (columns / tcolumns))
  (i2 : natlt trows)
  (j2 : natlt tcolumns)
  : Lemma
    (ensures
      macc
        (matmul
          (ematrix_subtile m1 trows shared i1 0)
          (ematrix_subtile m2 shared tcolumns 0 j1))
        i2 j2
      ==
      macc
        (ematrix_subtile
          (matmul m1 m2)
          trows tcolumns
          i1 j1)
        i2 j2)
  = calc (==) {
      macc (matmul (ematrix_subtile m1 trows shared i1 0)
                      (ematrix_subtile m2 shared tcolumns 0 j1))
           i2 j2;
      == {}
      matmul_single (ematrix_subtile m1 trows shared i1 0)
                    (ematrix_subtile m2 shared tcolumns 0 j1)
                    i2 j2;
      == {}
      __matmul_single
        (ematrix_subtile m1 trows shared i1 0)
        (ematrix_subtile m2 shared tcolumns 0 j1)
        i2 j2
        shared;
      == { __gmatmul_single_congr
            zero mul add
            (ematrix_subtile m1 trows shared i1 0)
            (ematrix_subtile m2 shared tcolumns 0 j1)
            m1 m2
            i2 j2
            (i1 * trows + i2) (j1 * tcolumns + j2) shared }
      __matmul_single
        m1
        m2
        (i1 * trows + i2)
        (j1 * tcolumns + j2)
        shared;
      == {}
      macc (matmul m1 m2)
           (i1 * trows + i2)
           (j1 * tcolumns + j2);
      == {}
      macc (ematrix_subtile (matmul m1 m2) trows tcolumns i1 j1)
           i2 j2;
  }

let matmul_decompose_lemma
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
= Classical.forall_intro_2 (__matmul_decompose_lemma m1 m2 trows tcolumns i j);
  assert (
    matmul
      (ematrix_subtile m1 trows shared i 0)
      (ematrix_subtile m2 shared tcolumns 0 j)
    `equal`
    ematrix_subtile
      (matmul m1 m2)
      trows tcolumns
      i j)

let matmul_tiles_lemma
  (#et : Type) {| scalar et |}
  (#rows #columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (acc : ematrix et trows tcols)
  (m1 : ematrix et rows shared)
  (m2 : ematrix et shared columns)
  (i : nat{i < rows/trows})
  (j : nat{j < columns/tcols})
: Lemma
  (ensures
    gmatmul_single acc matmul matplus
      (ematrix_tiled m1 trows tshared)
      (ematrix_tiled m2 tshared tcols)
      i j
    ==
    matplus acc (
      matmul
        (ematrix_subtile m1 trows shared i 0)
        (ematrix_subtile m2 shared tcols 0 j)
    ))
= admit()