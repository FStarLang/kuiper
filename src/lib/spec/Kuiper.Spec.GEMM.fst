module Kuiper.Spec.GEMM

module Chest = Kuiper.Chest
open Kuiper.Shape

let lincomb_approx2
  (#et:Type) {| scalar et, real_like et |}
  (alpha beta : et) (alpha_r beta_r : real)
  : Lemma (requires alpha %~ alpha_r /\ beta %~ beta_r)
          (ensures approx2 (lincomb alpha beta) (lincomb alpha_r beta_r))
          [SMTPat (approx2 (lincomb alpha beta) (lincomb alpha_r beta_r))]
  = ()

let rec __gmatmul_single
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot t3 (decreases to)
  =
  if reveal to = 0 then z
  else (
    add
      (__gmatmul_single z mul add m1 m2 row col (to - 1))
      (mul (acc2 m1 row (to - 1))
           (acc2 m2 (to - 1) col))
  )

let rec __gmatmul_single_congr
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared columns)
  (#rows' #columns' : nat)
  (m1' : chest2 t1 rows' shared)
  (m2' : chest2 t2 shared columns')
  (row : nat{row < rows})
  (col : nat{col < columns})
  (row' : nat{row' < rows'})
  (col' : nat{col' < columns'})
  (to : nat{to <= shared})
  : Lemma (requires (forall k. 0 <= k /\ k < to ==>
                        acc2 m1 row k == acc2 m1' row' k /\
                        acc2 m2 k col == acc2 m2' k col'))
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
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared cols)
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
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : pos{to <= shared})
: Lemma
  (ensures
    __gmatmul_single z mul add m1 m2 row col to ==
    add
      (__gmatmul_single z mul add m1 m2 row col (to - 1))
      (mul (acc2 m1 row (to - 1))
           (acc2 m2 (to - 1) col)))
= ()

let matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  = ()

let matmul_single_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat)
: Lemma
  (requires (0 < to /\ to <= shared))
  (ensures (
    __matmul_single m1 m2 row col to ==
    add
      (__matmul_single m1 m2 row col (to - 1))
      (mul (acc2 m1 row (to-1)) (acc2 m2 (to-1) col))
  ))
  = ()

let matmul_single_at
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (idx : nat{idx < rows * columns})
  : GTot et
=
  let row = idx / columns in
  let col = idx % columns in
  matmul_single m1 m2 row col

let matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
: chest2 et rows columns
= Kuiper.Chest.mk (rows @| columns @| INil) fun (i, (j, ())) -> matmul_single m1 m2 i j

let lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (acc2 (matmul m1 m2) i j == matmul_single m1 m2 i j)
= ()

let rec __matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  (to : nat{to <= shared / tk})
  : GTot (chest2 et tm tn) (decreases to)
  =
  if reveal to = 0 then const _ zero
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
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
: Lemma
  (ensures (
    __matmul_single_tile tm tn tk m1 m2 trow tcol 0 == const _ zero
  ))
  = ()

let matmul_single_tile_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
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
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (acc2 (matmul m1 m2) i j == matmul_single m1 m2 i j)
= ()

let matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : chest2 et rows columns)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]
  = assert equal (mmcomb comb2 m0 m1 m2) (matmul m1 m2)

(* If we take a full-width slice of A and a full-width slice of B, then
   the matmul of those slices is equal to the corresponding slice of the
   full matmul. *)
let __matmul_decompose_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : pos)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (trows : nat {trows /? rows})
  (tcolumns : nat {tcolumns /? columns})
  (i1 : natlt (rows / trows))
  (j1 : natlt (columns / tcolumns))
  (i2 : natlt trows)
  (j2 : natlt tcolumns)
  : Lemma
    (ensures
      acc2
        (matmul
          (ematrix_subtile m1 trows shared i1 0)
          (ematrix_subtile m2 shared tcolumns 0 j1))
        i2 j2
      ==
      acc2
        (ematrix_subtile
          (matmul m1 m2)
          trows tcolumns
          i1 j1)
        i2 j2)
  = calc (==) {
      acc2 (matmul (ematrix_subtile m1 trows shared i1 0)
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
      acc2 (matmul m1 m2)
           (i1 * trows + i2)
           (j1 * tcolumns + j2);
      == {}
      acc2 (ematrix_subtile (matmul m1 m2) trows tcolumns i1 j1)
           i2 j2;
  }

let matmul_decompose_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : pos)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
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

#push-options "--z3rlimit 40"
let rec __matmul_single_subtile_lemma'
  (#et : Type) {| scalar et |}
  (pf2 : (x: et -> squash (add x zero == x /\ add zero x == x)))
  (pf3 : (x: et -> y: et -> z: et -> squash (add x (add y z) == add (add x y) z)))
  (#rows : nat)
  (#columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{i < rows / trows})
  (j : nat{j < columns / tcols})
  (i' : natlt trows)
  (j' : natlt tcols)
  (to : natle (shared / tshared) { 0 < to })
  (k: natle tshared)
: Lemma
      (
          add (__matmul_single (ematrix_subtile m1 trows shared i 0)
              (ematrix_subtile m2 shared tcols 0 j)
              i'
              j'
              ((to - 1) * tshared))
          (__matmul_single
            (ematrix_subtile m1 trows tshared i (to - 1))
            (ematrix_subtile m2 tshared tcols (to - 1) j)
            i'
            j'
            k) ==
      __matmul_single (ematrix_subtile m1 trows shared i 0)
          (ematrix_subtile m2 shared tcols 0 j)
          i'
          j'
          ((to - 1) * tshared + k)
      )
= let m1' = (ematrix_subtile m1 trows tshared i (to - 1)) in
  let m2' = (ematrix_subtile m2 tshared tcols (to - 1) j) in
  let m10 = (ematrix_subtile m1 trows shared i 0) in
  let m20 = (ematrix_subtile m2 shared tcols 0 j) in
  let x = __matmul_single m10 m20 i' j' ((to - 1) * tshared) in
  if k = 0
  then begin
    matmul_zero_lemma m1' m2' i' j';
    pf2 x
  end
  else begin
    matmul_single_lemma m1' m2' i' j' k;
    matmul_single_lemma m10 m20 i' j' ((to - 1) * tshared + k);
    pf3 x (__matmul_single m1' m2' i' j' (k - 1)) (mul (acc2 m1' i' (k-1)) (acc2 m2' (k-1) j'));
    __matmul_single_subtile_lemma' pf2 pf3 trows tcols tshared m1 m2 i j i' j' to (k - 1)
  end

let __matmul_single_subtile_lemma
  (#et : Type) {| scalar et |}
  (pf2 : (x: et -> squash (add x zero == x /\ add zero x == x)))
  (pf3 : (x: et -> y: et -> z: et -> squash (add x (add y z) == add (add x y) z)))
  (#rows : nat)
  (#columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{i < rows / trows})
  (j : nat{j < columns / tcols})
  (i' : natlt trows)
  (j' : natlt tcols)
  (to : natle (shared / tshared) { 0 < to })
: Lemma
      (
          add (__matmul_single (ematrix_subtile m1 trows shared i 0)
              (ematrix_subtile m2 shared tcols 0 j)
              i'
              j'
              ((to - 1) * tshared))
          (acc2 (matmul (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
                  (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j))
              i'
              j') ==
      __matmul_single (ematrix_subtile m1 trows shared i 0)
          (ematrix_subtile m2 shared tcols 0 j)
          i'
          j'
          (to * tshared)
      )
= __matmul_single_subtile_lemma' pf2 pf3 trows tcols tshared m1 m2 i j i' j' to tshared
#pop-options

#push-options "--z3rlimit 40"
let rec __matmul_tiles_lemma
  (#et : Type) {| scalar et |}
  (pf2 : (x:et -> squash (add x zero == x /\ add zero x == x)))  // zero is additive identity
  (pf3 : (x:et -> y:et -> z:et -> squash (add x (add y z) == add (add x y) z)))  // add is associative
  (#rows #columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (acc : chest2 et trows tcols)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{i < rows/trows})
  (j : nat{j < columns/tcols})
  (i' : natlt trows)
  (j' : natlt tcols)
  (to : natle (shared / tshared))
  : Lemma (
      acc2 (__gmatmul_single acc matmul matplus
            (ematrix_tiled m1 trows tshared)
            (ematrix_tiled m2 tshared tcols)
            i j to)
           i' j'
      ==
      acc2 acc i' j'
      `add` __matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i' j' (to * tshared)
  )
  = if to = 0 then (
      calc (==) {
        acc2 (__gmatmul_single acc matmul matplus
              (ematrix_tiled m1 trows tshared)
              (ematrix_tiled m2 tshared tcols)
              i j 0)
             i' j';
        == {}
        acc2 acc i' j';
        == { pf2 (acc2 acc i' j') }
        acc2 acc i' j'
        `add` zero;
      }
    ) else (
      calc (==) {
        acc2 (__gmatmul_single acc matmul matplus
              (ematrix_tiled m1 trows tshared)
              (ematrix_tiled m2 tshared tcols)
              i j to)
             i' j';
        == { matmul_single_tile_lemma trows tcols tshared m1 m2 i j to }
        acc2 (matplus
               (__gmatmul_single acc matmul matplus
                 (ematrix_tiled m1 trows tshared)
                 (ematrix_tiled m2 tshared tcols)
                 i j (to - 1))
               (matmul
                 (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
                 (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j)))
             i' j';
        == { (* distr acc2 *) }
        acc2 (__gmatmul_single acc matmul matplus
              (ematrix_tiled m1 trows tshared)
              (ematrix_tiled m2 tshared tcols)
              i j (to - 1))
             i' j'
        `add`
        acc2 (matmul
               (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
               (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j))
             i' j';
        == { __matmul_tiles_lemma pf2 pf3 trows tcols tshared acc m1 m2 i j i' j' (to - 1) }
        (acc2 acc i' j'
         `add` __matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i' j' ((to - 1) * tshared))
        `add`
        acc2 (matmul
               (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
               (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j))
             i' j';
        == { pf3 (acc2 acc i' j')
                 (__matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i' j' ((to - 1) * tshared))
                 (acc2 (matmul
                         (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
                         (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j))
                       i' j') }
        acc2 acc i' j'
        `add`
         (__matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i' j' ((to - 1) * tshared)
          `add`
          acc2 (matmul
                (acc2 (ematrix_tiled m1 trows tshared) i (to - 1))
                (acc2 (ematrix_tiled m2 tshared tcols) (to - 1) j))
              i' j');
        == { __matmul_single_subtile_lemma pf2 pf3 trows tcols tshared m1 m2 i j i' j' to }
        acc2 acc i' j'
        `add` __matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i' j' (to * tshared);
      }
    )

let matmul_tiles_lemma
  (#et : Type) {| scalar et |}
  (pf2 : (x:et -> squash (add x zero == x /\ add zero x == x)))  // zero is additive identity
  (pf3 : (x:et -> y:et -> z:et -> squash (add x (add y z) == add (add x y) z)))  // add is associative
  (#rows #columns : nat)
  (#shared : pos)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? columns})
  (tshared : pos{tshared /? shared})
  (acc : chest2 et trows tcols)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
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
= let aux (i'j' : natlt trows & (natlt tcols & unit))
    : Lemma (
      ensures
        Chest.acc
          (gmatmul_single acc matmul matplus
            (ematrix_tiled m1 trows tshared)
            (ematrix_tiled m2 tshared tcols)
            i j)
          i'j'
        ==
        Chest.acc
          (matplus acc (
            matmul
              (ematrix_subtile m1 trows shared i 0)
              (ematrix_subtile m2 shared tcols 0 j)))
          i'j'
    )
  =
    calc (==) {
      Chest.acc
        (gmatmul_single acc matmul matplus
            (ematrix_tiled m1 trows tshared)
            (ematrix_tiled m2 tshared tcols)
            i j)
        i'j';
      == { __matmul_tiles_lemma pf2 pf3 trows tcols tshared acc m1 m2 i j i'j'._1 i'j'._2._1 (shared / tshared) }
      Chest.acc acc i'j'
      `add` matmul_single (ematrix_subtile m1 trows shared i 0) (ematrix_subtile m2 shared tcols 0 j) i'j'._1 i'j'._2._1;
      == { }
      Chest.acc (
        matplus acc (
            matmul
              (ematrix_subtile m1 trows shared i 0)
              (ematrix_subtile m2 shared tcols 0 j)))
           i'j';
    }
  in
  Classical.forall_intro aux;
  assert (
    gmatmul_single acc matmul matplus
      (ematrix_tiled m1 trows tshared)
      (ematrix_tiled m2 tshared tcols)
      i j
    `equal`
    matplus acc (
      matmul
        (ematrix_subtile m1 trows shared i 0)
        (ematrix_subtile m2 shared tcols 0 j)
    ))

let bmatmul_is_bgemm
  (#et:Type) {| scalar et |}
  (#batch #rows #shared #columns : nat)
  (m0 : chest3 et batch rows columns)
  (m1 : chest3 et batch rows shared)
  (m2 : chest3 et batch shared columns)
  : Lemma (bmmcomb comb2 m0 m1 m2 == batched_matmul m1 m2)
  = assert equal (bmmcomb comb2 m0 m1 m2) (batched_matmul m1 m2)
#pop-options
