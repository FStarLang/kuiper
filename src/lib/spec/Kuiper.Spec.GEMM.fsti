module Kuiper.Spec.GEMM

(* NOTE: this is for an "exact" matmul at the mathematical level. It does not
provide any weak approximate spec. *)

open Kuiper
open Kuiper.Chest
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Float.Casts

inline_for_extraction noextract
let comb2 (#et:Type) (x y : et) : et = y

// Out of place version of comb2 including casts
let comb2_to (#et_acc #et_cd : Type0)
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (x : et_cd) (y : et_acc)
  : et_cd
=
  let x_acc : et_acc = fcast x in
  fcast (comb2 x_acc y)

inline_for_extraction noextract
let lincomb
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (x y : et)
  (* x is the old value, y is the new computed value *)
  : et
  = add (mul beta x) (mul alpha y)

// Out of place version of lincomb, including casts
inline_for_extraction noextract
let lincomb_to
  (#et_acc #et_cd : Type0)
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (alpha beta : et_acc)
  (x : et_cd)
  (y : et_acc)
  : et_cd
=
  let x_acc : et_acc = fcast x in
  fcast (lincomb alpha beta x_acc y)

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

val lincomb_to_approx2
  (#et_acc #et_cd : Type0)
  {| scalar et_acc, real_like et_acc |}
  {| scalar et_cd, real_like et_cd |}
  {| float_cast et_cd et_acc, float_cast et_acc et_cd |}
  (alpha beta : et_acc)
  : Lemma (ensures
      approx2
        (lincomb_to #et_acc #et_cd alpha beta)
        (rlincomb (to_real alpha) (to_real beta)))

(* These functions defined a matmul over potentially
different types, which is useful to state a matmul
over a big matrix being a matmul over individual tiles.

In that case, we are multiplying something like
  chest2 (chest2 et tm tk) (rows/tm) (shared/tk)
with
  chest2 (chest2 et tk tn) (shared/tk) (cols/tn)
to get
  chest2 (chest2 et tm tn) (rows/tm) (cols/tn

Notably, the inner elements (chest2 et tm tk) and (chest2 et tk tn)
are not scalars. We therefore require a function to multiply them
into some other type (chest2 et tm tn), and a function to add
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
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared columns)
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

val __gmatmul_single_zero_lemma
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
  [SMTPat (__gmatmul_single z mul add m1 m2 i j 0)]

val __gmatmul_single_lemma
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

let gmatmul_single
  (#t1 #t2 #t3 : Type)
  (z : t3)
  (mul : t1 -> t2 -> t3)
  (add : t3 -> t3 -> t3)
  (#rows #shared #columns : nat)
  (m1 : chest2 t1 rows shared)
  (m2 : chest2 t2 shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot t3
  = __gmatmul_single z mul add m1 m2 row col shared

(* For scalars, we specialize the above functions, using
the canonical multiplication and addition. *)

let __matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  (to : nat{to <= shared})
  : GTot et
  = __gmatmul_single zero mul add m1 m2 row col to

let matmul_single
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = __matmul_single m1 m2 row col shared

let __matmul_up_to
  (#et : Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (to : nat{to <= shared})
: chest2 et rows columns
= mk2 fun i j -> __matmul_single m1 m2 i j to

let gemm_single
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (m0 : chest2 et rows columns)
  (row : nat{row < rows})
  (col : nat{col < columns})
  : GTot et
  = comb
      (acc2 m0 row col)
      (matmul_single m1 m2 row col)

val matmul_zero_lemma
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (row col : nat{row < rows /\ col < columns})
: Lemma
  (ensures (
    __matmul_single m1 m2 row col 0 == zero
  ))
  [SMTPat (__matmul_single m1 m2 row col 0)]

val matmul_single_lemma
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
  [SMTPat (__matmul_single m1 m2 row col to)]

[@@erasable] // avoid silly warning
val matmul
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
: chest2 et rows columns

let matplus
  (#et:Type) {| scalar et |}
  (#rows #columns : nat)
  (m1 m2 : chest2 et rows columns)
  : chest2 et rows columns
  = Kuiper.Chest.chest_comb add m1 m2

val lemma_matmul_index
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (i : nat{ i < rows })
  (j : nat{ j < columns })
: Lemma (acc2 (matmul m1 m2) i j == matmul_single m1 m2 i j)
        [SMTPat (matmul_single m1 m2 i j)]

val __matmul_single_tile
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
  : GTot (chest2 et tm tn)

let matmul_single_tile
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (tm : pos{tm /? rows})
  (tn : pos{tn /? columns})
  (tk : pos{tk /? shared})
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (trow : natlt (rows / tm))
  (tcol : natlt (columns / tn))
  : GTot (chest2 et tm tn)
  = __matmul_single_tile tm tn tk m1 m2 trow tcol (shared/tk)

val matmul_single_tile_zero_lemma
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

val matmul_single_tile_lemma
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

let mmcomb
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m0 : chest2 et rows columns)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  : chest2 et rows columns
  = chest_comb comb m0 (matmul m1 m2)

val matmul_is_gemm
  (#et:Type) {| scalar et |}
  (#rows #shared #columns : nat)
  (m0 : chest2 et rows columns)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  : Lemma (mmcomb comb2 m0 m1 m2 == matmul m1 m2)
          [SMTPat (mmcomb comb2 m0 m1 m2)]

let gemm
  (#et:Type) {| scalar et |}
  (alpha beta : et)
  (#rows #shared #columns : nat)
  (m0 : chest2 et rows columns)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  : chest2 et rows columns
  = chest_comb (lincomb alpha beta) m0 (matmul m1 m2)

(* If we take a full-width slice of A and a full-height slice of B, then
   the matmul of those slices is equal to the corresponding tile of the
   full matmul. *)
val matmul_decompose_lemma
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
  (z : chest2 et trows tcols)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
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

(* Per-page batched gemm spec. *)
let bmmcomb
  (#et:Type) {| scalar et |}
  (comb : binop et)
  (#batch #rows #shared #cols : nat)
  (c : chest3 et batch rows cols)
  (a : chest3 et batch rows shared)
  (b : chest3 et batch shared cols)
  : chest3 et batch rows cols
  = mk3 fun i j k ->
      acc2 (mmcomb comb (slice_page c i) (slice_page a i) (slice_page b i)) j k

(* Per-page batched matmul spec. *)
let batched_matmul
  (#et:Type) {| scalar et |}
  (#batch #rows #shared #cols : nat)
  (a : chest3 et batch rows shared)
  (b : chest3 et batch shared cols)
  : chest3 et batch rows cols
  = mk3 fun i j k ->
      acc2 (matmul (slice_page a i)
                   (slice_page b i)) j k

val bmatmul_is_bgemm
  (#et:Type) {| scalar et |}
  (#batch #rows #shared #columns : nat)
  (m0 : chest3 et batch rows columns)
  (m1 : chest3 et batch rows shared)
  (m2 : chest3 et batch shared columns)
  : Lemma (bmmcomb comb2 m0 m1 m2 == batched_matmul m1 m2)
          [SMTPat (bmmcomb comb2 m0 m1 m2)]

(* ===== General (multi-type, fused-map) GEMM spec =====

   These generalize the scalar spec above to four decoupled types:
     ta   : element type of input A
     tb   : element type of input B
     tc   : element type of output C
     tacc : accumulation type (with a [scalar] instance)
   plus fused elementwise pre-maps [mapA : ta -> tacc], [mapB : tb -> tacc]
   and a combine operation [comb : tc -> tacc -> tc].

   Each of the general functions reduces to the corresponding scalar
   function when [ta = tb = tc = tacc = et] and [mapA = mapB = (fun x -> x)];
   this is stated by the [*_id] lemmas below (with SMT patterns), which the
   kernel wrappers use to line up their pre/post separation-logic assertions. *)

(* Single output cell: comb (old C cell) (dot product in tacc of the mapped
   inputs).  The fused elementwise maps [mapA]/[mapB] are applied to the inputs
   with [chest_map], reducing the accumulation to the ordinary scalar
   [matmul_single] at the accumulation type [tacc]. *)
let ggemm_single
  (#ta #tb #tc #tacc : Type) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#rows #shared #columns : nat)
  (m1 : chest2 ta rows shared)
  (m2 : chest2 tb shared columns)
  (m0 : chest2 tc rows columns)
  (row : nat{row < rows}) (col : nat{col < columns})
  : GTot tc
  = comb (acc2 m0 row col)
         (matmul_single (chest_map mapA m1) (chest_map mapB m2) row col)

(* Rank-2 output. *)
let gmmcomb
  (#ta #tb #tc #tacc : Type) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#rows #shared #columns : nat)
  (m0 : chest2 tc rows columns)
  (m1 : chest2 ta rows shared)
  (m2 : chest2 tb shared columns)
  : chest2 tc rows columns
  = mk2 fun i j -> ggemm_single mapA mapB comb m1 m2 m0 i j

(* Rank-3 batched (per-page) output. *)
let gbmmcomb
  (#ta #tb #tc #tacc : Type) {| scalar tacc |}
  (mapA : ta -> tacc) (mapB : tb -> tacc)
  (comb : tc -> tacc -> tc)
  (#batch #rows #shared #cols : nat)
  (c : chest3 tc batch rows cols)
  (a : chest3 ta batch rows shared)
  (b : chest3 tb batch shared cols)
  : chest3 tc batch rows cols
  = mk3 fun i j k ->
      acc2 (gmmcomb mapA mapB comb (slice_page c i) (slice_page a i) (slice_page b i)) j k

(* Reduction to the scalar spec at the identity maps. *)
val ggemm_single_id
  (#et : Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  (m0 : chest2 et rows columns)
  (row : nat{row < rows}) (col : nat{col < columns})
  : Lemma (ggemm_single (fun (x:et) -> x) (fun (x:et) -> x) comb m1 m2 m0 row col
           == gemm_single comb m1 m2 m0 row col)
          [SMTPat (ggemm_single (fun (x:et) -> x) (fun (x:et) -> x) comb m1 m2 m0 row col)]

val gmmcomb_id
  (#et : Type) {| scalar et |}
  (comb : binop et)
  (#rows #shared #columns : nat)
  (m0 : chest2 et rows columns)
  (m1 : chest2 et rows shared)
  (m2 : chest2 et shared columns)
  : Lemma (gmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb m0 m1 m2
           == mmcomb comb m0 m1 m2)
          [SMTPat (gmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb m0 m1 m2)]

val gbmmcomb_id
  (#et : Type) {| scalar et |}
  (comb : binop et)
  (#batch #rows #shared #cols : nat)
  (c : chest3 et batch rows cols)
  (a : chest3 et batch rows shared)
  (b : chest3 et batch shared cols)
  : Lemma (gbmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb c a b
           == bmmcomb comb c a b)
          [SMTPat (gbmmcomb (fun (x:et) -> x) (fun (x:et) -> x) comb c a b)]
