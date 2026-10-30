module Kuiper.Sparse.Approx

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Sparse.Common
open Kuiper.Sparse.Matrix
open Kuiper.Sparse.Matrix.PtsTo
open Kuiper.Array.Vectorized

(* CSR models have real-valued stored entries and exact structural zeros.
   The concrete entries only approximate the stored entries of the model. *)
unfold
let smatrix_approx'
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (elems : lseq et m.nnz)
  (col_ind : lseq sz m.nnz)
  (row_off : lseq sz (rows + 1))
  (relems : lseq real m.nnz)
  (rA : chest2 real rows cols)
  : slprop
  = m.elems |-> Frac f elems **
    m.col_ind |-> Frac f col_ind **
    m.row_off |-> Frac f row_off **
    pure (valid_smatrix rows cols (cast_pos col_ind) (cast_pos row_off) /\
          elems %~ relems /\
          rA == matrix_unsparse rows cols relems (cast_pos col_ind) (cast_pos row_off))

(* Preserve the existential model as a named predicate until it is unpacked.
   Eagerly unfolding its pure relation loses the real-valued witness. *)
let smatrix_model
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (m : smatrix et rows cols) (#f : perm)
  (elems : lseq et m.nnz) (col_ind : lseq sz m.nnz)
  (row_off : lseq sz (rows + 1)) (relems : lseq real m.nnz)
  (eA : chest2 et rows cols)
  (rA : chest2 real rows cols) : slprop
  = smatrix_approx' m #f elems col_ind row_off relems rA **
    pure (valid_smatrix rows cols (cast_pos col_ind) (cast_pos row_off) /\
          eA == matrix_unsparse rows cols elems (cast_pos col_ind) (cast_pos row_off))

(* The actual input [eA] and explicit real model [rA] share the same CSR
   structure. In particular, every unstored position is exactly real zero in
   [rA]; stored values may approximate nonzero reals even when they are zero.
   This strengthens the ordinary exact input ownership without changing it. *)
let smatrix_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (eA : chest2 et rows cols)
  (rA : chest2 real rows cols)
  : slprop
  = exists* elems col_ind row_off relems.
      smatrix_model m #f elems col_ind row_off relems eA rA

ghost
fn smatrix_approx_forget
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (m : smatrix et rows cols) (#f : perm)
  (eA : chest2 et rows cols) (rA : chest2 real rows cols)
  requires smatrix_approx m #f eA rA
  ensures m |-> Frac f eA
{
  unfold smatrix_approx m #f eA rA;
  with elems col_ind row_off relems.
    unfold smatrix_model m #f elems col_ind row_off relems eA rA;
  fold smatrix_pts_to m #f eA;
}

(* Explicitly expose the structural-zero side condition of a CSR real model. *)
let csr_model_zero
  (#nnz rows cols : nat)
  (relems : lseq real nnz)
  (col_ind : lseq nat nnz)
  (row_off : lseq nat (rows + 1))
  (rA : chest2 real rows cols)
  (i : natlt rows) (j : natlt cols)
  : Lemma
    (requires valid_smatrix rows cols col_ind row_off /\
      rA == matrix_unsparse rows cols relems col_ind row_off /\
      not (mem_slice j col_ind (row_off @! i) (row_off @! (i + 1))))
    (ensures acc2 rA i j == 0.0R)
  = ()

let tile_approximates
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : nat)
  (rM : chest2 real rows cols)
  (row : natlt rows)
  (col : nat)
  (#n : nat)
  (values : lseq et n)
  (step : nat)
  : prop
  = forall (i : natlt n).
      let c = col + i / chunk et * step * chunk et + i % chunk et in
      c < cols ==> (values @! i) %~ acc2 rM row c

unfold
let cell_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : natlt cols)
  (r : real)
  : slprop
  = exists* (value : et).
      Cell m (idx2 row col) |-> value ** pure (value %~ r)

unfold
let cell_approx_in_bounds
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat)
  (rM : chest2 real rows cols)
  : slprop
  = when__ (col < cols) (fun _ -> cell_approx m row col (acc2 rM row col))

unfold
let vec_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : pos { chunk et /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { chunk et /? col })
  (rM : chest2 real rows cols)
  : slprop
  = forall+ (lane : natlt (chunk et)).
      cell_approx_in_bounds m row (col + lane) rM

unfold
let thread_vec_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : pos { chunk et /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { chunk et /? col })
  (rM : chest2 real rows cols) (nthr k : nat)
  : slprop
  = vec_approx m row (offset_chunk et col k nthr) rM

let thread_tile_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : pos { chunk et /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { chunk et /? col })
  (rM : chest2 real rows cols) (row_tile nthr : nat)
  : slprop
  = forall+ (k : natlt (row_tile / chunk et)).
      thread_vec_approx m row col rM nthr k

let tile_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat)
  (rM : chest2 real rows cols) (row_tile : nat)
  : slprop
  = forall+ (k : natlt row_tile).
      cell_approx_in_bounds m row (col + k) rM

let matrix_approx
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (m : array2 et l) (rM : chest2 real rows cols)
  : slprop
  = exists* (actual : chest2 et rows cols).
      m |-> actual ** pure (actual %~ rM)

ghost
fn smatrix_share_approx_n'
  (#et:Type0) {| d : scalar et, real_like et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (relems : lseq real m.nnz)
  (em : chest2 real rows cols)
  (k : pos)
  requires smatrix_approx' m #f v_elems v_col_ind v_row_off relems em
  ensures forall+ (_ : natlt k).
    smatrix_approx' m #(f /. k) v_elems v_col_ind v_row_off relems em
{
  Kuiper.Array.Extra.array_share m.elems k;
  Kuiper.Array.Extra.array_share m.col_ind k;
  Kuiper.Array.Extra.array_share m.row_off k;

  forevery_zip
    (fun _ -> pts_to m.col_ind #(f /. k) v_col_ind)
    (fun _ -> pts_to m.row_off #(f /. k) v_row_off);
  forevery_zip
    (fun _ -> pts_to m.elems #(f /.k) v_elems) _;

  forevery_map #(natlt k)
    (fun _ ->
      pts_to m.elems   #(f /. k) v_elems **
      pts_to m.col_ind #(f /. k) v_col_ind **
      pts_to m.row_off #(f /. k) v_row_off
    )
    (fun _ -> smatrix_approx' m #(f /. k) v_elems v_col_ind v_row_off relems em)
    fn _ {};
}


let forall_natlt_elim (n : pos) (p : prop)
: Lemma (requires forall (_ : natlt n). p) (ensures p)
= eliminate forall (_ : natlt n). p with 0

ghost
fn forevery_natlt_elim
  (n : pos) (p : prop)
  requires forall+ (_ : natlt n). pure p
  ensures pure p
{
  forevery_extract_pure #(natlt n)
    (fun _ -> pure p) (fun _ -> p) fn _ {};

  forall_natlt_elim n p;

  forevery_map #(natlt n) (fun _ -> pure p) (fun _ -> emp) fn _ {};
  forevery_emp_elim _;

}

ghost
fn smatrix_gather_approx_n'
  (#et:Type0) {| d : scalar et, real_like et |}
  #rows #cols
  (m : smatrix et rows cols)
  (#[full_default ()] f : perm)
  (v_elems   : lseq et m.nnz)
  (v_col_ind : lseq sz m.nnz)
  (v_row_off : lseq sz (rows + 1))
  (relems : lseq real m.nnz)
  (em : chest2 real rows cols)
  (k : pos)
  requires forall+ (_ : natlt k).
    smatrix_approx' m #(f /. k) v_elems v_col_ind v_row_off relems em
  ensures smatrix_approx' m #f v_elems v_col_ind v_row_off relems em
{
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;
  forevery_unzip #(natlt k) _ _;

  Kuiper.Array.Extra.array_gather m.elems   k;
  Kuiper.Array.Extra.array_gather m.col_ind k;
  Kuiper.Array.Extra.array_gather m.row_off k;

  forevery_natlt_elim k _;

  ();
}


let divides_mul_div (d : pos) (a : nat) (b : nat)
  : Lemma (requires d /? a) (ensures a * b / d == a / d * b)
  = Kuiper.Divides.lemma_divides_exact d a;
    FStar.Math.Lemmas.paren_mul_right d (a / d) b;
    FStar.Math.Lemmas.swap_mul d (a / d * b);
    FStar.Math.Lemmas.cancel_mul_div (a / d * b) d

let thread_offset_chunk
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (j : nat { chunk et /? j }) (nthr : pos) (x : nat)
  : Lemma (offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr ==
           j + x * chunk et)
  = FStar.Math.Lemmas.distributivity_add_left (x % nthr) (x / nthr * nthr) (chunk et);
    FStar.Math.Lemmas.euclidean_division_definition x nthr

#push-options "--z3rlimit 25"
ghost
fn thread_tile_approx_gather
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
  (em : chest2 real rows cols)
  (nthr : nat)
  (trow_tile : nat { chunk et /? trow_tile })
  requires forall+ (tid : natlt nthr).
    thread_tile_approx gm i (thread_offset et j tid) em trow_tile nthr
  ensures tile_approx gm i j em (trow_tile * nthr)
{
  forevery_commute _;
  divides_mul_div (chunk et) trow_tile nthr;
  forevery_unfactor' (trow_tile * nthr / chunk et)
    (trow_tile / chunk et) nthr
    (fun k tid -> thread_vec_approx gm i (thread_offset et j tid) em nthr k);

  forevery_map #(natlt (trow_tile * nthr / chunk et))
    (fun x ->
      thread_vec_approx gm i (thread_offset et j (x % nthr)) em nthr (x / nthr)
    )
    (fun x ->
      forall+ (y : natlt (chunk et)).
        cell_approx_in_bounds gm i (j + (x * chunk et + y)) em
    )
    fn x {
      let lem : squash (
        offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr ==
        j + x * chunk et
      ) = thread_offset_chunk #et j nthr x;
      rewrite each offset_chunk et (thread_offset et j (x % nthr)) (x / nthr) nthr
      as (j + x * chunk et);

      unfold vec_approx gm i (j + x * chunk et) em;

      forevery_map #(natlt (chunk et))
        (fun lane -> cell_approx_in_bounds gm i ((j + x * chunk et) + lane) em)
        (fun lane -> cell_approx_in_bounds gm i (j + (x * chunk et + lane)) em)
        fn lane {
          rewrite each ((j + x * chunk et) + lane) as (j + (x * chunk et + lane));
        };
    };

  lemma_divides_product_l (chunk et) trow_tile nthr;
  forevery_unfactor (trow_tile * nthr)
    (trow_tile * nthr / chunk et) (chunk et)
    (fun k ->
      cell_approx_in_bounds gm i (j + k) em
    );
  fold tile_approx gm i j em (trow_tile * nthr);
}

#pop-options

#push-options "--z3rlimit 25"
ghost
fn gather_approx_row
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat) (#l : layout2 rows cols)
  (gm : array2 et l) (i : natlt rows)
  (rM : chest2 real rows cols) (row_tile : pos)
  requires forall+ (b : natlt (cols `Kuiper.Sparse.Math.divup` row_tile)).
    tile_approx gm i (b * row_tile) rM row_tile
  ensures forall+ (j : natlt cols).
    cell_approx gm i j (acc2 rM i j)
{
  forevery_map #(natlt (cols `Kuiper.Sparse.Math.divup` row_tile))
    (fun b -> tile_approx gm i (b * row_tile) rM row_tile)
    (fun b -> forall+ (k : natlt row_tile { b * row_tile + k < cols }).
      cell_approx gm i (b * row_tile + k) (acc2 rM i (b * row_tile + k)))
    fn b {
      unfold tile_approx gm;
      forevery_refine_pred' #(natlt row_tile)
        (fun k -> b * row_tile + k < cols) _;
    };
  forevery_unfactor_ cols row_tile
    (fun j -> cell_approx gm i j (acc2 rM i j));
}

ghost
fn gather_approx_matrix
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : pos) (#l : layout2 rows cols)
  (gm : array2 et l) (rM : chest2 real rows cols) (row_tile : pos)
  requires pure (fits (tlayout_ulen l))
  requires forall+ (r : natlt rows) (b : natlt (cols `Kuiper.Sparse.Math.divup` row_tile)).
    tile_approx gm r (b * row_tile) rM row_tile
  ensures matrix_approx gm rM
{
  forevery_map #(natlt rows)
    (fun r -> forall+ (b : natlt (cols `Kuiper.Sparse.Math.divup` row_tile)).
      tile_approx gm r (b * row_tile) rM row_tile)
    (fun r -> forall+ (j : natlt cols). cell_approx gm r j (acc2 rM r j))
    fn r { gather_approx_row gm r rM row_tile; };
  let values = forevery_exists_2
    (fun r c (v : et) -> Cell gm (idx2 r c) |-> v ** pure (v %~ acc2 rM r c));
  forevery_extract_pure_2
    (fun r c -> Cell gm (idx2 r c) |-> values r c ** pure (values r c %~ acc2 rM r c))
    (fun r c -> values r c %~ acc2 rM r c)
    fn r c {};
  forevery_map_2
    (fun r c -> Cell gm (idx2 r c) |-> values r c ** pure (values r c %~ acc2 rM r c))
    (fun r c -> Cell gm (idx2 r c) |-> acc2 (mk2 values) r c)
    fn r c {};
  tensor_iraise2 gm #_ #(mk2 values);
  lemma_approximates_intro (mk2 values) rM;
  fold matrix_approx gm rM;
}

ghost
fn fold_thread_tile_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (#rows #cols : pos { chunk et /? cols }) (#l : layout2 rows cols)
  (gm : array2 et l) (i : natlt rows) (j : nat { chunk et /? j })
  (#n : nat { chunk et /? n }) (s : lseq et n)
  (rM : chest2 real rows cols) (nthr : nat)
  requires thread_pts_to_tile_vec_underspec gm i j s nthr
  requires pure (tile_approximates rM i j s nthr)
  ensures thread_tile_approx gm i j rM n nthr
{
  ghost
  fn collect_chunk (k : natlt (n / chunk et))
    requires thread_pts_to_vec_underspec gm i j s nthr k
    ensures thread_vec_approx gm i j rM nthr k
  {
      let off = offset_chunk et j k nthr;
      rewrite each (offset_chunk et j k nthr) as off;
      if (off < cols) {
        Kuiper.Sparse.Math.divides_leq (chunk et) cols off;
        unfold_matrix_pts_to_vec_in_bounds gm i off (seq_chunk s (k * chunk et));
        unfold matrix_pts_to_vec gm i off (seq_chunk s (k * chunk et));
        ghost fn collect_lane (lane : natlt (chunk et))
          requires Cell gm (idx2 i (off + lane <: natlt cols)) |->
            (seq_chunk s (k * chunk et) @! lane)
          ensures cell_approx_in_bounds gm i (off + lane) rM
        {
          Kuiper.Math.div_mod_of_mul_add (chunk et) k lane;
          assert pure ((s @! (k * chunk et + lane)) %~ acc2 rM i (off + lane));
          fold cell_approx gm i (off + lane) (acc2 rM i (off + lane));
          when__intro_true (off + lane < cols)
            (cell_approx gm i (off + lane) (acc2 rM i (off + lane)));
        };
        forevery_map #(natlt (chunk et))
          (fun lane -> Cell gm (idx2 i (off + lane <: natlt cols)) |->
            (seq_chunk s (k * chunk et) @! lane))
          (fun lane -> cell_approx_in_bounds gm i (off + lane) rM)
          collect_lane;
        rewrite each off as (offset_chunk et j k nthr);
      } else {
        unfold_matrix_pts_to_vec_not_in_bounds gm i off (seq_chunk s (k * chunk et));
        forevery_intro_fill #(natlt (chunk et))
          (fun lane -> cell_approx_in_bounds gm i (off + lane) rM)
          fn lane {
            when__intro_false (off + lane < cols)
              (fun _ -> cell_approx gm i (off + lane) (acc2 rM i (off + lane)));
          };
        rewrite each off as (offset_chunk et j k nthr);
      }
  };
  unfold thread_pts_to_tile_vec_underspec gm;
  forevery_map #(natlt (n / chunk et))
    (fun k -> thread_pts_to_vec_underspec gm i j s nthr k)
    (fun k -> thread_vec_approx gm i j rM nthr k)
    collect_chunk;
  fold thread_tile_approx gm i j rM n nthr;
}
#pop-options
