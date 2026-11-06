module Kuiper.Sparse.SPMM.Groups

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Sparse
open Kuiper.Sparse.Approx
open Kuiper.Array.Vectorized

// A working group is independent of the native B and C vector widths.
let group_offset (ch : pos) (j : nat { ch /? j }) (k nthr : nat)
  : Pure nat (requires true) (ensures divides ch)
  = lemma_divides_product ch (k * nthr);
    lemma_divides_sum ch j (k * nthr * ch);
    j + k * nthr * ch

let thread_group_offset (ch : pos) (j tid : nat)
  : Pure nat (requires ch /? j) (ensures divides ch)
  = Kuiper.Sparse.Math.lineal_divides ch j ch tid;
    j + tid * ch

unfold
let group_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (ch : pos)
  (#rows #cols : pos { ch /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { ch /? col })
  (rM : chest2 real rows cols)
  : slprop
  = forall+ (lane : natlt (ch)).
      cell_approx_in_bounds m row (col + lane) rM

unfold
let thread_group_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (ch : pos)
  (#rows #cols : pos { ch /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { ch /? col })
  (rM : chest2 real rows cols) (nthr k : nat)
  : slprop
  = group_approx ch m row (group_offset ch col k nthr) rM

let thread_groups_approx
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (ch : pos)
  (#rows #cols : pos { ch /? cols }) (#l : layout2 rows cols)
  (m : array2 et l) (row : natlt rows) (col : nat { ch /? col })
  (rM : chest2 real rows cols) (row_tile nthr : nat)
  : slprop
  = forall+ (k : natlt (row_tile / ch)).
      thread_group_approx ch m row col rM nthr k

let thread_group_offset_lemma
  (ch : pos)
  (j : nat { ch /? j }) (nthr : pos) (x : nat)
  : Lemma (group_offset ch (thread_group_offset ch j (x % nthr)) (x / nthr) nthr ==
           j + x * ch)
  = FStar.Math.Lemmas.distributivity_add_left (x % nthr) (x / nthr * nthr) (ch);
    FStar.Math.Lemmas.euclidean_division_definition x nthr

#push-options "--z3rlimit 20"
let share_groups_offset
  (ch : pos)
  (col : nat { ch /? col }) (k nthr tid : nat)
  : Lemma (col + (k * nthr + tid) * ch ==
      group_offset ch (thread_group_offset ch col tid) k nthr)
  = FStar.Math.Lemmas.distributivity_add_left (k * nthr) tid (ch)

let share_groups_lane_offset
  (ch : pos) (col : nat { ch /? col }) (k nthr tid lane : nat)
  : Lemma (col + ((k * nthr + tid) * ch + lane) ==
      group_offset ch (thread_group_offset ch col tid) k nthr + lane)
  = share_groups_offset ch col k nthr tid

(* Distribute a tile to threads while preserving its value approximation. *)
ghost
fn share_groups
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (ch : pos)
  (#rows #cols : pos { ch /? cols }) (#l : layout2 rows cols)
  (gm : array2 et l) (row : natlt rows)
  (col : natlt cols { ch /? col })
  (em : chest2 et rows cols) (rM : chest2 real rows cols { em %~ rM })
  (nthr : pos) (n : nat { ch /? n })
  requires pts_to_tile_in_matrix gm row col em (n * nthr)
  ensures forall+ (tid : natlt nthr).
    thread_groups_approx ch gm row (thread_group_offset ch col tid) rM n nthr
{
  unfold pts_to_tile_in_matrix gm;
  forevery_map #(natlt (n * nthr))
    (fun i -> matrix_pts_to_cell_in_matrix_in_bounds gm row (col + i) em)
    (fun i -> cell_approx_in_bounds gm row (col + i) rM)
    fn i {
      if (col + i < cols) {
        when__elim_true _ _;
        fold cell_approx gm row (col + i) (acc2 rM row (col + i));
        when__intro_true (col + i < cols)
          (cell_approx gm row (col + i) (acc2 rM row (col + i)));
      } else {
        when__elim_false _ _;
        when__intro_false (col + i < cols)
          (fun _ -> cell_approx gm row (col + i) (acc2 rM row (col + i)));
      }
    };
  lemma_divides_product_l (ch) n nthr;
  lemma_divides_exact (ch) n;
  divides_mul_div ch n nthr;
  lemma_divides_exact ch (n * nthr);
  FStar.Math.Lemmas.swap_mul ch (n / ch * nthr);
  forevery_factor (n * nthr) ((n / ch) * nthr) (ch) _;
  forevery_factor ((n / ch) * nthr) (n / ch) nthr _;
  forevery_map_2 #(natlt (n / ch)) #(natlt nthr)
    (fun k tid -> forall+ (lane : natlt (ch)).
      cell_approx_in_bounds gm row (col + ((k * nthr + tid) * ch + lane)) rM)
    (fun k tid -> thread_group_approx ch gm row (thread_group_offset ch col tid) rM nthr k)
    fn k tid {
      share_groups_offset ch col k nthr tid;
      forevery_map #(natlt (ch))
        (fun lane -> cell_approx_in_bounds gm row
          (col + ((k * nthr + tid) * ch + lane)) rM)
        (fun lane -> cell_approx_in_bounds gm row
          (group_offset ch (thread_group_offset ch col tid) k nthr + lane) rM)
        fn lane {
          let eq : squash (col + ((k * nthr + tid) * ch + lane) ==
            group_offset ch (thread_group_offset ch col tid) k nthr + lane) =
              share_groups_lane_offset ch col k nthr tid lane;
          rewrite each (col + ((k * nthr + tid) * ch + lane))
            as (group_offset ch (thread_group_offset ch col tid) k nthr + lane);
        };
      fold group_approx ch gm row (group_offset ch (thread_group_offset ch col tid) k nthr) rM;
    };
  forevery_commute _;
  forevery_map #(natlt nthr)
    (fun tid -> forall+ (k : natlt (n / ch)).
      thread_group_approx ch gm row (thread_group_offset ch col tid) rM nthr k)
    (fun tid -> thread_groups_approx ch gm row (thread_group_offset ch col tid) rM n nthr)
    fn tid { fold thread_groups_approx ch gm row (thread_group_offset ch col tid) rM n nthr; };
}

#pop-options

#push-options "--z3rlimit 25"
ghost
fn gather_groups
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et, real_like et |}
  (ch : pos)
  (#rows #cols : pos { ch /? cols })
  (#l : layout2 rows cols)
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { ch /? j })
  (em : chest2 real rows cols)
  (nthr : nat)
  (trow_tile : nat { ch /? trow_tile })
  requires forall+ (tid : natlt nthr).
    thread_groups_approx ch gm i (thread_group_offset ch j tid) em trow_tile nthr
  ensures tile_approx gm i j em (trow_tile * nthr)
{
  forevery_commute _;
  divides_mul_div (ch) trow_tile nthr;
  forevery_unfactor' (trow_tile * nthr / ch)
    (trow_tile / ch) nthr
    (fun k tid -> thread_group_approx ch gm i (thread_group_offset ch j tid) em nthr k);

  forevery_map #(natlt (trow_tile * nthr / ch))
    (fun x ->
      thread_group_approx ch gm i (thread_group_offset ch j (x % nthr)) em nthr (x / nthr)
    )
    (fun x ->
      forall+ (y : natlt (ch)).
        cell_approx_in_bounds gm i (j + (x * ch + y)) em
    )
    fn x {
      let lem : squash (
        group_offset ch (thread_group_offset ch j (x % nthr)) (x / nthr) nthr ==
        j + x * ch
      ) = thread_group_offset_lemma ch j nthr x;
      rewrite each group_offset ch (thread_group_offset ch j (x % nthr)) (x / nthr) nthr
      as (j + x * ch);

      unfold group_approx ch gm i (j + x * ch) em;

      forevery_map #(natlt (ch))
        (fun lane -> cell_approx_in_bounds gm i ((j + x * ch) + lane) em)
        (fun lane -> cell_approx_in_bounds gm i (j + (x * ch + lane)) em)
        fn lane {
          FStar.Math.Lemmas.paren_add_right j (x * ch) lane;
          rewrite each ((j + x * ch) + lane) as (j + (x * ch + lane));
        };
    };

  lemma_divides_product_l (ch) trow_tile nthr;
  forevery_unfactor (trow_tile * nthr)
    (trow_tile * nthr / ch) (ch)
    (fun k ->
      cell_approx_in_bounds gm i (j + k) em
    );
  fold tile_approx gm i j em (trow_tile * nthr);
}

#pop-options
