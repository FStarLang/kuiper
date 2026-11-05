module Kuiper.Sparse.SPMM.Epilogue

#lang-pulse
#set-options "--z3rlimit 20"

open Kuiper
open Kuiper.Tensor
open Kuiper.Sparse
open Kuiper.Sparse.SPMM.Groups
open Kuiper.Sparse.Approx
open Kuiper.Sparse.SPMM.Maps
open Kuiper.Array.Vectorized
open Kuiper.Array2.Strided
open Kuiper.Chest
open Kuiper.Sparse.Tensor

inline_for_extraction noextract
let chunk_offset
  (ch : szp)
  (col : sz { ch /? col })
  (k nthr : sz { fits (col + k * nthr * ch) })
  : Pure (o : sz { ch /? o })
      (requires true)
      (ensures fun o -> v o == group_offset ch col k nthr)
  = lemma_divides_product (ch) (v k * v nthr);
    lemma_divides_sum (ch) (v col) (v k * v nthr * ch);
    col +^ k *^ nthr *^ ch

let tile_chunk_bound (ch : pos) (n col : nat) (nthr : pos)
  (k : natlt (n / ch))
  : Lemma
    (requires ch /? n /\ fits (col + n * nthr))
    (ensures (k + 1) * ch <= n /\
      col + k * nthr * ch + ch <= col + n * nthr /\
      fits (col + k * nthr * ch) /\ fits (col + k * nthr * ch + ch))
  = lemma_divides_exact ch n;
    FStar.Math.Lemmas.lemma_mult_le_right ch (k + 1) (n / ch);
    FStar.Math.Lemmas.lemma_mult_le_right nthr ((k + 1) * ch) n;
    FStar.Math.Lemmas.lemma_mult_le_right ch 1 nthr;
    FStar.Math.Lemmas.distributivity_add_left k 1 ch;
    FStar.Math.Lemmas.distributivity_add_left (k * ch) ch nthr;
    FStar.Math.Lemmas.paren_mul_right k ch nthr;
    FStar.Math.Lemmas.paren_mul_right k nthr ch;
    FStar.Math.Lemmas.swap_mul ch nthr;
    assert (k * nthr * ch + nthr * ch == (k + 1) * ch * nthr);
    FStar.SizeT.fits_lte (col + k * nthr * ch) (col + n * nthr);
    FStar.SizeT.fits_lte (col + k * nthr * ch + ch) (col + n * nthr)

let chunk_values_approx
  (#tc : Type0) {| scalar tc, sized tc, has_vec_cpy tc, real_like tc |}
  (#rows #cols : pos)
  (rM : chest2 real rows cols) (row : natlt rows)
  (col : nat { col + chunk tc <= cols }) (values : lseq tc (chunk tc))
  : prop
  = forall (lane : natlt (chunk tc)).
      (values @! lane) %~ acc2 rM row (col + lane)

(* Convert the owned C cells into one contiguous slice, without reading them. *)
ghost
fn lower_chunk
  (#tc : Type0) {| scalar tc, sized tc, has_vec_cpy tc, real_like tc |}
  (#rows #cols : pos { chunk tc /? cols })
  (#lC : layout2 rows cols) {| strided : strided_row_major lC |}
  (gC : array2 tc lC) (row : natlt rows)
  (col : nat { chunk tc /? col /\ col + chunk tc <= cols })
  (rC : chest2 real rows cols)
  requires vec_approx gC row col rC
  ensures exists* (values : lseq tc (chunk tc)).
    pts_to_slice (core gC) (cell_of_pos lC row col)
      (cell_of_pos lC row col + chunk tc) values **
    pure (chunk_values_approx rC row col values)
{
  unfold vec_approx gC;
  strided.pf row col;
  forevery_map #(natlt (chunk tc))
    (fun lane -> cell_approx_in_bounds gC row (col + lane) rC)
    (fun lane -> exists* (value : tc).
      Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> value **
      pure (value %~ acc2 rC row (col + lane)))
    fn lane {
      when__elim_true _ _;
      with value. assert Kuiper.Tensor.pts_to_cell gC (idx2 #rows #cols row (col + lane)) value;
      pts_to_cell_eq gC (idx2 #rows #cols row (col + lane)) 1.0R value;
      strided.pf row (col + lane);
      rewrite Cell gC (idx2 #rows #cols row (col + lane)) |-> value
        as Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> value;
    };
  let values = forevery_exists #(natlt (chunk tc))
    (fun lane (value : tc) ->
      Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> value **
      pure (value %~ acc2 rC row (col + lane)));
  forevery_extract_pure #(natlt (chunk tc))
    (fun lane -> Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> values lane **
      pure (values lane %~ acc2 rC row (col + lane)))
    (fun lane -> values lane %~ acc2 rC row (col + lane))
    fn lane {};
  forevery_map #(natlt (chunk tc))
    (fun lane -> Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> values lane **
      pure (values lane %~ acc2 rC row (col + lane)))
    (fun lane -> Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |->
      (Seq.init_ghost (chunk tc) values @! lane))
    fn lane {};
  forevery_rw_size (chunk tc)
    ((cell_of_pos lC row col + chunk tc) - cell_of_pos lC row col);
  cells_to_nonempty_slice (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc) #(Seq.init_ghost (chunk tc) values);
}

ghost
fn raise_chunk
  (#tc : Type0) {| scalar tc, sized tc, has_vec_cpy tc, real_like tc |}
  (#rows #cols : pos { chunk tc /? cols })
  (#lC : layout2 rows cols) {| strided : strided_row_major lC |}
  (gC : array2 tc lC) (row : natlt rows)
  (col : nat { chunk tc /? col /\ col + chunk tc <= cols })
  (rC : chest2 real rows cols) (values : lseq tc (chunk tc))
  requires pts_to_slice (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc) values
  requires pure (chunk_values_approx rC row col values)
  ensures vec_approx gC row col rC
{
  strided.pf row col;
  slice_to_cells (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc);
  forevery_rw_size ((cell_of_pos lC row col + chunk tc) - cell_of_pos lC row col)
    (chunk tc);
  forevery_map #(natlt (chunk tc))
    (fun lane -> Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> (values @! lane))
    (fun lane -> cell_approx_in_bounds gC row (col + lane) rC)
    fn lane {
      pts_to_cell_eq gC (idx2 #rows #cols row (col + lane)) 1.0R (values @! lane);
      strided.pf row (col + lane);
      rewrite Cell (core gC <: array tc) (cell_of_pos lC row col + lane <: nat) |-> (values @! lane)
        as Cell gC (idx2 #rows #cols row (col + lane)) |-> (values @! lane);
      fold cell_approx gC row (col + lane) (acc2 rC row (col + lane));
      when__intro_true (col + lane < cols)
        (cell_approx gC row (col + lane) (acc2 rC row (col + lane)));
    };
  fold vec_approx gC row col rC;
}

inline_for_extraction noextract
fn combine_chunk
  (#tc #tacc : Type0)
  {| scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (#rows #cols : szp { chunk tc /? cols })
  (#lC : layout2 rows cols) {| ctlayout lC, strided : strided_row_major lC |}
  (gC : array2 tc lC)
  (row : szlt rows) (col : sz { chunk tc /? col /\ col + chunk tc <= cols })
  (rC rProduct : chest2 real rows cols)
  (#n : sz) (out : larray tacc n) (#values : lseq tacc n)
  (start : sz { start + chunk tc <= n })
  preserves gpu ** out |-> values
  requires vec_approx gC row col rC
  requires pure (aligned 16 (core gC))
  requires pure (aligned_strided_row_major (chunk tc) strided)
  requires pure (forall (lane : natlt (chunk tc)).
    (values @! (start + lane)) %~ acc2 rProduct row (col + lane))
  ensures vec_approx gC row col (chest_comb rcomb rC rProduct)
{
  lower_chunk gC row col rC;
  with old. assert pts_to_slice (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc) old;
  pts_to_slice_ref (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc);
  aligned_cell_strided_row_major gC row col;
  strided.pf row col;
  let address = strided.offset +^ strided.stride *^ row +^ col;
  let mut combined = [| zero #tc #_; chunk tc |];
  assume pure (aligned 16 combined);
  Pulse.Lib.Array.pts_to_len combined;
  array_vec_cpy combined 0sz (core gC) address;
  with initial. assert combined |-> initial;
  assert pure (Seq.equal initial old);
  let mut lane = 0sz;
  while (!lane <^ chunk tc)
    invariant exists* (vl : sz) (current : lseq tc (chunk tc)).
      lane |-> vl ** combined |-> current **
      pure (vl <= chunk tc /\
        (forall (i : natlt (chunk tc)). (current @! i) %~
          (if i < vl then rcomb (acc2 rC row (col + i)) (acc2 rProduct row (col + i))
           else acc2 rC row (col + i))))
    decreases (chunk tc - !lane)
  {
    open Pulse.Lib.Array;
    combined.(!lane) <- comb combined.(!lane) out.(start +^ !lane);
    lane := !lane +^ 1sz;
  };
  with result. assert combined |-> result;
  array_vec_cpy (core gC) address combined 0sz;
  with stored. assert pts_to_slice (core gC) (cell_of_pos lC row col)
    (cell_of_pos lC row col + chunk tc) stored;
  assert pure (Seq.equal stored result);
  raise_chunk gC row col (chest_comb rcomb rC rProduct) stored;
}

let accumulator_chunk
  (#tacc : Type0) {| scalar tacc, real_like tacc |}
  (ch : pos) (#rows #cols : pos)
  (rProduct : chest2 real rows cols) (row : natlt rows) (col : nat)
  (#n : nat) (values : lseq tacc n) (nthr : pos) (k : nat)
  : Lemma
    (requires (k + 1) * ch <= n /\ col + k * nthr * ch + ch <= cols /\
      tile_values_approx ch rProduct row col values nthr)
    (ensures forall (lane : natlt ch).
      (values @! (k * ch + lane)) %~ acc2 rProduct row (col + k * nthr * ch + lane))
  = introduce forall (lane : natlt ch).
      (values @! (k * ch + lane)) %~ acc2 rProduct row (col + k * nthr * ch + lane)
    with Kuiper.Math.div_mod_of_mul_add ch k lane

// Restrict the accumulator mapping to one native C vector, in a small
// pure context rather than asking the store loop to instantiate it implicitly.
let accumulator_subchunk
  (#tacc : Type0) {| scalar tacc, real_like tacc |}
  (ch c : pos) (#rows #cols : pos)
  (rProduct : chest2 real rows cols) (row : natlt rows) (col : nat)
  (#n : nat) (values : lseq tacc n) (nthr : pos) (k s : nat)
  : Lemma
    (requires (k + 1) * ch <= n /\ (s + 1) * c <= ch /\
      col + k * nthr * ch + ch <= cols /\
      tile_values_approx ch rProduct row col values nthr)
    (ensures forall (lane : natlt c).
      (values @! (k * ch + s * c + lane)) %~
        acc2 rProduct row (col + k * nthr * ch + s * c + lane))
  = introduce forall (lane : natlt c).
      (values @! (k * ch + s * c + lane)) %~
        acc2 rProduct row (col + k * nthr * ch + s * c + lane)
    with (
      FStar.Math.Lemmas.distributivity_add_left s 1 c;
      Kuiper.Math.div_mod_of_mul_add ch k (s * c + lane)
    )

let group_offset_value (ch : pos) (col : nat { ch /? col }) (k nthr : nat)
  : Lemma (group_offset ch col k nthr == col + k * nthr * ch)
  = ()

let native_subchunk_bound (whole block : pos) (k : natlt (whole / block))
  : Lemma (requires block /? whole)
          (ensures k * block + block <= whole /\ (k + 1) * block <= whole)
  = lemma_divides_exact block whole;
    FStar.Math.Lemmas.lemma_mult_le_right block (k + 1) (whole / block);
    FStar.Math.Lemmas.distributivity_add_left k 1 block

let accumulator_subchunk_bound (ch c : pos) (n k s : nat)
  : Lemma
    (requires (k + 1) * ch <= n /\ (s + 1) * c <= ch /\ fits n)
    (ensures k * ch + s * c + c <= n /\ fits (k * ch + s * c))
  = FStar.Math.Lemmas.distributivity_add_left k 1 ch;
    FStar.Math.Lemmas.distributivity_add_left s 1 c;
    FStar.SizeT.fits_lte (k * ch + s * c) n

(* Apply the combination locally, then copy a complete aligned vector into C.
   The bounds check covers the whole vector, including on partial tiles. *)
inline_for_extraction noextract
fn combine_store
  (#tc #tacc : Type0)
  {| scalar tc, sized tc, has_vec_cpy tc, real_like tc,
     scalar tacc, real_like tacc |}
  (ch : szp { chunk tc /? ch })
  (comb : tc -> tacc -> tc)
  (rcomb : real -> real -> real { approx2 comb rcomb })
  (#rows #cols : szp { ch /? cols /\ chunk tc /? cols })
  (#lC : layout2 rows cols) {| ctlayout lC, strided : strided_row_major lC |}
  (gC : array2 tc lC)
  (row : szlt rows) (col : sz { ch /? col })
  (rC rProduct : chest2 real rows cols)
  (n : sz { ch /? n })
  (out : larray tacc n) (#values : lseq tacc n)
  (nthr : szp)
  preserves gpu ** out |-> values
  requires thread_groups_approx ch gC row col rC n nthr
  requires pure (aligned 16 (core gC))
  requires pure (aligned_strided_row_major (chunk tc) strided)
  requires pure (fits (col + n * nthr))
  requires pure (tile_values_approx ch rProduct row col values nthr)
  ensures thread_groups_approx ch gC row col (chest_comb rcomb rC rProduct) n nthr
{
  unfold thread_groups_approx ch gC;
  forevery_rw_size (n / ch) (n /^ ch);
  foreach (n /^ ch)
    (fun k -> thread_group_approx ch gC row col rC nthr k)
    (fun k -> thread_group_approx ch gC row col (chest_comb rcomb rC rProduct) nthr k)
    #(gpu ** out |-> values)
    fn k {
      tile_chunk_bound ch n col nthr k;
      let off = chunk_offset ch col k nthr;
      group_offset_value ch col k nthr;
      rewrite each (group_offset ch col k nthr) as (v off);
      if (off <^ cols)
        requires group_approx ch gC row off rC
        ensures group_approx ch gC row off (chest_comb rcomb rC rProduct)
      {
        Kuiper.Sparse.Math.divides_leq ch cols off;
        accumulator_chunk ch rProduct row col values nthr k;
        unfold group_approx ch gC;
        lemma_divides_exact (chunk tc) ch;
        FStar.Math.Lemmas.swap_mul (chunk tc) (ch / chunk tc);
        forevery_factor ch (ch / chunk tc) (chunk tc) _;
        forevery_rw_size (ch / chunk tc) (ch /^ chunk tc);
        foreach (ch /^ chunk tc)
          (fun s -> forall+ (lane : natlt (chunk tc)).
            cell_approx_in_bounds gC row (off + (s * chunk tc + lane)) rC)
          (fun s -> forall+ (lane : natlt (chunk tc)).
            cell_approx_in_bounds gC row (off + (s * chunk tc + lane))
              (chest_comb rcomb rC rProduct))
          #(gpu ** out |-> values)
          fn s {
            native_subchunk_bound ch (chunk tc) s;
            lemma_divides_product (chunk tc) s;
            Kuiper.Sparse.Math.divides_chain (chunk tc) ch off;
            lemma_divides_sum (chunk tc) off (s * chunk tc);
            accumulator_subchunk_bound ch (chunk tc) n k s;
            let dst = off +^ s *^ chunk tc;
            let start = k *^ ch +^ s *^ chunk tc;
            forevery_map #(natlt (chunk tc))
              (fun lane -> cell_approx_in_bounds gC row (off + (s * chunk tc + lane)) rC)
              (fun lane -> cell_approx_in_bounds gC row (dst + lane) rC)
              fn lane {
                rewrite each (off + (s * chunk tc + lane)) as (v dst + lane);
              };
            fold vec_approx gC row dst rC;
            accumulator_subchunk ch (chunk tc) rProduct row col values nthr k s;
            combine_chunk comb rcomb gC row dst rC rProduct out start;
            unfold vec_approx gC;
            forevery_map #(natlt (chunk tc))
              (fun lane -> cell_approx_in_bounds gC row (dst + lane)
                (chest_comb rcomb rC rProduct))
              (fun lane -> cell_approx_in_bounds gC row (off + (s * chunk tc + lane))
                (chest_comb rcomb rC rProduct))
              fn lane {
                rewrite each (v dst + lane) as (off + (s * chunk tc + lane));
              };
          };
        forevery_rw_size (ch /^ chunk tc) (ch / chunk tc);
        forevery_unfactor ch (ch / chunk tc) (chunk tc)
          (fun lane -> cell_approx_in_bounds gC row (off + lane)
            (chest_comb rcomb rC rProduct));
        fold group_approx ch gC row off (chest_comb rcomb rC rProduct);
      } else {
        unfold group_approx ch gC;
        forevery_map #(natlt ch)
          (fun lane -> cell_approx_in_bounds gC row (off + lane) rC)
          (fun lane -> cell_approx_in_bounds gC row (off + lane)
            (chest_comb rcomb rC rProduct))
          fn lane {
            when__elim_false _ _;
            when__intro_false (off + lane < cols)
              (fun _ -> cell_approx gC row (off + lane)
                (acc2 (chest_comb rcomb rC rProduct) row (off + lane)));
          };
        fold group_approx ch gC row off (chest_comb rcomb rC rProduct);
      };
      rewrite each (v off) as (group_offset ch col k nthr);
    };
  forevery_rw_size (n /^ ch) (n / ch);
  fold thread_groups_approx ch gC row col (chest_comb rcomb rC rProduct) n nthr;
}
