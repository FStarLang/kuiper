module KPlay.Proof.Base.Matmul

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module C = Kuiper.Chest
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module M = Kuiper.TensorCore.MMA
module G = Kuiper.Spec.GEMM
module Sum = Kuiper.Sum
module EM = Kuiper.EMatrix
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module Algebra = KPlay.Proof.Base.Algebra
module Shape = Kuiper.Shape

inline_for_extraction noextract
let a_tile (#rows #columns:pos) (weights:P.matrix BF.t rows columns) (row depth:nat)
  : C.chest2 BF.t 16 16 =
  C.mk2 (fun i j ->
    if row + i < rows && depth + j < columns then weights (row + i) (depth + j)
    else F.zero #BF.t)

inline_for_extraction noextract
let b_tile (#columns:pos) (input:P.vector BF.t columns) (depth:nat)
  : C.chest2 BF.t 16 8 =
  C.mk2 (fun i _ -> if depth + i < columns then input (depth + i) else F.zero #BF.t)

inline_for_extraction noextract
let real_a_tile (#rows #columns:pos) (weights:P.matrix real rows columns) (row depth:nat)
  : C.chest2 real 16 16 =
  C.mk2 (fun i j ->
    if row + i < rows && depth + j < columns then weights (row + i) (depth + j)
    else 0.0R)

inline_for_extraction noextract
let real_b_tile (#columns:pos) (input:P.vector real columns) (depth:nat)
  : C.chest2 real 16 8 =
  C.mk2 (fun i _ -> if depth + i < columns then input (depth + i) else 0.0R)

inline_for_extraction noextract
let tiles_approx (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (wr:P.matrix real rows columns) (xr:P.vector real columns) (row depth:nat)
  : Lemma
      (requires O.matrix_approximates weights wr /\ O.vector_approximates input xr)
      (ensures C.chest_approximates (a_tile #rows #columns weights row depth) (real_a_tile #rows #columns wr row depth) /\
        C.chest_approximates (b_tile #columns input depth) (real_b_tile #columns xr depth)) =
  let at_a (i j:P.index 16)
    : Lemma (Approx.v_approximates
        (C.acc2 (a_tile #rows #columns weights row depth) i j) (C.acc2 (real_a_tile #rows #columns wr row depth) i j)) =
    if row + i < rows && depth + j < columns then
      assert (Approx.v_approximates (weights (row + i) (depth + j)) (wr (row + i) (depth + j)))
  in
  FStar.Classical.forall_intro_2 at_a;
  let all_a (idx:Shape.abs (Shape.ICons 16 (Shape.ICons 16 Shape.INil)))
    : Lemma (Approx.v_approximates
        (C.acc (a_tile #rows #columns weights row depth) idx) (C.acc (real_a_tile #rows #columns wr row depth) idx)) =
    let i, (j, ()) = idx in at_a i j
  in
  FStar.Classical.forall_intro all_a;
  let at_b (i:P.index 16) (j:P.index 8)
    : Lemma (Approx.v_approximates (C.acc2 (b_tile #columns input depth) i j) (C.acc2 (real_b_tile #columns xr depth) i j)) =
    if depth + i < columns then
      assert (Approx.v_approximates (input (depth + i)) (xr (depth + i)))
  in
  FStar.Classical.forall_intro_2 at_b;
  let all_b (idx:Shape.abs (Shape.ICons 16 (Shape.ICons 8 Shape.INil)))
    : Lemma (Approx.v_approximates
        (C.acc (b_tile #columns input depth) idx) (C.acc (real_b_tile #columns xr depth) idx)) =
    let i, (j, ()) = idx in at_b i j
  in
  FStar.Classical.forall_intro all_b

inline_for_extraction noextract
let rec accumulator (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (row start count:nat) : GTot (C.chest2 FP.t 16 8) (decreases count) =
  if count = 0 then C.mk2 (fun _ _ -> F.zero #FP.t)
  else
    let depth = start + 16 * (count - 1) in
    M.emma (accumulator #rows #columns weights input row start (count - 1))
      (a_tile #rows #columns weights row depth) (b_tile #columns input depth)

inline_for_extraction noextract
let rec real_accumulator (#rows #columns:pos)
  (weights:P.matrix real rows columns) (input:P.vector real columns)
  (row start count:nat) : GTot (C.chest2 real 16 8) (decreases count) =
  if count = 0 then C.mk2 (fun _ _ -> 0.0R)
  else
    let depth = start + 16 * (count - 1) in
    G.matplus (real_accumulator #rows #columns weights input row start (count - 1))
      (G.matmul (real_a_tile #rows #columns weights row depth) (real_b_tile #columns input depth))

inline_for_extraction noextract
let rec accumulator_approx (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (wr:P.matrix real rows columns) (xr:P.vector real columns)
  (row start count:nat)
  : Lemma
      (requires O.matrix_approximates weights wr /\ O.vector_approximates input xr)
      (ensures C.chest_approximates (accumulator #rows #columns weights input row start count)
        (real_accumulator #rows #columns wr xr row start count))
      (decreases count) =
  if count > 0 then (
    accumulator_approx #rows #columns weights input wr xr row start (count - 1);
    let depth = start + 16 * (count - 1) in
    tiles_approx #rows #columns weights input wr xr row depth;
    M.emma_approx_lemma (accumulator #rows #columns weights input row start (count - 1))
      (a_tile #rows #columns weights row depth) (b_tile #columns input depth)
      (real_accumulator #rows #columns wr xr row start (count - 1))
      (real_a_tile #rows #columns wr row depth) (real_b_tile #columns xr depth)
  )

inline_for_extraction noextract
let terms (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) (column:nat) : GTot real =
  if column < columns then weights row column *. input column else 0.0R

inline_for_extraction noextract
let tile_dot (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row depth:nat) (i:P.index 16{row + i < rows})
  : Lemma (
      G.__matmul_single (real_a_tile #rows #columns weights row depth) (real_b_tile #columns input depth) i 0 16 ==
      Sum.sum #real #Sum.real_add_semigroup depth (depth + 16) (terms #rows #columns weights input (row + i))) =
  let a = real_a_tile #rows #columns weights row depth in
  let b = real_b_tile #columns input depth in
  Algebra.matmul_sum a b i 0 16;
  let at_index (j:P.index 16)
    : Lemma (Algebra.dot_terms a b i 0 j == terms #rows #columns weights input (row + i) (depth + j)) =
    EM.macc_mkM #real #16 #16
      (fun r c -> if row + r < rows && depth + c < columns
        then weights (row + r) (depth + c) else 0.0R) i j;
    EM.macc_mkM #real #16 #8
      (fun r _ -> if depth + r < columns then input (depth + r) else 0.0R) j 0;
    norm_spec [delta_once ["KPlay.Proof.Base.Matmul.real_a_tile"]] a;
    norm_spec [delta_once ["KPlay.Proof.Base.Matmul.real_b_tile"]] b;
    norm_spec [delta_once ["KPlay.Proof.Base.Algebra.dot_terms"]] (Algebra.dot_terms a b i 0 j);
    norm_spec [delta_once ["KPlay.Proof.Base.Matmul.terms"]] (terms #rows #columns weights input (row + i) (depth + j))
  in
  FStar.Classical.forall_intro at_index;
  Algebra.sum_ext #16 (Algebra.dot_terms a b i 0)
    (fun j -> terms #rows #columns weights input (row + i) (depth + j)) 16;
  Algebra.shift_sum #(depth + 16) (terms #rows #columns weights input (row + i)) depth 16;
  Algebra.sum_ext #16 (fun j -> terms #rows #columns weights input (row + i) (depth + j))
    (Algebra.shifted #(depth + 16) (terms #rows #columns weights input (row + i)) depth) 16

inline_for_extraction noextract
let rec accumulator_dot (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row start count:nat) (i:P.index 16{row + i < rows})
  : Lemma
      (ensures C.acc2 (real_accumulator #rows #columns weights input row start count) i 0 ==
        Sum.sum #real #Sum.real_add_semigroup start (start + 16 * count) (terms #rows #columns weights input (row + i)))
      (decreases count) =
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.real_accumulator"]]
    (real_accumulator #rows #columns weights input row start count);
  if count > 0 then (
    accumulator_dot #rows #columns weights input row start (count - 1) i;
    let depth = start + 16 * (count - 1) in
    assert (depth + 16 == start + 16 * count);
    tile_dot #rows #columns weights input row depth i;
    G.lemma_matmul_index (real_a_tile #rows #columns weights row depth) (real_b_tile #columns input depth) i 0;
    Sum.sum_split #real #Sum.real_add_semigroup start (start + 16 * count) (terms #rows #columns weights input (row + i)) depth
  )

inline_for_extraction noextract
let rec zero_tail (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) (start:nat{columns <= start})
  (finish:nat{start <= finish})
  : Lemma
      (ensures Sum.sum #real #Sum.real_add_semigroup start finish (terms #rows #columns weights input row) == 0.0R)
      (decreases finish - start) =
  if start < finish then (
    Sum.sum_pop_left #real #Sum.real_add_semigroup start finish (terms #rows #columns weights input row);
    zero_tail #rows #columns weights input row (start + 1) finish
  )

inline_for_extraction noextract
let padded_sum (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows)
  (start:nat{start <= columns}) (finish:nat{columns <= finish})
  : Lemma (
      Sum.sum #real #Sum.real_add_semigroup start finish (terms #rows #columns weights input row) == Sum.sum #real #Sum.real_add_semigroup start columns (terms #rows #columns weights input row)) =
  Sum.sum_split #real #Sum.real_add_semigroup start finish (terms #rows #columns weights input row) columns;
  zero_tail #rows #columns weights input row columns finish

inline_for_extraction noextract
let split_count = n:pos{n = 1 \/ n = 2}

inline_for_extraction noextract
let part_tiles (columns:pos) (splits:split_count) : nat =
  ((columns / splits) + 15) / 16

inline_for_extraction noextract
let rec totals (#rows #columns:pos) (weights:P.matrix BF.t rows columns)
  (input:P.vector BF.t columns) (row:nat) (splits:split_count)
  (count:nat{count <= splits}) : GTot (C.chest2 FP.t 16 8) (decreases count) =
  if count = 0 then C.mk2 (fun _ _ -> F.zero #FP.t)
  else
    let part = accumulator #rows #columns weights input row ((count - 1) * (columns / splits)) (part_tiles columns splits) in
    let previous = totals #rows #columns weights input row splits (count - 1) in
    C.mk2 (fun i j -> F.add (C.acc2 previous i j) (C.acc2 part i j))

inline_for_extraction noextract
let output (#rows #columns:pos) (weights:P.matrix BF.t rows columns)
  (input:P.vector BF.t columns) (splits:split_count) (row:P.index rows) : GTot BF.t =
  Cast.cast_f32_to_bf16 (C.acc2 (totals #rows #columns weights input ((row / 16) * 16) splits splits) (row % 16) 0)

inline_for_extraction noextract
let split_valid (columns:pos) (splits:split_count) : prop =
  splits = 1 \/ (splits = 2 /\ columns % 32 = 0)

inline_for_extraction noextract
let split_geometry (columns:pos) (splits:split_count)
  : Lemma
      (requires split_valid columns splits)
      (ensures
        columns / splits > 0 /\ splits * (columns / splits) == columns /\
        (if splits = 1 then columns <= 16 * part_tiles columns splits /\
          16 * part_tiles columns splits < columns + 16
         else 16 * part_tiles columns splits == columns / splits)) = ()

inline_for_extraction noextract
let part_geometry (columns:pos) (splits:split_count) (part:nat{part < splits})
  : Lemma
      (requires split_valid columns splits)
      (ensures (
        if splits = 1 then part == 0 /\ columns / splits == columns /\
          part * (columns / splits) == 0 /\
          (part + 1) * (columns / splits) == columns /\
          columns <= part * (columns / splits) + 16 * part_tiles columns splits
        else part * (columns / splits) + 16 * part_tiles columns splits ==
          (part + 1) * (columns / splits))) =
  split_geometry columns splits

inline_for_extraction noextract
let part_sum (#rows #columns:pos) (wr:P.matrix real rows columns)
  (xr:P.vector real columns) (row:P.index rows) (splits:split_count) (part:nat{part < splits})
  : Lemma
      (requires split_valid columns splits)
      (ensures Sum.sum #real #Sum.real_add_semigroup (part * (columns / splits))
          (part * (columns / splits) + 16 * part_tiles columns splits) (terms #rows #columns wr xr row) ==
        Sum.sum #real #Sum.real_add_semigroup (part * (columns / splits)) ((part + 1) * (columns / splits)) (terms #rows #columns wr xr row)) =
  part_geometry columns splits part;
  if splits = 1 then
    padded_sum #rows #columns wr xr row (part * (columns / splits))
      (part * (columns / splits) + 16 * part_tiles columns splits)

inline_for_extraction noextract
let part_approx (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (wr:P.matrix real rows columns) (xr:P.vector real columns)
  (row:nat) (i:P.index 16{row + i < rows}) (splits:split_count)
  (part:nat{part < splits})
  : Lemma
      (requires O.matrix_approximates weights wr /\ O.vector_approximates input xr /\
        split_valid columns splits)
      (ensures Approx.v_approximates
        (C.acc2 (accumulator #rows #columns weights input row (part * (columns / splits)) (part_tiles columns splits)) i 0)
        (Sum.sum #real #Sum.real_add_semigroup (part * (columns / splits)) ((part + 1) * (columns / splits)) (terms #rows #columns wr xr (row + i)))) =
  split_geometry columns splits;
  let start = part * (columns / splits) in
  let count = part_tiles columns splits in
  accumulator_approx #rows #columns weights input wr xr row start count;
  accumulator_dot #rows #columns wr xr row start count i;
  assert (Approx.v_approximates
    (C.acc2 (accumulator #rows #columns weights input row start count) i 0)
    (C.acc2 (real_accumulator #rows #columns wr xr row start count) i 0));
  part_sum #rows #columns wr xr (row + i) splits part

inline_for_extraction noextract
let total_sum (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) (stop:nat) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 stop (terms #rows #columns weights input row)

inline_for_extraction noextract
let rec totals_approx (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (wr:P.matrix real rows columns) (xr:P.vector real columns)
  (row:nat) (i:P.index 16{row + i < rows}) (splits:split_count)
  (count:nat{count <= splits})
  : Lemma
      (requires O.matrix_approximates weights wr /\ O.vector_approximates input xr /\
        split_valid columns splits)
      (ensures Approx.v_approximates (C.acc2 (totals #rows #columns weights input row splits count) i 0)
        (total_sum #rows #columns wr xr (row + i) (count * (columns / splits))))
      (decreases count) =
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.totals"]] (totals #rows #columns weights input row splits count);
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.total_sum"]]
    (total_sum #rows #columns wr xr (row + i) (count * (columns / splits)));
  if count > 0 then (
    totals_approx #rows #columns weights input wr xr row i splits (count - 1);
    part_approx #rows #columns weights input wr xr row i splits (count - 1);
    let start = (count - 1) * (columns / splits) in
    norm_spec [delta_once ["KPlay.Proof.Base.Matmul.total_sum"]]
      (total_sum #rows #columns wr xr (row + i) start);
    let previous = totals #rows #columns weights input row splits (count - 1) in
    let part = accumulator #rows #columns weights input row start (part_tiles columns splits) in
    Approx.a_add (C.acc2 previous i 0) (C.acc2 part i 0)
      (Sum.sum #real #Sum.real_add_semigroup 0 start (terms #rows #columns wr xr (row + i)))
      (Sum.sum #real #Sum.real_add_semigroup start (count * (columns / splits)) (terms #rows #columns wr xr (row + i)));
    Sum.sum_split #real #Sum.real_add_semigroup 0 (count * (columns / splits)) (terms #rows #columns wr xr (row + i)) start
  )

inline_for_extraction noextract
let project (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) : GTot real =
  total_sum #rows #columns weights input row columns

inline_for_extraction noextract
let term_inside (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) (column:P.index columns)
  : Lemma (terms #rows #columns weights input row column == weights row column *. input column) = ()

inline_for_extraction noextract
let complete_sum (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows) (splits:split_count)
  : Lemma
      (requires split_valid columns splits)
      (ensures total_sum #rows #columns weights input row (splits * (columns / splits)) ==
        project #rows #columns weights input row) =
  split_geometry columns splits;
  assert (splits * (columns / splits) == columns);
  norm_spec [delta_once ["KPlay.Proof.Base.Matmul.project"]] (project #rows #columns weights input row)

inline_for_extraction noextract
let rec sum_restrict (#n:pos) (f:nat -> GTot real) (g:P.vector real n)
  (count:nat{count <= n})
  : Lemma
      (requires forall (i:P.index n). f i == g i)
      (ensures Sum.sum #real #Sum.real_add_semigroup 0 count f ==
        Sum.sum #real #Sum.real_add_semigroup 0 count g)
      (decreases count) =
  if count > 0 then (
    sum_restrict #n f g (count - 1);
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count f;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 count g
  )

inline_for_extraction noextract
let sum_restrict_full (n:pos) (f:nat -> GTot real) (g:P.vector real n)
  : Lemma
      (requires forall (i:P.index n). f i == g i)
      (ensures Sum.sum #real #Sum.real_add_semigroup 0 n f ==
        Sum.sum #real #Sum.real_add_semigroup 0 n g) =
  sum_restrict #n f g n

inline_for_extraction noextract
let project_expanded (#rows #columns:pos) (weights:P.matrix real rows columns)
  (input:P.vector real columns) (row:P.index rows)
  : Lemma (project #rows #columns weights input row ==
      Sum.sum #real #Sum.real_add_semigroup 0 columns (fun i -> weights row i *. input i)) =
  let at_index (i:P.index columns)
    : Lemma (terms #rows #columns weights input row i == weights row i *. input i) =
    term_inside #rows #columns weights input row i
  in
  FStar.Classical.forall_intro at_index;
  assert (project #rows #columns weights input row ==
    Sum.sum #real #Sum.real_add_semigroup 0 columns (fun i -> weights row i *. input i)) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Base.Matmul.project"; "KPlay.Proof.Base.Matmul.total_sum"]];
    FStar.Tactics.V2.apply_lemma (`sum_restrict_full)
  )

inline_for_extraction noextract
let output_approx (#rows #columns:pos)
  (weights:P.matrix BF.t rows columns) (input:P.vector BF.t columns)
  (wr:P.matrix real rows columns) (xr:P.vector real columns)
  (splits:split_count) (row:P.index rows)
  : Lemma
      (requires O.matrix_approximates weights wr /\ O.vector_approximates input xr /\
        split_valid columns splits)
      (ensures Approx.v_approximates (output #rows #columns weights input splits row) (project #rows #columns wr xr row)) =
  split_geometry columns splits;
  assert ((row / 16) * 16 + row % 16 == row);
  totals_approx #rows #columns weights input wr xr ((row / 16) * 16) (row % 16) splits splits;
  complete_sum #rows #columns wr xr ((row / 16) * 16 + row % 16) splits;
  assert (Approx.v_approximates
    (C.acc2 (totals #rows #columns weights input ((row / 16) * 16) splits splits) (row % 16) 0)
    (project #rows #columns wr xr row));
  Cast.cast_f32_to_bf16_ok
    (C.acc2 (totals #rows #columns weights input ((row / 16) * 16) splits splits) (row % 16) 0)
    (project #rows #columns wr xr row)
