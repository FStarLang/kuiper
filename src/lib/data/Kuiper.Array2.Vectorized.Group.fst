module Kuiper.Array2.Vectorized.Group
#lang-pulse

(* Chunk groups: a partition of the cells of a [rows x cols] matrix into
   runs of [w] consecutive cells of a single row (assuming [w] divides
   [cols]). Group [g] is the run starting at flat offset [g * w].

   This is what makes a vectorized access possible: the cells a thread
   owns must form such a run. *)

open Kuiper
open Kuiper.EMatrix
open Kuiper.Chest
open Kuiper.Array.Vectorized
open Kuiper.Array2.Vectorized
open Kuiper.Bijection

open Kuiper.Tensor { array2, layout2, idx2 }
open Kuiper.Array2.Strided
module T = Kuiper.Tensor
module ML = FStar.Math.Lemmas

(* The group a cell belongs to. *)
let group_of (w : pos) (cols : nat) (i j : nat) : nat = (i * cols + j) / w

(* Number of groups in a [rows x cols] matrix. *)
let ngroups (w : pos) (rows cols : nat) : nat = rows * cols / w

(* The first cell of group [g]. *)
let group_row (w : pos) (#rows : nat) (cols : pos) (g : nat) : nat = (g * w) / cols
let group_col (w : pos) (cols : pos) (g : nat) : nat = (g * w) % cols

let lemma_divides_rows_cols (w : pos) (rows cols : nat)
  : Lemma (requires w /?+ cols)
          (ensures w /?+ (rows * cols))
  = lemma_nat_divides_pos_divides w cols;
    lemma_divides_product_r w rows cols;
    lemma_nat_divides_pos_divides w (rows * cols)

let lemma_group_lt (w : pos) (rows cols : nat) (i : natlt rows) (j : natlt cols)
  : Lemma (requires w /?+ cols)
          (ensures group_of w cols i j < ngroups w rows cols)
  = lemma_divides_rows_cols w rows cols;
    lemma_nat_divides_pos_divides w (rows * cols);
    lemma_divides_exact w (rows * cols);
    let flat = i * cols + j in
    ML.distributivity_add_left i 1 cols;
    assert ((i + 1) * cols == i * cols + cols);
    ML.lemma_mult_le_right cols (i + 1) rows;
    assert ((i + 1) * cols <= rows * cols);
    assert (flat < rows * cols);
    ML.lemma_div_mod flat w;
    if rows * cols / w <= flat / w then
      ML.lemma_mult_le_left w (rows * cols / w) (flat / w)

#push-options "--z3rlimit 40"
let group_bounds (w : pos) (rows : pos) (cols : pos) (g : nat)
  : Lemma (requires w /?+ cols /\ g < ngroups w rows cols)
          (ensures group_row w #rows cols g < rows /\
                   w /?+ group_col w cols g /\
                   group_col w cols g + w <= cols)
  = lemma_divides_rows_cols w rows cols;
    lemma_nat_divides_pos_divides w (rows * cols);
    lemma_divides_exact w (rows * cols);
    ML.lemma_mult_le_right w (g + 1) (rows * cols / w);
    ML.distributivity_add_left g 1 w;
    assert (g * w + w <= rows * cols);
    ML.lemma_div_mod (g * w) cols;
    if rows <= (g * w) / cols then
      ML.lemma_mult_le_left cols rows ((g * w) / cols);
    assert (group_row w #rows cols g < rows);
    lemma_nat_divides_pos_divides w cols;
    lemma_divides_product w g;
    lemma_divides_mod_op w (g * w) cols;
    lemma_nat_divides_pos_divides w (group_col w cols g);
    lemma_divides_exact w (group_col w cols g);
    let c = group_col w cols g / w in
    let cc = cols / w in
    assert (w * c == group_col w cols g /\ w * cc == cols);
    if cc <= c then ML.lemma_mult_le_left w cc c;
    assert (c < cc);
    ML.lemma_mult_le_left w (c + 1) cc;
    ML.distributivity_add_right w c 1
#pop-options

#push-options "--z3rlimit 40"
let group_mem (w : pos) (rows : pos) (cols : pos) (g : nat) (x : natlt w)
  : Lemma (requires w /?+ cols /\ g < ngroups w rows cols)
          (ensures (let i = group_row w #rows cols g in
                    let j = group_col w cols g in
                    i < rows /\ j + w <= cols /\
                    group_of w cols i (j + x) == g))
  = group_bounds w rows cols g;
    ML.euclidean_division_definition (g * w) cols;
    assert (group_row w #rows cols g * cols + group_col w cols g == g * w);
    ML.lemma_div_plus x g w;
    ML.small_div x w
#pop-options

#push-options "--z3rlimit 40"
let group_char (w : pos) (rows : pos) (cols : pos) (g : nat)
  (i : natlt rows) (j : natlt cols)
  : Lemma (requires w /?+ cols /\ g < ngroups w rows cols /\
                    group_of w cols i j == g)
          (ensures (let gi = group_row w #rows cols g in
                    let gj = group_col w cols g in
                    i == gi /\ gj <= j /\ j < gj + w))
  = group_bounds w rows cols g;
    let gi = group_row w #rows cols g in
    let gj = group_col w cols g in
    let flat = i * cols + j in
    ML.euclidean_division_definition (g * w) cols;
    assert (gi * cols + gj == g * w);
    ML.lemma_div_mod flat w;
    assert (g * w <= flat /\ flat < g * w + w);
    ML.division_definition flat cols gi;
    ML.division_definition flat cols i
#pop-options

(* Refined versions of [group_row] / [group_col], carrying the bounds
   that make them usable as indices. *)
let grow (w : pos) (rows : pos) (cols : pos { w /?+ cols })
  (g : natlt (ngroups w rows cols))
  : natlt rows
  = group_bounds w rows cols g;
    group_row w #rows cols g

let gcol (w : pos) (rows : pos) (cols : pos { w /?+ cols })
  (g : natlt (ngroups w rows cols))
  : (j : natlt cols { j + w <= cols })
  = group_bounds w rows cols g;
    group_col w cols g

let group_mem' (w : pos) (rows : pos) (cols : pos { w /?+ cols })
  (g : natlt (ngroups w rows cols)) (x : natlt w)
  : Lemma (group_of w cols (grow w rows cols g) (gcol w rows cols g + x) == g)
          [SMTPat (group_of w cols (grow w rows cols g) (gcol w rows cols g + x))]
  = group_mem w rows cols g x

let group_char' (w : pos) (rows : pos) (cols : pos { w /?+ cols })
  (g : natlt (ngroups w rows cols)) (i : natlt rows) (j : natlt cols)
  : Lemma (requires group_of w cols i j == g)
          (ensures i == grow w rows cols g /\
                   gcol w rows cols g <= j /\ j < gcol w rows cols g + w)
  = group_char w rows cols g i j

(* The values held by the cells of group [g]. *)
let group_seq
  (#et : Type0) (#rows : pos) (#cols : pos)
  (w : pos { w /?+ cols })
  (em : chest2 et rows cols)
  (g : natlt (ngroups w rows cols))
  : GTot (v : seq et { Seq.length v == w })
  = Seq.init_ghost w (fun (x : natlt w) ->
      acc2 em (grow w rows cols g) (gcol w rows cols g + x))

let group_of_lt (w : pos) (rows : pos) (cols : pos) (i : natlt rows) (j : natlt cols)
  : Lemma (requires w /?+ cols)
          (ensures group_of w cols i j < ngroups w rows cols)
          [SMTPat (group_of w cols i j); SMTPat (ngroups w rows cols)]
  = lemma_group_lt w rows cols i j

(* Cell [ij] is selected by [sel] and belongs to group [g]. *)
let group_rel
  (#rows #cols : pos)
  (w : pos)
  (sel : (natlt rows & natlt cols) -> prop)
  (g : nat)
  (ij : natlt rows & natlt cols)
  : prop
  = sel ij /\ group_of w cols ij._1 ij._2 == g

(* The cells of group [g], as a subset of the cells satisfying [sel]. *)
let group_cell
  (#rows #cols : pos)
  (w : pos)
  (sel : (natlt rows & natlt cols) -> prop)
  (g : nat)
  = ij : (natlt rows & natlt cols) { group_rel w sel g ij }

unfold
let bij_group
  (#rows #cols : pos)
  (w : pos { w /?+ cols })
  (sel : (natlt rows & natlt cols) -> prop)
  (g : natlt (ngroups w rows cols))
  (sq : squash (forall (ij : natlt rows & natlt cols).
                  group_of w cols ij._1 ij._2 == g ==> sel ij))
  : (group_cell w sel g =~ natlt w)
  = {
      ff = (fun (ij : group_cell w sel g) ->
              group_char' w rows cols g ij._1 ij._2;
              ((ij._2 - gcol w rows cols g) <: natlt w));
      gg = (fun (x : natlt w) ->
              ((grow w rows cols g), ((gcol w rows cols g + x) <: natlt cols))
                <: group_cell w sel g);
      ff_gg = (fun (x : natlt w) ->
                 group_char' w rows cols g (grow w rows cols g)
                                           (gcol w rows cols g + x));
      gg_ff = (fun (ij : group_cell w sel g) ->
                 group_char' w rows cols g ij._1 ij._2);
    }

let group_residual_lemma
  (#rows #cols : pos)
  (w : pos { w /?+ cols })
  (sel : (natlt rows & natlt cols) -> prop)
  (g : natlt (ngroups w rows cols))
  : Lemma (forall (x : natlt rows & natlt cols).
             (exists (i : natlt (ngroups w rows cols)
                        { ~(eq2 #(natlt (ngroups w rows cols)) i g) }).
                group_rel w sel i x)
             <==> (sel x /\ group_of w cols x._1 x._2 =!= g))
  = introduce forall (x : natlt rows & natlt cols).
        (sel x /\ group_of w cols x._1 x._2 =!= g) ==>
        (exists (i : natlt (ngroups w rows cols)
                   { ~(eq2 #(natlt (ngroups w rows cols)) i g) }).
           group_rel w sel i x)
    with introduce _ ==> _
    with _. (
      let i : (i : natlt (ngroups w rows cols)
                 { ~(eq2 #(natlt (ngroups w rows cols)) i g) })
        = group_of w cols x._1 x._2 in
      introduce exists (i' : natlt (ngroups w rows cols)
                          { ~(eq2 #(natlt (ngroups w rows cols)) i' g) }).
                  group_rel w sel i' x
      with i and ()
    )

(* Split a family of per-cell permissions indexed by a selector [sel] into
   the run of [w] consecutive cells forming group [g] --- ready for a
   vectorized access --- and the rest. [sel] must contain all of group [g]. *)
ghost
fn cells_extract_group
  (#et : Type0) {| sized et |}
  (#rows #cols : pos)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#f : perm)
  (w : pos { w /?+ cols })
  (em : chest2 et rows cols)
  (sel : (natlt rows & natlt cols) -> prop)
  (g : natlt (ngroups w rows cols))
  (#_ : squash (forall (ij : natlt rows & natlt cols).
                  group_of w cols ij._1 ij._2 == g ==> sel ij))
  ()
  requires
    forall+ (ij : (natlt rows & natlt cols) { sel ij }).
      T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)
  ensures
    row_cells m f (grow w rows cols g) (gcol w rows cols g) w (group_seq w em g)
  ensures
    forall+ (ij : (natlt rows & natlt cols)
              { sel ij /\ group_of w cols ij._1 ij._2 =!= g }).
      T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)
{
  assert pure (forall (ij : natlt rows & natlt cols).
                 group_of w cols ij._1 ij._2 < ngroups w rows cols);
  forevery_refine_ext
    (fun (ij : natlt rows & natlt cols) ->
       exists (i : natlt (ngroups w rows cols)). group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_split_or_n #_ #(natlt (ngroups w rows cols))
    (group_rel w sel)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_remove
    (fun (i : natlt (ngroups w rows cols)) ->
       forall+ (ij : (natlt rows & natlt cols) { group_rel w sel i ij }).
         T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    g;
  forevery_iso
    (bij_group w sel g ())
    (fun (ij : group_cell w sel g) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_ext
    (fun (x : natlt w) ->
       T.tensor_pts_to_cell m #f
         (idx2 (grow w rows cols g) (gcol w rows cols g + x))
         (acc2 em (grow w rows cols g) (gcol w rows cols g + x)))
    (fun (x : natlt w) ->
       T.tensor_pts_to_cell m #f
         (idx2 (grow w rows cols g) (gcol w rows cols g + x))
         (Seq.index (group_seq w em g) x));
  fold (row_cells m f (grow w rows cols g) (gcol w rows cols g) w
          (group_seq w em g));
  forevery_join_or_n
    (fun (i : natlt (ngroups w rows cols)
            { ~(eq2 #(natlt (ngroups w rows cols)) i g) })
         (ij : natlt rows & natlt cols) -> group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  group_residual_lemma w sel g;
  forevery_refine_ext
    (fun (ij : natlt rows & natlt cols) ->
       sel ij /\ group_of w cols ij._1 ij._2 =!= g)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
}

(* Inverse of [cells_extract_group]: put the run of cells forming group
   [g] back into the family of per-cell permissions. [em] is the (possibly
   updated) contents of the whole matrix. *)
ghost
fn cells_restore_group
  (#et : Type0) {| sized et |}
  (#rows #cols : pos)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (#f : perm)
  (w : pos { w /?+ cols })
  (em : chest2 et rows cols)
  (sel : (natlt rows & natlt cols) -> prop)
  (g : natlt (ngroups w rows cols))
  (#_ : squash (forall (ij : natlt rows & natlt cols).
                  group_of w cols ij._1 ij._2 == g ==> sel ij))
  ()
  requires
    row_cells m f (grow w rows cols g) (gcol w rows cols g) w (group_seq w em g)
  requires
    forall+ (ij : (natlt rows & natlt cols)
              { sel ij /\ group_of w cols ij._1 ij._2 =!= g }).
      T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)
  ensures
    forall+ (ij : (natlt rows & natlt cols) { sel ij }).
      T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2)
{
  assert pure (forall (ij : natlt rows & natlt cols).
                 group_of w cols ij._1 ij._2 < ngroups w rows cols);
  unfold (row_cells m f (grow w rows cols g) (gcol w rows cols g) w
            (group_seq w em g));
  forevery_ext
    (fun (x : natlt w) ->
       T.tensor_pts_to_cell m #f
         (idx2 (grow w rows cols g) (gcol w rows cols g + x))
         (Seq.index (group_seq w em g) x))
    (fun (x : natlt w) ->
       T.tensor_pts_to_cell m #f
         (idx2 (grow w rows cols g) (gcol w rows cols g + x))
         (acc2 em (grow w rows cols g) (gcol w rows cols g + x)));
  forevery_iso_back
    (bij_group w sel g ())
    (fun (ij : group_cell w sel g) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  group_residual_lemma w sel g;
  forevery_refine_ext
    #_
    #(fun (ij : natlt rows & natlt cols) ->
        sel ij /\ group_of w cols ij._1 ij._2 =!= g)
    (fun (ij : natlt rows & natlt cols) ->
       exists (i : natlt (ngroups w rows cols)
                 { ~(eq2 #(natlt (ngroups w rows cols)) i g) }).
         group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_split_or_n
    (fun (i : natlt (ngroups w rows cols)
            { ~(eq2 #(natlt (ngroups w rows cols)) i g) })
         (ij : natlt rows & natlt cols) -> group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_insert
    (fun (i : natlt (ngroups w rows cols)) ->
       forall+ (ij : (natlt rows & natlt cols) { group_rel w sel i ij }).
         T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    g;
  forevery_unrefine
    (fun (i : natlt (ngroups w rows cols)) ->
       forall+ (ij : (natlt rows & natlt cols) { group_rel w sel i ij }).
         T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_join_or_n
    (fun (i : natlt (ngroups w rows cols))
         (ij : natlt rows & natlt cols) -> group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  forevery_refine_ext
    #_
    #(fun (ij : natlt rows & natlt cols) ->
        exists (i : natlt (ngroups w rows cols)). group_rel w sel i ij)
    (fun (ij : natlt rows & natlt cols) -> sel ij)
    (fun (ij : natlt rows & natlt cols) ->
       T.tensor_pts_to_cell m #f (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
}
