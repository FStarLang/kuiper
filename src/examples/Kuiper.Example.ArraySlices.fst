module Kuiper.Example.ArraySlices

#lang-pulse

open Kuiper
open Kuiper.Tensor

module IV = Kuiper.IView
module IA = Kuiper.IArray
module VA = Kuiper.VArray

(* Includes zero-length arrays and arbitrary fractional permissions. *)
ghost
fn array_round_trip
  (#a : Type0) (#sz : nat)
  (arr : larray a sz) (f : perm)
  (v : lseq a sz)
  requires arr |-> Frac f v
  ensures arr |-> Frac f v
{
  array_slice_1 arr #f #v;
  array_unslice_1_with_exists arr #f #v;
}

(* Empty subranges retain ownership at their original offset. *)
ghost
fn empty_subrange_round_trip
  (#a : Type0) (arr : array a)
  (f : perm) (i : nat)
  requires pts_to_slice arr #f i i seq![]
  ensures pts_to_slice arr #f i i seq![]
{
  slice_to_cells arr #f i i #seq![];
  cells_to_slice arr #f i i #seq![];
}

(* A returned abstract view owns its backing-array witness internally. *)
inline_for_extraction noextract
fn empty_iarray_round_trip
  (#a : Type0) (arr : larray a 0) (#f : perm)
  requires arr |-> Frac f (seq![] <: seq a)
  returns result : larray a 0
  ensures result |-> Frac f (seq![] <: seq a)
{
  let va = IA.iarray_begin arr #f #seq![];
  let result = IA.iarray_end va;
  assert pure (Seq.equal (Seq.init_ghost 0 (IA.g_seq_acc (seq![] <: lseq a 0))) seq![]);
  result;
}

(* Ordinary location mapping suffices, including for empty views. *)
ghost
fn located_view_round_trip
  (#a : Type0) (#sz : nat)
  (arr : larray a sz) (f : perm) (v : lseq a sz) (l : loc_id)
  requires on l (arr |-> Frac f v)
  ensures on l (arr |-> Frac f v)
{
  map_loc l
    #(arr |-> Frac f v)
    #(VA.from_array (VA.raw_view #a #sz) arr |-> Frac f v)
    fn () { VA.varray_begin_ arr; };
  map_loc l
    #(VA.from_array (VA.raw_view #a #sz) arr |-> Frac f v)
    #(arr |-> Frac f v)
    fn () {
      VA.varray_end_ (VA.from_array (VA.raw_view #a #sz) arr);
      rewrite each VA.core (VA.from_array (VA.raw_view #a #sz) arr) as arr;
    };
}

(* A nonempty collection of cells already carries allocation ownership. *)
ghost
fn nonempty_subrange_from_cells
  (#a : Type0) (arr : array a)
  (f : perm) (i : nat) (j : nat{i < j})
  (v : lseq a (j - i))
  requires forall+ (k : natlt (j - i)). pts_to_cell arr #f (i + k) (v @! k)
  ensures pts_to_slice arr #f i j v
{
  cells_to_nonempty_slice arr #f i j #v;
}

(* Knowing the pointer and its length cannot manufacture empty ownership. *)
[@@expect_failure [228]]
ghost
fn cannot_reassemble_unowned_empty_array
  (#a : Type0) (arr : larray a 0)
  requires emp
  ensures arr |-> seq![]
{
  forevery_intro_false #(natlt 0) (fun i -> pts_to_cell arr #1.0R i (seq![] @! i));
  forevery_unrefine _;
  array_unslice_1_with_exists arr #1.0R #seq![];
}

[@@expect_failure [228]]
ghost
fn cannot_reassemble_unowned_empty_subrange
  (#a : Type0) (#sz : nat)
  (arr : larray a sz) (i : natle sz)
  requires emp
  ensures pts_to_slice arr #1.0R i i seq![]
{
  forevery_intro_false #(natlt (i - i))
    (fun k -> pts_to_cell arr #1.0R (i + k) (seq![] @! k));
  forevery_unrefine _;
  cells_to_slice arr #1.0R i i #seq![];
}

(* The permission-free witness duplicates through the registered instance. *)
ghost
fn duplicate_existence (#a : Type0) (arr : array a)
  requires array_exists arr
  ensures array_exists arr ** array_exists arr
{
}

(* An empty slice at any valid offset can use any fraction. No cell permission
   changes, and the original slice remains owned. *)
ghost
fn empty_slice_any_fraction
  (#a : Type0) (arr : array a) (f g : perm)
  (i : nat) (j : nat{j <= Pulse.Lib.Array.length arr})
  preserves pts_to_slice arr #f i i seq![]
  ensures pts_to_slice arr #g j j seq![]
{
  slice_array_exists arr i i;
  empty_slice arr g j;
}

(* Sharing and gathering a view copies the witness, regardless of its size. *)
ghost
fn shared_view_round_trip
  (#a : Type0) (#sz : nat) (arr : larray a sz)
  (f : perm) (v : lseq a sz) (n : pos)
  requires arr |-> Frac f v
  ensures arr |-> Frac f v
{
  VA.varray_begin_ arr;
  let va = VA.from_array (VA.raw_view #a #sz) arr;
  assert rewrites_to va (VA.from_array (VA.raw_view #a #sz) arr);
  VA.varray_share_n va n;
  VA.varray_gather_n va n;
  VA.varray_end_ va;
  rewrite each VA.core va as arr;
}

[@@expect_failure [228]]
ghost
fn cannot_forge_existence (#a : Type0) (arr : array a)
  requires emp
  ensures array_exists arr
{
}

[@@expect_failure [228]]
ghost
fn cannot_forge_empty_view
  (#a : Type0) (arr : larray a 0) (v : natlt 0 -> GTot a)
  requires emp
  ensures IA.from_array (IV.raw_view #0) arr |-> v
{
  let va = IA.from_array (IV.raw_view #0) arr;
  assert rewrites_to va (IA.from_array (IV.raw_view #0) arr);
  forevery_intro_empty (fun (i : (IV.raw_view #0).ait) ->
    IA.iarray_pts_to_cell va #1.0R i (v i));
  IA.iarray_implode_with_exists va;
}

inline_for_extraction noextract
let empty_view (sz : nat) : IV.aiview = {
  len = sz;
  ait = natlt 0;
  step = { imap = { f = (fun (i : natlt 0) -> (i <: natlt sz)); }; };
}

(* An empty view of a nonempty allocation still carries its own witness. *)
ghost
fn empty_partial_view_shared
  (#a : Type0) (#sz : pos) (arr : larray a sz)
  (f : perm) (v : lseq a sz) (ev : natlt 0 -> GTot a)
  preserves arr |-> Frac f v
  ensures IA.from_array (empty_view sz) arr |-> Frac f ev
{
  array_to_slice arr;
  pts_to_slice_ref arr 0 sz;
  slice_array_exists arr 0 sz;
  slice_to_array arr;
  let va = IA.from_array (empty_view sz) arr;
  assert rewrites_to va (IA.from_array (empty_view sz) arr);
  forevery_intro_empty (fun (i : (empty_view sz).ait) -> IA.iarray_pts_to_cell va #f i (ev i));
  rewrite array_exists arr as array_exists (IA.core va);
  IA.iarray_implode_with_exists va;
  IA.iarray_share_n va 3;
  IA.iarray_gather_n va 3;
  IA.iarray_explode va;
  IA.iarray_implode_with_exists va;
}

(* Existence grants no access to the values stored in a nonempty array. *)
[@@expect_failure [228]]
ghost
fn existence_does_not_own_cells
  (#a : Type0) (arr : array a) (i : nat) (v : a)
  requires array_exists arr
  ensures pts_to_cell arr #1.0R i v
{
}

(* The tile collection may be empty when either matrix dimension is zero. *)
ghost
fn tiled_tensor_round_trip
  (#a : Type0) (#rows #cols : nat)
  (#l : layout2 rows cols) (m : array2 a l)
  (trows : pos{trows /? rows}) (tcols : pos{tcols /? cols})
  (#f : perm) (#v : chest2 a rows cols)
  requires m |-> Frac f v
  ensures m |-> Frac f v
{
  tensor_pts_to_ref m;
  Kuiper.Tensor.Tiling.array2_tile m trows tcols;
  Kuiper.Tensor.Tiling.array2_untile_with_exists m trows tcols;
}

(* A pure nonemptiness fact suffices even for an abstract index type. *)
ghost
fn nonempty_view_from_cells
  (#et : Type0) (#vw : IV.aiview) (a : IA.iarray et vw)
  (#f : perm) (#v : vw.ait -> GTot et)
  requires pure (Kuiper.SizeT.fits vw.len /\ nonempty vw.ait)
  requires forall+ (i : vw.ait). IA.iarray_pts_to_cell a #f i (v i)
  ensures a |-> Frac f v
{
  IA.iarray_implode a;
}

(* Nonempty tile collections recover their witness without a separate input. *)
ghost
fn nonempty_tiled_tensor_from_views
  (#et : Type0) (#rows #cols : pos)
  (#l : layout2 rows cols) (m : array2 et l)
  (trows : pos{trows /? rows}) (tcols : pos{tcols /? cols})
  (#f : perm) (#v : chest2 et rows cols)
  requires pure (Kuiper.SizeT.fits l.ulen)
  requires forall+ (tr : natlt (rows / trows)) (tc : natlt (cols / tcols)).
    Kuiper.Tensor.Tiling.array2_subtile m trows tcols tr tc
      |-> Frac f (Kuiper.Tensor.Tiling.ematrix_subtile v trows tcols tr tc)
  ensures m |-> Frac f v
{
  Kuiper.Tensor.Tiling.array2_untile m trows tcols;
}

(* Ordinary implosion still rejects an empty index type. *)
[@@expect_failure [19]]
ghost
fn cannot_implode_empty_index_type
  (#et : Type0) (arr : larray et 0) (v : natlt 0 -> GTot et)
  requires emp
  ensures IA.from_array (IV.raw_view #0) arr |-> v
{
  let va = IA.from_array (IV.raw_view #0) arr;
  assert rewrites_to va (IA.from_array (IV.raw_view #0) arr);
  forevery_intro_empty (fun (i : (IV.raw_view #0).ait) ->
    IA.iarray_pts_to_cell va #1.0R i (v i));
  IA.iarray_implode va;
}
