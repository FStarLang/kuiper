module KPlay.Proof.Base.Scratch

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.ForEvery
open Kuiper.Array2.Strided

module C = Kuiper.Chest
module B = Kuiper.Bijection
module Native = KPlay.Proof.Base.NativeProjection
module SZ = Kuiper.SizeT
module Transfer = KPlay.Proof.Base.Transfer
module M = Kuiper.TensorCore.MMA
module V = Kuiper.TensorCore.MMA.Views
module SH = Kuiper.SHMem
module A = Pulse.Lib.Array

let base_shape = 16000 @| 1 @| INil
let base_index = abs base_shape

let in_window (offset count:nat) (index:base_index) : prop =
  let i, (_, ()) = index in offset <= i /\ i < offset + count

let window_bij (offset count:nat{offset + count <= 16000})
  : B.bijection (index:base_index{in_window offset count index})
      (abs (count @| 1 @| INil)) = {
  B.ff = (fun (index:base_index{in_window offset count index}) ->
    let i, (j, ()) = index in ((i - offset, (j, ())) <: abs (count @| 1 @| INil)));
  B.gg = (fun (index:abs (count @| 1 @| INil)) ->
    let i, (j, ()) = index in
    ((offset + i, (j, ())) <: index:base_index{in_window offset count index}));
}

let window_values (v:chest2 bf16 16000 1)
  (offset count:nat{offset + count <= 16000}) : chest2 bf16 count 1 =
  mk2 (fun i _ -> acc2 v (offset + i) 0)

let replace_window (v:chest2 bf16 16000 1)
  (offset count:nat{offset + count <= 16000}) (next:chest2 bf16 count 1)
  : chest2 bf16 16000 1 =
  mk2 (fun i _ ->
    if offset <= i && i < offset + count
    then acc2 next (i - offset) 0
    else acc2 v i 0)

let window_layout (offset count:nat{offset + count <= 16000}) =
  l:layout2 count 1{
    l.ulen == 16000 /\
    (forall (i:natlt count) (j:natlt 1). cell_of_pos l i j == offset + i)}

inline_for_extraction noextract
let window_view (hidden:array2 bf16 Native.hidden_layout)
  (offset count:nat{offset + count <= 16000})
  (layout:window_layout offset count) : array2 bf16 layout =
  from_array layout (core hidden)

let frame (hidden:array2 bf16 Native.hidden_layout)
  (offset count:nat{offset + count <= 16000})
  (fraction:perm) (v:chest2 bf16 16000 1) : slprop =
  forall+ (index:base_index{~(in_window offset count index)}).
    Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
      (Native.hidden_layout.imap.f index) (acc v index)

let window_cell (hidden:array2 bf16 Native.hidden_layout)
  (offset count:nat{offset + count <= 16000})
  (layout:window_layout offset count) (v:chest2 bf16 16000 1)
  (index:abs (count @| 1 @| INil))
  : Lemma (
    core (window_view hidden offset count layout) == core hidden /\
    Native.hidden_layout.imap.f ((window_bij offset count).B.gg index) ==
      layout.imap.f index /\
    acc v ((window_bij offset count).B.gg index) ==
      acc (window_values v offset count) index) =
  let i, (j, ()) = index in
  assert (cell_of_pos layout i j == offset + i);
  assert (Native.hidden_layout.imap.f (offset + i, (j, ())) == offset + i);
  assert (acc (window_values v offset count) index == acc2 v (offset + i) 0)

let replaced_cell (v:chest2 bf16 16000 1)
  (offset count:nat{offset + count <= 16000}) (next:chest2 bf16 count 1)
  (index:abs (count @| 1 @| INil))
  : Lemma (
    acc (replace_window v offset count next) ((window_bij offset count).B.gg index) ==
      acc next index) =
  let i, (j, ()) = index in
  assert (offset <= offset + i /\ offset + i < offset + count)

ghost
fn extract_window
  (hidden:array2 bf16 Native.hidden_layout)
  (offset count:nat{offset + count <= 16000})
  (layout:window_layout offset count)
  (#fraction:perm) (#v:chest2 bf16 16000 1)
  requires hidden |-> Frac fraction v
  ensures window_view hidden offset count layout |->
    Frac fraction (window_values v offset count)
  ensures frame hidden offset count fraction v
  ensures Kuiper.Array.Core.array_exists (core hidden)
{
  assert pure (core (window_view hidden offset count layout) == core hidden);
  tensor_ilower hidden;
  forevery_refine_split
    (fun (index:base_index) -> Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
      (Native.hidden_layout.imap.f index) (acc v index))
    (in_window offset count);
  forevery_iso (window_bij offset count)
    (fun (index:base_index{in_window offset count index}) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f index) (acc v index));
  FStar.Classical.forall_intro (window_cell hidden offset count layout v);
  forevery_ext
    (fun (index:abs (count @| 1 @| INil)) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f ((window_bij offset count).B.gg index))
        (acc v ((window_bij offset count).B.gg index)))
    (fun (index:abs (count @| 1 @| INil)) ->
      Kuiper.Array.Core.pts_to_cell (core (window_view hidden offset count layout))
        #fraction (layout.imap.f index) (acc (window_values v offset count) index));
  Pulse.Class.Duplicable.dup (Kuiper.Array.Core.array_exists (core hidden)) ();
  rewrite Kuiper.Array.Core.array_exists (core hidden)
    as Kuiper.Array.Core.array_exists (core (window_view hidden offset count layout));
  tensor_iraise (window_view hidden offset count layout);
  fold frame hidden offset count fraction v;
}

ghost
fn restore_window
  (hidden:array2 bf16 Native.hidden_layout)
  (offset count:nat{offset + count <= 16000})
  (layout:window_layout offset count)
  (#fraction:perm) (#v:chest2 bf16 16000 1) (#next:chest2 bf16 count 1)
  requires window_view hidden offset count layout |-> Frac fraction next
  requires frame hidden offset count fraction v
  requires Kuiper.Array.Core.array_exists (core hidden)
  ensures hidden |-> Frac fraction (replace_window v offset count next)
{
  assert pure (core (window_view hidden offset count layout) == core hidden);
  tensor_ilower (window_view hidden offset count layout);
  FStar.Classical.forall_intro (window_cell hidden offset count layout v);
  FStar.Classical.forall_intro (replaced_cell v offset count next);
  forevery_ext
    (fun (index:abs (count @| 1 @| INil)) ->
      Kuiper.Array.Core.pts_to_cell (core (window_view hidden offset count layout))
        #fraction (layout.imap.f index) (acc next index))
    (fun (index:abs (count @| 1 @| INil)) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f ((window_bij offset count).B.gg index))
        (acc (replace_window v offset count next) ((window_bij offset count).B.gg index)));
  forevery_iso_back (window_bij offset count)
    (fun (index:base_index{in_window offset count index}) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f index) (acc (replace_window v offset count next) index));
  unfold frame hidden offset count fraction v;
  forevery_ext
    (fun (index:base_index{~(in_window offset count index)}) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f index) (acc v index))
    (fun (index:base_index{~(in_window offset count index)}) ->
      Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
        (Native.hidden_layout.imap.f index) (acc (replace_window v offset count next) index));
  forevery_refine_join
    (fun (index:base_index) -> Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
      (Native.hidden_layout.imap.f index) (acc (replace_window v offset count next) index))
    (in_window offset count) (fun index -> ~(in_window offset count index));
  forevery_unrefine
    (fun (index:base_index) -> Kuiper.Array.Core.pts_to_cell (core hidden) #fraction
      (Native.hidden_layout.imap.f index) (acc (replace_window v offset count next) index));
  tensor_iraise hidden;
}

let replace_frame (v:chest2 bf16 16000 1)
  (offset count:nat{offset + count <= 16000}) (next:chest2 bf16 count 1)
  (i:natlt 16000)
  : Lemma (
    (~(offset <= i /\ i < offset + count) ==>
      acc2 (replace_window v offset count next) i 0 == acc2 v i 0) /\
    (offset <= i /\ i < offset + count ==>
      acc2 (replace_window v offset count next) i 0 == acc2 next (i - offset) 0)) = ()

inline_for_extraction noextract
fn owned_projection_block
  (row:sz{SZ.v row % 16 == 0 /\ SZ.v row + 16 <= 248320})
  (weight:array2 bf16 Native.weight_layout)
  (hidden:array2 bf16 Native.hidden_layout)
  (accum part:M.fragment)
  (#vw:chest2 bf16 752393024 1) (#vh:chest2 bf16 16000 1)
  (#initial #vc:chest2 float 16 8)
  (#weight_fraction:perm)
  preserves gpu ** weight |-> Frac weight_fraction vw
  requires hidden |-> Frac (1.0R /. 32) vh
  requires accum |-> initial ** part |-> vc
  ensures part |-> Transfer.block_accumulator
    (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64
  ensures accum |-> V.added 5 (V.filled 5 initial (zero #float))
    (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64)
  ensures hidden |-> Frac (1.0R /. 32) (replace_window vh 12500 16
    (Native.stored_column (Transfer.block_accumulator
      (Native.packed_values vw) (Native.hidden_values vh) (SZ.v row) 64)))
{
  Native.final_projection_block row weight hidden accum part;
  extract_window hidden 12500 16 Native.score_layout;
  Native.publish_block hidden accum part;
  restore_window hidden 12500 16 Native.score_layout;
}

(* This is the existing kernel allocation resource, not a claim that source
   code initialized every cell. Read-before-write is tracked separately. *)
ghost
fn from_shared_allocation
  (hidden:SH.c_shmem (SH.SHArray bf16 16000sz))
  requires SH.live_c_shmem #(SH.SHArray bf16 16000sz) hidden
  ensures exists* (v:chest2 bf16 16000 1).
    from_array Native.hidden_layout hidden |-> v
{
  SH.unfold_live_c_shmem hidden;
  with s. assert A.pts_to hidden s;
  A.pts_to_len hidden;
  tensor_abs' Native.hidden_layout hidden;
}
