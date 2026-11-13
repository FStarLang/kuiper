module KPlay.Functional

#lang-pulse

open Kuiper

module Shared = KPlay.SharedSpec
module Seq = FStar.Seq
module FP = Kuiper.Floating
module I32 = FStar.Int32
module I64 = FStar.Int64
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let prefix_step (#e:pos) (row:row_values e) (n:pos{n < e})
  : Lemma (
      Shared.best_prefix (logical_input row 0) (n + 1) ==
        (let previous = Shared.best_prefix (logical_input row 0) n in
         if FP.lt #f32 (Seq.index row previous) (Seq.index row n)
         then n else previous))
  = ()

inline_for_extraction noextract
fn copy_exact
  (#a:Type0)
  (n:sz)
  (#contents:erased (lseq a (SZ.v n)))
  (input output:larray a (SZ.v n))
  preserves gpu ** PA.pts_to input contents
  requires PA.pts_to output 'old
  ensures PA.pts_to output contents
{
  pts_to_len output;
  let mut cursor = 0sz;
  while (!cursor <^ n)
    invariant exists* (j:sz{j <= n}) (values:lseq a (SZ.v n)).
      cursor |-> j ** PA.pts_to output values **
      pure (forall (k:nat{k < SZ.v j}).
        Seq.index values k == Seq.index contents k)
    decreases (n - !cursor)
  {
    let j = !cursor;
    let value = input.(j);
    output.(j) <- value;
    cursor := j +^ 1sz;
  };
  with values. assert PA.pts_to output values;
  Seq.lemma_eq_intro values contents;
  Seq.lemma_eq_elim values contents;
}

inline_for_extraction noextract
fn select_index
  (n:sz{0 < SZ.v n /\ SZ.v n <= 2147483648})
  (#row:erased (row_values (SZ.v n)))
  (input:larray f32 (SZ.v n))
  (output:larray i32 1)
  preserves gpu ** PA.pts_to input row
  requires PA.pts_to output 'old
  ensures exists* (selected:selected_values (SZ.v n)).
    PA.pts_to output selected **
    pure (selected_index selected ==
      Shared.top1_index (logical_input row 0))
{
  let mut next = 1sz;
  let mut chosen = 0sz;
  while (!next <^ n)
    invariant exists*
      (j:sz{0 < SZ.v j /\ j <= n})
      (best:sz{best < j}).
      next |-> j ** chosen |-> best **
      pure (SZ.v best == Shared.best_prefix (logical_input row 0) (SZ.v j))
    decreases (n - !next)
  {
    let j = !next;
    let best = !chosen;
    let previous = input.(best);
    let candidate = input.(j);
    prefix_step row (SZ.v j);
    let updated:sz = (if FP.lt #f32 previous candidate then j else best);
    chosen := updated;
    next := j +^ 1sz;
  };
  let best = !chosen;
  let result = FStar.Int.Cast.uint32_to_int32 (SZ.sizet_to_u32 best);
  assert pure (I32.v result == SZ.v best);
  pts_to_len output;
  output.(0sz) <- result;
}

inline_for_extraction noextract
fn select_and_copy
  (experts:sz{0 < SZ.v experts /\ SZ.v experts <= 2147483648})
  (slots history:sz)
  (#row0:erased (row_values (SZ.v experts)))
  (#state0:erased
    (routing_state (SZ.v experts) (SZ.v slots) (SZ.v history)))
  (#old_selected:erased (seq i32))
  (#old_row:erased (seq f32))
  (#old_counts #old_indices:erased (seq i64))
  (#old_probabilities:erased (seq f32))
  (#old_survival:erased (seq i32))
  (input_row:device_array f32 (SZ.v experts))
  (input_counts:device_array i64 (SZ.v experts))
  (input_indices:device_array i64 (SZ.v slots))
  (input_probabilities:device_array f32 (SZ.v slots))
  (input_survival:device_array i32 (SZ.v history))
  (output_selected:device_array i32 1)
  (output_row:device_array f32 (SZ.v experts))
  (output_counts:device_array i64 (SZ.v experts))
  (output_indices:device_array i64 (SZ.v slots))
  (output_probabilities:device_array f32 (SZ.v slots))
  (output_survival:device_array i32 (SZ.v history))
  preserves gpu
  requires
    PA.pts_to input_row row0 **
    PA.pts_to input_counts state0.counts **
    PA.pts_to input_indices state0.indices **
    PA.pts_to input_probabilities state0.probabilities **
    PA.pts_to input_survival state0.survival **
    PA.pts_to output_selected old_selected **
    PA.pts_to output_row old_row **
    PA.pts_to output_counts old_counts **
    PA.pts_to output_indices old_indices **
    PA.pts_to output_probabilities old_probabilities **
    PA.pts_to output_survival old_survival
  ensures exists* (selected:selected_values (SZ.v experts)).
    PA.pts_to input_row row0 **
    PA.pts_to input_counts state0.counts **
    PA.pts_to input_indices state0.indices **
    PA.pts_to input_probabilities state0.probabilities **
    PA.pts_to input_survival state0.survival **
    PA.pts_to output_selected selected **
    PA.pts_to output_row row0 **
    PA.pts_to output_counts state0.counts **
    PA.pts_to output_indices state0.indices **
    PA.pts_to output_probabilities state0.probabilities **
    PA.pts_to output_survival state0.survival **
    pure (semantic_post row0 state0 selected row0 state0)
{
  select_index experts #row0 input_row output_selected;
  with selected. assert PA.pts_to output_selected selected;
  copy_exact experts
    #(hide (reveal row0 <: lseq f32 (SZ.v experts))) input_row output_row;
  copy_exact experts #state0.counts input_counts output_counts;
  copy_exact slots #state0.indices input_indices output_indices;
  copy_exact slots #state0.probabilities input_probabilities output_probabilities;
  copy_exact history #state0.survival input_survival output_survival;
  Shared.best_prefix_correct (logical_input row0 0) (SZ.v experts);
  copied_selection_refines row0 state0 selected;
}

inline_for_extraction noextract
instance send_global_contents
  (#a:Type0)
  (arr:array a{is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (PA.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (experts:u64{0 < U64.v experts /\ U64.v experts <= 2147483648})
  (slots:u64{
    U64.v slots < 4294967296 /\ U64.v slots % U64.v experts == 0
  })
  (history:u64{U64.v history < 4294967296})
  (token:i64{
    0 <= I64.v token /\
    U64.v history == (I64.v token + 1) * U64.v experts
  })
  (#row0:erased (row_values (U64.v experts)))
  (#state0:erased
    (routing_state (U64.v experts) (U64.v slots) (U64.v history)))
  (input_row:device_array f32 (U64.v experts))
  (input_counts:device_array i64 (U64.v experts))
  (input_indices:device_array i64 (U64.v slots))
  (input_probabilities:device_array f32 (U64.v slots))
  (input_survival:device_array i32 (U64.v history))
  (output_selected:device_array i32 1)
  (output_row:device_array f32 (U64.v experts))
  (output_counts:device_array i64 (U64.v experts))
  (output_indices:device_array i64 (U64.v slots))
  (output_probabilities:device_array f32 (U64.v slots))
  (output_survival:device_array i32 (U64.v history))
  requires cpu ** on gpu_loc (
    PA.pts_to input_row row0 **
    PA.pts_to input_counts state0.counts **
    PA.pts_to input_indices state0.indices **
    PA.pts_to input_probabilities state0.probabilities **
    PA.pts_to input_survival state0.survival **
    PA.pts_to output_selected 'old_selected **
    PA.pts_to output_row 'old_row **
    PA.pts_to output_counts 'old_counts **
    PA.pts_to output_indices 'old_indices **
    PA.pts_to output_probabilities 'old_probabilities **
    PA.pts_to output_survival 'old_survival)
  ensures exists* (selected:selected_values (U64.v experts)).
    cpu ** on gpu_loc (
      PA.pts_to input_row row0 **
      PA.pts_to input_counts state0.counts **
      PA.pts_to input_indices state0.indices **
      PA.pts_to input_probabilities state0.probabilities **
      PA.pts_to input_survival state0.survival **
      PA.pts_to output_selected selected **
      PA.pts_to output_row row0 **
      PA.pts_to output_counts state0.counts **
      PA.pts_to output_indices state0.indices **
      PA.pts_to output_probabilities state0.probabilities **
      PA.pts_to output_survival state0.survival) **
    pure (semantic_post row0 state0 selected row0 state0)
{
  SZ.fits_ok (U64.v experts);
  SZ.fits_ok (U64.v slots);
  SZ.fits_ok (U64.v history);
  let width = FStar.SizeT.uint64_to_sizet experts;
  let packed = FStar.SizeT.uint64_to_sizet slots;
  let rows = FStar.SizeT.uint64_to_sizet history;
  launch_kernel_1 (fun () ->
    select_and_copy width packed rows
      #(hide (reveal row0 <: row_values (SZ.v width)))
      #(hide (reveal state0 <:
        routing_state (SZ.v width) (SZ.v packed) (SZ.v rows)))
      #'old_selected #'old_row #'old_counts #'old_indices
      #'old_probabilities #'old_survival
      input_row input_counts input_indices input_probabilities input_survival
      output_selected output_row output_counts output_indices
      output_probabilities output_survival);
}
