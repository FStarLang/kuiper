module KPlay.Functional

#lang-pulse

open Kuiper

module Shared = KPlay.SharedSpec
module C = Kuiper.Chest
module S = FStar.Seq
module A = Kuiper.Approximates.Base
module FP = Kuiper.Floating
module SM = Kuiper.Spec.Softmax
module I64 = FStar.Int64
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module KA = Kuiper.Array
module PA = Pulse.Lib.Array
module Cast = Kuiper.Float.Casts
module Fast = Kuiper.Float32.FastMath

inline_for_extraction noextract
fn fill_all
  (#a:Type0)
  (n:sz)
  (output:larray a (SZ.v n))
  (#old:erased (lseq a (SZ.v n)))
  (value:a)
  preserves gpu
  requires PA.pts_to output old
  ensures PA.pts_to output (S.create (SZ.v n) value)
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (values:lseq a (SZ.v n)).
      i |-> j ** PA.pts_to output values **
      pure (forall (k:natlt (SZ.v j)). S.index values k == value)
    decreases (n - !i)
  {
    let j = !i;
    output.(j) <- value;
    i := j +^ 1sz;
  };
  with values. assert PA.pts_to output values;
  assert pure (S.equal values (S.create (SZ.v n) value));
}

inline_for_extraction noextract
fn copy_prefix
  (#a:Type0)
  (#m:erased nat)
  (n:sz{SZ.v n <= m})
  (input:larray a (SZ.v n))
  (output:larray a m)
  (#source:erased (lseq a (SZ.v n)))
  (#old:erased (lseq a m))
  preserves gpu ** PA.pts_to input source
  requires PA.pts_to output old
  ensures exists* (values:lseq a m).
    PA.pts_to output values **
    pure (forall (k:natlt (SZ.v n)).
      S.index values k == S.index source k)
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (values:lseq a m).
      i |-> j ** PA.pts_to output values **
      pure (forall (k:natlt (SZ.v j)).
        S.index values k == S.index source k)
    decreases (n - !i)
  {
    let j = !i;
    let value = input.(j);
    output.(j) <- value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn copy_all
  (#a:Type0)
  (n:sz)
  (input output:larray a (SZ.v n))
  (#source:erased (lseq a (SZ.v n)))
  (#old:erased (lseq a (SZ.v n)))
  preserves gpu ** PA.pts_to input source
  requires PA.pts_to output old
  ensures PA.pts_to output source
{
  copy_prefix n input output #source #old;
  with values. assert PA.pts_to output values;
  assert pure (S.equal values source);
}

inline_for_extraction noextract
fn prepare_frame
  (e:szp{SZ.fits (packed_size (SZ.v e))})
  (t:sz{SZ.fits ((SZ.v t + 1) * SZ.v e)})
  (#before:erased (input_frame (SZ.v e) (SZ.v t)))
  (input_counts:inactive_or_device_array i64 (active_size (SZ.v t) (SZ.v e)))
  (input_indices:inactive_or_device_array i64
    (active_size (SZ.v t) (packed_size (SZ.v e))))
  (input_probabilities:inactive_or_device_array f32
    (active_size (SZ.v t) (packed_size (SZ.v e))))
  (input_survival:inactive_or_device_array i32 (SZ.v t * SZ.v e))
  (output_counts:device_array i64 (SZ.v e))
  (output_indices:device_array i64 (packed_size (SZ.v e)))
  (output_probabilities:device_array f32 (packed_size (SZ.v e)))
  (output_survival:device_array i32 ((SZ.v t + 1) * SZ.v e))
  preserves gpu **
    input_frame_ownership input_counts input_indices input_probabilities input_survival before
  requires
    PA.pts_to output_counts 'old_counts **
    PA.pts_to output_indices 'old_indices **
    PA.pts_to output_probabilities 'old_probabilities **
    PA.pts_to output_survival 'old_survival
  ensures exists* (survival:lseq i32 ((SZ.v t + 1) * SZ.v e)).
    PA.pts_to output_counts (next_frame before).counts **
    PA.pts_to output_indices (next_frame before).indices **
    PA.pts_to output_probabilities (next_frame before).probabilities **
    PA.pts_to output_survival survival **
    pure (forall (k:natlt (SZ.v t * SZ.v e)).
      S.index survival k == S.index before.previous_survival k)
{
  pts_to_len output_counts;
  pts_to_len output_indices;
  pts_to_len output_probabilities;
  pts_to_len output_survival;
  let packed = e *^ 32sz;
  if (t = 0sz) {
    fill_all e output_counts 0L;
    fill_all packed output_indices (-1L);
    fill_all packed output_probabilities (FP.zero #f32);
  } else {
    rewrite
      (input_frame_ownership input_counts input_indices input_probabilities input_survival before)
      as
      (PA.pts_to input_counts before.previous_counts **
       PA.pts_to input_indices before.previous_indices **
       PA.pts_to input_probabilities before.previous_probabilities **
       PA.pts_to input_survival before.previous_survival);
    copy_all e input_counts output_counts;
    copy_all packed input_indices output_indices;
    copy_all packed input_probabilities output_probabilities;
    let previous = t *^ e;
    copy_prefix previous input_survival output_survival;
    rewrite
      (PA.pts_to input_counts before.previous_counts **
       PA.pts_to input_indices before.previous_indices **
       PA.pts_to input_probabilities before.previous_probabilities **
       PA.pts_to input_survival before.previous_survival)
      as
      (input_frame_ownership input_counts input_indices input_probabilities input_survival before);
  };
}

inline_for_extraction noextract
fn dot_product
  (h:szp)
  (e:szp{SZ.fits (SZ.v e * SZ.v h)})
  (j:szlt (SZ.v e))
  (#x:erased (lseq bf16 (SZ.v h)))
  (#w:erased (lseq bf16 (SZ.v e * SZ.v h)))
  (input_token:device_array bf16 (SZ.v h))
  (input_weights:device_array bf16 (SZ.v e * SZ.v h))
  preserves gpu ** PA.pts_to input_token x ** PA.pts_to input_weights w
  returns score:f32
  ensures pure (score %~ projection_prefix x w (SZ.v j) (SZ.v h))
{
  projection_prefix_zero x w (SZ.v j);
  let mut k = 0sz;
  let mut accumulator:f32 = FP.zero #f32;
  while (!k <^ h)
    invariant exists* (n:sz{n <= h}) (value:f32).
      k |-> n ** accumulator |-> value **
      pure (value %~ projection_prefix x w (SZ.v j) (SZ.v n))
    decreases (h - !k)
  {
    let n = !k;
    let xv = input_token.(n);
    let wv = input_weights.(j *^ h +^ n);
    let xf = Cast.fcast #bf16 #f32 xv;
    let wf = Cast.fcast #bf16 #f32 wv;
    A.to_real_ok xv;
    A.to_real_ok wv;
    Cast.fcast_approx #bf16 #f32 xv (A.to_real xv);
    Cast.fcast_approx #bf16 #f32 wv (A.to_real wv);
    let value = !accumulator;
    let next = Fast.fma_rn xf wf value;
    projection_prefix_step x w (SZ.v j) (SZ.v n);
    assert pure (next %~ projection_prefix x w (SZ.v j) (SZ.v n + 1));
    accumulator := next;
    k := n +^ 1sz;
  };
  !accumulator
}

inline_for_extraction noextract
fn project_row
  (h:szp)
  (e:szp{SZ.fits (SZ.v e * SZ.v h)})
  (t:sz{SZ.fits ((SZ.v t + 1) * SZ.v e)})
  (#x:erased (lseq bf16 (SZ.v h)))
  (#w:erased (lseq bf16 (SZ.v e * SZ.v h)))
  (#past:erased (lseq i32 (SZ.v t * SZ.v e)))
  (input_token:device_array bf16 (SZ.v h))
  (input_weights:device_array bf16 (SZ.v e * SZ.v h))
  (output_row:device_array f32 (SZ.v e))
  (output_survival:device_array i32 ((SZ.v t + 1) * SZ.v e))
  (#old_row:erased (lseq f32 (SZ.v e)))
  (#old_survival:erased (lseq i32 ((SZ.v t + 1) * SZ.v e)))
  preserves gpu ** PA.pts_to input_token x ** PA.pts_to input_weights w
  requires
    PA.pts_to output_row old_row **
    PA.pts_to output_survival old_survival **
    pure (forall (k:natlt (SZ.v t * SZ.v e)).
      S.index old_survival k == S.index past k)
  ensures exists* (scores:lseq f32 (SZ.v e)).
    PA.pts_to output_row scores **
    PA.pts_to output_survival (S.append past (S.create (SZ.v e) 0l)) **
    pure (C.seq_to_chest1 scores %~ projection_row x w)
{
  let offset = t *^ e;
  let mut expert = 0sz;
  while (!expert <^ e)
    invariant exists* (j:sz{j <= e})
      (scores:lseq f32 (SZ.v e))
      (survival:lseq i32 ((SZ.v t + 1) * SZ.v e)).
      expert |-> j **
      PA.pts_to output_row scores ** PA.pts_to output_survival survival **
      pure (
        (forall (k:natlt (SZ.v t * SZ.v e)).
          S.index survival k == S.index past k) /\
        (forall (q:natlt (SZ.v e)). q < SZ.v j ==>
          S.index survival (SZ.v t * SZ.v e + q) == 0l /\
          S.index scores q %~
            projection_prefix #(SZ.v h) #(SZ.v e) x w q (SZ.v h)))
    decreases (e - !expert)
  {
    let j = !expert;
    let score = dot_product h e j #x #w input_token input_weights;
    output_row.(j) <- score;
    output_survival.(offset +^ j) <- 0l;
    expert := j +^ 1sz;
  };
  with scores survival.
    assert PA.pts_to output_row scores ** PA.pts_to output_survival survival;
  assert pure (S.equal survival (S.append past (S.create (SZ.v e) 0l)));
  assert pure (C.seq_to_chest1 scores %~ projection_row x w);
}

inline_for_extraction noextract
let maximum_index_step (#e:pos) (scores:lseq f32 e) (n:pos{n < e})
  : Lemma (
      maximum_index scores (n + 1) ==
      (if FP.lt #f32 (S.index scores (maximum_index scores n)) (S.index scores n)
       then n else maximum_index scores n))
  = ()

inline_for_extraction noextract
let sum_prefix_step (#e:pos) (terms:lseq f32 e) (n:natlt e)
  : Lemma (
      sum_prefix terms (n + 1) ==
      FP.add #f32 (sum_prefix terms n) (S.index terms n))
  = ()

inline_for_extraction noextract
fn stored_maximum
  (e:szp)
  (row:device_array f32 (SZ.v e))
  (#scores:erased (lseq f32 (SZ.v e)))
  preserves gpu ** PA.pts_to row scores
  returns maximum:f32
  ensures pure (maximum == S.index scores (maximum_index scores (SZ.v e)))
{
  let first = row.(0sz);
  let mut maximum = first;
  let mut expert = 1sz;
  while (!expert <^ e)
    invariant exists* (j:sz{0 < j /\ j <= e}) (value:f32).
      expert |-> j ** maximum |-> value **
      pure (value == S.index scores (maximum_index scores (SZ.v j)))
    decreases (e - !expert)
  {
    let j = !expert;
    let current = row.(j);
    let previous = !maximum;
    maximum_index_step scores (SZ.v j);
    if (FP.lt #f32 previous current) {
      maximum := current;
    };
    expert := j +^ 1sz;
  };
  !maximum
}

inline_for_extraction noextract
fn exponentiate_row
  (e:szp)
  (row:device_array f32 (SZ.v e))
  (#scores:erased (lseq f32 (SZ.v e)))
  (maximum:f32{
    maximum == S.index scores (maximum_index scores (SZ.v e))})
  preserves gpu
  requires PA.pts_to row scores
  returns sum:f32
  ensures PA.pts_to row (exponentials scores) **
    pure (sum == sum_prefix (exponentials scores) (SZ.v e))
{
  let mut expert = 0sz;
  let mut sum:f32 = FP.zero #f32;
  while (!expert <^ e)
    invariant exists* (j:sz{j <= e}) (value:f32) (values:lseq f32 (SZ.v e)).
      expert |-> j ** sum |-> value ** PA.pts_to row values **
      pure (
        value == sum_prefix (exponentials scores) (SZ.v j) /\
        (forall (q:natlt (SZ.v e)).
          S.index values q ==
          (if q < SZ.v j then S.index (exponentials scores) q
           else S.index scores q)))
    decreases (e - !expert)
  {
    let j = !expert;
    let score = row.(j);
    let numerator = FP.fexp #f32 (FP.sub #f32 score maximum);
    row.(j) <- numerator;
    let previous = !sum;
    sum_prefix_step (exponentials scores) (SZ.v j);
    sum := FP.add #f32 previous numerator;
    expert := j +^ 1sz;
  };
  with values. assert PA.pts_to row values;
  assert pure (S.equal values (exponentials scores));
  !sum
}

inline_for_extraction noextract
fn normalize_row
  (e:szp)
  (row:device_array f32 (SZ.v e))
  (#scores:erased (lseq f32 (SZ.v e)))
  preserves gpu
  requires PA.pts_to row scores
  ensures PA.pts_to row (normalized_values scores)
{
  let maximum = stored_maximum e row #scores;
  let sum = exponentiate_row e row #scores maximum;
  let inverse = FP.div #f32 (FP.one #f32) sum;
  let mut expert = 0sz;
  while (!expert <^ e)
    invariant exists* (j:sz{j <= e}) (values:lseq f32 (SZ.v e)).
      expert |-> j ** PA.pts_to row values **
      pure (forall (q:natlt (SZ.v e)).
        S.index values q ==
        (if q < SZ.v j then S.index (normalized_values scores) q
         else S.index (exponentials scores) q))
    decreases (e - !expert)
  {
    let j = !expert;
    let numerator = row.(j);
    row.(j) <- FP.mul #f32 numerator inverse;
    expert := j +^ 1sz;
  };
  with values. assert PA.pts_to row values;
  assert pure (S.equal values (normalized_values scores));
}

inline_for_extraction noextract
fn row_kernel
  (h:szp)
  (e:szp{SZ.fits (SZ.v e * SZ.v h) /\ SZ.fits (packed_size (SZ.v e))})
  (t:sz{SZ.fits ((SZ.v t + 1) * SZ.v e)})
  (#x:erased (lseq bf16 (SZ.v h)))
  (#w:erased (lseq bf16 (SZ.v e * SZ.v h)))
  (#before:erased (input_frame (SZ.v e) (SZ.v t)))
  (#old_row:erased (seq f32))
  (#old_counts:erased (seq i64))
  (#old_indices:erased (seq i64))
  (#old_probabilities:erased (seq f32))
  (#old_survival:erased (seq i32))
  (input_token:device_array bf16 (SZ.v h))
  (input_weights:device_array bf16 (SZ.v e * SZ.v h))
  (input_counts:inactive_or_device_array i64 (active_size (SZ.v t) (SZ.v e)))
  (input_indices:inactive_or_device_array i64
    (active_size (SZ.v t) (packed_size (SZ.v e))))
  (input_probabilities:inactive_or_device_array f32
    (active_size (SZ.v t) (packed_size (SZ.v e))))
  (input_survival:inactive_or_device_array i32 (SZ.v t * SZ.v e))
  (output_row:device_array f32 (SZ.v e))
  (output_counts:device_array i64 (SZ.v e))
  (output_indices:device_array i64 (packed_size (SZ.v e)))
  (output_probabilities:device_array f32 (packed_size (SZ.v e)))
  (output_survival:device_array i32 ((SZ.v t + 1) * SZ.v e))
  preserves gpu **
    PA.pts_to input_token x ** PA.pts_to input_weights w **
    input_frame_ownership input_counts input_indices input_probabilities input_survival before
  requires pure (row_domain x w) **
    PA.pts_to output_row old_row **
    PA.pts_to output_counts old_counts **
    PA.pts_to output_indices old_indices **
    PA.pts_to output_probabilities old_probabilities **
    PA.pts_to output_survival old_survival
  ensures exists* (row:lseq f32 (SZ.v e)).
    PA.pts_to output_row row **
    PA.pts_to output_counts (next_frame before).counts **
    PA.pts_to output_indices (next_frame before).indices **
    PA.pts_to output_probabilities (next_frame before).probabilities **
    PA.pts_to output_survival (next_frame before).survival **
    pure (exists (modeled:C.chest1 real (SZ.v e)).
      semantic_post x w before x w row (next_frame before) modeled)
{
  pts_to_len output_row;
  prepare_frame e t #before
    input_counts input_indices input_probabilities input_survival
    output_counts output_indices output_probabilities output_survival;
  project_row h e t #x #w #before.previous_survival
    input_token input_weights output_row output_survival;
  with scores. assert PA.pts_to output_row scores;
  normalize_row e output_row #scores;
  completed_row_models x w before scores;
  FStar.Classical.exists_intro
    (fun modeled -> semantic_post x w before x w
      (normalized_values scores) (next_frame before) modeled)
    (SM.softmax_real (projection_row x w));
}

inline_for_extraction noextract
instance send_global_contents
  (#a:Type0)
  (arr:array a{KA.is_global_array arr})
  (#f:perm)
  (contents:seq a)
  : is_send_across gpu_of (PA.pts_to arr #f contents)
  = KA.is_send_pts_to arr #f contents

inline_for_extraction noextract
instance send_input_frame
  (#e:pos) (#t:nat)
  (counts:inactive_or_device_array i64 (active_size t e))
  (indices:inactive_or_device_array i64 (active_size t (packed_size e)))
  (probabilities:inactive_or_device_array f32 (active_size t (packed_size e)))
  (survival:inactive_or_device_array i32 (t * e))
  (contents:input_frame e t)
  : is_send_across gpu_of
      (input_frame_ownership counts indices probabilities survival contents)
  = if t = 0 then
      let result:is_send_across gpu_of emp = solve in result
    else
      let result:is_send_across gpu_of (
        PA.pts_to counts contents.previous_counts **
        PA.pts_to indices contents.previous_indices **
        PA.pts_to probabilities contents.previous_probabilities **
        PA.pts_to survival contents.previous_survival) = solve in result

fn kplay_run
  (hidden:u64{0 < U64.v hidden /\ U64.v hidden < 4294967296})
  (experts:u64{
    0 < U64.v experts /\
    U64.v experts * U64.v hidden < 4294967296 /\
    packed_size (U64.v experts) < 4294967296})
  (token:i64{
    0 <= I64.v token /\ (I64.v token + 1) * U64.v experts < 4294967296})
  (#x:erased (lseq bf16 (U64.v hidden)))
  (#w:erased (lseq bf16 (U64.v experts * U64.v hidden)))
  (#before:erased (input_frame (U64.v experts) (I64.v token)))
  (input_token:device_array bf16 (U64.v hidden))
  (input_weights:device_array bf16 (U64.v experts * U64.v hidden))
  (input_counts:inactive_or_device_array i64 (active_size (I64.v token) (U64.v experts)))
  (input_indices:inactive_or_device_array i64
    (active_size (I64.v token) (packed_size (U64.v experts))))
  (input_probabilities:inactive_or_device_array f32
    (active_size (I64.v token) (packed_size (U64.v experts))))
  (input_survival:inactive_or_device_array i32 (I64.v token * U64.v experts))
  (output_row:device_array f32 (U64.v experts))
  (output_counts:device_array i64 (U64.v experts))
  (output_indices:device_array i64 (packed_size (U64.v experts)))
  (output_probabilities:device_array f32 (packed_size (U64.v experts)))
  (output_survival:device_array i32 ((I64.v token + 1) * U64.v experts))
  requires cpu ** pure (row_domain x w) ** on gpu_loc (
    PA.pts_to input_token x ** PA.pts_to input_weights w **
    input_frame_ownership input_counts input_indices input_probabilities input_survival before **
    PA.pts_to output_row 'old_row **
    PA.pts_to output_counts 'old_counts **
    PA.pts_to output_indices 'old_indices **
    PA.pts_to output_probabilities 'old_probabilities **
    PA.pts_to output_survival 'old_survival)
  ensures exists* (row:lseq f32 (U64.v experts)).
    cpu ** on gpu_loc (
      PA.pts_to input_token x ** PA.pts_to input_weights w **
      input_frame_ownership input_counts input_indices input_probabilities input_survival before **
      PA.pts_to output_row row **
      PA.pts_to output_counts (next_frame before).counts **
      PA.pts_to output_indices (next_frame before).indices **
      PA.pts_to output_probabilities (next_frame before).probabilities **
      PA.pts_to output_survival (next_frame before).survival) **
    pure (exists (modeled:C.chest1 real (U64.v experts)).
      semantic_post x w before x w row (next_frame before) modeled)
{
  let h = FStar.SizeT.uint64_to_sizet hidden;
  let e = FStar.SizeT.uint64_to_sizet experts;
  let token32 = FStar.Int.Cast.int64_to_uint32 token;
  let t = SZ.uint32_to_sizet token32;
  launch_kernel_1 (fun () ->
    row_kernel h e t #x #w #before
      #'old_row #'old_counts #'old_indices #'old_probabilities #'old_survival
      input_token input_weights input_counts input_indices input_probabilities input_survival
      output_row output_counts output_indices output_probabilities output_survival);
}
