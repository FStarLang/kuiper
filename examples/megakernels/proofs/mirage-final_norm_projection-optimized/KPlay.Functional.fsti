module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module R = FStar.Real
module Sqrt = FStar.Math.Sqrt
module U32 = FStar.UInt32
module U64 = FStar.UInt64
module SZ = Kuiper.SizeT
module Shared = KPlay.SharedSpec

inline_for_extraction noextract
let epsilon : Sqrt.rpos = 0.000001R

(* Exact real summation; no floating-point reduction tree is specified. *)
inline_for_extraction noextract
let rec sum_squares
  (x : seq real)
  (start : nat)
  (count : nat{start + count <= S.length x})
  : GTot Sqrt.rnonneg (decreases count)
  = if count = 0 then 0.0R
    else
      let v = S.index x (start + count - 1) in
      sum_squares x start (count - 1) +. v *. v

inline_for_extraction noextract
let rms_scale
  (x : seq real)
  (start : nat)
  (width : pos{start + width <= S.length x})
  : GTot Sqrt.rpos
  = Sqrt.rsqrt (sum_squares x start width /. R.of_int width +. epsilon)

inline_for_extraction noextract
let normalized_cell
  (input_length : nat)
  (width : pos)
  (rows : nat{rows * width <= input_length})
  (x : lseq real input_length)
  (gamma : lseq real width)
  (row : nat{row < rows})
  (column : nat{column < width})
  : GTot real
  = S.index x (row * width + column) *.
      (S.index gamma column *. rms_scale x (row * width) width)

inline_for_extraction noextract
let padded_weight
  (width : pos)
  (vocab_rows : nat)
  (weights : lseq real (vocab_rows * width))
  (vocab_row : nat)
  (column : nat{column < width})
  : GTot real
  = if vocab_row < vocab_rows
    then S.index weights (vocab_row * width + column)
    else 0.0R

inline_for_extraction noextract
let projected_cell
  (width : pos)
  (vocab_rows rows : nat)
  (hidden : lseq real (rows * width))
  (weights : lseq real (vocab_rows * width))
  (row : nat{row < rows})
  (vocab_row : nat)
  : GTot real
  = rsum (S.init_ghost width (fun column ->
      S.index hidden (row * width + column) *.
        padded_weight width vocab_rows weights vocab_row column))

inline_for_extraction noextract
let normalization_post
  (input_length : nat)
  (width : pos)
  (rows : nat{rows * width <= input_length})
  (x : lseq bf16 input_length)
  (gamma : lseq bf16 width)
  (hidden : lseq bf16 (rows * width))
  : prop
  = forall (rx : lseq real input_length) (rgamma : lseq real width).
      x %~ rx /\ gamma %~ rgamma ==>
      (forall (row : nat{row < rows}) (column : nat{column < width}).
        S.index hidden (row * width + column) %~
          normalized_cell input_length width rows rx rgamma row column)

(* Projection models the returned BF16 hidden buffer, not an unrounded shortcut. *)
inline_for_extraction noextract
let projection_post
  (width : pos)
  (vocab_rows rows : nat)
  (columns : nat{vocab_rows <= columns})
  (hidden : lseq bf16 (rows * width))
  (weights : lseq bf16 (vocab_rows * width))
  (scores : lseq bf16 (rows * columns))
  : prop
  = forall (rhidden : lseq real (rows * width))
           (rweights : lseq real (vocab_rows * width)).
      hidden %~ rhidden /\ weights %~ rweights ==>
      (forall (row : nat{row < rows}) (column : nat{column < columns}).
        S.index scores (row * columns + column) %~
          projected_cell width vocab_rows rows rhidden rweights row column)

inline_for_extraction noextract
let final_norm_projection_post
  (input_length : nat)
  (width : pos)
  (vocab_rows : nat)
  (rows : nat{rows * width <= input_length})
  (columns : nat{vocab_rows <= columns})
  (x : lseq bf16 input_length)
  (gamma : lseq bf16 width)
  (weights : lseq bf16 (vocab_rows * width))
  (hidden : lseq bf16 (rows * width))
  (scores : lseq bf16 (rows * columns))
  : prop
  = normalization_post input_length width rows x gamma hidden /\
    projection_post width vocab_rows rows columns hidden weights scores

inline_for_extraction noextract
let addressable
  (input_length width vocab_rows rows columns : nat)
  : prop
  = SZ.fits input_length /\ SZ.fits width /\
    SZ.fits (vocab_rows * width) /\
    SZ.fits (rows * width) /\ SZ.fits (rows * columns)

(* The model fixes the logical extents; the CUDA ABI adds one input row and
  layout-3 vocabulary padding. Neither padding region restricts input values. *)
inline_for_extraction noextract
let input_elements : pos = 8 * Shared.hidden_size

inline_for_extraction noextract
let hidden_elements : pos = Shared.prompt_length * Shared.hidden_size

inline_for_extraction noextract
let embedding_elements : pos = Shared.vocab_size * Shared.hidden_size

inline_for_extraction noextract
let score_columns : pos = 153600

inline_for_extraction noextract
let score_elements : pos = Shared.prompt_length * score_columns

inline_for_extraction noextract
type score_column = c:nat{c < score_columns}

inline_for_extraction noextract
let input_representation
 (x:lseq bf16 input_elements)
 (gamma:lseq bf16 Shared.hidden_size)
 (weights:lseq bf16 embedding_elements)
 (rx:lseq real input_elements)
 (rgamma:lseq real Shared.hidden_size)
 (rweights:lseq real embedding_elements) : prop =
 x %~ rx /\ gamma %~ rgamma /\ weights %~ rweights

inline_for_extraction noextract
let native_denotation (#n:nat) (values:lseq bf16 n) : GTot (lseq real n) =
 S.init_ghost n (fun i -> to_real (S.index values i))

inline_for_extraction noextract
let canonical_input_is_represented
 (x:lseq bf16 input_elements)
 (gamma:lseq bf16 Shared.hidden_size)
 (weights:lseq bf16 embedding_elements)
 : Lemma (input_representation x gamma weights
     (native_denotation x) (native_denotation gamma) (native_denotation weights))
 = ()

inline_for_extraction noextract
let map_inputs
 (rx:lseq real input_elements)
 (rgamma:lseq real Shared.hidden_size)
 (rweights:lseq real embedding_elements) : GTot Shared.inputs =
 {
   hidden_in = (fun p d -> S.index rx (p * Shared.hidden_size + d));
   norm_weight = (fun d -> S.index rgamma d);
   embedding = (fun v d -> S.index rweights (v * Shared.hidden_size + d))
 }

inline_for_extraction noextract
noeq type output_denotation = {
 hidden: Shared.hidden_rows;
 scores: Shared.prompt_index -> score_column -> GTot real
}

inline_for_extraction noextract
let output_representation
 (hidden:lseq bf16 hidden_elements)
 (scores:lseq bf16 score_elements)
 (logical:output_denotation) : GTot prop =
 (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
   S.index hidden (p * Shared.hidden_size + d) %~ logical.hidden p d) /\
 (forall (p:Shared.prompt_index) (c:score_column).
   S.index scores (p * score_columns + c) %~ logical.scores p c)

inline_for_extraction noextract
let map_outputs (logical:output_denotation) : GTot Shared.outputs =
 {
   normalized_hidden = logical.hidden;
   final_logits = (fun v -> logical.scores (Shared.prompt_length - 1) v)
 }

inline_for_extraction noextract
let map_state
 (x:lseq bf16 input_elements)
 (gamma:lseq bf16 Shared.hidden_size)
 (weights:lseq bf16 embedding_elements)
 : GTot (seq bf16 & seq bf16 & seq bf16) =
 (x, gamma, weights)

inline_for_extraction noextract
let all_scores (x:Shared.inputs) (observed:output_denotation) : GTot prop =
 forall (p:Shared.prompt_index) (c:score_column).
   observed.scores p c ==
     (if c < Shared.vocab_size
      then Shared.tied_projection x.embedding (observed.hidden p) c
      else 0.0R)

inline_for_extraction noextract
let expected_outputs (x:Shared.inputs) : GTot output_denotation =
 let hidden = Shared.final_rms_norm x in
 {
   hidden = hidden;
   scores = (fun p c ->
     if c < Shared.vocab_size
     then Shared.tied_projection x.embedding (hidden p) c
     else 0.0R)
 }

inline_for_extraction noextract
let expected_outputs_refine_shared (#ambient:Type0)
 (x:Shared.inputs) (state:ambient)
 : Lemma (
     Shared.postcondition x state (map_outputs (expected_outputs x)) state /\
     all_scores x (expected_outputs x))
 = ()

(* Quantifying over every related input denotation permits producer composition.
  The existential observation must represent the actual computed buffers. *)
inline_for_extraction noextract
let shared_refinement_post
 (x:lseq bf16 input_elements)
 (gamma:lseq bf16 Shared.hidden_size)
 (weights:lseq bf16 embedding_elements)
 (hidden:lseq bf16 hidden_elements)
 (scores:lseq bf16 score_elements) : GTot prop =
 forall (rx:lseq real input_elements)
        (rgamma:lseq real Shared.hidden_size)
        (rweights:lseq real embedding_elements).
   input_representation x gamma weights rx rgamma rweights ==>
   (exists (observed:output_denotation).
     output_representation hidden scores observed /\
     Shared.postcondition (map_inputs rx rgamma rweights)
       (map_state x gamma weights) (map_outputs observed)
       (map_state x gamma weights) /\
     all_scores (map_inputs rx rgamma rweights) observed)

fn kplay_run
 (input_length : u64{U64.v input_length == input_elements})
 (width : u64{U64.v width == Shared.hidden_size})
 (vocab_rows : u64{U64.v vocab_rows == Shared.vocab_size})
 (rows : u32{U32.v rows == Shared.prompt_length})
 (columns : u32{U32.v columns == score_columns})
  (input0 : larray bf16 (U64.v input_length){is_global_array input0})
  (input1 : larray bf16 (U64.v width){is_global_array input1})
  (input2 : larray bf16 (U64.v vocab_rows * U64.v width){is_global_array input2})
  (output0 : larray bf16 (U32.v rows * U64.v width){is_global_array output0})
  (output1 : larray bf16 (U32.v rows * U32.v columns){is_global_array output1})
  (#x : erased (lseq bf16 (U64.v input_length)))
  (#gamma : erased (lseq bf16 (U64.v width)))
  (#weights : erased (lseq bf16 (U64.v vocab_rows * U64.v width)))
  (#old_hidden : erased (lseq bf16 (U32.v rows * U64.v width)))
  (#old_scores : erased (lseq bf16 (U32.v rows * U32.v columns)))
  (#fx #fgamma #fweights : perm)
  preserves cpu
  preserves on gpu_loc (
    input0 |-> Frac fx x **
    input1 |-> Frac fgamma gamma **
    input2 |-> Frac fweights weights)
  requires pure (addressable
    (U64.v input_length) (U64.v width) (U64.v vocab_rows)
    (U32.v rows) (U32.v columns))
  requires on gpu_loc (output0 |-> old_hidden ** output1 |-> old_scores)
  ensures exists* (hidden : lseq bf16 (U32.v rows * U64.v width))
                  (scores : lseq bf16 (U32.v rows * U32.v columns)).
    on gpu_loc (output0 |-> hidden ** output1 |-> scores) **
    pure (final_norm_projection_post
      (U64.v input_length) (U64.v width) (U64.v vocab_rows)
      (U32.v rows) (U32.v columns) x gamma weights hidden scores /\
      shared_refinement_post x gamma weights hidden scores)
