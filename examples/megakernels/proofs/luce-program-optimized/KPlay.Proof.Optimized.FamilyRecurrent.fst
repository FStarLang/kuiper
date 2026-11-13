module KPlay.Proof.Optimized.FamilyRecurrent

#lang-pulse

open Kuiper
module S = KPlay.Proof.Shared.Recurrent
module Seq = FStar.Seq
module Approx = Kuiper.Approximates.Base
module I64 = FStar.Int64

// These are model dimensions, not dimensions inferred from a trace.
inline_for_extraction noextract
let hidden_width : pos = 1024

inline_for_extraction noextract
let intermediate_width : pos = 3584

inline_for_extraction noextract
let heads : pos = 16

inline_for_extraction noextract
let head_width : pos = 128

inline_for_extraction noextract
let conv_width : pos = 4

inline_for_extraction noextract
let mixer_width : pos = heads * head_width

inline_for_extraction noextract
let qkv_width : pos = 3 * mixer_width

inline_for_extraction noextract
let recurrent_count : pos = heads * head_width * head_width

inline_for_extraction noextract
let convolution_count : pos = qkv_width * conv_width

inline_for_extraction noextract
let input_norm_offset : nat = 0

inline_for_extraction noextract
let post_norm_offset : nat = input_norm_offset + hidden_width

inline_for_extraction noextract
let mlp_gate_offset : nat = post_norm_offset + hidden_width

inline_for_extraction noextract
let mlp_up_offset : nat = mlp_gate_offset + intermediate_width * hidden_width

inline_for_extraction noextract
let mlp_down_offset : nat = mlp_up_offset + intermediate_width * hidden_width

inline_for_extraction noextract
let qkv_offset : nat = mlp_down_offset + hidden_width * intermediate_width

inline_for_extraction noextract
let z_offset : nat = qkv_offset + qkv_width * hidden_width

inline_for_extraction noextract
let beta_offset : nat = z_offset + mixer_width * hidden_width

inline_for_extraction noextract
let a_offset : nat = beta_offset + heads * hidden_width

inline_for_extraction noextract
let conv_kernel_offset : nat = a_offset + heads * hidden_width

inline_for_extraction noextract
let a_log_offset : nat = conv_kernel_offset + convolution_count

inline_for_extraction noextract
let dt_bias_offset : nat = a_log_offset + heads

inline_for_extraction noextract
let gated_norm_offset : nat = dt_bias_offset + heads

inline_for_extraction noextract
let out_projection_offset : nat = gated_norm_offset + head_width

inline_for_extraction noextract
let weight_count : pos = out_projection_offset + hidden_width * mixer_width

inline_for_extraction noextract
let valid_position (position:int) : prop =
  0 <= position /\ position < 2048

inline_for_extraction noextract
let previous_elements (position:int) (count:nat) : nat =
  if position = 0 then 0 else count

inline_for_extraction noextract
let physical_extents ()
  : Lemma
      (weight_count == 21555360 /\
       recurrent_count == 262144 /\ convolution_count == 24576 /\
       mlp_gate_offset == 2048 /\ mlp_up_offset == 3672064 /\
       mlp_down_offset == 7342080 /\ qkv_offset == 11012096 /\
       z_offset == 17303552 /\ beta_offset == 19400704 /\
       a_offset == 19417088 /\ conv_kernel_offset == 19433472 /\
       a_log_offset == 19458048 /\ dt_bias_offset == 19458064 /\
       gated_norm_offset == 19458080 /\ out_projection_offset == 19458208)
  = ()

inline_for_extraction noextract
let row_major (#rows:nat) (#columns:pos)
  (row:S.fin rows) (column:S.fin columns)
  : S.fin (rows * columns)
  = row * columns + column

inline_for_extraction noextract
let convolution_index (channel:S.fin qkv_width) (tap:S.fin conv_width)
  : S.fin convolution_count
  = row_major channel tap

inline_for_extraction noextract
let recurrent_index
  (head:S.fin heads) (key value:S.fin head_width)
  : S.fin recurrent_count
  = (head * head_width + value) * head_width + key

inline_for_extraction noextract
let convolution_index_covers (i:S.fin convolution_count)
  : Lemma (convolution_index (i / conv_width) (i % conv_width) == i)
  = ()

inline_for_extraction noextract
let recurrent_index_covers (i:S.fin recurrent_count)
  : Lemma
      (recurrent_index (i / (head_width * head_width))
         (i % head_width) ((i / head_width) % head_width) == i)
  = ()

inline_for_extraction noextract
let bf16_vector_representation (#n:nat)
  (physical:lseq bf16 n) (logical:S.vector n)
  : GTot prop
  = forall (i:S.fin n). Approx.v_approximates (Seq.index physical i) (logical i)

inline_for_extraction noextract
let weight_vector_representation
  (width:nat) (offset:nat{offset + width <= weight_count})
  (physical:lseq bf16 weight_count) (logical:S.vector width)
  : GTot prop
  =
  forall (i:S.fin width).
    Approx.v_approximates (Seq.index physical (offset + i)) (logical i)

inline_for_extraction noextract
let weight_matrix_representation
  (rows:nat) (columns:pos)
  (offset:nat{offset + rows * columns <= weight_count})
  (physical:lseq bf16 weight_count) (logical:S.matrix rows columns)
  : GTot prop
  =
  forall (row:S.fin rows) (column:S.fin columns).
    Approx.v_approximates
      (Seq.index physical (offset + row_major row column)) (logical row column)

inline_for_extraction noextract
let weights_representation
  (physical:lseq bf16 weight_count) (logical:S.weights)
  : GTot prop
  =
  weight_vector_representation hidden_width input_norm_offset
    physical logical.input_norm /\
  weight_vector_representation hidden_width post_norm_offset
    physical logical.post_attention_norm /\
  weight_matrix_representation qkv_width hidden_width qkv_offset
    physical logical.in_qkv /\
  weight_matrix_representation mixer_width hidden_width z_offset
    physical logical.in_z /\
  weight_matrix_representation heads hidden_width beta_offset
    physical logical.in_beta /\
  weight_matrix_representation heads hidden_width a_offset
    physical logical.in_a /\
  weight_matrix_representation qkv_width conv_width conv_kernel_offset
    physical logical.conv_kernel /\
  weight_vector_representation heads a_log_offset physical logical.a_log /\
  weight_vector_representation heads dt_bias_offset physical logical.dt_bias /\
  weight_vector_representation head_width gated_norm_offset
    physical logical.gated_norm /\
  weight_matrix_representation hidden_width mixer_width out_projection_offset
    physical logical.out_projection /\
  weight_matrix_representation intermediate_width hidden_width mlp_gate_offset
    physical logical.mlp_gate /\
  weight_matrix_representation intermediate_width hidden_width mlp_up_offset
    physical logical.mlp_up /\
  weight_matrix_representation hidden_width intermediate_width mlp_down_offset
    physical logical.mlp_down

inline_for_extraction noextract
let convolution_representation
  (physical:lseq f32 convolution_count) (logical:S.convolution_state)
  : GTot prop
  =
  forall (channel:S.fin qkv_width) (tap:S.fin conv_width).
    Approx.v_approximates
      (Seq.index physical (convolution_index channel tap)) (logical channel tap)

inline_for_extraction noextract
let recurrent_representation
  (physical:lseq f32 recurrent_count) (logical:S.recurrent_state)
  : GTot prop
  =
  forall (head:S.fin heads) (key value:S.fin head_width).
    Approx.v_approximates
      (Seq.index physical (recurrent_index head key value)) (logical head key value)

// The candidate initializes absent histories at their first consumer.
inline_for_extraction noextract
let old_convolution
  (call:S.invocation) (before:S.model_cache{S.cache_domain call before})
  : GTot S.convolution_state
  =
  if S.has_previous_state before
  then S.present_value (before.conv_slots call.layer_index)
  else S.zero_convolution ()

inline_for_extraction noextract
let old_recurrent
  (call:S.invocation) (before:S.model_cache{S.cache_domain call before})
  : GTot S.recurrent_state
  =
  if S.has_previous_state before
  then S.present_value (before.recurrent_slots call.layer_index)
  else S.zero_recurrent ()

inline_for_extraction noextract
let input_representation
  (call:S.invocation) (weights:S.weights) (hidden:S.vector hidden_width)
  (before:S.model_cache{S.cache_domain call before})
  (physical_weights:lseq bf16 weight_count)
  (physical_hidden:lseq bf16 hidden_width)
  (position:int)
  (physical_recurrent:lseq f32 (previous_elements position recurrent_count))
  (physical_convolution:lseq f32 (previous_elements position convolution_count))
  : GTot prop
  =
  call.cache_position == position /\
  weights_representation physical_weights weights /\
  bf16_vector_representation physical_hidden hidden /\
  (if position = 0 then True
   else
     recurrent_representation physical_recurrent (old_recurrent call before) /\
     convolution_representation physical_convolution (old_convolution call before))

// The physical ABI is hidden, recurrent, convolution, not shared-output order.
inline_for_extraction noextract
let output_representation
  (physical_hidden:lseq bf16 hidden_width)
  (physical_recurrent:lseq f32 recurrent_count)
  (physical_convolution:lseq f32 convolution_count)
  (logical:S.layer_values)
  : GTot prop
  =
  bf16_vector_representation physical_hidden logical.hidden_value /\
  recurrent_representation physical_recurrent logical.recurrent_value /\
  convolution_representation physical_convolution logical.convolution_value

inline_for_extraction noextract
let logical_result
  (call:S.invocation) (before:S.model_cache) (values:S.layer_values)
  : GTot S.result
  =
  {
    S.hidden = values.hidden_value;
    S.cache = S.install_state call.layer_index before values;
  }

inline_for_extraction noextract
let logical_result_frames_cache
  (call:S.invocation) (before:S.model_cache) (values:S.layer_values)
  : Lemma
      (S.cache_frame call.layer_index before (logical_result call before values).cache)
  = ()

inline_for_extraction noextract
let call_postcondition
  (call:S.invocation) (weights:S.weights) (hidden:S.vector hidden_width)
  (before:S.model_cache{S.cache_domain call before})
  (physical_hidden:lseq bf16 hidden_width)
  (physical_recurrent:lseq f32 recurrent_count)
  (physical_convolution:lseq f32 convolution_count)
  : GTot prop
  =
  exists (values:S.layer_values).
    output_representation physical_hidden physical_recurrent physical_convolution values /\
    S.postcondition call weights hidden before (logical_result call before values)

inline_for_extraction noextract
let decoder_values_establish_call_postcondition
  (call:S.invocation) (weights:S.weights) (hidden:S.vector hidden_width)
  (before:S.model_cache{S.cache_domain call before})
  (physical_hidden:lseq bf16 hidden_width)
  (physical_recurrent:lseq f32 recurrent_count)
  (physical_convolution:lseq f32 convolution_count)
  : Lemma
      (requires
        output_representation physical_hidden physical_recurrent physical_convolution
          (S.decoder_values call weights hidden before))
      (ensures
        call_postcondition call weights hidden before
          physical_hidden physical_recurrent physical_convolution)
  =
  let expected = S.decoder_values call weights hidden before in
  S.recurrent_decoder_satisfies_postcondition call weights hidden before;
  assert (S.postcondition call weights hidden before (logical_result call before expected));
  ()

// Quantification retains caller context without adding runtime arguments or
// choosing a single to_real image of an approximately represented live state.
inline_for_extraction noextract
let semantic_postcondition
  (physical_weights:lseq bf16 weight_count)
  (physical_hidden:lseq bf16 hidden_width)
  (position:int)
  (physical_recurrent:lseq f32 (previous_elements position recurrent_count))
  (physical_convolution:lseq f32 (previous_elements position convolution_count))
  (next_hidden:lseq bf16 hidden_width)
  (next_recurrent:lseq f32 recurrent_count)
  (next_convolution:lseq f32 convolution_count)
  : GTot prop
  =
  forall (call:S.invocation) (weights:S.weights) (hidden:S.vector hidden_width)
         (before:S.model_cache{S.cache_domain call before}).
    input_representation call weights hidden before
      physical_weights physical_hidden position physical_recurrent physical_convolution ==>
    call_postcondition call weights hidden before next_hidden next_recurrent next_convolution

inline_for_extraction noextract
let first_use_is_zero
  (call:S.invocation{call.cache_position == 0})
  (before:S.model_cache{S.cache_domain call before})
  : Lemma
      (old_convolution call before == S.zero_convolution () /\
       old_recurrent call before == S.zero_recurrent ())
  = ()
