module KPlay.SharedSpec

open FStar.Real
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

(* The trusted base.py call and declared configuration, not test dimensions.
   The boundary is entry to the first decoder layer, not the model's return.
   All views below are logical; the singleton batch dimension is implicit. *)

inline_for_extraction noextract
let prompt_length : pos = 7

inline_for_extraction noextract
let vocab_size : pos = 151936

inline_for_extraction noextract
let hidden_size : pos = 1024

inline_for_extraction noextract
let head_dim : pos = 128

inline_for_extraction noextract
let half_dim : pos = 64

inline_for_extraction noextract
let kv_heads : pos = 8

inline_for_extraction noextract
let layer_count : pos = 28

inline_for_extraction noextract
let max_positions : pos = 40960

inline_for_extraction noextract
let rope_theta : r:real{r >. 0.0R} = 1000000.0R

inline_for_extraction noextract
let attention_factor : real = 1.0R

inline_for_extraction noextract
let prompt_index = i:nat{i < prompt_length}

inline_for_extraction noextract
let token_id = i:nat{i < vocab_size}

inline_for_extraction noextract
let hidden_index = i:nat{i < hidden_size}

inline_for_extraction noextract
let rotary_index = i:nat{i < head_dim}

inline_for_extraction noextract
let frequency_index = i:nat{i < half_dim}

inline_for_extraction noextract
let kv_head_index = i:nat{i < kv_heads}

inline_for_extraction noextract
let layer_index = i:nat{i < layer_count}

(* The datum type is opaque, so lookup preserves even exceptional BF16
   representations. No real-valued/finite-weight precondition is imposed.
   The physical-to-logical interpretation of the input is role-specific. *)
noextract noeq
type inputs (a:Type0) = {
  embedding_weight: token_id -> hidden_index -> a;
  tokens: prompt_index -> token_id;
  decode_steps: n:nat{n == 1}
}

noextract noeq
type kv_row (a:Type0) = {
  key: kv_head_index -> rotary_index -> a;
  value: kv_head_index -> rotary_index -> a
}

noextract noeq
type decoder_entry_state (a:Type0) = {
  layer_history: layer_index -> list (kv_row a);
  inv_freq: frequency_index -> real;
  original_inv_freq: frequency_index -> real;
  attention_scaling: real;
  max_seq_len_cached: nat;
  original_max_seq_len: nat
}

(* Field order agrees with the logical output indices in M.json.
   Functions are finite indexed views, not buffer or storage requirements. *)
noextract noeq
type outputs (a:Type0) = {
  hidden_rows: prompt_index -> hidden_index -> a;
  position_ids: prompt_index -> nat;
  cache_positions: prompt_index -> nat;
  causal_bias: prompt_index -> prompt_index -> real;
  rope_cos: prompt_index -> rotary_index -> real;
  rope_sin: prompt_index -> rotary_index -> real;
  decoder_state: decoder_entry_state a
}

(* Algebraic FP model: these are real targets, not rounded native values.
   Arithmetic/cast approximation belongs to each role's representation
   proof; no error bounds or universal bitwise arithmetic equality is used.
   Pow and Trig are existing authorized mathematical library APIs. *)
inline_for_extraction noextract
let inverse_frequency (r:frequency_index) : real =
  1.0R /. Pow.pow rope_theta (of_int (2 * r) /. of_int head_dim)

inline_for_extraction noextract
let frequency_of_coordinate (d:rotary_index) : frequency_index =
  if d < half_dim then d else d - half_dim

inline_for_extraction noextract
let angle (p:prompt_index) (d:rotary_index) : real =
  inverse_frequency (frequency_of_coordinate d) *. of_int p

inline_for_extraction noextract
let cosine (p:prompt_index) (d:rotary_index) : real =
  Trig.cos (angle p d) *. attention_factor

inline_for_extraction noextract
let sine (p:prompt_index) (d:rotary_index) : real =
  Trig.sin (angle p d) *. attention_factor

(* eager_mask uses finfo(BF16).min, not negative infinity:
   -(2 - 2^(-7)) * 2^127 = -255 * 2^120. *)
inline_for_extraction noextract
let bf16_min : real =
  0.0R -. 338953138925153547590470800371487866880.0R

inline_for_extraction noextract
let full_attention_bias (p k:prompt_index) : real =
  if k <= p then 0.0R else bf16_min

(* Fresh DynamicCache has no logical entries. Default RoPE leaves its
   initialized frequencies, scaling and cached-length metadata unchanged. *)
inline_for_extraction noextract
let initial_decoder_state (#a:Type0) () : decoder_entry_state a = {
  layer_history = (fun _ -> []);
  inv_freq = inverse_frequency;
  original_inv_freq = inverse_frequency;
  attention_scaling = attention_factor;
  max_seq_len_cached = max_positions;
  original_max_seq_len = max_positions
}

inline_for_extraction noextract
let prefill (#a:Type0) (input:inputs a) : outputs a = {
  hidden_rows = (fun p h -> input.embedding_weight (input.tokens p) h);
  position_ids = (fun p -> p);
  cache_positions = (fun p -> p);
  causal_bias = full_attention_bias;
  rope_cos = cosine;
  rope_sin = sine;
  decoder_state = initial_decoder_state ()
}

(* Pointwise postconditions avoid requiring equality of representations
   of indexed functions. Every logical field, including state, is fixed. *)
inline_for_extraction noextract
let state_postcondition (#a:Type0) (state:decoder_entry_state a) : prop =
  (forall (l:layer_index). state.layer_history l == []) /\
  (forall (r:frequency_index).
    state.inv_freq r == inverse_frequency r /\
    state.original_inv_freq r == inverse_frequency r) /\
  state.attention_scaling == attention_factor /\
  state.max_seq_len_cached == max_positions /\
  state.original_max_seq_len == max_positions

inline_for_extraction noextract
let postcondition (#a:Type0) (input:inputs a) (output:outputs a) : prop =
  (forall (p:prompt_index) (h:hidden_index).
    output.hidden_rows p h == input.embedding_weight (input.tokens p) h) /\
  (forall (p:prompt_index).
    output.position_ids p == p /\ output.cache_positions p == p) /\
  (forall (p k:prompt_index).
    output.causal_bias p k == full_attention_bias p k) /\
  (forall (p:prompt_index) (d:rotary_index).
    output.rope_cos p d == cosine p d /\
    output.rope_sin p d == sine p d) /\
  state_postcondition output.decoder_state

inline_for_extraction noextract
let prefill_satisfies_postcondition (#a:Type0) (input:inputs a)
  : Lemma (postcondition input (prefill input)) =
  ()

inline_for_extraction noextract
let duplicated_rotary_halves (p:prompt_index) (r:frequency_index)
  : Lemma (
      cosine p r == cosine p (r + half_dim) /\
      sine p r == sine p (r + half_dim)) =
  ()

inline_for_extraction noextract
let rotary_at_position_zero (d:rotary_index)
  : Lemma (cosine 0 d == 1.0R /\ sine 0 d == 0.0R) =
  Trig.cos_zero ();
  Trig.sin_zero ()

inline_for_extraction noextract
let causal_branch_exact (p k:prompt_index)
  : Lemma (
      (k <= p ==> full_attention_bias p k == 0.0R) /\
      (p < k ==> full_attention_bias p k == bf16_min)) =
  ()
