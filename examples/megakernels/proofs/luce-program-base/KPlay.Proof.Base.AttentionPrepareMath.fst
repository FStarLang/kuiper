module KPlay.Proof.Base.AttentionPrepareMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
module Seq = FStar.Seq
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module Norm = KPlay.Proof.Base.Norm
module Math = KPlay.Proof.Base.AttentionAlgebra
module Sum = KPlay.Proof.Base.ReductionMath
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Config = KPlay.Proof.Base.RMSConfigs
module Gates = KPlay.Proof.Base.AttentionGateCopy
module Rotation = KPlay.Proof.Base.AttentionRotation
module Prepare = KPlay.Proof.Base.AttentionPrepare
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let rms_bridge (n:Norm.width) (weight input:A.vector n) (i:P.index n)
  : Lemma (P.normalized_for_logits #n P.reference_epsilon input weight i == A.rms_norm n weight input i) =
  Math.sum_squares_bridge n input;
  Sum.sum_ext n (Norm.squares #n input) (fun j -> input j *. input j);
  Sum.sum_kuiper n (Norm.squares #n input);
  norm_spec [delta_once ["KPlay.Proof.Shared.Projection.real_sum"]]
    (P.real_sum #n (Norm.squares #n input));
  assert (P.real_sum #n (Norm.squares #n input) == A.sum_squares n input);
  Norm.argument_formula n input;
  assert_norm (P.reference_epsilon == 0.000001R);
  norm_spec [delta_only [
    "KPlay.Proof.Shared.Projection.normalized_for_logits"; "KPlay.Proof.Shared.Attention.rms_norm"]]
    (P.normalized_for_logits #n P.reference_epsilon input weight i == A.rms_norm n weight input i)

inline_for_extraction noextract
let represented (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) : prop =
  (forall (i:P.index 4096). Approx.v_approximates (Seq.index before (2048 + i)) (q i)) /\
  (forall (i:P.index 512).
    Approx.v_approximates (Seq.index before (6144 + i)) (k i) /\
    Approx.v_approximates (Seq.index before (6656 + i)) (v i)) /\
  (forall (i:P.index 256).
    Approx.v_approximates (Seq.index weights (Inputs.layer_base layer + 16254976 + i)) (w.A.query_norm i) /\
    Approx.v_approximates (Seq.index weights (Inputs.layer_base layer + 16255232 + i)) (w.A.key_norm i)) /\
  (forall (i:P.index 32).
    Approx.v_approximates (Seq.index before (15600 + i)) (A.rope_cos position i) /\
    Approx.v_approximates (Seq.index before (15632 + i)) (A.rope_sin position i))

inline_for_extraction noextract
let queried (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  : GTot Transfer.scratch =
  RMS.after (Config.query_norm layer) (Gates.writes before 2048) weights 8

inline_for_extraction noextract
let queried_frame (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  (i:P.index 16000)
  : Lemma
      (requires ~(2048 <= i /\ i < 4096) /\ ~(8192 <= i /\ i < 10240))
      (ensures Seq.index (queried weights before layer) i == Seq.index before i) =
  RMSMath.outside_frame (Config.query_norm layer) (Gates.writes before 2048) weights 8 i

inline_for_extraction noextract
let normalized_frame (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  (i:P.index 16000)
  : Lemma
      (requires ~(2048 <= i /\ i < 4096) /\ ~(6144 <= i /\ i < 6656) /\ ~(8192 <= i /\ i < 10240))
      (ensures Seq.index (Prepare.normalized weights before layer) i == Seq.index before i) =
  queried_frame weights before layer i;
  RMSMath.outside_frame (Config.key_norm layer) (queried weights before layer) weights 2 i

inline_for_extraction noextract
let normalized_query (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 8) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.normalized weights before layer) (2048 + head * 256 + i))
        (A.rms_norm 256 w.A.query_norm (A.split_queries q head) i)) =
  RMSMath.complete_row_approx (Config.query_norm layer) (Gates.writes before 2048) weights head
    (A.split_queries q head) w.A.query_norm i;
  rms_bridge 256 w.A.query_norm (A.split_queries q head) i;
  RMSMath.outside_frame (Config.key_norm layer) (queried weights before layer) weights 2 (2048 + head * 256 + i)

inline_for_extraction noextract
let normalized_key (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 2) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.normalized weights before layer) (6144 + head * 256 + i))
        (A.rms_norm 256 w.A.key_norm (A.reshape_kv k head) i)) =
  let at_index (d:P.index 256)
    : Lemma (Seq.index (queried weights before layer) (6144 + head * 256 + d) ==
        Seq.index before (6144 + head * 256 + d)) =
    queried_frame weights before layer (6144 + head * 256 + d)
  in
  FStar.Classical.forall_intro at_index;
  RMSMath.complete_row_approx (Config.key_norm layer) (queried weights before layer) weights head
    (A.reshape_kv k head) w.A.key_norm i;
  rms_bridge 256 w.A.key_norm (A.reshape_kv k head) i

inline_for_extraction noextract
let rotary_preserved (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (i:P.index 32)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures
        Approx.v_approximates (Seq.index (Prepare.normalized weights before layer) (15600 + i)) (A.rope_cos position i) /\
        Approx.v_approximates (Seq.index (Prepare.normalized weights before layer) (15632 + i)) (A.rope_sin position i)) =
  normalized_frame weights before layer (15600 + i);
  normalized_frame weights before layer (15632 + i)

inline_for_extraction noextract
let query_approx (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 8) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.prepared weights before layer) (2048 + head * 256 + i))
        (A.rotate_partial position (A.rms_norm 256 w.A.query_norm (A.split_queries q head)) i)) =
  FStar.Classical.forall_intro (normalized_query weights before layer position w q k v head);
  FStar.Classical.forall_intro (rotary_preserved weights before layer position w q k v);
  Rotation.complete_head_approx (Prepare.normalized weights before layer) head position
    (A.rms_norm 256 w.A.query_norm (A.split_queries q head)) i

inline_for_extraction noextract
let key_approx (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 2) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.prepared weights before layer) (6144 + head * 256 + i))
        (A.rotate_partial position (A.rms_norm 256 w.A.key_norm (A.reshape_kv k head)) i)) =
  FStar.Classical.forall_intro (normalized_key weights before layer position w q k v head);
  FStar.Classical.forall_intro (rotary_preserved weights before layer position w q k v);
  Rotation.complete_head_approx (Prepare.normalized weights before layer) (8 + head) position
    (A.rms_norm 256 w.A.key_norm (A.reshape_kv k head)) i

inline_for_extraction noextract
let outside_frame (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  (i:P.index 16000)
  : Lemma
      (requires ~(2048 <= i /\ i < 4096) /\ ~(6144 <= i /\ i < 6656) /\ ~(8192 <= i /\ i < 10240))
      (ensures Seq.index (Prepare.prepared weights before layer) i == Seq.index before i) =
  normalized_frame weights before layer i;
  Rotation.outside_frame (Prepare.normalized weights before layer) 10 i

inline_for_extraction noextract
let gate_cell (weights:Transfer.packed) (before:Transfer.scratch) (layer:A.full_attention_layer)
  (head:P.index 8) (i:P.index 256)
  : Lemma (
      Seq.index (Prepare.prepared weights before layer) (8192 + head * 256 + i) ==
      Seq.index before (2048 + head * 512 + 256 + i)) =
  RMSMath.outside_frame (Config.query_norm layer) (Gates.writes before 2048) weights 8 (8192 + head * 256 + i);
  RMSMath.outside_frame (Config.key_norm layer) (queried weights before layer) weights 2 (8192 + head * 256 + i);
  Rotation.outside_frame (Prepare.normalized weights before layer) 10 (8192 + head * 256 + i)

inline_for_extraction noextract
let gates_approx (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 8) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.prepared weights before layer) (8192 + head * 256 + i))
        (A.split_gates q head i)) =
  gate_cell weights before layer head i

inline_for_extraction noextract
let value_approx (weights:Transfer.packed) (before:Transfer.scratch)
  (layer:A.full_attention_layer) (position:nat) (w:A.layer_weights)
  (q:A.vector 4096) (k v:A.vector 512) (head:P.index 2) (i:P.index 256)
  : Lemma
      (requires represented weights before layer position w q k v)
      (ensures Approx.v_approximates
        (Seq.index (Prepare.prepared weights before layer) (6656 + head * 256 + i))
        (A.reshape_kv v head i)) =
  outside_frame weights before layer (6656 + head * 256 + i)
