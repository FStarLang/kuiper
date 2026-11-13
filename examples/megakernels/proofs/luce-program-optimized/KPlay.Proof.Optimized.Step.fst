module KPlay.Proof.Optimized.Step

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module G = FStar.Ghost
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module O = KPlay.Proof.Shared.Observations
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Layout = KPlay.Proof.Optimized.Layout
module State = KPlay.Proof.Optimized.State
module Layer = KPlay.Proof.Optimized.Layer
module Book = KPlay.Proof.Optimized.LayerBook
module Memory = KPlay.Proof.Optimized.Memory
module Embedding = KPlay.Proof.Optimized.Embedding
module History = KPlay.Proof.Optimized.History
module Scores = KPlay.Proof.Optimized.Scores
module SO = KPlay.Proof.Optimized.ScoreObservation
module FS = KPlay.Proof.Optimized.FamilySelection

noextract
noeq type data (position:nat{position < 2048}) = {
  consumed: D.token 248320;
  layers: b:Book.book position{b.Book.count == 24};
  head: Scores.observation;
  chosen: Embedding.token;
}

inline_for_extraction noextract
let observed (#position:nat{position < 2048}) (step:data position)
  : GTot (O.model_observation O.cache_configuration bf16 f32 f32 bf16 f32 f32) = {
  O.position = position;
  O.consumed = step.consumed;
  O.stages = (fun next -> State.logical_stage (Book.at step.layers next));
  O.updates = (fun layer -> Layer.update_of (S.index step.layers.Book.entries layer));
  O.norm_input = (fun i -> S.index (Book.last step.layers).State.hidden i);
  O.normalized = (fun i -> S.index step.head.Scores.normalized i);
  O.produced_scores = (fun i -> S.index step.head.Scores.scores i);
  O.score_keys = SO.keys step.head.Scores.scores;
  O.captured = (Book.last step.layers).State.hidden;
  O.cache_after = (Book.last step.layers).State.cache;
}

inline_for_extraction noextract
let well_formed (#position:nat{position < 2048}) (w:Layout.packed) (step:data position) : prop =
  Book.well_formed step.layers /\
  step.layers.Book.initial.State.hidden == Embedding.row w step.consumed /\
  Scores.post w (Book.last step.layers).State.hidden step.chosen step.head

let keys_equal (scores:FS.score_vector)
  : Lemma (SO.keys scores == FS.logical_scores #248320 scores) =
  assert (SO.keys scores == FS.logical_scores #248320 scores)
  by (FStar.Tactics.V2.norm [delta_only [`%SO.keys; `%FS.logical_scores]; zeta; iota];
      FStar.Tactics.V2.trefl ())

let structure (#position:nat{position < 2048}) (w:Layout.packed) (step:data position)
  : Lemma
      (requires well_formed w step)
      (ensures
        O.structure (Layout.parameter_values w) position step.consumed
          step.layers.Book.initial.State.cache (observed step) /\
        D.maximum (observed step).O.score_keys (I32.v step.chosen)) =
  Book.complete step.layers;
  Embedding.row_binding w step.consumed;
  SO.represented step.head.Scores.scores;
  keys_equal step.head.Scores.scores;
  FS.logical_scores_represent #248320 step.head.Scores.scores;
  FS.maximum_representation #248320 step.head.Scores.scores
    (FS.logical_scores #248320 step.head.Scores.scores) (I32.v step.chosen)

noextract
fn source (call:Model.invocation)
  (weights:larray bf16 752393024) (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048)
  (states:larray f32 4718592) (convolution:larray f32 442368)
  (keys values:larray bf16 6291456)
  (tokens:larray I64.t call.decode_steps) (history:larray bf16 (call.decode_steps * 1024))
  (position:sz{SZ.v position < Model.calls call /\ position < 2048})
  (length:sz{SZ.v length == call.prompt_length}) (selected:Embedding.token)
  (#wv:erased (lseq bf16 752393024)) (#incoming:erased State.native_cache) (#memory:erased State.physical)
  (#hv #nv:erased (lseq (option bf16) 1024)) (#pv:erased (lseq (option f32) 8192))
  (#av:erased (lseq (option f32) 2048))
  (#tv:erased (lseq (option I64.t) call.decode_steps))
  (#history_before:erased (lseq (option bf16) (call.decode_steps * 1024))) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires A.pts_to states memory.State.states ** A.pts_to convolution memory.State.convolution **
    A.pts_to keys memory.State.keys ** A.pts_to values memory.State.values **
    A.pts_to_mask hidden hv Memory.all_cells ** A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells ** A.pts_to_mask activation av Memory.all_cells **
    A.pts_to_mask tokens tv Memory.all_cells ** A.pts_to_mask history history_before Memory.all_cells **
    pure (State.represents (SZ.v position) 0 incoming memory)
  returns result:(Embedding.token & erased (data (SZ.v position)))
  ensures (
    let step = G.reveal #(data (SZ.v position)) (snd result) in
    hidden |-> (Book.last step.layers).State.hidden **
    A.pts_to states (Book.last step.layers).State.memory.State.states **
    A.pts_to convolution (Book.last step.layers).State.memory.State.convolution **
    A.pts_to keys (Book.last step.layers).State.memory.State.keys **
    A.pts_to values (Book.last step.layers).State.memory.State.values **
    A.pts_to_mask history
      (History.hidden_after call (SZ.v position) history_before (Book.last step.layers).State.hidden) Memory.all_cells **
    A.pts_to_mask tokens
      (History.tokens_after call (SZ.v position) tv (IC.int32_to_int64 (fst result))) Memory.all_cells **
    pure (step.consumed == I32.v selected /\ step.chosen == fst result /\
      step.layers.Book.initial.State.cache == incoming /\ well_formed wv step))
  ensures exists* (next_norm:lseq (option bf16) 1024)
                  (next_projected:lseq (option f32) 8192)
                  (next_activation:lseq (option f32) 2048).
    A.pts_to_mask normalized next_norm Memory.all_cells **
    A.pts_to_mask projected next_projected Memory.all_cells **
    A.pts_to_mask activation next_activation Memory.all_cells **
    pure (Memory.ready next_norm 0 1024 /\ Memory.ready next_projected 0 8192 /\
      Memory.ready next_activation 0 2048)
{
  Embedding.copy weights hidden selected;
  let initial:erased State.stage = G.hide ({
    State.hidden = Embedding.row wv (I32.v selected);
    State.cache = G.reveal #State.native_cache incoming;
    State.memory = G.reveal #State.physical memory;
  } <: State.stage);
  let layers = Book.source weights hidden normalized projected activation
    states convolution keys values position #wv #initial;
  History.capture call position length hidden history;
  let head_result = Scores.source_head weights hidden activation;
  let chosen:Embedding.token = fst head_result;
  let head = snd head_result;
  History.emit call position length (IC.int32_to_int64 chosen) tokens;
  let step:erased (data (SZ.v position)) = G.hide ({
    consumed = I32.v selected;
    layers = G.reveal #(Book.book (SZ.v position)) layers;
    head = G.reveal #Scores.observation head;
    chosen;
  } <: data (SZ.v position));
  (chosen, step)
}
