module KPlay.Proof.Optimized.Layer

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module A = Pulse.Lib.Array
module S = FStar.Seq
module G = FStar.Ghost
module SZ = Kuiper.SizeT
module Model = KPlay.Proof.Shared.Model
module SharedAttention = KPlay.Proof.Shared.Attention
module O = KPlay.Proof.Shared.Observations
module State = KPlay.Proof.Optimized.State
module Layout = KPlay.Proof.Optimized.Layout
module Memory = KPlay.Proof.Optimized.Memory
module Recurrent = KPlay.Proof.Optimized.Recurrent
module Attention = KPlay.Proof.Optimized.Attention

noextract
noeq type receipt (position:nat{position < 2048}) =
  | LinearCall : layer:Model.linear_layer -> before:State.stage ->
      produced:Recurrent.observation -> receipt position
  | AttentionCall : layer:SharedAttention.full_attention_layer -> before:State.stage ->
      produced:Attention.observation position -> receipt position

inline_for_extraction noextract
let layer_of (#position:nat{position < 2048}) (r:receipt position) : SharedAttention.layer_index =
  match r with | LinearCall l _ _ -> l | AttentionCall l _ _ -> l

inline_for_extraction noextract
let before_of (#position:nat{position < 2048}) (r:receipt position) : State.stage =
  match r with | LinearCall _ before _ -> before | AttentionCall _ before _ -> before

inline_for_extraction noextract
let after_of (#position:nat{position < 2048}) (r:receipt position) : GTot State.stage =
  match r with
  | LinearCall l before produced -> State.linear_after before l produced
  | AttentionCall l before produced -> State.attention_after position before l produced

inline_for_extraction noextract
let update_of (#position:nat{position < 2048}) (r:receipt position) : O.layer_update f32 f32 bf16 =
  match r with
  | LinearCall l before produced -> State.linear_update before l produced
  | AttentionCall _ _ produced -> State.attention_update position produced

inline_for_extraction noextract
let structural (#position:nat{position < 2048}) (r:receipt position) : prop =
  O.layer_structure position (layer_of r) (before_of r).State.cache
    (after_of r).State.cache (update_of r) /\
  State.represents position (layer_of r + 1) (after_of r).State.cache (after_of r).State.memory

noextract
fn source (weights:larray bf16 752393024) (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048)
  (states:larray f32 4718592) (convolution:larray f32 442368)
  (keys values:larray bf16 6291456)
  (position:sz{position < 2048}) (layer:sz{layer < 24})
  (#wv:erased (lseq bf16 752393024)) (#before:erased State.stage)
  (#nv:erased (lseq (option bf16) 1024)) (#pv:erased (lseq (option f32) 8192))
  (#av:erased (lseq (option f32) 2048)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> before.State.hidden **
    A.pts_to states before.State.memory.State.states **
    A.pts_to convolution before.State.memory.State.convolution **
    A.pts_to keys before.State.memory.State.keys ** A.pts_to values before.State.memory.State.values **
    A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells ** A.pts_to_mask activation av Memory.all_cells **
    pure (State.represents (SZ.v position) (SZ.v layer) before.State.cache before.State.memory)
  returns observed:erased (receipt (SZ.v position))
  ensures hidden |-> (after_of observed).State.hidden **
    A.pts_to states (after_of observed).State.memory.State.states **
    A.pts_to convolution (after_of observed).State.memory.State.convolution **
    A.pts_to keys (after_of observed).State.memory.State.keys **
    A.pts_to values (after_of observed).State.memory.State.values
  ensures exists* (next_norm:lseq (option bf16) 1024)
                  (next_projected:lseq (option f32) 8192)
                  (next_activation:lseq (option f32) 2048).
    A.pts_to_mask normalized next_norm Memory.all_cells **
    A.pts_to_mask projected next_projected Memory.all_cells **
    A.pts_to_mask activation next_activation Memory.all_cells **
    pure (Memory.ready next_norm 0 1024 /\ Memory.ready next_activation 0 2048 /\
      ((SZ.v layer % 4 < 3 \/ Memory.ready pv 0 8192) ==> Memory.ready next_projected 0 8192))
  ensures pure (layer_of observed == SZ.v layer /\ before_of observed == before /\ structural observed)
{
  Layout.layer_partition (SZ.v layer);
  let base = SZ.uint32_to_sizet 254280704ul
    +^ (layer /^ 4sz) *^ (SZ.uint32_to_sizet 83018720ul)
    +^ (layer %^ 4sz) *^ (SZ.uint32_to_sizet 21555360ul);
  if (layer %^ 4sz <^ 3sz) {
    let slot = (layer /^ 4sz) *^ 3sz +^ layer %^ 4sz;
    let sw = SZ.uint32_to_sizet 21555360ul;
    let sc = SZ.uint32_to_sizet 24576ul;
    let sr = SZ.uint32_to_sizet 262144ul;
    let local_weights = Memory.take weights base sw;
    let local_states = Memory.take states (slot *^ sr) sr;
    let local_conv = Memory.take convolution (slot *^ sc) sc;
    let produced = Recurrent.source_block local_weights hidden normalized projected activation local_states local_conv;
    Memory.give states (SZ.v slot * SZ.v sr) (SZ.v sr) #before.State.memory.State.states
      #produced.Recurrent.recurrent #1.0R;
    Memory.give convolution (SZ.v slot * SZ.v sc) (SZ.v sc) #before.State.memory.State.convolution
      #produced.Recurrent.convolution #1.0R;
    Memory.give weights (SZ.v base) (SZ.v sw) #wv
      #(S.slice wv (SZ.v base) (SZ.v base + SZ.v sw)) #fw;
    Memory.overwrite_readonly wv (SZ.v base) (SZ.v sw);
    State.linear_represents (SZ.v position) (SZ.v layer) before produced;
    State.linear_structure (SZ.v position) (SZ.v layer) before produced;
    let observed:erased (receipt (SZ.v position)) =
      G.hide (LinearCall (SZ.v layer) (G.reveal #State.stage before) (G.reveal #Recurrent.observation produced));
    observed
  } else {
    let slot = layer /^ 4sz;
    let sw = SZ.uint32_to_sizet 18352640ul;
    let sk = SZ.uint32_to_sizet 1048576ul;
    let local_weights = Memory.take weights base sw;
    let local_keys = Memory.take keys (slot *^ sk) sk;
    let local_values = Memory.take values (slot *^ sk) sk;
    let produced = Attention.source_block local_weights hidden normalized projected activation local_keys local_values position;
    Memory.give keys (SZ.v slot * SZ.v sk) (SZ.v sk) #before.State.memory.State.keys
      #produced.Attention.keys #1.0R;
    Memory.give values (SZ.v slot * SZ.v sk) (SZ.v sk) #before.State.memory.State.values
      #produced.Attention.values #1.0R;
    Memory.give weights (SZ.v base) (SZ.v sw) #wv
      #(S.slice wv (SZ.v base) (SZ.v base + SZ.v sw)) #fw;
    Memory.overwrite_readonly wv (SZ.v base) (SZ.v sw);
    State.attention_represents (SZ.v position) (SZ.v layer) before produced;
    State.attention_structure (SZ.v position) (SZ.v layer) before produced;
    let observed:erased (receipt (SZ.v position)) =
      G.hide (AttentionCall (SZ.v layer) (G.reveal #State.stage before)
        (G.reveal #(Attention.observation (SZ.v position)) produced));
    observed
  }
}
