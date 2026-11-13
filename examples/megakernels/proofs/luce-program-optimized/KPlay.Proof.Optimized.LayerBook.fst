module KPlay.Proof.Optimized.LayerBook

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module C = FStar.Classical
module G = FStar.Ghost
module GR = Pulse.Lib.GhostReference
module SZ = Kuiper.SizeT
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations
module SharedAttention = KPlay.Proof.Shared.Attention
module State = KPlay.Proof.Optimized.State
module Layer = KPlay.Proof.Optimized.Layer
module Memory = KPlay.Proof.Optimized.Memory

noextract
noeq type book (position:nat{position < 2048}) = {
  count: Model.cursor;
  initial: State.stage;
  entries: lseq (Layer.receipt position) count;
}

inline_for_extraction noextract
let at (#position:nat{position < 2048}) (b:book position)
  (next:nat{next <= b.count}) : GTot State.stage =
  if next = 0 then b.initial else Layer.after_of (S.index b.entries (next - 1))

inline_for_extraction noextract
let last (#position:nat{position < 2048}) (b:book position) : GTot State.stage =
  at b b.count

inline_for_extraction noextract
let empty (position:nat{position < 2048}) (initial:State.stage) : book position =
  {count = 0; initial; entries = S.empty}

inline_for_extraction noextract
let append (#position:nat{position < 2048}) (b:book position{b.count < 24})
  (r:Layer.receipt position) : book position = {
  count = b.count + 1;
  initial = b.initial;
  entries = S.append b.entries (S.create 1 r);
}

inline_for_extraction noextract
let well_formed (#position:nat{position < 2048}) (b:book position) : prop =
  State.represents position 0 b.initial.State.cache b.initial.State.memory /\
  (forall (i:nat{i < b.count}).
    Layer.layer_of (S.index b.entries i) == i /\
    Layer.before_of (S.index b.entries i) == at b i /\
    Layer.structural (S.index b.entries i))

let last_represents (#position:nat{position < 2048}) (b:book position)
  : Lemma
      (requires well_formed b)
      (ensures State.represents position b.count (last b).State.cache (last b).State.memory) =
  if b.count > 0 then
    assert (Layer.structural (S.index b.entries (b.count - 1)) /\
      Layer.layer_of (S.index b.entries (b.count - 1)) == b.count - 1)

let append_checked (#position:nat{position < 2048}) (b:book position{b.count < 24})
  (r:Layer.receipt position)
  : Lemma
      (requires well_formed b /\ Layer.layer_of r == b.count /\
        Layer.before_of r == last b /\ Layer.structural r)
      (ensures well_formed (append b r) /\ last (append b r) == Layer.after_of r) =
  let related (i:nat{i < b.count + 1})
    : Lemma (
        Layer.layer_of (S.index (append b r).entries i) == i /\
        Layer.before_of (S.index (append b r).entries i) == at (append b r) i /\
        Layer.structural (S.index (append b r).entries i)) =
    if i < b.count then
      assert (at (append b r) i == at b i)
    else assert (S.index (append b r).entries i == r /\ at (append b r) i == last b)
  in C.forall_intro related

let complete (#position:nat{position < 2048}) (b:book position{b.count == 24})
  : Lemma
      (requires well_formed b)
      (ensures forall (l:SharedAttention.layer_index).
        O.layer_structure position l (at b l).State.cache (at b (l + 1)).State.cache
          (Layer.update_of (S.index b.entries l))) =
  let related (l:SharedAttention.layer_index)
    : Lemma (O.layer_structure position l (at b l).State.cache (at b (l + 1)).State.cache
        (Layer.update_of (S.index b.entries l))) =
    assert (Layer.structural (S.index b.entries l))
  in C.forall_intro related

noextract
fn source (weights:larray bf16 752393024) (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048)
  (states:larray f32 4718592) (convolution:larray f32 442368)
  (keys values:larray bf16 6291456) (position:sz{position < 2048})
  (#wv:erased (lseq bf16 752393024)) (#initial:erased State.stage)
  (#nv:erased (lseq (option bf16) 1024)) (#pv:erased (lseq (option f32) 8192))
  (#av:erased (lseq (option f32) 2048)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> initial.State.hidden **
    A.pts_to states initial.State.memory.State.states **
    A.pts_to convolution initial.State.memory.State.convolution **
    A.pts_to keys initial.State.memory.State.keys ** A.pts_to values initial.State.memory.State.values **
    A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells ** A.pts_to_mask activation av Memory.all_cells **
    pure (State.represents (SZ.v position) 0 initial.State.cache initial.State.memory)
  returns observed:erased (book (SZ.v position))
  ensures hidden |-> (last observed).State.hidden **
    A.pts_to states (last observed).State.memory.State.states **
    A.pts_to convolution (last observed).State.memory.State.convolution **
    A.pts_to keys (last observed).State.memory.State.keys **
    A.pts_to values (last observed).State.memory.State.values
  ensures exists* (next_norm:lseq (option bf16) 1024)
                  (next_projected:lseq (option f32) 8192)
                  (next_activation:lseq (option f32) 2048).
    A.pts_to_mask normalized next_norm Memory.all_cells **
    A.pts_to_mask projected next_projected Memory.all_cells **
    A.pts_to_mask activation next_activation Memory.all_cells **
    pure (Memory.ready next_norm 0 1024 /\ Memory.ready next_projected 0 8192 /\
      Memory.ready next_activation 0 2048)
  ensures pure (observed.count == 24 /\ observed.initial == initial /\ well_formed observed)
{
  let trace = GR.alloc #(book (SZ.v position)) (empty (SZ.v position) initial);
  let mut layer = 0sz;
  while (!layer <^ 24sz)
    invariant exists* (l:sz{l <= 24}) (current:book (SZ.v position))
                      (nn:lseq (option bf16) 1024)
                      (np:lseq (option f32) 8192) (na:lseq (option f32) 2048).
      layer |-> l ** trace |-> current **
      hidden |-> (last current).State.hidden **
      A.pts_to states (last current).State.memory.State.states **
      A.pts_to convolution (last current).State.memory.State.convolution **
      A.pts_to keys (last current).State.memory.State.keys **
      A.pts_to values (last current).State.memory.State.values **
      pure (current.count == SZ.v l /\ current.initial == initial /\ well_formed current) **
      A.pts_to_mask normalized nn Memory.all_cells **
      A.pts_to_mask projected np Memory.all_cells ** A.pts_to_mask activation na Memory.all_cells **
      pure (SZ.v l > 0 ==> Memory.ready nn 0 1024 /\ Memory.ready np 0 8192 /\ Memory.ready na 0 2048)
    decreases (24 - !layer)
  {
    let l = !layer;
    let current = GR.read trace;
    last_represents current;
    let result = Layer.source weights hidden normalized projected activation
      states convolution keys values position l #wv #(G.hide (last current));
    append_checked current result;
    let next:erased (book (SZ.v position)) = G.hide (append current result);
    GR.write trace next;
    layer := l +^ 1sz;
  };
  let observed = GR.read trace;
  GR.free trace;
  observed
}
