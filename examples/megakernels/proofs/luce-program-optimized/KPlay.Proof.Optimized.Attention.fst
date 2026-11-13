module KPlay.Proof.Optimized.Attention

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module G = FStar.Ghost
module GR = Pulse.Lib.GhostReference
module Norm = KPlay.Proof.Optimized.Norm
module Matrix = KPlay.Proof.Optimized.Matrix
module Dot = KPlay.Proof.Optimized.Dot
module Memory = KPlay.Proof.Optimized.Memory
module Ops = KPlay.Proof.Optimized.AttentionOps
module Online = KPlay.Proof.Optimized.Online
module Block = KPlay.Proof.Optimized.Block

noextract
noeq type observation (position:nat{position < 2048}) = {
  hidden: lseq bf16 1024;
  keys: lseq bf16 1048576;
  values: lseq bf16 1048576;
  scores: lseq f32 (8 * (position + 1));
}

inline_for_extraction noextract
let flatten_scores (#position:nat{position < 2048})
  (heads:lseq (lseq f32 (position + 1)) 8) : GTot (lseq f32 (8 * (position + 1))) =
  S.init_ghost (8 * (position + 1))
    (fun i -> S.index (S.index heads (i / (position + 1))) (i % (position + 1)))

noextract
fn source_block
  (weights:larray bf16 18352640) (hidden normalized:larray bf16 1024)
  (projected:larray f32 8192) (activation:larray f32 2048)
  (keys values:larray bf16 1048576) (position:sz{position < 2048})
  (#wv:erased (lseq bf16 18352640)) (#hv:erased (lseq bf16 1024))
  (#kv #vv:erased (lseq bf16 1048576))
  (#nv:erased (lseq (option bf16) 1024)) (#pv:erased (lseq (option f32) 8192))
  (#av:erased (lseq (option f32) 2048)) (#fw:perm)
  preserves gpu ** weights |-> Frac fw wv
  requires hidden |-> hv ** keys |-> kv ** values |-> vv **
    A.pts_to_mask normalized nv Memory.all_cells **
    A.pts_to_mask projected pv Memory.all_cells **
    A.pts_to_mask activation av Memory.all_cells
  returns observed:erased (observation (SZ.v position))
  ensures hidden |-> observed.hidden ** keys |-> observed.keys ** values |-> observed.values
  ensures exists* (next_norm:lseq (option bf16) 1024)
                  (next_projected:lseq (option f32) 8192)
                  (next_activation:lseq (option f32) 2048).
    A.pts_to_mask normalized next_norm Memory.all_cells **
    A.pts_to_mask projected next_projected Memory.all_cells **
    A.pts_to_mask activation next_activation Memory.all_cells **
    pure (Memory.ready next_norm 0 1024 /\ Memory.ready next_activation 0 2048 /\
      (Memory.ready pv 0 8192 ==> Memory.ready next_projected 0 8192))
  ensures pure (Ops.cache_frame (SZ.v position) kv observed.keys /\
    Ops.cache_frame (SZ.v position) vv observed.values)
{
  Norm.normalize_hidden weights hidden normalized 0sz;
  Matrix.project_bf16 weights normalized projected 4096sz
    (SZ.uint32_to_sizet 11012096ul) 0sz;
  let mut row = 0sz;
  while (!row <^ 512sz)
    invariant exists* (r:sz{r <= 512}) (current:lseq (option f32) 8192).
      row |-> r ** A.pts_to_mask projected current Memory.all_cells **
      pure (Memory.ready current 0 4096 /\
        Memory.ready current 4096 (SZ.v r) /\ Memory.ready current 4608 (SZ.v r)) **
      pure (forall (i:nat{5120 <= i /\ i < 8192}). S.index current i == S.index pv i)
    decreases (512 - !row)
  {
    let r = !row;
    let k = Dot.dot_bf16 weights normalized (SZ.uint32_to_sizet 15206400ul +^ r *^ 1024sz);
    A.mask_write projected (4096sz +^ r) k;
    let v = Dot.dot_bf16 weights normalized (SZ.uint32_to_sizet 15730688ul +^ r *^ 1024sz);
    A.mask_write projected (4608sz +^ r) v;
    row := r +^ 1sz;
  };
  with raw. assert A.pts_to_mask projected raw Memory.all_cells;
  let initialized : squash (Memory.ready raw 0 5120) = ();
  let live = Memory.borrow projected 0sz 5120sz #raw #1.0R initialized;
  Ops.write_keys weights live keys values position;
  let trace = GR.alloc #(lseq (option (lseq f32 (SZ.v position + 1))) 8)
    (S.create 8 None);
  let mut head = 0sz;
  while (!head <^ 8sz)
    invariant exists* (h:sz{h <= 8}) (current:lseq f32 5120)
                     (current_activation:lseq (option f32) 2048)
                     (scores:lseq (option (lseq f32 (SZ.v position + 1))) 8).
      head |-> h ** live |-> current **
      A.pts_to_mask activation current_activation Memory.all_cells ** trace |-> scores **
      pure (Memory.ready current_activation 0 (SZ.v h * 256) /\
        Memory.ready scores 0 (SZ.v h))
    decreases (8 - !head)
  {
    let h = !head;
    Ops.transform_query weights live h position;
    with before_activation. assert A.pts_to_mask activation before_activation Memory.all_cells;
    let output = Memory.take_mask activation (h *^ 256sz) 256sz;
    let scores = Online.source_head live keys values output h position;
    with after_head. assert output |-> after_head;
    Memory.restore activation (SZ.v h * 256) 256 #before_activation #after_head #1.0R;
    let previous = GR.read trace;
    GR.write trace (G.hide (S.upd previous (SZ.v h)
      (Some (G.reveal #(lseq f32 (SZ.v position + 1)) scores))));
    head := h +^ 1sz;
  };
  let captured = GR.read trace;
  let head_scores:erased (lseq (lseq f32 (SZ.v position + 1)) 8) =
    G.hide (Memory.values captured 0 8 ());
  let scores:erased (lseq f32 (8 * (SZ.v position + 1))) =
    G.hide (flatten_scores #(SZ.v position)
      (G.reveal #(lseq (lseq f32 (SZ.v position + 1)) 8) head_scores));
  GR.free trace;
  with next_live. assert live |-> next_live;
  Memory.restore projected 0 5120 #raw #next_live #1.0R;
  A.from_mask activation;
  A.to_mask normalized;
  Block.finish weights activation hidden normalized projected
    (SZ.uint32_to_sizet 16255488ul) 1024sz 2048sz
    (SZ.uint32_to_sizet 3672064ul) (SZ.uint32_to_sizet 7342080ul);
  with after_hidden. assert hidden |-> after_hidden;
  with after_keys. assert keys |-> after_keys;
  with after_values. assert values |-> after_values;
  let observed:erased (observation (SZ.v position)) = G.hide ({
    hidden = after_hidden;
    keys = after_keys;
    values = after_values;
    scores = G.reveal #(lseq f32 (8 * (SZ.v position + 1))) scores;
  } <: observation (SZ.v position));
  A.to_mask normalized;
  A.to_mask activation;
  observed
}
