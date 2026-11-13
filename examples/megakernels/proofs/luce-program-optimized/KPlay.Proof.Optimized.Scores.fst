module KPlay.Proof.Optimized.Scores

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64
module IC = FStar.Int.Cast
module G = FStar.Ghost
module GR = Pulse.Lib.GhostReference
module Final = KPlay.Proof.Optimized.Final
module Memory = KPlay.Proof.Optimized.Memory
module Norm = KPlay.Proof.Optimized.Norm
module Sel = KPlay.Proof.Optimized.Selector
module Math = KPlay.Proof.Optimized.ScoreMath
module SM = KPlay.Proof.Optimized.SelectionMath
module FS = KPlay.Proof.Optimized.FamilySelection

noextract
fn scan_warp (weights:larray bf16 752393024) (activation:larray f32 2048)
  (block:sz{block < 512}) (warp:sz{warp < 32}) (trace:GR.ref FS.score_vector)
  (#wv:erased (lseq bf16 752393024)) (#av:erased (lseq (option f32) 2048))
  (#before:erased FS.score_vector) (#fw #fa:perm)
  (initialized:squash (Memory.ready av 0 1024))
  preserves gpu ** weights |-> Frac fw wv ** A.pts_to_mask activation #fa av Memory.all_cells
  requires trace |-> before **
    pure (Math.modeled wv (Memory.values av 0 1024 initialized) before
      (Math.before_warp (SZ.v block) (SZ.v warp)))
  returns result:Sel.candidate
  ensures exists* (after:FS.score_vector).
    trace |-> after **
    pure (Math.certified after (SM.lane_region (SZ.v block) 32 (SZ.v warp)) result /\
      Math.modeled wv (Memory.values av 0 1024 initialized) after
        (Math.before_warp (SZ.v block) (SZ.v warp + 1)) /\
      (forall (i:nat{i < 248320}).
        not (SM.lane_region (SZ.v block) 32 (SZ.v warp) i) ==>
          S.index after i == S.index before i))
{
  let mut maximum:f32 = FS.negative_infinity;
  let mut token = -1l;
  let start = block *^ 485sz +^ warp;
  let finish = (block +^ 1sz) *^ 485sz;
  let mut row = start;
  if (warp <^ 8sz) {
    SM.empty_best before (SM.scan_region (SZ.v block) (SZ.v warp) (SZ.v start));
    while (!row <^ finish)
      invariant exists* (r:sz{
          SZ.v start <= SZ.v r /\ SZ.v r < SZ.v finish + 8 /\
          (SZ.v r - SZ.v start) % 8 == 0})
                       (v:f32) (i:I32.t) (scores:FS.score_vector).
        row |-> r ** maximum |-> v ** token |-> i ** trace |-> scores **
        pure (Math.certified scores (SM.scan_region (SZ.v block) (SZ.v warp) (SZ.v r))
          {Sel.value = v; Sel.index = i}) **
        pure (Math.modeled wv (Memory.values av 0 1024 initialized) scores
          (Math.produced (SZ.v block) (SZ.v warp) (SZ.v r))) **
        pure (forall (j:nat{j < 248320}).
          not (SM.lane_region (SZ.v block) 32 (SZ.v warp) j) ==>
            S.index scores j == S.index before j)
      decreases (SZ.v finish + 8 - SZ.v !row)
    {
      let r = !row;
      let value = Final.score weights activation r #wv #av #fw #fa initialized;
      let index = IC.uint32_to_int32 (SZ.sizet_to_u32 r);
      let scores = GR.read trace;
      let current:Sel.candidate = {value = !maximum; index = !token};
      Math.scan_record scores (SZ.v block) (SZ.v warp) (SZ.v r) current value index;
      Math.model_record wv (Memory.values av 0 1024 initialized)
        scores (SZ.v block) (SZ.v warp) (SZ.v r) value;
      GR.write trace (G.hide (S.upd scores (SZ.v r) value));
      Sel.accumulate maximum token value index;
      row := r +^ 8sz;
    };
    let scores = GR.read trace;
    let result:Sel.candidate = {value = !maximum; index = !token};
    Math.warp_finished wv (Memory.values av 0 1024 initialized)
      scores (SZ.v block) (SZ.v warp) (SZ.v !row) result;
    result
  } else {
    SM.empty_best before (SM.lane_region (SZ.v block) 32 (SZ.v warp));
    {Sel.value = !maximum; Sel.index = !token}
  }
}

noextract
fn scan_block (weights:larray bf16 752393024) (activation:larray f32 2048)
  (block:sz{block < 512}) (trace:GR.ref FS.score_vector)
  (#wv:erased (lseq bf16 752393024)) (#av:erased (lseq (option f32) 2048))
  (#before:erased FS.score_vector) (#fw #fa:perm)
  (initialized:squash (Memory.ready av 0 1024))
  preserves gpu ** weights |-> Frac fw wv ** A.pts_to_mask activation #fa av Memory.all_cells
  requires trace |-> before **
    pure (Math.modeled wv (Memory.values av 0 1024 initialized) before
      (Math.before_warp (SZ.v block) 0))
  returns result:Sel.candidate
  ensures exists* (after:FS.score_vector).
    trace |-> after **
    pure (Math.certified after (fun row -> row / 485 == SZ.v block) result /\
      Math.modeled wv (Memory.values av 0 1024 initialized) after
        (Math.before_warp (SZ.v block + 1) 0) /\
      (forall (i:nat{i < 248320}). i / 485 <> SZ.v block ==> S.index after i == S.index before i))
{
  let mut values:A.array f32 = [| 32sz |];
  let mut indices:A.array I32.t = [| 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 32sz)
    invariant exists* (w:sz{w <= 32}) (vv:lseq (option f32) 32)
                     (iv:lseq (option I32.t) 32) (scores:FS.score_vector).
      warp |-> w ** A.pts_to_mask values vv Memory.all_cells **
      A.pts_to_mask indices iv Memory.all_cells ** trace |-> scores **
      pure (Math.masked_certificates scores (SM.lane_region (SZ.v block) 32) vv iv (SZ.v w)) **
      pure (Math.modeled wv (Memory.values av 0 1024 initialized) scores
        (Math.before_warp (SZ.v block) (SZ.v w))) **
      pure (forall (i:nat{i < 248320}). i / 485 <> SZ.v block ==> S.index scores i == S.index before i)
    decreases (32 - !warp)
  {
    let w = !warp;
    let previous = GR.read trace;
    with vv. assert A.pts_to_mask values vv Memory.all_cells;
    with iv. assert A.pts_to_mask indices iv Memory.all_cells;
    let candidate = scan_warp weights activation block w trace #wv #av #previous #fw #fa initialized;
    let scores = GR.read trace;
    Math.frame_certificates previous scores (SM.lane_region (SZ.v block) 32) vv iv (SZ.v w);
    Math.push_certificate scores (SM.lane_region (SZ.v block) 32) vv iv (SZ.v w) candidate;
    A.mask_write values w candidate.Sel.value;
    A.mask_write indices w candidate.Sel.index;
    warp := w +^ 1sz;
  };
  let scores = GR.read trace;
  with mv. assert A.pts_to_mask values mv Memory.all_cells;
  with mi. assert A.pts_to_mask indices mi Memory.all_cells;
  A.from_mask values;
  A.from_mask indices;
  with vv. assert values |-> vv;
  with iv. assert indices |-> iv;
  Math.plain_certificates #32 scores (SM.lane_region (SZ.v block) 32) mv mi vv iv;
  let result = Sel.reduce #32 values indices 32sz;
  Math.block_tree scores (SZ.v block) (Sel.candidates #32 vv iv);
  A.to_mask values;
  A.to_mask indices;
  result
}

noextract
noeq type observation = {
  normalized: lseq f32 1024;
  scores: FS.score_vector;
}

inline_for_extraction noextract
let post (weights:lseq bf16 752393024) (hidden:lseq bf16 1024)
  (selected:I32.t) (observed:observation) : prop =
  if 0 <= I32.v selected && I32.v selected < 248320 then
    (forall (m:Norm.interpretation 752393024). Norm.inputs weights hidden m ==>
      observed.normalized %~ Norm.normalized m 254279680) /\
    Math.modeled weights observed.normalized observed.scores (fun _ -> True) /\
    FS.native_maximum #248320 observed.scores (I32.v selected)
  else False

noextract
fn source_head (weights:larray bf16 752393024) (hidden:larray bf16 1024)
  (activation:larray f32 2048)
  (#wv:erased (lseq bf16 752393024)) (#hv:erased (lseq bf16 1024))
  (#av:erased (lseq (option f32) 2048)) (#fw #fh:perm)
  preserves gpu ** weights |-> Frac fw wv ** hidden |-> Frac fh hv
  requires A.pts_to_mask activation av Memory.all_cells
  returns result:(I32.t & erased observation)
  ensures exists* (after:lseq (option f32) 2048).
    A.pts_to_mask activation after Memory.all_cells **
    pure (post wv hv (fst result) (G.reveal #observation (snd result)) /\
      Memory.ready after 0 1024 /\
      (Memory.ready av 0 2048 ==> Memory.ready after 0 2048))
{
  Final.normalize weights hidden activation;
  with norm_values. assert A.pts_to_mask activation norm_values Memory.all_cells;
  let initialized : squash (Memory.ready norm_values 0 1024) = ();
  let normalized:erased (lseq f32 1024) = G.hide (Memory.values norm_values 0 1024 initialized);
  let trace = GR.alloc #FS.score_vector (S.create 248320 (zero #f32));
  let mut values:A.array f32 = [| 512sz |];
  let mut indices:A.array I32.t = [| 512sz |];
  let mut block = 0sz;
  while (!block <^ 512sz)
    invariant exists* (b:sz{b <= 512}) (vv:lseq (option f32) 512)
                     (iv:lseq (option I32.t) 512) (scores:FS.score_vector).
      block |-> b ** A.pts_to_mask values vv Memory.all_cells **
      A.pts_to_mask indices iv Memory.all_cells ** trace |-> scores **
      pure (Math.masked_certificates scores (fun b row -> row / 485 == b) vv iv (SZ.v b)) **
      pure (Math.modeled wv normalized scores (Math.before_warp (SZ.v b) 0))
    decreases (512 - !block)
  {
    let b = !block;
    let previous = GR.read trace;
    with vv. assert A.pts_to_mask values vv Memory.all_cells;
    with iv. assert A.pts_to_mask indices iv Memory.all_cells;
    let candidate = scan_block weights activation b trace #wv #norm_values #previous #fw #1.0R initialized;
    let scores = GR.read trace;
    Math.frame_certificates previous scores (fun b row -> row / 485 == b) vv iv (SZ.v b);
    Math.push_certificate scores (fun b row -> row / 485 == b) vv iv (SZ.v b) candidate;
    A.mask_write values b candidate.Sel.value;
    A.mask_write indices b candidate.Sel.index;
    block := b +^ 1sz;
  };
  let scores = GR.read trace;
  with mv. assert A.pts_to_mask values mv Memory.all_cells;
  with mi. assert A.pts_to_mask indices mi Memory.all_cells;
  A.from_mask values;
  A.from_mask indices;
  with vv. assert values |-> vv;
  with iv. assert indices |-> iv;
  Math.plain_certificates #512 scores (fun b row -> row / 485 == b) mv mi vv iv;
  let selected = Sel.reduce #512 values indices 512sz;
  Math.vocabulary_tree scores (Sel.candidates #512 vv iv);
  SM.best_member scores (fun _ -> True) selected.Sel.value (I32.v selected.Sel.index) 0;
  let observed:erased observation = G.hide ({
    normalized = G.reveal #(lseq f32 1024) normalized;
    scores = G.reveal #FS.score_vector scores;
  } <: observation);
  GR.free trace;
  A.to_mask values;
  A.to_mask indices;
  (selected.Sel.index, observed)
}
