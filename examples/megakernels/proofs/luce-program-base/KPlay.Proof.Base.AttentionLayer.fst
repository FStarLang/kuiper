module KPlay.Proof.Base.AttentionLayer

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module C = Kuiper.Chest
module T = Kuiper.Tensor
module GR = Pulse.Lib.GhostReference
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module SZ = Kuiper.SizeT
module Attention = KPlay.Proof.Shared.Attention
module Model = KPlay.Proof.Shared.Model
module Pipeline = KPlay.Proof.Base.AttentionPipeline
module Stages = KPlay.Proof.Base.AttentionStages
module Heads = KPlay.Proof.Base.AttentionHeadStages
module Meta = KPlay.Proof.Base.AttentionBook
module Book = KPlay.Proof.Base.WarpBook
module Phase = KPlay.Proof.Base.PhaseMMA
module Head = KPlay.Proof.Base.Head
module Hidden = KPlay.Proof.Base.WarpHidden
module Native = KPlay.Proof.Base.NativeProjection
module Scratch = KPlay.Proof.Base.LinearBook
module Arrays = KPlay.Proof.Base.SourceArrays
module Cache = KPlay.Proof.Base.CacheNative
module CacheMath = KPlay.Proof.Base.CacheMath
module CacheProjection = KPlay.Proof.Base.CacheProjection
module Init = KPlay.Proof.Base.Initialization
module Transfer = KPlay.Proof.Base.Transfer
module RMS = KPlay.Proof.Base.RMS
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Prepare = KPlay.Proof.Base.AttentionPrepare
module Softmax = KPlay.Proof.Base.SoftmaxLocal
module Algebra = KPlay.Proof.Base.AttentionAlgebra
module Schedule = KPlay.Proof.Base.Schedule

inline_for_extraction noextract
let local_owners (#extra:Type0) (extra_owners:extra -> slprop)
  (m:Meta.metadata extra) : slprop =
  extra_owners m.Meta.root.Meta.payload

inline_for_extraction noextract
fn simple_prelude
  (layer:Attention.full_attention_layer)
  (phase:sz{SZ.v phase < 23 /\ SZ.v phase <> 3 /\ (SZ.v phase < 3 \/ 19 <= SZ.v phase \/ SZ.v phase % 2 = 1)})
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures A.pts_to hidden (Meta.prelude layer (SZ.v phase) vw vh)
{
  if (FStar.SizeT.eq phase 0sz) {
    RMS.source_apply (Config.model_norm layer false) weights hidden;
  } else if (FStar.SizeT.eq phase 19sz) {
    Scalar.apply Scalar.AttentionGate hidden;
  } else if (FStar.SizeT.eq phase 20sz) {
    Scalar.apply Scalar.Residual hidden;
    RMS.source_apply (Config.model_norm layer true) weights hidden;
  } else if (FStar.SizeT.eq phase 22sz) {
    Scalar.apply Scalar.Mlp hidden;
  }
}

inline_for_extraction noextract
fn prelude
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Attention.full_attention_layer)
  (phase:sz{SZ.v phase < 23})
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912)
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm) (#scope:prop)
  preserves gpu ** weights |-> Frac fraction vw
  requires Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase)))
  ensures Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)))
{
  unfold Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase)));
  with m before keys. assert GR.pts_to book m ** hidden |-> before ** A.pts_to_mask cache keys Init.all_cells;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase)) m before keys);
  unfold local_owners extra_owners m;
  Arrays.weight_to_array weights;
  Head.hidden_to_array hidden #before #1.0R;
  if (FStar.SizeT.eq phase 3sz) {
    Prepare.source_apply (T.core weights) (T.core hidden) layer;
    let source_layer = SZ.uint_to_t layer;
    let source_position = SZ.uint_to_t position;
    Cache.append cache (T.core hidden) source_layer source_position;
    Meta.prepare_step call position layer (Native.packed_values vw) scope anchor m before keys;
    let result : erased Transfer.scratch =
      hide #Transfer.scratch (Prepare.prepared (Native.packed_values vw) (Native.hidden_values before) layer);
    let next : erased (Meta.metadata extra) = hide #(Meta.metadata extra) (Meta.with_prepared m result);
    GR.write book next;
    Scratch.tensor_values result;
    Head.array_to_hidden hidden #(Scratch.tensor result) #1.0R;
    rewrite extra_owners (reveal #(Meta.metadata extra) m).Meta.root.Meta.payload
      as extra_owners (reveal #(Meta.metadata extra) next).Meta.root.Meta.payload;
    fold local_owners extra_owners next;
    Arrays.array_to_weight weights;
    fold Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
  } else if (3sz <=^ phase && phase <^ 19sz && FStar.SizeT.eq (phase %^ 2sz) 0sz) {
    let length = SZ.uint_to_t (position + 1);
    Heads.softmax_input scope position (SZ.v phase)
      (reveal #(Meta.metadata extra) m).Meta.root.Meta.logical
      (reveal #(Meta.metadata extra) m).Meta.root.Meta.stages (Native.hidden_values before);
    Softmax.apply length 13500sz (T.core hidden) #(Native.hidden_values before)
      #(Algebra.score_dots position
        (reveal #(Meta.metadata extra) m).Meta.root.Meta.stages.Pipeline.queries
        (reveal #(Meta.metadata extra) m).Meta.root.Meta.stages.Pipeline.history (Heads.head (SZ.v phase))) #scope;
    with result. assert A.pts_to (T.core hidden) result;
    Meta.softmax_step call position layer (SZ.v phase) (Native.packed_values vw) scope anchor m before keys result;
    Scratch.tensor_values result;
    Head.array_to_hidden hidden #(Scratch.tensor result) #1.0R;
    fold local_owners extra_owners m;
    Arrays.array_to_weight weights;
    fold Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
  } else {
    simple_prelude layer phase (T.core weights) (T.core hidden);
    Meta.prelude_valid call position layer (SZ.v phase) (Native.packed_values vw) scope anchor m before keys;
    Scratch.tensor_values (Meta.prelude layer (SZ.v phase) (Native.packed_values vw) (Native.hidden_values before));
    Head.array_to_hidden hidden
      #(Scratch.tensor (Meta.prelude layer (SZ.v phase) (Native.packed_values vw) (Native.hidden_values before))) #1.0R;
    fold local_owners extra_owners m;
    Arrays.array_to_weight weights;
    fold Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
  }
}

inline_for_extraction noextract
fn simt
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Attention.full_attention_layer)
  (phase:Heads.phase{not (Schedule.mma_path layer phase position)})
  (hidden:array2 BF.t Native.hidden_layout) (cache:larray BF.t 12582912)
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#scope:prop)
  preserves gpu
  requires Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input phase))
  ensures Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output phase))
{
  unfold Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input phase));
  with m before keys. assert GR.pts_to book m ** hidden |-> before ** A.pts_to_mask cache keys Init.all_cells;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionBook.valid"]]
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input phase) m before keys);
  unfold local_owners extra_owners m;
  Meta.simt_ready call position layer phase (Native.packed_values vw) scope anchor m before keys ();
  Head.hidden_to_array hidden #before #1.0R;
  CacheProjection.simt call position layer phase cache (T.core hidden)
    #keys #(Native.hidden_values before)
    #(CacheMath.matrix position (reveal #(Meta.metadata extra) m).Meta.root.Meta.stages.Pipeline.history
      (Heads.head phase) (Heads.score phase))
    #(Heads.input (reveal #(Meta.metadata extra) m).Meta.root.Meta.stages phase) #1.0R #scope ();
  with result. assert A.pts_to (T.core hidden) result;
  Meta.simt_step call position layer phase (Native.packed_values vw) scope anchor m before keys result;
  Scratch.tensor_values result;
  Head.array_to_hidden hidden #(Scratch.tensor result) #1.0R;
  fold local_owners extra_owners m;
  fold Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output phase));
}

inline_for_extraction noextract
fn postlude
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Attention.full_attention_layer)
  (phase:sz{SZ.v phase < 23}) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912)
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#scope:prop)
  preserves gpu
  requires Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)))
  ensures Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase + 1)))
{
  unfold Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)));
  with m before keys. assert GR.pts_to book m ** hidden |-> before ** A.pts_to_mask cache keys Init.all_cells;
  Head.hidden_to_array hidden #before #1.0R;
  if (FStar.SizeT.eq phase 22sz) {
    Scalar.apply Scalar.Residual (T.core hidden);
  };
  Meta.postlude_valid call position layer (SZ.v phase) (Native.packed_values vw) scope anchor m before keys;
  Scratch.tensor_values (Meta.postlude (SZ.v phase) (Native.hidden_values before));
  Head.array_to_hidden hidden #(Scratch.tensor (Meta.postlude (SZ.v phase) (Native.hidden_values before))) #1.0R;
  fold Book.full book hidden cache (local_owners extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase + 1)));
}

inline_for_extraction noextract
fn matrix
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Attention.full_attention_layer)
  (phase:sz{SZ.v phase < 23})
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (tid:sz{SZ.v tid < 32})
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm) (#scope:prop)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))
  ensures Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase))))
{
  if (3sz <=^ phase && phase <^ 19sz) {
    let length = SZ.uint_to_t (position + 1);
    if (FStar.SizeT.ne (length %^ 8sz) 0sz) {
      Book.keep_wait (fun i -> Head.lane_only i
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))) #(SZ.v tid);
      if (FStar.SizeT.eq tid 0sz) {
        rewrite Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))
          as Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
        simt book call position layer (SZ.v phase) hidden cache extra_owners anchor #vw #scope;
        rewrite Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)))
          as Head.lane_only (SZ.v tid)
            (Book.full book hidden cache (local_owners extra_owners)
              (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase))));
      } else {
        rewrite Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))
          as Head.lane_only (SZ.v tid)
            (Book.full book hidden cache (local_owners extra_owners)
              (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase))));
      };
      Book.keep_wait (fun i -> Head.lane_only i
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase))))) #(SZ.v tid);
    } else {
      FStar.Classical.forall_intro_3
        (Meta.initialized call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
      Phase.cache book call position layer (SZ.v phase) hidden cache tid (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))) ();
      FStar.Classical.forall_intro_3
        (Meta.cache_post call position layer (SZ.v phase) (Native.packed_values vw) scope anchor);
      Book.strengthen book hidden cache (local_owners extra_owners)
        (Phase.cache_post call position layer (SZ.v phase)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)))
        (SZ.v tid) ();
    }
  } else {
    Phase.weight book call position layer (SZ.v phase) weights hidden cache tid (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
    FStar.Classical.forall_intro_3 (Meta.weight_post call position layer (SZ.v phase) vw scope anchor);
    Book.strengthen book hidden cache (local_owners extra_owners)
      (Phase.weight_post call position layer (SZ.v phase) vw
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase))))
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase))) (SZ.v tid) ();
  }
}

inline_for_extraction noextract
fn apply
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Attention.full_attention_layer)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (tid:sz{SZ.v tid < 32})
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm) (#scope:prop)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor 0))
  ensures Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor 37))
{
  let mut phase = 0sz;
  while (!phase <^ 23sz)
    invariant exists* (p:sz{SZ.v p <= 23}).
      phase |-> p ** Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
    invariant gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
    decreases (23 - SZ.v !phase)
  {
    let p = !phase;
    if (FStar.SizeT.eq tid 0sz) {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
        as Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p)));
      prelude book call position layer p weights hidden cache extra_owners anchor #vw #fraction #scope;
      rewrite Book.full book hidden cache (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p)))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p))));
    } else {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p))));
    };
    matrix book call position layer p weights hidden cache tid extra_owners anchor #vw #fraction #scope;
    if (FStar.SizeT.eq tid 0sz) {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p))))
        as Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p)));
      postlude book call position layer p hidden cache extra_owners anchor #vw #scope;
      rewrite Book.full book hidden cache (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1)))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))));
    } else {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p))))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))));
    };
    Book.keep_wait (fun i -> Head.lane_only i
      (Book.full book hidden cache (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))))) #(SZ.v tid);
    let next = SZ.add p 1sz;
    rewrite Head.lane_only (SZ.v tid)
      (Book.full book hidden cache (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))))
      as Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v next))));
    phase := next;
  };
  let completed = !phase;
  assert pure (SZ.v completed == 23);
  rewrite Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v completed))))
    as Head.lane_only (SZ.v tid)
      (Book.full book hidden cache (local_owners extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor 37));
}
