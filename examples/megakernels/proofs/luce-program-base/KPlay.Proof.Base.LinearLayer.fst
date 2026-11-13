module KPlay.Proof.Base.LinearLayer

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
module Model = KPlay.Proof.Shared.Model
module Stages = KPlay.Proof.Base.LinearStages
module Meta = KPlay.Proof.Base.LinearBook
module Book = KPlay.Proof.Base.WarpBook
module Phase = KPlay.Proof.Base.PhaseMMA
module Head = KPlay.Proof.Base.Head
module Hidden = KPlay.Proof.Base.WarpHidden
module Native = KPlay.Proof.Base.NativeProjection
module Arrays = KPlay.Proof.Base.SourceArrays
module Cache = KPlay.Proof.Base.CacheNative
module Init = KPlay.Proof.Base.Initialization
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer
module RMS = KPlay.Proof.Base.RMS
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Mixer = KPlay.Proof.Base.RecurrentMixer

inline_for_extraction noextract
let local_owners (#extra:Type0) (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (extra_owners:extra -> slprop) (m:Meta.metadata extra) : slprop =
  state |-> m.Meta.state ** A.pts_to conv m.Meta.convolution **
  extra_owners m.Meta.root.Meta.payload

inline_for_extraction noextract
fn simple_prelude
  (layer:Model.linear_layer) (phase:sz{SZ.v phase < 8 /\ SZ.v phase <> 4})
  (weights:larray BF.t 752393024) (hidden:larray BF.t 16000)
  (#vw:erased Transfer.packed) (#vh:erased Transfer.scratch) (#fraction:perm)
  preserves gpu ** A.pts_to weights #fraction vw
  requires A.pts_to hidden vh
  ensures A.pts_to hidden (Meta.prelude layer (SZ.v phase) vw vh)
{
  if (FStar.SizeT.eq phase 0sz) {
    RMS.source_apply (Config.model_norm layer false) weights hidden;
  } else if (FStar.SizeT.eq phase 5sz) {
    Scalar.apply Scalar.Residual hidden;
    RMS.source_apply (Config.model_norm layer true) weights hidden;
  } else if (FStar.SizeT.eq phase 7sz) {
    Scalar.apply Scalar.Mlp hidden;
  }
}

inline_for_extraction noextract
fn prelude
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:sz{SZ.v phase < 8})
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm) (#scope:prop)
  preserves gpu ** weights |-> Frac fraction vw
  requires Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase)))
  ensures Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)))
{
  unfold Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase)));
  with m before keys. assert GR.pts_to book m ** hidden |-> before ** A.pts_to_mask cache keys Init.all_cells;
  unfold local_owners state conv extra_owners m;
  Arrays.weight_to_array weights;
  Head.hidden_to_array hidden #before #1.0R;
  if (FStar.SizeT.eq phase 4sz) {
    Meta.mixer_ready call position layer (Native.packed_values vw) scope anchor m before keys;
    Inputs.layer_extent layer;
    let base = SZ.uint_to_t (Inputs.layer_base layer);
    let slot = SZ.uint_to_t (Meta.slot layer);
    Mixer.apply (T.core weights) (T.core hidden) state conv base slot
      #(reveal #(Meta.metadata extra) m).Meta.root.Meta.logical
      #(reveal #(Meta.metadata extra) m).Meta.root.Meta.stages
      #(Native.packed_values vw) #(Native.hidden_values before)
      #(reveal #(Meta.metadata extra) m).Meta.state #(reveal #(Meta.metadata extra) m).Meta.convolution
      #fraction #scope;
    with result updated. assert A.pts_to (T.core hidden) result ** state |-> updated;
    Meta.mixer_step call position layer (Native.packed_values vw) scope anchor m before keys result updated;
    let next : erased (Meta.metadata extra) =
      hide #(Meta.metadata extra) (Meta.mixed layer m (Native.hidden_values before) result updated);
    GR.write book next;
    Meta.tensor_values result;
    Head.array_to_hidden hidden #(Meta.tensor result) #1.0R;
    rewrite extra_owners (reveal #(Meta.metadata extra) m).Meta.root.Meta.payload
      as extra_owners (reveal #(Meta.metadata extra) next).Meta.root.Meta.payload;
    fold local_owners state conv extra_owners next;
    Arrays.array_to_weight weights;
    fold Book.full book hidden cache (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
  } else {
    simple_prelude layer phase (T.core weights) (T.core hidden);
    Meta.prelude_valid call position layer (SZ.v phase) (Native.packed_values vw) scope anchor m before keys;
    Meta.tensor_values (Meta.prelude layer (SZ.v phase) (Native.packed_values vw) (Native.hidden_values before));
    Head.array_to_hidden hidden
      #(Meta.tensor (Meta.prelude layer (SZ.v phase) (Native.packed_values vw) (Native.hidden_values before))) #1.0R;
    fold local_owners state conv extra_owners m;
    Arrays.array_to_weight weights;
    fold Book.full book hidden cache (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v phase)));
  };
}

inline_for_extraction noextract
fn postlude
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:sz{SZ.v phase < 8}) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#scope:prop)
  preserves gpu
  requires Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)))
  ensures Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase + 1)))
{
  unfold Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v phase)));
  with m before keys. assert GR.pts_to book m ** hidden |-> before ** A.pts_to_mask cache keys Init.all_cells;
  Head.hidden_to_array hidden #before #1.0R;
  if (FStar.SizeT.eq phase 7sz) {
    Scalar.apply Scalar.Residual (T.core hidden);
  };
  Meta.postlude_valid call position layer (SZ.v phase) (Native.packed_values vw) scope anchor m before keys;
  Meta.tensor_values (Meta.postlude (SZ.v phase) (Native.hidden_values before));
  Head.array_to_hidden hidden #(Meta.tensor (Meta.postlude (SZ.v phase) (Native.hidden_values before))) #1.0R;
  fold Book.full book hidden cache (local_owners state conv extra_owners)
    (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v phase + 1)));
}

inline_for_extraction noextract
fn apply
  (#extra:Type0) (book:GR.ref (Meta.metadata extra))
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (weights:array2 BF.t Native.weight_layout) (hidden:array2 BF.t Native.hidden_layout)
  (cache:larray BF.t 12582912) (state:larray FP.t 4718592) (conv:larray BF.t 442368)
  (tid:sz{SZ.v tid < 32})
  (extra_owners:extra -> slprop) (anchor:Meta.root extra -> prop)
  (#vw:C.chest2 BF.t 752393024 1) (#fraction:perm) (#scope:prop)
  preserves gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
  requires Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor 0))
  ensures Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor 13))
{
  let mut phase = 0sz;
  while (!phase <^ 8sz)
    invariant exists* (p:sz{SZ.v p <= 8}).
      phase |-> p ** Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
    invariant gpu ** thread_id 32 (SZ.v tid) ** weights |-> Frac fraction vw
    decreases (8 - SZ.v !phase)
  {
    let p = !phase;
    if (FStar.SizeT.eq tid 0sz) {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
        as Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p)));
      prelude book call position layer p weights hidden cache state conv extra_owners anchor #vw #fraction #scope;
      rewrite Book.full book hidden cache (local_owners state conv extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p)))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners state conv extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p))));
    } else {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p))))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners state conv extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p))));
    };
    Phase.weight book call position layer (SZ.v p) weights hidden cache tid
      (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p)));
    FStar.Classical.forall_intro_3 (Meta.matrix_post call position layer (SZ.v p) vw scope anchor);
    Book.strengthen book hidden cache (local_owners state conv extra_owners)
      (Phase.weight_post call position layer (SZ.v p) vw
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_input (SZ.v p))))
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p)))
      (SZ.v tid) ();
    if (FStar.SizeT.eq tid 0sz) {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p))))
        as Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p)));
      postlude book call position layer p hidden cache state conv extra_owners anchor #vw #scope;
      rewrite Book.full book hidden cache (local_owners state conv extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1)))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners state conv extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))));
    } else {
      rewrite Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.matrix_output (SZ.v p))))
        as Head.lane_only (SZ.v tid)
          (Book.full book hidden cache (local_owners state conv extra_owners)
            (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))));
    };
    Book.keep_wait (fun i -> Head.lane_only i
      (Book.full book hidden cache (local_owners state conv extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))))) #(SZ.v tid);
    let next = SZ.add p 1sz;
    rewrite Head.lane_only (SZ.v tid)
      (Book.full book hidden cache (local_owners state conv extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v p + 1))))
      as Head.lane_only (SZ.v tid)
        (Book.full book hidden cache (local_owners state conv extra_owners)
          (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v next))));
    phase := next;
  };
  let completed = !phase;
  assert pure (SZ.v completed == 8);
  rewrite Head.lane_only (SZ.v tid)
    (Book.full book hidden cache (local_owners state conv extra_owners)
      (Meta.valid call position layer (Native.packed_values vw) scope anchor (Stages.checkpoint (SZ.v completed))))
    as Head.lane_only (SZ.v tid)
      (Book.full book hidden cache (local_owners state conv extra_owners)
        (Meta.valid call position layer (Native.packed_values vw) scope anchor 13));
}
