module KPlay.Proof.Base.LinearBook

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
module Seq = FStar.Seq
module C = Kuiper.Chest
module T = Kuiper.Tensor
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module Approx = Kuiper.Approximates.Base
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent
module O = KPlay.Proof.Shared.Observations
module Model = KPlay.Proof.Shared.Model
module Prior = KPlay.Proof.Base.RecurrentPipeline
module Stages = KPlay.Proof.Base.LinearStages
module Transfer = KPlay.Proof.Base.Transfer
module Inputs = KPlay.Proof.Base.Inputs
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Head = KPlay.Proof.Base.Head
module Cache = KPlay.Proof.Base.CacheNative
module Conv = KPlay.Proof.Base.Convolution
module Convolution = KPlay.Proof.Base.ConvolutionLoop
module Mixer = KPlay.Proof.Base.RecurrentMixer
module Heads = KPlay.Proof.Base.RecurrentHeads
module Math = KPlay.Proof.Base.ScratchMath
module Source = KPlay.Proof.Base.SourceProjection
module Schedule = KPlay.Proof.Base.Schedule
module Writes = KPlay.Proof.Base.MatvecWrites
module SZ = Kuiper.SizeT
module RMS = KPlay.Proof.Base.RMS
module RMSMath = KPlay.Proof.Base.RMSMath
module Config = KPlay.Proof.Base.RMSConfigs
module Scalar = KPlay.Proof.Base.Scalar
module Phase = KPlay.Proof.Base.PhaseMMA
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let slot (layer:Model.linear_layer) : (n:nat{n < 18}) = layer - layer / 4

noeq type root (extra:Type0) = {
  payload:extra;
  logical:Prior.logical_call;
  stages:Prior.decoder_stages;
  hidden:Transfer.scratch;
  state:lseq FP.t 4718592;
  convolution:Conv.cells;
  cache:Cache.snapshot;
}

noeq type metadata (extra:Type0) = {
  root:root extra;
  state:lseq FP.t 4718592;
  convolution:Conv.cells;
  mixer_input:Transfer.scratch;
  mixer_result:Transfer.scratch;
}

inline_for_extraction noextract
let tensor (values:Transfer.scratch) : Hidden.snapshot =
  T.from_seq Native.hidden_layout values

inline_for_extraction noextract
let tensor_values (values:Transfer.scratch)
  : Lemma (Native.hidden_values (tensor values) == values) =
  Head.flat_hidden (tensor values);
  T.to_from Native.hidden_layout values

inline_for_extraction noextract
let tensor_roundtrip (values:Hidden.snapshot)
  : Lemma (tensor (Native.hidden_values values) == values) =
  tensor_values (Native.hidden_values values);
  Hidden.values_agree (tensor (Native.hidden_values values)) values

inline_for_extraction noextract
let same_rope (before after:Transfer.scratch) : prop =
  forall (i:P.index 64). Seq.index after (15600 + i) == Seq.index before (15600 + i)

inline_for_extraction noextract
let initial_numeric (#extra:Type0) (r:root extra) : prop =
  let c = r.logical in
  let layer = c.Prior.call.R.layer_index in
  Math.represents #1024 r.hidden 0 c.Prior.input /\
  (forall (channel:P.index 6144) (tap:P.index 4).
    Approx.v_approximates (Seq.index r.convolution (Conv.address (slot layer) channel tap))
      (Prior.old_convolution c.Prior.call c.Prior.before channel tap)) /\
  (forall (head:P.index 16) (i j:P.index 128).
    Approx.v_approximates (Seq.index r.state (Heads.offset (slot layer) head + i * 128 + j))
      (Prior.old_recurrent c.Prior.call c.Prior.before head i j))

inline_for_extraction noextract
let initial_numeric_views (#extra:Type0) (r:root extra)
  : Lemma
      (requires Math.represents #1024 r.hidden 0 r.logical.Prior.input /\
        O.matrix_approximates (Init.convolution_layer r.convolution r.logical.Prior.call.R.layer_index)
          (Prior.old_convolution r.logical.Prior.call r.logical.Prior.before) /\
        O.tensor_approximates (Init.state_layer r.state r.logical.Prior.call.R.layer_index)
          (Prior.old_recurrent r.logical.Prior.call r.logical.Prior.before))
      (ensures initial_numeric r) =
  let layer = r.logical.Prior.call.R.layer_index in
  let convolution_cell (channel:P.index 6144) (tap:P.index 4)
    : Lemma (Approx.v_approximates (Seq.index r.convolution (Conv.address (slot layer) channel tap))
        (Prior.old_convolution r.logical.Prior.call r.logical.Prior.before channel tap)) = ()
  in
  FStar.Classical.forall_intro_2 convolution_cell;
  let state_cell (head:P.index 16) (i j:P.index 128)
    : Lemma (Approx.v_approximates (Seq.index r.state (Heads.offset (slot layer) head + i * 128 + j))
        (Prior.old_recurrent r.logical.Prior.call r.logical.Prior.before head i j)) = ()
  in
  FStar.Classical.forall_intro_3 state_cell

inline_for_extraction noextract
let mixer_numeric (#extra:Type0) (m:metadata extra) : prop =
  Stages.live m.root.logical m.root.stages m.mixer_input 5 /\
  Heads.completed m.state m.mixer_result (slot m.root.logical.Prior.call.R.layer_index)
    (Mixer.head_models m.root.stages) 16

[@@"opaque_to_smt"]
inline_for_extraction noextract
let valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (point:Stages.point) (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot) : prop =
  let r = m.root in
  anchor r /\
  r.logical.Prior.call == Model.recurrent_call call position layer /\
  r.logical.Prior.weight == O.model_linear ((Inputs.parameter_values weights).O.linear layer) /\
  r.stages == Prior.build_stages r.logical /\
  cache == r.cache /\
  (point = 0 ==> Native.hidden_values h == r.hidden) /\
  same_rope r.hidden (Native.hidden_values h) /\
  (if point < 6 then m.state == r.state /\ m.convolution == r.convolution
   else Heads.state_frame r.state m.state (slot layer) 16 /\
     m.convolution == Convolution.after r.convolution m.mixer_input (slot layer) 6144) /\
  (scope ==> initial_numeric r /\
    Stages.live r.logical r.stages (Native.hidden_values h) point /\
    (6 <= point ==> mixer_numeric m))

inline_for_extraction noextract
let hold (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (point next:Stages.point{next > 0 /\ (point < 6 <==> next < 6)})
  (m:metadata extra) (before after:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor point m before cache /\
        same_rope (Native.hidden_values before) (Native.hidden_values after) /\
        (scope ==> Stages.live m.root.logical m.root.stages (Native.hidden_values after) next))
      (ensures valid call position layer weights scope anchor next m after cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor point m before cache);
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor next m after cache)

inline_for_extraction noextract
let matrix_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:Stages.phase) (weights:C.chest2 BF.t 752393024 1) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (cache:Cache.snapshot) (before:Hidden.snapshot)
  : Lemma (
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase) m before cache ==>
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_output phase) m
        (Writes.after (Source.weight_geometry call position layer phase) weights before
          (Schedule.projection_at call position layer phase).Schedule.rows) cache) =
  let result = Writes.after (Source.weight_geometry call position layer phase) weights before
    (Schedule.projection_at call position layer phase).Schedule.rows in
  let under (_:valid call position layer (Native.packed_values weights) scope anchor
      (Stages.matrix_input phase) m before cache)
    : Lemma (valid call position layer (Native.packed_values weights) scope anchor
        (Stages.matrix_output phase) m result cache) =
    norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
      (valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase) m before cache);
    Stages.projection call position layer phase weights before m.root.logical m.root.stages scope;
    Math.mma_frame (Source.weight_geometry call position layer phase) weights before;
    Schedule.bounds call position layer phase;
    assert (same_rope (Native.hidden_values before) (Native.hidden_values result));
    hold call position layer (Native.packed_values weights) scope anchor
      (Stages.matrix_input phase) (Stages.matrix_output phase) m before result cache
  in
  FStar.Classical.impl_intro under

inline_for_extraction noextract
let matrix_post (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:Stages.phase) (weights:C.chest2 BF.t 752393024 1) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma (
      Phase.weight_post call position layer phase weights
        (valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_input phase)) m h cache ==>
      valid call position layer (Native.packed_values weights) scope anchor (Stages.matrix_output phase) m h cache) =
  FStar.Classical.forall_intro (matrix_step call position layer phase weights scope anchor m cache)

inline_for_extraction noextract
let mixer_ready (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (h:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor 5 m h cache)
      (ensures scope ==> Mixer.inputs m.root.logical m.root.stages weights (Native.hidden_values h)
        m.state m.convolution (Inputs.layer_base layer) (slot layer)) =
  Inputs.layer_extent layer;
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor 5 m h cache);
  let weight_cell (i:P.index 752393024)
    : Lemma (Approx.v_approximates (Seq.index weights i) (Approx.to_real (Seq.index weights i))) =
    Approx.to_real_ok (Seq.index weights i)
  in
  FStar.Classical.forall_intro weight_cell;
  norm_spec [delta_once ["KPlay.Proof.Base.RecurrentMixer.inputs"]]
    (Mixer.inputs m.root.logical m.root.stages weights (Native.hidden_values h)
      m.state m.convolution (Inputs.layer_base layer) (slot layer))

inline_for_extraction noextract
let mixed (#extra:Type0) (layer:Model.linear_layer) (before:metadata extra) (input result:Transfer.scratch)
  (state:lseq FP.t 4718592) : GTot (metadata extra) = {
  before with
  state;
  convolution = Convolution.after before.convolution input (slot layer) 6144;
  mixer_input = input;
  mixer_result = result;
}

inline_for_extraction noextract
let mixer_step (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  (result:Transfer.scratch) (state:lseq FP.t 4718592)
  : Lemma
      (requires valid call position layer weights scope anchor 5 m before cache /\
        Heads.state_frame m.state state (slot layer) 16 /\ Mixer.outside (Native.hidden_values before) result /\
        (scope ==> Heads.completed state result (slot layer) (Mixer.head_models m.root.stages) 16))
      (ensures valid call position layer weights scope anchor 6
        (mixed layer m (Native.hidden_values before) result state) (tensor result) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor 5 m before cache);
  tensor_values result;
  Stages.mixer scope (Native.hidden_values before) result state (slot layer) m.root.logical m.root.stages;
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor 6
      (mixed layer m (Native.hidden_values before) result state) (tensor result) cache)

inline_for_extraction noextract
let prelude (layer:Model.linear_layer) (phase:Stages.phase{phase <> 4})
  (weights:Transfer.packed) (before:Transfer.scratch) : GTot Transfer.scratch =
  if phase = 0 then RMS.after (Config.model_norm layer false) before weights 1
  else if phase = 5 then RMS.after (Config.model_norm layer true) (Scalar.writes Scalar.Residual before 1024) weights 1
  else if phase = 7 then Scalar.writes Scalar.Mlp before 3584
  else before

inline_for_extraction noextract
let prelude_rope (layer:Model.linear_layer) (phase:Stages.phase{phase <> 4})
  (weights:Transfer.packed) (before:Transfer.scratch)
  : Lemma (same_rope before (prelude layer phase weights before)) =
  let at_index (i:P.index 64)
    : Lemma (Seq.index (prelude layer phase weights before) (15600 + i) == Seq.index before (15600 + i)) =
    if phase = 0 then
      RMSMath.outside_frame (Config.model_norm layer false) before weights 1 (15600 + i)
    else if phase = 5 then (
      Scalar.outside_frame Scalar.Residual before (15600 + i);
      RMSMath.outside_frame (Config.model_norm layer true) (Scalar.writes Scalar.Residual before 1024) weights 1 (15600 + i)
    ) else if phase = 7 then Scalar.outside_frame Scalar.Mlp before (15600 + i)
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let prelude_valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:Stages.phase{phase <> 4}) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache)
      (ensures valid call position layer weights scope anchor (Stages.matrix_input phase)
        m (tensor (prelude layer phase weights (Native.hidden_values before))) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor (Stages.checkpoint phase) m before cache);
  let h = Native.hidden_values before in
  let result = prelude layer phase weights h in
  tensor_values result;
  if phase = 0 then (
    Stages.first_norm scope layer weights h m.root.logical m.root.stages
  ) else if phase = 5 then (
    Stages.post_attention scope layer weights h m.root.logical m.root.stages
  ) else if phase = 7 then (
    Stages.product scope h m.root.logical m.root.stages
  );
  prelude_rope layer phase weights h;
  hold call position layer weights scope anchor (Stages.checkpoint phase) (Stages.matrix_input phase)
    m before (tensor result) cache

inline_for_extraction noextract
let postlude (phase:Stages.phase) (before:Transfer.scratch) : GTot Transfer.scratch =
  if phase = 7 then Scalar.writes Scalar.Residual before 1024 else before

inline_for_extraction noextract
let postlude_valid (#extra:Type0)
  (call:Model.invocation) (position:nat{position < Model.calls call}) (layer:Model.linear_layer)
  (phase:Stages.phase) (weights:Transfer.packed) (scope:prop) (anchor:root extra -> prop)
  (m:metadata extra) (before:Hidden.snapshot) (cache:Cache.snapshot)
  : Lemma
      (requires valid call position layer weights scope anchor (Stages.matrix_output phase) m before cache)
      (ensures valid call position layer weights scope anchor (Stages.checkpoint (phase + 1))
        m (tensor (postlude phase (Native.hidden_values before))) cache) =
  norm_spec [delta_once ["KPlay.Proof.Base.LinearBook.valid"]]
    (valid call position layer weights scope anchor (Stages.matrix_output phase) m before cache);
  let h = Native.hidden_values before in
  let result = postlude phase h in
  tensor_values result;
  if phase = 7 then (
    Stages.final_residual scope h m.root.logical m.root.stages;
    Math.scalar_frame Scalar.Residual h
  );
  assert (same_rope h result);
  hold call position layer weights scope anchor (Stages.matrix_output phase) (Stages.checkpoint (phase + 1))
    m before (tensor result) cache
