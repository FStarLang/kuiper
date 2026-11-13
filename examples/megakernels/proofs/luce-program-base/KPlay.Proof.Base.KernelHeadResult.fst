module KPlay.Proof.Base.KernelHeadResult

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module I64 = FStar.Int64
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Head = KPlay.Proof.Base.KernelHead
module Trace = KPlay.Proof.Base.Trace
module Inputs = KPlay.Proof.Base.Inputs
module Transfer = KPlay.Proof.Base.Transfer
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module Linear = KPlay.Proof.Base.LinearBook
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Order = KPlay.Proof.Base.Order

[@@"opaque_to_smt"]
inline_for_extraction noextract
let projected (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot) : prop =
  K.Normalized? m.K.phase /\
  K.head_ready call weights prompt position m (Linear.tensor (Head.values call m)) kv /\
  h == Vocabulary.scratch_after weights (Linear.tensor (Head.values call m)) 15520

inline_for_extraction noextract
let projection_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.head_ready call weights prompt position m h kv)
      (ensures projected call weights prompt position m (Vocabulary.scratch_after weights h 15520) kv) =
  Head.ready_facts call weights prompt position m h kv;
  norm_spec [delta_once ["KPlay.Proof.Base.KernelHeadResult.projected"]]
    (projected call weights prompt position m (Vocabulary.scratch_after weights h 15520) kv)

inline_for_extraction noextract
let projected_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires projected call weights prompt position m h kv)
      (ensures K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24 /\
        K.head_ready call weights prompt position m (Linear.tensor (Head.values call m)) kv /\
        h == Vocabulary.scratch_after weights (Linear.tensor (Head.values call m)) 15520) =
  norm_spec [delta_once ["KPlay.Proof.Base.KernelHeadResult.projected"]]
    (projected call weights prompt position m h kv);
  Head.ready_facts call weights prompt position m (Linear.tensor (Head.values call m)) kv

inline_for_extraction noextract
let finish (call:Model.invocation) (weights:Transfer.packed)
  (m:K.metadata call{K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24})
  (winner:Order.selection{winner.Order.token < 248320}) (word:I64.t)
  : GTot (K.metadata call) =
  {m with K.memory = {m.K.memory with K.best = winner.Order.best; K.next = word};
    K.phase = K.Emitting (Head.observation call weights (Head.current call m) (Head.values call m)) winner.Order.token}

inline_for_extraction noextract
let finish_memory (call:Model.invocation) (weights:Transfer.packed)
  (m:K.metadata call{K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24})
  (winner:Order.selection{winner.Order.token < 248320}) (word:I64.t)
  : Lemma ((finish call weights m winner word).K.memory ==
      {m.K.memory with K.best = winner.Order.best; K.next = word}) = ()

inline_for_extraction noextract
let finish_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  (winner:Order.selection{winner.Order.token < 248320}) (word:I64.t)
  : Lemma
      (requires
        K.Normalized? m.K.phase /\
        projected call weights prompt position m h kv /\
        winner == Vocabulary.scan_blocks weights (Head.values call m) 15520 /\
        I64.v word == winner.Order.token)
      (ensures
        (Head.current call m).K.layer == 24 /\
        K.emitting call weights prompt position (finish call weights m winner word) h kv) =
  projected_facts call weights prompt position m h kv;
  Head.ready_facts call weights prompt position m (Linear.tensor (Head.values call m)) kv;
  Head.observation_valid call weights prompt position m (Linear.tensor (Head.values call m)) kv;
  Trace.consumed call (Inputs.parameter_values weights) prompt K.configuration m.K.history m.K.completed;
  Head.same_history call weights prompt m (finish call weights m winner word);
  norm_spec [delta_once ["KPlay.Proof.Base.KernelState.emitting"]]
    (K.emitting call weights prompt position (finish call weights m winner word) h kv)
