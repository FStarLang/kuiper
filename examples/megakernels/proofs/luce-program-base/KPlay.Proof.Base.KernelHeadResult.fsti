module KPlay.Proof.Base.KernelHeadResult

module I64 = FStar.Int64
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module K = KPlay.Proof.Base.KernelState
module Head = KPlay.Proof.Base.KernelHead
module Transfer = KPlay.Proof.Base.Transfer
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module Linear = KPlay.Proof.Base.LinearBook
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Order = KPlay.Proof.Base.Order

inline_for_extraction noextract
val projected (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot) : prop

inline_for_extraction noextract
val projection_valid (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires K.head_ready call weights prompt position m h kv)
      (ensures projected call weights prompt position m (Vocabulary.scratch_after weights h 15520) kv)

inline_for_extraction noextract
val projected_facts (call:Model.invocation) (weights:Transfer.packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (position:nat{position < Model.calls call})
  (m:K.metadata call) (h:Hidden.snapshot) (kv:Cache.snapshot)
  : Lemma
      (requires projected call weights prompt position m h kv)
      (ensures K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24 /\
        K.head_ready call weights prompt position m (Linear.tensor (Head.values call m)) kv /\
        h == Vocabulary.scratch_after weights (Linear.tensor (Head.values call m)) 15520)

inline_for_extraction noextract
val finish (call:Model.invocation) (weights:Transfer.packed)
  (m:K.metadata call{K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24})
  (winner:Order.selection{winner.Order.token < 248320}) (word:I64.t)
  : GTot (K.metadata call)

inline_for_extraction noextract
val finish_memory (call:Model.invocation) (weights:Transfer.packed)
  (m:K.metadata call{K.Normalized? m.K.phase /\ (Head.current call m).K.layer == 24})
  (winner:Order.selection{winner.Order.token < 248320}) (word:I64.t)
  : Lemma ((finish call weights m winner word).K.memory ==
      {m.K.memory with K.best = winner.Order.best; K.next = word})

inline_for_extraction noextract
val finish_valid (call:Model.invocation) (weights:Transfer.packed)
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
        K.emitting call weights prompt position (finish call weights m winner word) h kv)
