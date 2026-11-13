module KPlay.Proof.Base.CacheInvariant

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Seq = FStar.Seq
module Model = KPlay.Proof.Shared.Model
module A = KPlay.Proof.Shared.Attention
module Cache = KPlay.Proof.Base.CacheNative
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let ready (before:Cache.snapshot) (position:nat{position < 2048}) (next:Model.cursor) : prop =
  forall (layer:A.full_attention_layer).
    Cache.initialized before layer (Model.visits position next layer)

inline_for_extraction noextract
let initial (before:Cache.snapshot) : Lemma (ready before 0 0) = ()

inline_for_extraction noextract
let other_prefix (before:Cache.snapshot) (hidden:Transfer.scratch)
  (layer other:A.full_attention_layer{layer <> other})
  (position:nat{position < 2048}) (length:nat{length <= 2048})
  : Lemma
      (requires Cache.initialized before other length)
      (ensures Cache.initialized (Cache.appended before hidden layer position 512) other length) =
  let at_index (value:bool) (i:nat{i < length * 512})
    : Lemma (
        Some? (Seq.index (Cache.appended before hidden layer position 512) (Cache.plane_offset other value + i))) =
    Cache.plane_bounds other value;
    Cache.append_frame before hidden layer position (Cache.plane_offset other value + i);
    assert (Some? (Seq.index before (Cache.plane_offset other value + i)))
  in
  FStar.Classical.forall_intro_2 at_index

inline_for_extraction noextract
let append_advances (before:Cache.snapshot) (hidden:Transfer.scratch)
  (position:nat{position < 2048}) (layer:A.full_attention_layer)
  : Lemma
      (requires ready before position layer)
      (ensures ready (Cache.appended before hidden layer position 512) position (layer + 1)) =
  let at_layer (other:A.full_attention_layer)
    : Lemma (
        Cache.initialized (Cache.appended before hidden layer position 512) other
          (Model.visits position (layer + 1) other)) =
    if other = layer then Cache.append_extends before hidden layer position
    else (
      other_prefix before hidden layer other position (Model.visits position layer other);
      assert (Model.visits position layer other == Model.visits position (layer + 1) other)
    )
  in
  FStar.Classical.forall_intro at_layer

inline_for_extraction noextract
let linear_advances (before:Cache.snapshot) (position:nat{position < 2048})
  (layer:Model.linear_layer)
  : Lemma
      (requires ready before position layer)
      (ensures ready before position (layer + 1)) =
  let at_layer (other:A.full_attention_layer)
    : Lemma (Cache.initialized before other (Model.visits position (layer + 1) other)) =
    assert (other <> layer);
    assert (Model.visits position layer other == Model.visits position (layer + 1) other)
  in
  FStar.Classical.forall_intro at_layer

inline_for_extraction noextract
let boundary (before:Cache.snapshot) (position:nat{position + 1 < 2048})
  : Lemma
      (requires ready before position 24)
      (ensures ready before (position + 1) 0) = ()

inline_for_extraction noextract
let current_prefix (before:Cache.snapshot) (position:nat{position < 2048})
  (layer:A.full_attention_layer)
  : Lemma
      (requires ready before position (layer + 1))
      (ensures Cache.initialized before layer (position + 1)) = ()
