module KPlay.Proof.Base.CacheNative

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module P = KPlay.Proof.Shared.Projection
module Attention = KPlay.Proof.Shared.Attention
module Cache = KPlay.Proof.Base.Cache
module Transfer = KPlay.Proof.Base.Transfer
module Init = KPlay.Proof.Base.Initialization

inline_for_extraction noextract
let snapshot = lseq (option BF.t) 12582912

inline_for_extraction noextract
let plane_offset (layer:Attention.full_attention_layer) (value:bool) : nat =
  ((layer / 4) * 4096 + (if value then 2048 else 0)) * 512

inline_for_extraction noextract
let plane_bounds (layer:Attention.full_attention_layer) (value:bool)
  : Lemma (plane_offset layer value + 2048 * 512 <= 12582912) = ()

inline_for_extraction noextract
let initialized (before:snapshot) (layer:Attention.full_attention_layer)
  (length:nat{length <= 2048}) : prop =
  forall (value:bool) (i:nat{i < length * 512}).
    Some? (Seq.index before (plane_offset layer value + i))

inline_for_extraction noextract
let appended (before:snapshot) (hidden:Transfer.scratch) (layer:Attention.full_attention_layer)
  (position:nat{position < 2048}) (count:nat{count <= 512}) : GTot snapshot =
  let key = plane_offset layer false + position * 512 in
  let value = plane_offset layer true + position * 512 in
  Seq.init_ghost 12582912 (fun i ->
    if key <= i && i < key + count then Some (Seq.index hidden (6144 + i - key))
    else if value <= i && i < value + count then Some (Seq.index hidden (6656 + i - value))
    else Seq.index before i)

inline_for_extraction noextract
let appended_zero (before:snapshot) (hidden:Transfer.scratch) (layer:Attention.full_attention_layer)
  (position:nat{position < 2048})
  : Lemma (appended before hidden layer position 0 == before) =
  assert (Seq.equal (appended before hidden layer position 0) before);
  Seq.lemma_eq_elim (appended before hidden layer position 0) before

inline_for_extraction noextract
let appended_step (before:snapshot) (hidden:Transfer.scratch) (layer:Attention.full_attention_layer)
  (position:nat{position < 2048}) (i:P.index 512)
  : Lemma (
      Seq.upd (Seq.upd (appended before hidden layer position i)
        (Cache.key_address layer position i) (Some (Seq.index hidden (6144 + i))))
        (Cache.value_address layer position i) (Some (Seq.index hidden (6656 + i))) ==
      appended before hidden layer position (i + 1)) =
  assert (Seq.equal
    (Seq.upd (Seq.upd (appended before hidden layer position i)
      (Cache.key_address layer position i) (Some (Seq.index hidden (6144 + i))))
      (Cache.value_address layer position i) (Some (Seq.index hidden (6656 + i))))
    (appended before hidden layer position (i + 1)));
  Seq.lemma_eq_elim
    (Seq.upd (Seq.upd (appended before hidden layer position i)
      (Cache.key_address layer position i) (Some (Seq.index hidden (6144 + i))))
      (Cache.value_address layer position i) (Some (Seq.index hidden (6656 + i))))
    (appended before hidden layer position (i + 1))

inline_for_extraction noextract
let append_extends (before:snapshot) (hidden:Transfer.scratch) (layer:Attention.full_attention_layer)
  (position:nat{position < 2048})
  : Lemma
      (requires initialized before layer position)
      (ensures initialized (appended before hidden layer position 512) layer (position + 1)) =
  let at_index (value:bool) (i:nat{i < (position + 1) * 512})
    : Lemma (Some? (Seq.index (appended before hidden layer position 512) (plane_offset layer value + i))) =
    if i < position * 512 then
      assert (Some? (Seq.index before (plane_offset layer value + i)))
  in
  FStar.Classical.forall_intro_2 at_index

inline_for_extraction noextract
let append_frame (before:snapshot) (hidden:Transfer.scratch) (layer:Attention.full_attention_layer)
  (position:nat{position < 2048}) (i:P.index 12582912)
  : Lemma (
      (~(plane_offset layer false + position * 512 <= i /\
         i < plane_offset layer false + (position + 1) * 512) /\
       ~(plane_offset layer true + position * 512 <= i /\
         i < plane_offset layer true + (position + 1) * 512)) ==>
      Seq.index (appended before hidden layer position 512) i == Seq.index before i) = ()

inline_for_extraction noextract
fn append
  (cache:larray BF.t 12582912) (hidden:larray BF.t 16000)
  (layer:sz{SZ.v layer < 24 /\ SZ.v layer % 4 = 3})
  (position:sz{SZ.v position < 2048})
  (#before:erased snapshot) (#vh:erased Transfer.scratch) (#hidden_fraction:perm)
  preserves gpu ** A.pts_to hidden #hidden_fraction vh
  requires A.pts_to_mask cache before Init.all_cells
  ensures A.pts_to_mask cache (appended before vh (SZ.v layer) (SZ.v position) 512) Init.all_cells
{
  appended_zero before vh (SZ.v layer) (SZ.v position);
  let base = SZ.mul (SZ.mul (layer /^ 4sz) 4096sz) 512sz;
  let key = SZ.add base (SZ.mul position 512sz);
  let value = SZ.add (SZ.add base (SZ.mul 2048sz 512sz)) (SZ.mul position 512sz);
  let mut i = 0sz;
  while (!i <^ 512sz)
    invariant exists* (j:sz{SZ.v j <= 512}).
      i |-> j ** A.pts_to_mask cache (appended before vh (SZ.v layer) (SZ.v position) (SZ.v j)) Init.all_cells
    invariant gpu ** A.pts_to hidden #hidden_fraction vh
    decreases (512 - SZ.v !i)
  {
    let j = !i;
    appended_step before vh (SZ.v layer) (SZ.v position) (SZ.v j);
    let k = hidden.(SZ.add 6144sz j);
    A.mask_write cache (SZ.add key j) k;
    let v = hidden.(SZ.add 6656sz j);
    A.mask_write cache (SZ.add value j) v;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
let outside (offset length:nat) (i:nat) : prop =
  ~(offset <= i /\ i < offset + length)

inline_for_extraction noextract
let prefix_view (cache:larray BF.t 12582912) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool) : GTot (larray BF.t (length * 512)) =
  A.gsub cache (plane_offset layer value) (plane_offset layer value + length * 512)

inline_for_extraction noextract
let prefix_binding (before:snapshot) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool) (contents:lseq BF.t (length * 512)) : prop =
  forall (i:nat{i < length * 512}).
    Seq.index before (plane_offset layer value + i) == Some (Seq.index contents i)

inline_for_extraction noextract
let plane_initialized (before:snapshot) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool)
  : Lemma
      (requires initialized before layer length)
      (ensures forall (i:nat{i < length * 512}).
        Some? (Seq.index
          (Seq.slice before (plane_offset layer value) (plane_offset layer value + length * 512)) i)) =
  plane_bounds layer value;
  let at_index (i:nat{i < length * 512})
    : Lemma (Some? (Seq.index
        (Seq.slice before (plane_offset layer value) (plane_offset layer value + length * 512)) i)) =
    assert (Some? (Seq.index before (plane_offset layer value + i)))
  in
  FStar.Classical.forall_intro at_index

inline_for_extraction noextract
let initialized_value (value:option BF.t{Some? value}) : BF.t =
  let Some result = value in result

inline_for_extraction noextract
let prefix_contents (before:snapshot) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool) (ready:squash (initialized before layer length))
  : GTot (lseq BF.t (length * 512)) =
  Seq.init_ghost (length * 512) (fun i ->
    initialized_value (Seq.index before (plane_offset layer value + i)))

inline_for_extraction noextract
let prefix_unique (before:snapshot) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool) (ready:squash (initialized before layer length))
  (contents:lseq BF.t (length * 512))
  : Lemma
      (requires prefix_binding before layer length value contents)
      (ensures contents == prefix_contents before layer length value ready) =
  assert (Seq.equal contents (prefix_contents before layer length value ready));
  Seq.lemma_eq_elim contents (prefix_contents before layer length value ready)

inline_for_extraction noextract
fn borrow_prefix
  (cache:larray BF.t 12582912) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool)
  (#before:erased snapshot) (#fraction:perm)
  preserves gpu
  requires A.pts_to_mask cache #fraction before Init.all_cells
  requires pure (initialized before layer length)
  returns prefix:larray BF.t (length * 512)
  ensures rewrites_to prefix (prefix_view cache layer length value)
  ensures A.pts_to_mask cache #fraction before (outside (plane_offset layer value) (length * 512))
  ensures exists* (contents:lseq BF.t (length * 512)).
    A.pts_to prefix #fraction contents ** pure (prefix_binding before layer length value contents)
{
  plane_bounds layer value;
  let offset = SZ.uint_to_t (plane_offset layer value);
  let prefix = A.sub cache offset (plane_offset layer value + length * 512);
  A.mask_mext cache #fraction #before
    #(fun i -> Init.all_cells i /\ ~(SZ.v offset <= i /\ i < plane_offset layer value + length * 512))
    (outside (plane_offset layer value) (length * 512));
  plane_initialized before layer length value;
  A.from_mask prefix;
  with contents. assert A.pts_to prefix #fraction contents;
  A.pts_to_len prefix;
  prefix
}

ghost
fn restore_prefix
  (cache:larray BF.t 12582912) (layer:Attention.full_attention_layer)
  (length:pos{length <= 2048}) (value:bool)
  (#before:erased snapshot) (#contents:erased (lseq BF.t (length * 512))) (#fraction:perm)
  requires A.pts_to_mask cache #fraction before (outside (plane_offset layer value) (length * 512))
  requires A.pts_to (prefix_view cache layer length value) #fraction contents
  requires pure (prefix_binding before layer length value contents)
  ensures A.pts_to_mask cache #fraction before Init.all_cells
{
  A.to_mask (prefix_view cache layer length value);
  with masked. assert A.pts_to_mask (prefix_view cache layer length value) #fraction masked (fun _ -> True);
  let original : erased (Seq.seq (option BF.t)) =
    hide #(Seq.seq (option BF.t)) (reveal #snapshot before);
  A.return_sub cache #fraction #original #masked
    #(outside (plane_offset layer value) (length * 512)) #(fun _ -> True)
    #(plane_offset layer value) #(plane_offset layer value + length * 512);
  with result. assert A.pts_to_mask cache #fraction result
    (fun i -> outside (plane_offset layer value) (length * 512) i \/
      (plane_offset layer value <= i /\ i < plane_offset layer value + length * 512));
  A.mask_mext cache #fraction #result
    #(fun i -> outside (plane_offset layer value) (length * 512) i \/
      (plane_offset layer value <= i /\ i < plane_offset layer value + length * 512))
    Init.all_cells;
  assert pure (Seq.equal result before);
  Seq.lemma_eq_elim result before;
  rewrite A.pts_to_mask cache #fraction result Init.all_cells
    as A.pts_to_mask cache #fraction before Init.all_cells;
}
