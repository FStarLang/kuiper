module KPlay.Proof.Base.Cache

module S = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module P = KPlay.Proof.Shared.Projection
module A = KPlay.Proof.Shared.Attention
module R = KPlay.Proof.Shared.Recurrent
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module O = KPlay.Proof.Shared.Observations
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let native_cache = O.cache O.cache_configuration BF.t FP.t BF.t

(* None records an uninitialized physical cell; no read or ownership of an
   untouched torch.empty suffix is manufactured by this observation map. *)
inline_for_extraction noextract
let buffer = P.index 12582912 -> GTot (option BF.t)

inline_for_extraction noextract
let position = p:nat{p < 2048}

inline_for_extraction noextract
let key_base (layer:A.full_attention_layer) (p:position) : nat =
  ((layer / 4) * 4096 + p) * 512

inline_for_extraction noextract
let value_base (layer:A.full_attention_layer) (p:position) : nat =
  ((layer / 4) * 4096 + 2048 + p) * 512

inline_for_extraction noextract
let key_address (layer:A.full_attention_layer) (p:position) (i:P.index 512)
  : P.index 12582912 = key_base layer p + i

inline_for_extraction noextract
let value_address (layer:A.full_attention_layer) (p:position) (i:P.index 512)
  : P.index 12582912 = value_base layer p + i

inline_for_extraction noextract
let key_value_disjoint (layer:A.full_attention_layer) (p:position)
  : Lemma (
      key_base layer p + 512 <= value_base layer p /\
      value_base layer p + 512 <= (layer / 4 + 1) * 4096 * 512 /\
      (layer / 4 + 1) * 4096 * 512 <= 12582912) = ()

inline_for_extraction noextract
let write (before:buffer) (address:P.index 12582912) (value:BF.t) : buffer =
  fun i -> if i = address then Some value else before i

(* Exact interleaved K[i],V[i] store order of kernel.cu:260-263. *)
inline_for_extraction noextract
let rec append_loop (before:buffer) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position) (count:nat{count <= 512})
  : GTot buffer (decreases count) =
  if count = 0 then before
  else
    let previous = append_loop before hidden layer p (count - 1) in
    let key_written = write previous (key_address layer p (count - 1))
      (S.index hidden (6144 + count - 1)) in
    write key_written (value_address layer p (count - 1))
      (S.index hidden (6656 + count - 1))

inline_for_extraction noextract
let written_value (before:buffer) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position)
  (count:nat{count <= 512}) (i:P.index 12582912) : GTot (option BF.t) =
  if key_base layer p <= i && i < key_base layer p + count then
    Some (S.index hidden (6144 + i - key_base layer p))
  else if value_base layer p <= i && i < value_base layer p + count then
    Some (S.index hidden (6656 + i - value_base layer p))
  else before i

inline_for_extraction noextract
let rec append_loop_cell (before:buffer) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position) (count:nat{count <= 512})
  (i:P.index 12582912)
  : Lemma
      (ensures append_loop before hidden layer p count i ==
        written_value before hidden layer p count i)
      (decreases count) =
  key_value_disjoint layer p;
  if count > 0 then append_loop_cell before hidden layer p (count - 1) i

inline_for_extraction noextract
let produced_key (hidden:Transfer.scratch) : O.matrix BF.t 2 256 =
  fun head d -> S.index hidden (6144 + head * 256 + d)

inline_for_extraction noextract
let produced_value (hidden:Transfer.scratch) : O.matrix BF.t 2 256 =
  fun head d -> S.index hidden (6656 + head * 256 + d)

inline_for_extraction noextract
let appended_history (p:position)
  (old:option (O.kv_history BF.t){O.history_length old == p})
  (hidden:Transfer.scratch) : history:O.kv_history BF.t{history.length == p + 1} = {
  O.length = p + 1;
  O.keys = (fun head t d ->
    if t = p then produced_key hidden head d
    else (R.present_value old).O.keys head t d);
  O.values = (fun head t d ->
    if t = p then produced_value hidden head d
    else (R.present_value old).O.values head t d);
}

inline_for_extraction noextract
let source_after (before:native_cache) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position)
  : Pure native_cache
      (requires O.cache_shape p layer before)
      (ensures fun _ -> True) = {
  O.metadata = before.metadata;
  O.convolution = before.convolution;
  O.recurrent = before.recurrent;
  O.kv = (fun l ->
    if l = layer then Some (appended_history p (before.kv layer) hidden)
    else before.kv l);
}

inline_for_extraction noextract
let represents_slot (physical:buffer) (layer:A.full_attention_layer)
  (history:option (O.kv_history BF.t)) : GTot prop =
  match history with
  | None -> True
  | Some old ->
      old.length <= 2048 /\
      (forall (head:P.index 2) (t:nat{t < old.length /\ t < 2048}) (d:P.index 256).
        physical (key_address layer t (head * 256 + d)) == Some (old.keys head t d) /\
        physical (value_address layer t (head * 256 + d)) == Some (old.values head t d))

inline_for_extraction noextract
let append_represents (physical:buffer) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position)
  (old:option (O.kv_history BF.t){O.history_length old == p})
  : Lemma
      (requires represents_slot physical layer old)
      (ensures represents_slot (append_loop physical hidden layer p 512)
        layer (Some (appended_history p old hidden))) =
  let each (head:P.index 2) (t:nat{t < p + 1 /\ t < 2048}) (d:P.index 256)
    : Lemma (
        append_loop physical hidden layer p 512
          (key_address layer t (head * 256 + d)) ==
          Some ((appended_history p old hidden).O.keys head t d) /\
        append_loop physical hidden layer p 512
          (value_address layer t (head * 256 + d)) ==
          Some ((appended_history p old hidden).O.values head t d)) =
    append_loop_cell physical hidden layer p 512
      (key_address layer t (head * 256 + d));
    append_loop_cell physical hidden layer p 512
      (value_address layer t (head * 256 + d));
    if t < p then (
      assert (Some? old);
      assert ((R.present_value old).O.length == p)
    )
  in
  FStar.Classical.forall_intro_3 each

inline_for_extraction noextract
let append_frames (physical:buffer) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position)
  (i:P.index 12582912)
  : Lemma
      (requires
        ~(key_base layer p <= i /\ i < key_base layer p + 512) /\
        ~(value_base layer p <= i /\ i < value_base layer p + 512))
      (ensures append_loop physical hidden layer p 512 i == physical i) =
  append_loop_cell physical hidden layer p 512 i

inline_for_extraction noextract
let source_structure (before:native_cache) (hidden:Transfer.scratch)
  (layer:A.full_attention_layer) (p:position)
  : Lemma
      (requires O.cache_shape p layer before)
      (ensures O.layer_structure p layer before (source_after before hidden layer p)
        (O.AttentionUpdate (produced_key hidden) (produced_value hidden))) =
  let each (l:A.layer_index)
    : Lemma (
        let after = source_after before hidden layer p in
        let count = Model.visits p (layer + 1) l in
        if l % 4 < 3 then
          after.kv l == None /\
          (Some? (after.convolution l) <==> count > 0) /\
          (Some? (after.recurrent l) <==> count > 0)
        else
          after.convolution l == None /\
          after.recurrent l == None /\
          O.history_length (after.kv l) == count) =
    assert (Model.visits p layer l == p + (if l < layer then 1 else 0))
  in
  FStar.Classical.forall_intro each
