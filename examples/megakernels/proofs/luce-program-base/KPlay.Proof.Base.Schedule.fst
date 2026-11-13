module KPlay.Proof.Base.Schedule

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

module Model = KPlay.Proof.Shared.Model
module A = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Cache = KPlay.Proof.Base.Cache

inline_for_extraction noextract
let phases (layer:Model.cursor) : pos =
  if layer = 24 then 1 else if layer % 4 <> 3 then 8 else 23

inline_for_extraction noextract
let phase (layer:Model.cursor) = p:nat{p < phases layer}

inline_for_extraction noextract
let base (layer:Model.cursor) : nat =
  254280704 + (layer / 4) * 83018720 + (layer % 4) * 21555360

inline_for_extraction noextract
let attention (layer:Model.cursor) (p:phase layer) : bool =
  layer <> 24 && layer % 4 = 3 && 3 <= p && p < 19

inline_for_extraction noextract
let kind (layer:Model.cursor) (p:phase layer) : nat =
  if layer % 4 = 3 && p >= 19 then p - 15 else p

noeq
type projection = {
  rows: pos;
  columns: pos;
  input_offset: nat;
  output_offset: nat;
  weight_offset: nat;
  matrix: m:nat{m <= 2};
  splits: s:pos{s = 1 \/ s = 2};
}

inline_for_extraction noextract
let projection_at (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor) (p:phase layer) : projection =
  let linear = layer % 4 <> 3 in
  let k = kind layer p in
  let b = base layer in
  if layer = 24 then
    {rows=248320; columns=1024; input_offset=1024; output_offset=12500;
     weight_offset=0; matrix=0; splits=1}
  else if attention layer p then
    let head = (p - 3) / 2 in
    if p % 2 = 1 then
      {rows=position+1; columns=256; input_offset=2048+head*256; output_offset=13500;
       weight_offset=0; matrix=1; splits=1}
    else
      {rows=256; columns=position+1; input_offset=13500; output_offset=10400+head*256;
       weight_offset=0; matrix=2; splits=1}
  else if k = 0 then
    {rows=(if linear then 6144 else 4096); columns=1024; input_offset=1024; output_offset=2048;
     weight_offset=b+11012096; matrix=0; splits=1}
  else if k = 1 then
    {rows=(if linear then 2048 else 512); columns=1024; input_offset=1024;
     output_offset=(if linear then 8192 else 6144);
     weight_offset=b+(if linear then 17303552 else 15206400); matrix=0; splits=1}
  else if k = 2 then
    {rows=(if linear then 16 else 512); columns=1024; input_offset=1024;
     output_offset=(if linear then 10240 else 6656);
     weight_offset=b+(if linear then 19400704 else 15730688); matrix=0; splits=1}
  else if k = 3 then
    {rows=16; columns=1024; input_offset=1024; output_offset=10256;
     weight_offset=b+19417088; matrix=0; splits=1}
  else if k = 4 then
    {rows=1024; columns=2048; input_offset=10400; output_offset=12500;
     weight_offset=b+(if linear then 19458208 else 16255488); matrix=0; splits=1}
  else if k = 5 || k = 6 then
    {rows=3584; columns=1024; input_offset=1024;
     output_offset=(if k = 5 then 2048 else 5632);
     weight_offset=b+(if k = 5 then 2048 else 3672064); matrix=0; splits=1}
  else
    {rows=1024; columns=3584; input_offset=2048; output_offset=12500;
     weight_offset=b+7342080; matrix=0; splits=2}

inline_for_extraction noextract
let phase_kinds (layer:Model.cursor) (p:phase layer)
  : Lemma (
      (layer = 24 ==> p = 0) /\
      (layer < 24 /\ ~(attention layer p) ==> kind layer p < 8) /\
      (layer < 24 /\ ~(attention layer p) /\ kind layer p = 3 ==> layer % 4 <> 3) /\
      (attention layer p ==> layer < 24 /\ layer % 4 = 3 /\ 3 <= p /\ p < 19 /\
        (p - 3) / 2 < 8)) = ()

inline_for_extraction noextract
let bounds (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor) (p:phase layer)
  : Lemma (
      let d = projection_at call position layer p in
      d.input_offset + d.columns <= 16000 /\
      d.output_offset + (if layer = 24 then 16 else d.rows) <= 16000 /\
      (d.matrix = 0 ==> d.weight_offset + d.rows * d.columns <= 752393024 /\
        d.rows % 16 = 0 /\ d.columns % (16 * d.splits) = 0) /\
      (d.matrix = 1 ==> d.rows = position + 1 /\ d.columns = 256) /\
      (d.matrix = 2 ==> d.rows = 256 /\ d.columns = position + 1)) =
  phase_kinds layer p;
  if layer = 24 then ()
  else if attention layer p then ()
  else (
    let k = kind layer p in
    if k = 0 then ()
    else if k = 1 then ()
    else if k = 2 then ()
    else if k = 3 then ()
    else if k = 4 then ()
    else if k = 5 then ()
    else if k = 6 then ()
    else assert (k = 7)
  )

inline_for_extraction noextract
let source_weight_address (call:Model.invocation) (position:nat{position < Model.calls call})
  (layer:Model.cursor) (p:phase layer)
  (row:nat{row < (projection_at call position layer p).rows})
  (column:nat{column < (projection_at call position layer p).columns})
  : Pure (P.index 752393024)
      (requires (projection_at call position layer p).matrix = 0)
      (ensures fun _ -> True) =
  bounds call position layer p;
  let d = projection_at call position layer p in
  d.weight_offset + row * d.columns + column

inline_for_extraction noextract
let source_cache_address (layer:A.full_attention_layer)
  (head:P.index 8) (length:pos{length <= 2048}) (score:bool)
  (row:nat{row < (if score then length else 256)})
  (column:nat{column < (if score then 256 else length)})
  : P.index 12582912 =
  if score then
    Cache.key_address layer row ((head / 4) * 256 + column)
  else
    Cache.value_address layer column ((head / 4) * 256 + row)

inline_for_extraction noextract
let cache_orientations (layer:A.full_attention_layer)
  (head:P.index 8) (length:pos{length <= 2048})
  (time:P.index length) (channel:P.index 256)
  : Lemma (
      source_cache_address layer head length true time channel ==
        ((layer / 4) * 4096 + time) * 512 + (head / 4) * 256 + channel /\
      source_cache_address layer head length false channel time ==
        ((layer / 4) * 4096 + 2048 + time) * 512 + (head / 4) * 256 + channel) = ()

inline_for_extraction noextract
let mma_path (layer:Model.cursor) (p:phase layer) (position:nat) : bool =
  not (attention layer p) || (position + 1) % 8 = 0

inline_for_extraction noextract
let simt_groups (score:bool) (length:pos) : pos =
  if score then (if 57 <= length && length <= 139 then 32 else 16)
  else if length < 426 then 32 else 16

inline_for_extraction noextract
let simt_linear_finish (score:bool) (length:pos) : bool =
  score && ((388 <= length && length <= 415) || 444 <= length)

inline_for_extraction noextract
let simt_chunk (score:bool) (length:pos) : pos =
  if simt_linear_finish score length || (score && 417 <= length && length <= 443) then 1
  else if score then 256 / simt_groups score length
  else if length < 426 then 4 else (length + 15) / 16

inline_for_extraction noextract
let source_loop_domain (call:Model.invocation)
  : Lemma (
      1 <= Model.calls call /\ Model.calls call <= 647 /\
      (forall (position:nat{position < Model.calls call}).
        position < 2048 /\ position + 1 <= 647)) = ()
