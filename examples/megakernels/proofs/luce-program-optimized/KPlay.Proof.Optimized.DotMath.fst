module KPlay.Proof.Optimized.DotMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module C = FStar.Classical
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base

inline_for_extraction noextract
let interpretation (nw n:nat) = lseq real nw & lseq real n

inline_for_extraction noextract
let float_inputs (#nw #n:nat) (w:lseq bf16 nw) (x:lseq f32 n)
  (model:interpretation nw n) : prop =
  w %~ fst model /\ x %~ snd model

inline_for_extraction noextract
let bf16_inputs (#nw:nat) (w:lseq bf16 nw) (x:lseq bf16 1024)
  (model:interpretation nw 1024) : prop =
  w %~ fst model /\ x %~ snd model

inline_for_extraction noextract
let term (#nw #n:nat) (model:interpretation nw n)
  (offset:nat{offset + n <= nw}) (column:nat) : GTot real =
  if column < n then
    S.index (fst model) (offset + column) *. S.index (snd model) column
  else 0.0R

inline_for_extraction noextract
let block (#nw #n:nat) (model:interpretation nw n)
  (offset:nat{offset + n <= nw}) (lane group:nat) : GTot real =
  prefix 8 (fun j -> term model offset (group * 256 + (lane * 8 + j)))

inline_for_extraction noextract
let lane_sum (#nw #n:nat) (model:interpretation nw n)
  (offset:nat{offset + n <= nw}) (lane:nat) : GTot real =
  prefix (n / 256) (block model offset lane)

inline_for_extraction noextract
let partial (#nw #n:nat) (model:interpretation nw n)
  (offset:nat{offset + n <= nw}) (lane group count:nat) (chunks:bool) : GTot real =
  (if chunks then 0.0R else prefix group (block model offset lane)) +.
    prefix count (fun j -> term model offset (group * 256 + (lane * 8 + j)))

let partial_step (#nw #n:nat) (offset:nat{offset + n <= nw})
  (lane group count:nat) (chunks:bool) (model:interpretation nw n)
  : Lemma (partial model offset lane group (count + 1) chunks ==
      partial model offset lane group count chunks +.
      term model offset (group * 256 + (lane * 8 + count))) =
  prefix_step count (fun j -> term model offset (group * 256 + (lane * 8 + j)))

let group_end (#nw #n:nat) (offset:nat{offset + n <= nw})
  (lane group:nat) (model:interpretation nw n)
  : Lemma (
      partial model offset lane group 8 false ==
        prefix (group + 1) (block model offset lane) /\
      prefix group (block model offset lane) +. partial model offset lane group 8 true ==
        prefix (group + 1) (block model offset lane)) =
  prefix_step group (block model offset lane)

let group_sum (#nw #n:nat) (model:interpretation nw n)
  (offset:nat{offset + n <= nw}) (group:nat)
  : Lemma (
      prefix 32 (fun lane -> block model offset lane group) ==
      prefix 256 (fun column -> term model offset (group * 256 + column))) =
  prefix_blocks 32 8 (fun column -> term model offset (group * 256 + column));
  prefix_ext 32 (fun lane -> block model offset lane group)
    (fun lane -> prefix 8 (fun j -> term model offset (group * 256 + (lane * 8 + j))))

let all_columns (#nw:nat) (#n:pos{n % 256 = 0})
  (offset:nat{offset + n <= nw}) (model:interpretation nw n)
  : Lemma (
      prefix 32 (lane_sum model offset) == rsum (S.init_ghost n (term model offset))) =
  let expand_lane (lane:nat)
    : Lemma (lane_sum model offset lane ==
        prefix (n / 256) (fun group -> block model offset lane group)) =
    prefix_ext (n / 256) (block model offset lane)
      (fun group -> block model offset lane group)
  in
  C.forall_intro expand_lane;
  prefix_ext 32 (lane_sum model offset)
    (fun lane -> prefix (n / 256) (fun group -> block model offset lane group));
  prefix_swap 32 (n / 256) (fun lane group -> block model offset lane group);
  C.forall_intro (group_sum model offset);
  prefix_ext (n / 256)
    (fun group -> prefix 32 (fun lane -> block model offset lane group))
    (fun group -> prefix 256 (fun column -> term model offset (group * 256 + column)));
  prefix_blocks (n / 256) 256 (term model offset);
  prefix_init n (term model offset)

let source_partition (n:pos{n % 256 = 0})
  (lane:nat{lane < 32}) (group:nat{group < n / 256}) (element:nat{element < 8})
  : Lemma (
      let column = lane * 8 + group * 256 + element in
      column < n /\ column / 256 == group /\
      (column % 256) / 8 == lane /\ column % 8 == element) = ()

let source_partition_covers (n:pos{n % 256 = 0}) (column:nat{column < n})
  : Lemma (
      column == ((column % 256) / 8) * 8 + (column / 256) * 256 + column % 8 /\
      (column % 256) / 8 < 32 /\ column / 256 < n / 256 /\ column % 8 < 8) = ()

let float_loads (#nw #n:nat) (w:lseq bf16 nw) (x:lseq f32 n)
  (offset:nat{offset + n <= nw}) (column:nat{column < n})
  : Lemma (
      forall (model:interpretation nw n). float_inputs w x model ==>
        Cast.cast_bf16_to_f32 (S.index w (offset + column)) %~
          S.index (fst model) (offset + column) /\
        S.index x column %~ S.index (snd model) column) =
  let related (model:interpretation nw n{float_inputs w x model})
    : Lemma (
        Cast.cast_bf16_to_f32 (S.index w (offset + column)) %~
          S.index (fst model) (offset + column) /\
        S.index x column %~ S.index (snd model) column) =
    Cast.cast_bf16_to_f32_ok (S.index w (offset + column))
      (S.index (fst model) (offset + column))
  in C.forall_intro related

let bf16_loads (#nw:nat) (w:lseq bf16 nw) (x:lseq bf16 1024)
  (offset:nat{offset + 1024 <= nw}) (column:nat{column < 1024})
  : Lemma (
      forall (model:interpretation nw 1024). bf16_inputs w x model ==>
        Cast.cast_bf16_to_f32 (S.index w (offset + column)) %~
          S.index (fst model) (offset + column) /\
        Cast.cast_bf16_to_f32 (S.index x column) %~ S.index (snd model) column) =
  let related (model:interpretation nw 1024{bf16_inputs w x model})
    : Lemma (
        Cast.cast_bf16_to_f32 (S.index w (offset + column)) %~
          S.index (fst model) (offset + column) /\
        Cast.cast_bf16_to_f32 (S.index x column) %~ S.index (snd model) column) =
    Cast.cast_bf16_to_f32_ok (S.index w (offset + column))
      (S.index (fst model) (offset + column));
    Cast.cast_bf16_to_f32_ok (S.index x column) (S.index (snd model) column)
  in C.forall_intro related
