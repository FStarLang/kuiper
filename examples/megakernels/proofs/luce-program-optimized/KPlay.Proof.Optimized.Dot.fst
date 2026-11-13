module KPlay.Proof.Optimized.Dot

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open KPlay.Proof.Optimized.Sum
open KPlay.Proof.Optimized.DotMath
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module Cast = Kuiper.Float.Casts.Base
module C = FStar.Classical

noextract
fn bf16_lane (#nw:erased pos{nw <= 752393024})
  (weights:larray bf16 nw) (values:larray bf16 1024)
  (offset:sz{SZ.v offset + 1024 <= nw}) (lane:sz{lane < 32})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq bf16 1024)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result:f32
  ensures pure (forall (model:interpretation nw 1024).
    bf16_inputs wv xv model ==> result %~ lane_sum model (SZ.v offset) (SZ.v lane))
{
  let mut accumulator:f32 = zero;
  let mut group = 0sz;
  while (!group <^ 4sz)
    invariant exists* (g:sz{g <= 4}) (value:f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model:interpretation nw 1024). bf16_inputs wv xv model ==>
        value %~ prefix (SZ.v g) (block model (SZ.v offset) (SZ.v lane)))
    decreases (4 - !group)
  {
    let g = !group;
    let previous = !accumulator;
    let mut chunk:f32 = zero;
    let mut element = 0sz;
    while (!element <^ 8sz)
      invariant exists* (j:sz{j <= 8}) (value:f32).
        element |-> j ** chunk |-> value **
        pure (forall (model:interpretation nw 1024). bf16_inputs wv xv model ==>
          value %~ partial model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) true)
      decreases (8 - !element)
    {
      let j = !element;
      let column = lane *^ 8sz +^ g *^ 256sz +^ j;
      let w = Cast.cast_bf16_to_f32 (weights.(offset +^ column));
      let x = Cast.cast_bf16_to_f32 (values.(column));
      bf16_loads wv xv (SZ.v offset) (SZ.v column);
      let previous_chunk = !chunk;
      let value = FP.fma_rn_ftz w x previous_chunk;
      lift_fma #(interpretation nw 1024) (bf16_inputs wv xv)
        (fun model -> S.index (fst model) (SZ.v offset + SZ.v column))
        (fun model -> S.index (snd model) (SZ.v column))
        (fun model -> partial model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) true)
        w x previous_chunk value;
      C.forall_intro (partial_step #nw #1024 (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) true);
      chunk := value;
      element := j +^ 1sz;
    };
    let last = !chunk;
    let value = FP.add_rn_ftz previous last;
    lift_add #(interpretation nw 1024) (bf16_inputs wv xv)
      (fun model -> prefix (SZ.v g) (block model (SZ.v offset) (SZ.v lane)))
      (fun model -> partial model (SZ.v offset) (SZ.v lane) (SZ.v g) 8 true)
      previous last value;
    C.forall_intro (group_end #nw #1024 (SZ.v offset) (SZ.v lane) (SZ.v g));
    accumulator := value;
    group := g +^ 1sz;
  };
  !accumulator
}

noextract
fn float_lane (#nw:erased pos{nw <= 752393024})
  (#n:erased pos{n % 256 = 0 /\ n <= 3584})
  (weights:larray bf16 nw) (values:larray f32 n)
  (count:sz{SZ.v count == n}) (offset:sz{SZ.v offset + n <= nw})
  (lane:sz{lane < 32})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq f32 n)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result:f32
  ensures pure (forall (model:interpretation nw n).
    float_inputs wv xv model ==> result %~ lane_sum model (SZ.v offset) (SZ.v lane))
{
  let mut accumulator:f32 = zero;
  let mut group = 0sz;
  while (!group <^ count /^ 256sz)
    invariant exists* (g:sz{g <= n / 256}) (value:f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model:interpretation nw n). float_inputs wv xv model ==>
        value %~ prefix (SZ.v g) (block model (SZ.v offset) (SZ.v lane)))
    decreases (n / 256 - !group)
  {
    let g = !group;
    let mut element = 0sz;
    while (!element <^ 8sz)
      invariant exists* (j:sz{j <= 8}) (value:f32).
        element |-> j ** accumulator |-> value **
        pure (forall (model:interpretation nw n). float_inputs wv xv model ==>
          value %~ partial model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) false)
      decreases (8 - !element)
    {
      let j = !element;
      let column = lane *^ 8sz +^ g *^ 256sz +^ j;
      let w = Cast.cast_bf16_to_f32 (weights.(offset +^ column));
      let x = values.(column);
      float_loads wv xv (SZ.v offset) (SZ.v column);
      let previous = !accumulator;
      let value = FP.fma_rn_ftz w x previous;
      lift_fma #(interpretation nw n) (float_inputs wv xv)
        (fun model -> S.index (fst model) (SZ.v offset + SZ.v column))
        (fun model -> S.index (snd model) (SZ.v column))
        (fun model -> partial model (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) false)
        w x previous value;
      C.forall_intro (partial_step #nw #n (SZ.v offset) (SZ.v lane) (SZ.v g) (SZ.v j) false);
      accumulator := value;
      element := j +^ 1sz;
    };
    C.forall_intro (group_end #nw #n (SZ.v offset) (SZ.v lane) (SZ.v g));
    group := g +^ 1sz;
  };
  !accumulator
}

inline_for_extraction noextract
let all_cells (_:nat) : prop = True

inline_for_extraction noextract
let initialized (values:lseq (option f32) 32) (count:nat{count <= 32}) : prop =
  forall (i:nat{i < count}). Some? (S.index values i)

inline_for_extraction noextract
let option_models (value:option f32) (r:real) : prop =
  match value with | None -> False | Some v -> v %~ r

noextract
fn dot_bf16 (#nw:erased pos{nw <= 752393024})
  (weights:larray bf16 nw) (values:larray bf16 1024)
  (offset:sz{SZ.v offset + 1024 <= nw})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq bf16 1024)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result:f32
  ensures pure (forall (model:interpretation nw 1024). bf16_inputs wv xv model ==>
    result %~ rsum (S.init_ghost 1024 (term model (SZ.v offset))))
{
  let mut partials:A.array f32 = [| 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (pv:lseq (option f32) 32).
      lane |-> i ** A.pts_to_mask partials pv all_cells **
      pure (initialized pv (SZ.v i)) **
      pure (forall (model:interpretation nw 1024). bf16_inputs wv xv model ==>
        (forall (j:nat{j < SZ.v i}).
          option_models (S.index pv j) (lane_sum model (SZ.v offset) j)))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = bf16_lane weights values offset i;
    A.mask_write partials i value;
    lane := i +^ 1sz;
  };
  A.from_mask partials;
  with before. assert partials |-> before;
  let result = sum32 partials;
  lift_sum32 #(interpretation nw 1024) (bf16_inputs wv xv)
    (fun model -> lane_sum model (SZ.v offset)) before result;
  C.forall_intro (all_columns #nw #1024 (SZ.v offset));
  A.to_mask partials;
  result
}

noextract
fn dot_float (#nw:erased pos{nw <= 752393024})
  (#n:erased pos{n % 256 = 0 /\ n <= 3584})
  (weights:larray bf16 nw) (values:larray f32 n)
  (count:sz{SZ.v count == n}) (offset:sz{SZ.v offset + n <= nw})
  (#wv:erased (lseq bf16 nw)) (#xv:erased (lseq f32 n)) (#fw #fx:perm)
  preserves gpu ** weights |-> Frac fw wv ** values |-> Frac fx xv
  returns result:f32
  ensures pure (forall (model:interpretation nw n). float_inputs wv xv model ==>
    result %~ rsum (S.init_ghost n (term model (SZ.v offset))))
{
  let mut partials:A.array f32 = [| 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (pv:lseq (option f32) 32).
      lane |-> i ** A.pts_to_mask partials pv all_cells **
      pure (initialized pv (SZ.v i)) **
      pure (forall (model:interpretation nw n). float_inputs wv xv model ==>
        (forall (j:nat{j < SZ.v i}).
          option_models (S.index pv j) (lane_sum model (SZ.v offset) j)))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = float_lane weights values count offset i;
    A.mask_write partials i value;
    lane := i +^ 1sz;
  };
  A.from_mask partials;
  with before. assert partials |-> before;
  let result = sum32 partials;
  lift_sum32 #(interpretation nw n) (float_inputs wv xv)
    (fun model -> lane_sum model (SZ.v offset)) before result;
  C.forall_intro (all_columns #nw #n (SZ.v offset));
  A.to_mask partials;
  result
}
