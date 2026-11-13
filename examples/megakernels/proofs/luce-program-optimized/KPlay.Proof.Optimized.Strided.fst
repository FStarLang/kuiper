module KPlay.Proof.Optimized.Strided

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module FP = Kuiper.Float32
module C = FStar.Classical
module Masks = KPlay.Proof.Optimized.Dot

inline_for_extraction noextract
let width = n:pos{n = 128 \/ n = 256}

inline_for_extraction noextract
let interpretation (n:width) = lseq real n & lseq real n

inline_for_extraction noextract
let inputs (#n:width) (left right:lseq f32 n) (model:interpretation n) : prop =
  left %~ fst model /\ right %~ snd model

inline_for_extraction noextract
let term (#n:width) (model:interpretation n) (i:nat) : GTot real =
  if i < n then S.index (fst model) i *. S.index (snd model) i else 0.0R

inline_for_extraction noextract
let lane_sum (#n:width) (model:interpretation n) (lane count:nat) : GTot real =
  prefix count (fun group -> term model (group * 32 + lane))

let all_columns (#n:width) (model:interpretation n)
  : Lemma (
      prefix 32 (fun lane -> lane_sum model lane (n / 32)) ==
        rsum (S.init_ghost n (term model))) =
  prefix_ext 32 (fun lane -> lane_sum model lane (n / 32))
    (fun lane -> prefix (n / 32) (fun group -> term model (group * 32 + lane)));
  prefix_swap 32 (n / 32) (fun lane group -> term model (group * 32 + lane));
  prefix_blocks (n / 32) 32 (term model);
  prefix_init n (term model)

let lane_step (#n:width) (lane group:nat) (model:interpretation n)
  : Lemma (
      lane_sum model lane (group + 1) ==
        lane_sum model lane group +. term model (group * 32 + lane)) =
  prefix_step group (fun g -> term model (g * 32 + lane))

noextract
fn source_lane (#n:erased width)
  (left right:larray f32 n) (count:sz{SZ.v count == n}) (lane:sz{lane < 32})
  (#lv #rv:erased (lseq f32 n)) (#fl #fr:perm)
  preserves gpu ** left |-> Frac fl lv ** right |-> Frac fr rv
  returns result:f32
  ensures pure (forall (model:interpretation n). inputs lv rv model ==>
    result %~ lane_sum model (SZ.v lane) (n / 32))
{
  let mut accumulator:f32 = zero;
  let mut group = 0sz;
  while (!group <^ count /^ 32sz)
    invariant exists* (g:sz{g <= n / 32}) (value:f32).
      group |-> g ** accumulator |-> value **
      pure (forall (model:interpretation n). inputs lv rv model ==>
        value %~ lane_sum model (SZ.v lane) (SZ.v g))
    decreases (n / 32 - !group)
  {
    let g = !group;
    let index = lane +^ g *^ 32sz;
    let a = left.(index);
    let b = right.(index);
    let previous = !accumulator;
    let value = FP.fma_rn_ftz a b previous;
    lift_fma #(interpretation n) (inputs lv rv)
      (fun model -> S.index (fst model) (SZ.v index))
      (fun model -> S.index (snd model) (SZ.v index))
      (fun model -> lane_sum model (SZ.v lane) (SZ.v g))
      a b previous value;
    C.forall_intro (lane_step #n (SZ.v lane) (SZ.v g));
    accumulator := value;
    group := g +^ 1sz;
  };
  !accumulator
}

noextract
fn strided_dot (#n:erased width)
  (left right:larray f32 n) (count:sz{SZ.v count == n})
  (#lv #rv:erased (lseq f32 n)) (#fl #fr:perm)
  preserves gpu ** left |-> Frac fl lv ** right |-> Frac fr rv
  returns result:f32
  ensures pure (forall (model:interpretation n). inputs lv rv model ==>
    result %~ rsum (S.init_ghost n (term model)))
{
  let mut partials:A.array f32 = [| 32sz |];
  let mut lane = 0sz;
  while (!lane <^ 32sz)
    invariant exists* (i:sz{i <= 32}) (pv:lseq (option f32) 32).
      lane |-> i ** A.pts_to_mask partials pv Masks.all_cells **
      pure (Masks.initialized pv (SZ.v i)) **
      pure (forall (model:interpretation n). inputs lv rv model ==>
        (forall (j:nat{j < SZ.v i}).
          Masks.option_models (S.index pv j) (lane_sum model j (n / 32))))
    decreases (32 - !lane)
  {
    let i = !lane;
    let value = source_lane left right count i;
    A.mask_write partials i value;
    lane := i +^ 1sz;
  };
  A.from_mask partials;
  with before. assert partials |-> before;
  let result = sum32 partials;
  lift_sum32 #(interpretation n) (inputs lv rv)
    (fun model lane -> lane_sum model lane (n / 32)) before result;
  C.forall_intro (all_columns #n);
  A.to_mask partials;
  result
}
