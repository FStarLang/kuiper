module KPlay.Proof.Optimized.Online

#lang-pulse

#set-options "--fuel 3 --ifuel 1 --z3rlimit 60"

open Kuiper
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module F = Kuiper.Floating
module FP = Kuiper.Float32
module Neg = Kuiper.Float32.NativeNeg
module G = FStar.Ghost
module GR = Pulse.Lib.GhostReference
module Memory = KPlay.Proof.Optimized.Memory
module Scalar = KPlay.Proof.Optimized.Scalar
module Ops = KPlay.Proof.Optimized.AttentionOps
module Cast = Kuiper.Float.Casts.Base

inline_for_extraction noextract
let negative_infinity : f32 = F.neg (F.infinity #f32)

inline_for_extraction noextract
let visited (partition count:nat) (time:nat) : prop =
  time % 16 < partition \/ (time % 16 == partition /\ time < count)

noextract
fn merge_partition (partials:larray f32 4096) (denominators:larray f32 16) (output:larray f32 256)
  (warp:sz{warp < 16}) (part_max maximum denominator:FP.t)
  (#pv:erased (lseq f32 4096)) (#dv:erased (lseq f32 16))
  (#ov:erased (lseq f32 256)) (#fp #fd:perm)
  preserves gpu ** partials |-> Frac fp pv ** denominators |-> Frac fd dv
  requires output |-> ov
  returns result:FP.t
  ensures exists* (after:lseq f32 256). output |-> after
{
  if (F.lt negative_infinity part_max) {
    let argument = FP.add_rn_ftz part_max (Neg.neg maximum);
    let scale = Scalar.exp32 argument;
    let next_denominator = FP.fma_rn_ftz (denominators.(warp)) scale denominator;
    let mut coordinate = 0sz;
    while (!coordinate <^ 256sz)
      invariant exists* (i:sz{i <= 256}) (current:lseq f32 256).
        coordinate |-> i ** output |-> current
      decreases (256 - !coordinate)
    {
      let i = !coordinate;
      let part = partials.(warp *^ 256sz +^ i);
      output.(i) <- FP.fma_rn_ftz part scale (output.(i));
      coordinate := i +^ 1sz;
    };
    next_denominator
  } else {
    denominator
  }
}

noextract
fn source_head (projected:larray f32 5120) (keys values:larray bf16 1048576)
  (output:larray f32 256) (head:sz{head < 8}) (position:sz{position < 2048})
  (#pv:erased (lseq f32 5120)) (#kv #vv:erased (lseq bf16 1048576))
  (#ov:erased (lseq (option f32) 256)) (#fp #fk #fv:perm)
  preserves gpu ** projected |-> Frac fp pv ** keys |-> Frac fk kv ** values |-> Frac fv vv
  requires A.pts_to_mask output ov Memory.all_cells
  returns observed:erased (lseq f32 (SZ.v position + 1))
  ensures exists* (after:lseq f32 256). output |-> after
{
  let trace = GR.alloc #(lseq (option f32) (SZ.v position + 1))
    (S.create (SZ.v position + 1) None);
  let mut partials:A.array f32 = [| 4096sz |];
  let mut maxima:A.array f32 = [| 16sz |];
  let mut denominators:A.array f32 = [| 16sz |];
  let mut warp = 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w:sz{w <= 16}) (mv dv:lseq (option f32) 16)
                     (partial:lseq (option f32) 4096)
                     (scores:lseq (option f32) (SZ.v position + 1)).
      warp |-> w ** A.pts_to_mask maxima mv Memory.all_cells **
      A.pts_to_mask denominators dv Memory.all_cells **
      A.pts_to_mask partials partial Memory.all_cells ** trace |-> scores **
      pure (Memory.ready mv 0 (SZ.v w) /\ Memory.ready dv 0 (SZ.v w) /\
        Memory.ready partial 0 (SZ.v w * 256) /\
        (forall (t:nat{t <= SZ.v position}). t % 16 < SZ.v w ==> Some? (S.index scores t)))
    decreases (16 - !warp)
  {
    let w = !warp;
    let mut maximum:f32 = negative_infinity;
    let mut denominator:f32 = zero;
    let mut coordinate = 0sz;
    while (!coordinate <^ 256sz)
      invariant exists* (i:sz{i <= 256}) (partial:lseq (option f32) 4096).
        coordinate |-> i ** A.pts_to_mask partials partial Memory.all_cells **
        pure (Memory.ready partial 0 (SZ.v w * 256 + SZ.v i))
      decreases (256 - !coordinate)
    {
      let i = !coordinate;
      A.mask_write partials (w *^ 256sz +^ i) (zero #f32);
      coordinate := i +^ 1sz;
    };
    let mut time = w;
    while (!time <=^ position)
      invariant exists* (t:sz{SZ.v t <= SZ.v position + 16 /\ SZ.v t % 16 == SZ.v w})
                       (m d:f32) (partial:lseq (option f32) 4096)
                       (scores:lseq (option f32) (SZ.v position + 1)).
        time |-> t ** maximum |-> m ** denominator |-> d **
        A.pts_to_mask partials partial Memory.all_cells ** trace |-> scores **
        pure (Memory.ready partial 0 ((SZ.v w + 1) * 256) /\
          (forall (p:nat{p <= SZ.v position}).
            visited (SZ.v w) (SZ.v t) p ==> Some? (S.index scores p)))
      decreases (if SZ.v !time <= SZ.v position then SZ.v position + 1 - SZ.v !time else 0)
    {
      let t = !time;
      let score = Ops.score projected keys head t;
      let scores = GR.read trace;
      GR.write trace (G.hide (S.upd scores (SZ.v t) (Some score)));
      let previous = !maximum;
      let next_maximum = F.fmax previous score;
      maximum := next_maximum;
      let correction_argument = FP.add_rn_ftz previous (Neg.neg next_maximum);
      let correction = Scalar.exp32 correction_argument;
      let weight_argument = FP.add_rn_ftz score (Neg.neg next_maximum);
      let weight = Scalar.exp32 weight_argument;
      denominator := FP.fma_rn_ftz (!denominator) correction weight;
      coordinate := 0sz;
      while (!coordinate <^ 256sz)
        invariant exists* (i:sz{i <= 256}) (partial:lseq (option f32) 4096).
          coordinate |-> i ** A.pts_to_mask partials partial Memory.all_cells **
          pure (Memory.ready partial 0 ((SZ.v w + 1) * 256))
        decreases (256 - !coordinate)
      {
        let i = !coordinate;
        let old = A.mask_read partials (w *^ 256sz +^ i);
        let value = Cast.cast_bf16_to_f32
          (values.((head /^ 4sz) *^ (SZ.uint32_to_sizet 524288ul) +^ t *^ 256sz +^ i));
        let weighted = FP.mul_rn_ftz weight value;
        let next = FP.fma_rn_ftz old correction weighted;
        A.mask_write partials (w *^ 256sz +^ i) next;
        coordinate := i +^ 1sz;
      };
      time := t +^ 16sz;
    };
    A.mask_write maxima w (!maximum);
    A.mask_write denominators w (!denominator);
    warp := w +^ 1sz;
  };
  A.from_mask maxima;
  A.from_mask denominators;
  A.from_mask partials;
  let mut maximum:f32 = negative_infinity;
  warp := 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w:sz{w <= 16}) (m:f32). warp |-> w ** maximum |-> m
    decreases (16 - !warp)
  {
    let w = !warp;
    let m = maxima.(w);
    if (F.lt negative_infinity m) {
      maximum := F.fmax (!maximum) m;
    };
    warp := w +^ 1sz;
  };
  let mut denominator:f32 = zero;
  let mut coordinate = 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i:sz{i <= 256}) (current:lseq (option f32) 256).
      coordinate |-> i ** A.pts_to_mask output current Memory.all_cells **
      pure (Memory.ready current 0 (SZ.v i))
    decreases (256 - !coordinate)
  {
    let i = !coordinate;
    A.mask_write output i (zero #f32);
    coordinate := i +^ 1sz;
  };
  A.from_mask output;
  warp := 0sz;
  while (!warp <^ 16sz)
    invariant exists* (w:sz{w <= 16}) (d:f32) (current:lseq f32 256).
      warp |-> w ** denominator |-> d ** output |-> current
    decreases (16 - !warp)
  {
    let w = !warp;
    let m = maxima.(w);
    let next = merge_partition partials denominators output w m (!maximum) (!denominator);
    denominator := next;
    warp := w +^ 1sz;
  };
  let inverse = FP.rcp_approx_ftz (!denominator);
  coordinate := 0sz;
  while (!coordinate <^ 256sz)
    invariant exists* (i:sz{i <= 256}) (current:lseq f32 256).
      coordinate |-> i ** output |-> current
    decreases (256 - !coordinate)
  {
    let i = !coordinate;
    let normalized = FP.mul_rn_ftz (output.(i)) inverse;
    let gate = Scalar.sigmoid32 (projected.(head *^ 512sz +^ 256sz +^ i));
    output.(i) <- FP.mul_rn_ftz normalized gate;
    coordinate := i +^ 1sz;
  };
  let captured = GR.read trace;
  assert pure (Memory.ready captured 0 (SZ.v position + 1));
  let observed:erased (lseq f32 (SZ.v position + 1)) =
    G.hide (Memory.values captured 0 (SZ.v position + 1) ());
  GR.free trace;
  A.to_mask maxima;
  A.to_mask denominators;
  A.to_mask partials;
  observed
}
