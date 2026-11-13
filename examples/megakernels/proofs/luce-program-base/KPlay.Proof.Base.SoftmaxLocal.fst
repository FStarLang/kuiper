module KPlay.Proof.Base.SoftmaxLocal

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Pulse.Lib.Array
open FStar.Ghost
open FStar.Real

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module S = KPlay.Proof.Shared.Attention
module P = KPlay.Proof.Shared.Projection
module Math = KPlay.Proof.Base.AttentionAlgebra
module Scale = KPlay.Proof.Base.SoftmaxScale
module Stripes = KPlay.Proof.Base.SoftmaxStripes
module Softmax = KPlay.Proof.Base.Softmax
module Local = KPlay.Proof.Base.LocalArray
module Numeric = KPlay.Proof.Base.LocalReductionMath
module Conditional = KPlay.Proof.Base.ConditionalNumerics
module Init = KPlay.Proof.Base.Initialization
module Norm = KPlay.Proof.Base.Norm
module Vocabulary = KPlay.Proof.Base.Vocabulary
module Transfer = KPlay.Proof.Base.Transfer

inline_for_extraction noextract
let initialized (n:pos{n <= 647}) = s:Local.snapshot 2048{Local.ready s n}

// The padded suffix is a pure model, never ownership or a native read.
inline_for_extraction noextract
let view (n:pos{n <= 647}) (before:initialized n) : GTot (lseq FP.t 2048) =
  Seq.init_ghost 2048 (fun i -> if i < n then Local.present (Seq.index before i) else F.zero #FP.t)

inline_for_extraction noextract
let view_write (n:pos{n <= 647}) (before:initialized n) (i:P.index n) (value:FP.t)
  : Lemma (
      Local.ready #2048 (Seq.upd before i (Some value)) n /\
      view n (Seq.upd before i (Some value)) == Seq.upd (view n before) i value) =
  assert (Seq.equal (view n (Seq.upd before i (Some value))) (Seq.upd (view n before) i value));
  Seq.lemma_eq_elim (view n (Seq.upd before i (Some value))) (Seq.upd (view n before) i value)

inline_for_extraction noextract
let scaled_values (n:pos{n <= 647}) (hidden:Transfer.scratch) (offset:nat{offset + n <= 16000}) : P.vector FP.t n =
  fun i -> Scale.scaled (Seq.index hidden (offset + i))

inline_for_extraction noextract
let scaled_ready (n:pos{n <= 647}) (before:Local.snapshot 2048) (hidden:Transfer.scratch)
  (offset:nat{offset + n <= 16000})
  : Lemma (
      Local.ready #2048 (Local.written #2048 #n before (scaled_values n hidden offset) n) n /\
      view n (Local.written #2048 #n before (scaled_values n hidden offset) n) ==
        Scale.writes (Seq.create 2048 (F.zero #FP.t)) hidden offset n) =
  Local.written_ready #2048 #n before (scaled_values n hidden offset) n;
  assert (Seq.equal (view n (Local.written #2048 #n before (scaled_values n hidden offset) n))
    (Scale.writes (Seq.create 2048 (F.zero #FP.t)) hidden offset n));
  Seq.lemma_eq_elim (view n (Local.written #2048 #n before (scaled_values n hidden offset) n))
    (Scale.writes (Seq.create 2048 (F.zero #FP.t)) hidden offset n)

inline_for_extraction noextract
fn scale_and_maximum
  (hidden:larray BF.t 16000) (values:larray FP.t 2048)
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (#sx:erased Transfer.scratch) (#sv:erased (Local.snapshot 2048)) (#fraction:perm)
  preserves gpu ** A.pts_to hidden #fraction sx
  requires A.pts_to_mask values sv Init.all_cells
  returns result:FP.t
  ensures A.pts_to_mask values (Local.written #2048 #(SZ.v n) sv (scaled_values (SZ.v n) sx (SZ.v offset)) (SZ.v n)) Init.all_cells
  ensures pure (result == Scale.maximum sx (SZ.v offset) (SZ.v n))
{
  Local.written_zero #2048 #(SZ.v n) sv (scaled_values (SZ.v n) sx (SZ.v offset));
  let mut maximum:FP.t = Vocabulary.negative_infinity;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (current:FP.t).
      i |-> j ** maximum |-> current ** A.pts_to_mask values
        (Local.written #2048 #(SZ.v n) sv (scaled_values (SZ.v n) sx (SZ.v offset)) (SZ.v j)) Init.all_cells **
      pure (current == Scale.maximum sx (SZ.v offset) (SZ.v j))
    invariant gpu ** A.pts_to hidden #fraction sx
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    let input = hidden.(SZ.add offset j);
    let value = Scale.scaled input;
    Local.written_step #2048 #(SZ.v n) sv (scaled_values (SZ.v n) sx (SZ.v offset)) (SZ.v j);
    A.mask_write values j value;
    let old = !maximum;
    norm_spec [delta_once ["KPlay.Proof.Base.SoftmaxScale.maximum"]] (Scale.maximum sx (SZ.v offset) (SZ.v j + 1));
    if (F.lt old value) { maximum := value; };
    i := SZ.add j 1sz;
  };
  !maximum
}

inline_for_extraction noextract
let partial_frame (before after:Local.snapshot 32) (group:P.index 32) : prop =
  forall (i:P.index 32). i <> group ==> Seq.index after i == Seq.index before i

inline_for_extraction noextract
fn stripe
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (width:sz{0 < SZ.v width /\ SZ.v width <= 32})
  (group:sz{SZ.v group < SZ.v width}) (values:larray FP.t 2048) (partial:larray FP.t 32) (maximum:FP.t)
  (#sv:erased (initialized (SZ.v n))) (#sp:erased (Local.snapshot 32))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu
  requires A.pts_to_mask values sv Init.all_cells ** A.pts_to_mask partial sp Init.all_cells
  requires pure (scope ==> Stripes.stage (SZ.v n) (SZ.v width) (SZ.v group) (view (SZ.v n) sv) scores (reveal #real shift) /\
    Approx.v_approximates maximum (reveal #real shift))
  ensures exists* (next:initialized (SZ.v n))
    (temporary:Local.snapshot 32{Some? (Seq.index temporary (SZ.v group))}).
    A.pts_to_mask values next Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells **
    pure (partial_frame sp temporary (SZ.v group) /\
      (scope ==> Stripes.stage (SZ.v n) (SZ.v width) (SZ.v group + 1) (view (SZ.v n) next) scores (reveal #real shift) /\
        Approx.v_approximates (Local.present (Seq.index temporary (SZ.v group))) (Math.group_sums (SZ.v n) (SZ.v width)
          (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)) (SZ.v group))))
{
  Math.slots_bounds (SZ.v n) (SZ.v width);
  Stripes.mixed_start scope (SZ.v n) (SZ.v width) (SZ.v group) (view (SZ.v n) sv) scores (reveal #real shift);
  A.mask_write partial group (F.zero #FP.t);
  let mut j = group;
  while (!j <^ n)
    invariant exists* (i:sz{SZ.v group <= SZ.v i /\ SZ.v i < SZ.v n + SZ.v width /\
      SZ.v i % SZ.v width = SZ.v group /\ (SZ.v i - SZ.v group) / SZ.v width <= Math.slots (SZ.v n) (SZ.v width)})
      (current:initialized (SZ.v n)) (temporary:Local.snapshot 32{Some? (Seq.index temporary (SZ.v group))}).
      j |-> i ** A.pts_to_mask values current Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells **
      pure (partial_frame sp temporary (SZ.v group) /\
        (scope ==> Stripes.mixed (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) (view (SZ.v n) current) scores (reveal #real shift) /\
          Approx.v_approximates (Local.present (Seq.index temporary (SZ.v group)))
            (Stripes.sum_prefix (SZ.v n) (SZ.v width) (SZ.v group) scores (reveal #real shift)
            ((SZ.v i - SZ.v group) / SZ.v width))))
    invariant gpu
    decreases (SZ.v n + SZ.v width - SZ.v !j)
  {
    let i = !j;
    with current temporary. assert A.pts_to_mask values current Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells;
    let input = A.mask_read values i;
    let difference = Stripes.subtract input maximum #(Stripes.score_at #(SZ.v n) scores (SZ.v i)) #shift #scope;
    let value = F.fexp difference;
    Conditional.exponential scope difference (Stripes.score_at #(SZ.v n) scores (SZ.v i) -. reveal #real shift);
    Stripes.mixed_step scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) (view (SZ.v n) current) scores (reveal #real shift) value;
    view_write (SZ.v n) current (SZ.v i) value;
    A.mask_write values i value;
    let old = A.mask_read partial group;
    Stripes.advanced_sum scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) scores (reveal #real shift) old value;
    let combined = F.add old value;
    A.mask_write partial group combined;
    Stripes.cursor_step (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i);
    j := SZ.add i width;
  };
  let i = !j;
  with current temporary. assert A.pts_to_mask values current Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells;
  Stripes.mixed_finish scope (SZ.v n) (SZ.v width) (SZ.v group) (SZ.v i) (view (SZ.v n) current) scores (reveal #real shift);
  Math.stripe_tail_zero (SZ.v n) (SZ.v width) (SZ.v group) ((SZ.v i - SZ.v group) / SZ.v width)
    (Stripes.exponential_values (SZ.v n) scores (reveal #real shift));
  Math.zero_suffix_sum (Math.slots (SZ.v n) (SZ.v width)) ((SZ.v i - SZ.v group) / SZ.v width)
    (Math.stripe_terms (SZ.v n) (SZ.v width) (SZ.v group) (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)));
  ()
}

inline_for_extraction noextract
fn exponentiate_and_reduce
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647})
  (width:sz{0 < SZ.v width /\ SZ.v width <= 32 /\ Norm.tree_size (SZ.v width)})
  (values:larray FP.t 2048) (partial:larray FP.t 32) (maximum:FP.t)
  (#sv:erased (initialized (SZ.v n))) (#sp:erased (Local.snapshot 32))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu
  requires A.pts_to_mask values sv Init.all_cells ** A.pts_to_mask partial sp Init.all_cells
  requires pure (scope ==> Scale.values_rep (SZ.v n) (view (SZ.v n) sv) scores /\ Approx.v_approximates maximum (reveal #real shift))
  ensures exists* (next:initialized (SZ.v n)) (temporary:Local.snapshot 32{Some? (Seq.index temporary 0)}).
    A.pts_to_mask values next Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells **
    pure (scope ==> Scale.values_rep (SZ.v n) (view (SZ.v n) next) (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)) /\
      Approx.v_approximates (Local.present (Seq.index temporary 0)) (Math.exponential_sum (SZ.v n) scores (reveal #real shift)))
{
  let mut group = 0sz;
  while (!group <^ width)
    invariant exists* (g:sz{SZ.v g <= SZ.v width}) (current:initialized (SZ.v n)) (temporary:Local.snapshot 32).
      group |-> g ** A.pts_to_mask values current Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells **
      pure (Local.ready temporary (SZ.v g) /\
        (scope ==> Stripes.stage (SZ.v n) (SZ.v width) (SZ.v g) (view (SZ.v n) current) scores (reveal #real shift) /\
          Numeric.partial_rep temporary (SZ.v g) (Math.group_sums (SZ.v n) (SZ.v width)
            (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)))))
    invariant gpu
    decreases (SZ.v width - SZ.v !group)
  {
    let g = !group;
    stripe n width g values partial maximum #_ #_ #scores #shift #scope;
    group := SZ.add g 1sz;
  };
  with current temporary. assert A.pts_to_mask values current Init.all_cells ** A.pts_to_mask partial temporary Init.all_cells;
  let denominator = Local.reduce #32 partial width false #temporary ();
  Numeric.reduce_approx #32 scope temporary (SZ.v width) ()
    (Math.group_sums (SZ.v n) (SZ.v width) (Stripes.exponential_values (SZ.v n) scores (reveal #real shift))) false denominator;
  Math.stripes_sum (SZ.v n) (SZ.v width) (Stripes.exponential_values (SZ.v n) scores (reveal #real shift));
  Stripes.exponential_binding (SZ.v n) scores (reveal #real shift);
  ()
}

inline_for_extraction noextract
fn finish
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (hidden:larray BF.t 16000) (values:larray FP.t 2048) (partial:larray FP.t 32)
  (#sx:erased Transfer.scratch) (#sv:erased (initialized (SZ.v n)))
  (#sp:erased (s:Local.snapshot 32{Some? (Seq.index s 0)}))
  (#scores:erased (S.vector (SZ.v n))) (#shift:erased real) (#scope:prop)
  preserves gpu ** A.pts_to_mask values sv Init.all_cells ** A.pts_to_mask partial sp Init.all_cells
  requires A.pts_to hidden sx
  requires pure (scope ==> Scale.values_rep (SZ.v n) (view (SZ.v n) sv) (Stripes.exponential_values (SZ.v n) scores (reveal #real shift)) /\
    Approx.v_approximates (Local.present (Seq.index sp 0)) (Math.exponential_sum (SZ.v n) scores (reveal #real shift)))
  ensures exists* (result:Transfer.scratch). A.pts_to hidden result **
    pure (Softmax.frame sx result (SZ.v offset) (SZ.v n) /\ (scope ==> Softmax.produced (SZ.v n) (SZ.v offset) result scores (SZ.v n)))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{SZ.v j <= SZ.v n}) (current:Transfer.scratch).
      i |-> j ** A.pts_to hidden current **
      pure (Softmax.frame sx current (SZ.v offset) (SZ.v j) /\ (scope ==> Softmax.produced (SZ.v n) (SZ.v offset) current scores (SZ.v j)))
    invariant gpu ** A.pts_to_mask values sv Init.all_cells ** A.pts_to_mask partial sp Init.all_cells
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    with current. assert A.pts_to hidden current;
    let value = A.mask_read values j;
    let denominator = A.mask_read partial 0sz;
    let quotient = F.div value denominator;
    Softmax.quotient_approx scope (SZ.v n) scores (reveal #real shift) (SZ.v j) value denominator;
    let rounded = Cast.cast_f32_to_bf16 quotient;
    Softmax.store_step scope (SZ.v n) (SZ.v offset) sx current scores (SZ.v j) rounded;
    hidden.(SZ.add offset j) <- rounded;
    i := SZ.add j 1sz;
  }
}

inline_for_extraction noextract
fn apply
  (n:sz{0 < SZ.v n /\ SZ.v n <= 647}) (offset:sz{SZ.v offset + SZ.v n <= 16000})
  (hidden:larray BF.t 16000) (#sx:erased Transfer.scratch)
  (#input:erased (S.vector (SZ.v n))) (#scope:prop)
  preserves gpu
  requires A.pts_to hidden sx
  requires pure (scope ==> Scale.hidden_rep (SZ.v n) sx (SZ.v offset) input)
  ensures exists* (result:Transfer.scratch). A.pts_to hidden result **
    pure (Softmax.frame sx result (SZ.v offset) (SZ.v n) /\
      (scope ==> Softmax.produced (SZ.v n) (SZ.v offset) result (Math.scaled_scores (SZ.v n) input) (SZ.v n)))
{
  let mut values : A.array FP.t = [| 2048sz |];
  let mut partial : A.array FP.t = [| 32sz |];
  with initial_values initial_partial.
    assert A.pts_to_mask values initial_values Init.all_cells ** A.pts_to_mask partial initial_partial Init.all_cells;
  let width = Scale.source_width n;
  let maximum = scale_and_maximum hidden values n offset;
  scaled_ready (SZ.v n) initial_values sx (SZ.v offset);
  Scale.values_approx scope (SZ.v n) (Seq.create 2048 (F.zero #FP.t)) sx (SZ.v offset) input;
  Approx.to_real_ok maximum;
  let shift : erased real = hide #real (Approx.to_real maximum);
  exponentiate_and_reduce n width values partial maximum #_ #_
    #(Math.scaled_scores (SZ.v n) input) #shift #scope;
  finish n offset hidden values partial #sx #_ #_
    #(Math.scaled_scores (SZ.v n) input) #shift #scope;
}
