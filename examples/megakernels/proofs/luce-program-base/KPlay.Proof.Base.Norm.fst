module KPlay.Proof.Base.Norm

#set-options "--fuel 2 --ifuel 1 --z3rlimit 20"

open FStar.Real
module S = FStar.Seq
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Cast = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module Lit = Kuiper.Float32.Literals
module Sum = Kuiper.Sum
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module O = KPlay.Proof.Shared.Observations
module Algebra = KPlay.Proof.Base.Algebra
module Transfer = KPlay.Proof.Base.Transfer
module Math = KPlay.Proof.Base.ReductionMath
module Attention = KPlay.Proof.Shared.Attention

inline_for_extraction noextract
let width = n:pos{n = 128 \/ n = 256 \/ n = 1024}

inline_for_extraction noextract
let tree_size (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/
  n = 32 \/ n = 64 \/ n = 128 \/ n = 256

inline_for_extraction noextract
let half_size (n:pos{tree_size n /\ n > 1})
  : Lemma (n % 2 == 0 /\ tree_size (n / 2) /\ 0 < n / 2 /\ n / 2 < n) = ()

inline_for_extraction noextract
let epsilon : FP.t = F.of_literal "0.000001"

inline_for_extraction noextract
let epsilon_approx ()
  : Lemma (Approx.v_approximates epsilon P.reference_epsilon) =
  assert_norm (Lit.common_normal_cell "0.000001" 1 1000000 1 1000000 8796093 (-43));
  FP.literal_interval_approx "0.000001" 1 1000000 1 1000000 8796093 (-43)
    P.reference_epsilon

inline_for_extraction noextract
let reciprocal (n:width) : FP.t =
  F.div (F.one #FP.t) (F.of_int #FP.t (FStar.Int64.int_to_t n))

inline_for_extraction noextract
let reciprocal_approx (n:width)
  : Lemma (Approx.v_approximates (reciprocal n) (1.0R /. of_int n)) =
  Approx.of_int_approx #FP.t (FStar.Int64.int_to_t n);
  Approx.div_approx (F.one #FP.t) (F.of_int #FP.t (FStar.Int64.int_to_t n))
    1.0R (of_int n)

inline_for_extraction noextract
let square (x:BF.t) : FP.t =
  let wide = Cast.cast_bf16_to_f32 x in
  F.mul wide wide

inline_for_extraction noextract
let square_approx (x:BF.t) (r:real)
  : Lemma
      (requires Approx.v_approximates x r)
      (ensures Approx.v_approximates (square x) (r *. r)) =
  Cast.cast_bf16_to_f32_ok x r;
  Approx.a_mul (Cast.cast_bf16_to_f32 x) (Cast.cast_bf16_to_f32 x) r r

inline_for_extraction noextract
let rec group (n:width) (x:P.vector BF.t n)
  (g:nat{4 * g + 4 <= n}) (count:nat{count <= 4})
  : GTot FP.t (decreases count) =
  if count = 0 then F.zero #FP.t
  else F.add (group n x g (count - 1)) (square (x (4 * g + count - 1)))

inline_for_extraction noextract
let squares (#n:nat) (x:P.vector real n) : P.vector real n =
  fun i -> x i *. x i

inline_for_extraction noextract
let rec group_approx (n:width) (x:P.vector BF.t n) (r:P.vector real n)
  (g:nat{4 * g + 4 <= n}) (count:nat{count <= 4})
  : Lemma
      (requires O.vector_approximates x r)
      (ensures Approx.v_approximates (group n x g count)
        (Sum.sum (4 * g) (4 * g + count) (squares r)))
      (decreases count) =
  if count > 0 then (
    group_approx n x r g (count - 1);
    let i = 4 * g + count - 1 in
    square_approx (x i) (r i);
    Sum.sum_pop_right (4 * g) (4 * g + count) (squares r);
    assert (group n x g count == F.add (group n x g (count - 1)) (square (x i)));
    assert (Sum.sum (4 * g) (4 * g + count) (squares r) ==
      Sum.sum (4 * g) (4 * g + (count - 1)) (squares r) +. r i *. r i);
    Approx.a_add (group n x g (count - 1)) (square (x i))
      (Sum.sum (4 * g) (4 * g + (count - 1)) (squares r)) (r i *. r i)
  )

inline_for_extraction noextract
let paired_float (n:pos{n <= 128}) (x:P.vector FP.t 256) : P.vector FP.t 256 =
  fun i -> if i < n then F.add (x i) (x (i + n)) else x i

inline_for_extraction noextract
let paired_real (n:pos{n <= 128}) (x:P.vector real 256) : P.vector real 256 =
  fun i -> if i < n then x i +. x (i + n) else x i

inline_for_extraction noextract
let rec tree (n:pos{tree_size n}) (x:P.vector FP.t 256)
  : GTot FP.t (decreases n) =
  if n = 1 then x 0
  else (
    half_size n;
    tree (n / 2) (paired_float (n / 2) x)
  )

#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"

inline_for_extraction noextract
let partial_sum (n:nat{n <= 256}) (x:P.vector real 256) : GTot real =
  Attention.sum n (fun i -> x i)

inline_for_extraction noextract
let sum_pairs (n:pos{n <= 128}) (x:P.vector real 256)
  : Lemma (
      partial_sum n (paired_real n x) == partial_sum (2 * n) x) =
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.partial_sum"]] (partial_sum n (paired_real n x));
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.partial_sum"]] (partial_sum (2 * n) x);
  Math.sum_halves n (fun i -> x i);
  Math.sum_ext n (fun i -> paired_real n x i) (fun i -> x i +. x (i + n))

inline_for_extraction noextract
let rec tree_approx (n:pos{tree_size n}) (x:P.vector FP.t 256) (r:P.vector real 256)
  : Lemma
      (requires forall (i:nat{i < n}). Approx.v_approximates (x i) (r i))
      (ensures Approx.v_approximates (tree n x) (partial_sum n r))
      (decreases n) =
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.tree"]] (tree n x);
  if n = 1 then (
    assert (Approx.v_approximates (x 0) (r 0));
    Math.sum_step 1 (fun i -> r i);
    Math.sum_zero (fun i -> r i);
    norm_spec [delta_once ["KPlay.Proof.Base.Norm.partial_sum"]] (partial_sum 1 r)
  ) else (
    half_size n;
    assert (n == 2 * (n / 2));
    let each (i:nat{i < n / 2})
      : Lemma (Approx.v_approximates (paired_float (n / 2) x i)
          (paired_real (n / 2) r i)) =
      Approx.a_add (x i) (x (i + n / 2)) (r i) (r (i + n / 2))
    in
    FStar.Classical.forall_intro each;
    tree_approx (n / 2) (paired_float (n / 2) x) (paired_real (n / 2) r);
    sum_pairs (n / 2) r
  )

inline_for_extraction noextract
let grouped (n:width) (x:P.vector BF.t n) : P.vector FP.t 256 =
  fun i -> if i < n / 4 then group n x i 4 else F.zero #FP.t

inline_for_extraction noextract
let real_groups (n:width) (r:P.vector real n) : P.vector real 256 =
  fun i -> if i < n / 4 then Sum.sum (4 * i) (4 * i + 4) (squares r) else 0.0R

inline_for_extraction noextract
let rec groups_prefix (n:width) (r:P.vector real n) (count:nat{count <= n / 4})
  : Lemma
      (ensures Sum.sum 0 count (real_groups n r) == Sum.sum 0 (4 * count) (squares r))
      (decreases count) =
  if count > 0 then (
    groups_prefix n r (count - 1);
    Sum.sum_pop_right 0 count (real_groups n r);
    Sum.sum_split 0 (4 * count) (squares r) (4 * (count - 1));
    norm_spec [delta_once ["KPlay.Proof.Base.Norm.real_groups"]]
      (real_groups n r (count - 1))
  )

inline_for_extraction noextract
let groups_sum (n:width) (r:P.vector real n)
  : Lemma (partial_sum (n / 4) (real_groups n r) == P.real_sum (squares r)) =
  assert (4 * (n / 4) == n);
  Math.sum_kuiper_prefix #256 (real_groups n r) (n / 4);
  groups_prefix n r (n / 4);
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.partial_sum"]]
    (partial_sum (n / 4) (real_groups n r));
  norm_spec [delta_once ["KPlay.Proof.Shared.Projection.real_sum"]]
    (P.real_sum (squares r))

inline_for_extraction noextract
let source_sum (n:width) (x:P.vector BF.t n) : GTot FP.t =
  tree (n / 4) (grouped n x)

inline_for_extraction noextract
let source_sum_approx (n:width) (x:P.vector BF.t n) (r:P.vector real n)
  : Lemma
      (requires O.vector_approximates x r)
      (ensures Approx.v_approximates (source_sum n x) (P.real_sum (squares r))) =
  let each (i:nat{i < n / 4})
    : Lemma (Approx.v_approximates (grouped n x i) (real_groups n r i)) =
    group_approx n x r i 4
  in
  FStar.Classical.forall_intro each;
  tree_approx (n / 4) (grouped n x) (real_groups n r);
  groups_sum n r

inline_for_extraction noextract
let source_scale (n:width) (x:P.vector BF.t n) : GTot FP.t =
  F.rsqrt (F.add (F.mul (source_sum n x) (reciprocal n)) epsilon)

inline_for_extraction noextract
let argument_formula (n:width) (r:P.vector real n)
  : Lemma (P.rms_argument #n P.reference_epsilon r ==
      P.real_sum #n (squares #n r) /. of_int n +. P.reference_epsilon) =
  assert (P.rms_argument #n P.reference_epsilon r ==
    P.real_sum #n (squares #n r) /. of_int n +. P.reference_epsilon) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only [
      "KPlay.Proof.Shared.Projection.rms_argument"; "KPlay.Proof.Base.Norm.squares"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let multiply_reciprocal (value:real) (n:width)
  : Lemma (value /. of_int n == value *. (1.0R /. of_int n)) = ()

inline_for_extraction noextract
let source_scale_approx (n:width) (x:P.vector BF.t n) (r:P.vector real n)
  : Lemma
      (requires O.vector_approximates x r)
      (ensures Approx.v_approximates (source_scale n x)
        (Root.rsqrt (P.rms_argument P.reference_epsilon r))) =
  source_sum_approx n x r;
  reciprocal_approx n;
  epsilon_approx ();
  argument_formula n r;
  multiply_reciprocal (P.real_sum #n (squares #n r)) n;
  Approx.a_mul (source_sum n x) (reciprocal n)
    (P.real_sum (squares r)) (1.0R /. of_int n);
  assert (P.rms_argument P.reference_epsilon r ==
    P.real_sum (squares r) *. (1.0R /. of_int n) +. P.reference_epsilon);
  Approx.a_add (F.mul (source_sum n x) (reciprocal n)) epsilon
    (P.real_sum (squares r) *. (1.0R /. of_int n)) P.reference_epsilon;
  Approx.rsqrt_approx (F.add (F.mul (source_sum n x) (reciprocal n)) epsilon)
    (P.rms_argument P.reference_epsilon r)

inline_for_extraction noextract
let source_cell (x weight:BF.t) (scale:FP.t) : BF.t =
  Cast.cast_f32_to_bf16
    (F.mul (F.mul (Cast.cast_bf16_to_f32 x) scale)
      (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 weight)))

inline_for_extraction noextract
let source_cell_approx (x weight:BF.t) (scale:FP.t) (xr wr sr:real)
  : Lemma
      (requires Approx.v_approximates x xr /\ Approx.v_approximates weight wr /\
        Approx.v_approximates scale sr)
      (ensures Approx.v_approximates (source_cell x weight scale) ((xr *. sr) *. (1.0R +. wr))) =
  Cast.cast_bf16_to_f32_ok x xr;
  Cast.cast_bf16_to_f32_ok weight wr;
  Approx.a_mul (Cast.cast_bf16_to_f32 x) scale xr sr;
  Approx.a_add (F.one #FP.t) (Cast.cast_bf16_to_f32 weight) 1.0R wr;
  Approx.a_mul (F.mul (Cast.cast_bf16_to_f32 x) scale)
    (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 weight)) (xr *. sr) (1.0R +. wr);
  Cast.cast_f32_to_bf16_ok
    (F.mul (F.mul (Cast.cast_bf16_to_f32 x) scale)
      (F.add (F.one #FP.t) (Cast.cast_bf16_to_f32 weight))) ((xr *. sr) *. (1.0R +. wr))

inline_for_extraction noextract
let source_normalized (n:width) (x weight:P.vector BF.t n) : P.vector BF.t n =
  fun i -> source_cell (x i) (weight i) (source_scale n x)

inline_for_extraction noextract
let source_normalized_approx (n:width) (x weight:P.vector BF.t n) (xr wr:P.vector real n)
  : Lemma
      (requires O.vector_approximates x xr /\ O.vector_approximates weight wr)
      (ensures O.vector_approximates (source_normalized n x weight)
        (P.normalized_for_logits P.reference_epsilon xr wr)) =
  source_scale_approx n x xr;
  let each (i:P.index n)
    : Lemma (Approx.v_approximates (source_normalized n x weight i)
      (P.normalized_for_logits P.reference_epsilon xr wr i)) =
    source_cell_approx (x i) (weight i) (source_scale n x) (xr i) (wr i)
      (Root.rsqrt (P.rms_argument P.reference_epsilon xr))
  in
  FStar.Classical.forall_intro each

inline_for_extraction noextract
let residual_values (before:Transfer.scratch) : P.vector BF.t 1024 =
  fun i -> S.index before i

inline_for_extraction noextract
let norm_weights (weights:Transfer.packed) : P.vector BF.t 1024 =
  fun i -> S.index weights (254279680 + i)

inline_for_extraction noextract
let final_value (before:Transfer.scratch) (weights:Transfer.packed) (i:P.index 16000)
  : GTot BF.t =
  if 1024 <= i && i < 2048 then
    source_normalized 1024 (residual_values before) (norm_weights weights) (i - 1024)
  else S.index before i

inline_for_extraction noextract
let final_writes (before:Transfer.scratch) (weights:Transfer.packed)
  : GTot Transfer.scratch =
  S.init_ghost 16000 (final_value before weights)

inline_for_extraction noextract
let final_writes_cell (before:Transfer.scratch) (weights:Transfer.packed) (i:P.index 16000)
  : Lemma (
      S.index (final_writes before weights) i == final_value before weights i) =
  S.init_ghost_index_ 16000 (final_value before weights) i;
  norm_spec [delta_once ["KPlay.Proof.Base.Norm.final_writes"]] (final_writes before weights)

inline_for_extraction noextract
let final_connection (before:Transfer.scratch) (weights:Transfer.packed)
  (residual:P.vector real 1024)
  : Lemma
      (requires O.vector_approximates (fun i -> S.index before i) residual)
      (ensures
        Transfer.normalized_input (final_writes before weights)
          (P.normalized_for_logits P.reference_epsilon residual
            (fun i -> Approx.to_real (S.index weights (254279680 + i)))) /\
        (forall (i:P.index 16000).
          ~(1024 <= i /\ i < 2048) ==> S.index (final_writes before weights) i == S.index before i)) =
  let wr = fun (i:P.index 1024) -> Approx.to_real (S.index weights (254279680 + i)) in
  let actual = norm_weights weights in
  let each (i:P.index 1024) : Lemma (Approx.v_approximates (actual i) (wr i)) =
    Approx.to_real_ok (actual i)
  in
  FStar.Classical.forall_intro each;
  source_normalized_approx 1024 (residual_values before) actual residual wr;
  let copied (i:P.index 1024)
    : Lemma (Approx.v_approximates (S.index (final_writes before weights) (1024 + i))
        (P.normalized_for_logits P.reference_epsilon residual wr i)) =
    final_writes_cell before weights (1024 + i);
    norm_spec [delta_once ["KPlay.Proof.Base.Norm.final_value"]] (final_value before weights (1024 + i));
    assert (Approx.v_approximates
      (source_normalized 1024 (residual_values before) actual i)
      (P.normalized_for_logits P.reference_epsilon residual wr i))
  in
  FStar.Classical.forall_intro copied;
  let frame (i:P.index 16000)
    : Lemma (~(1024 <= i /\ i < 2048) ==>
      S.index (final_writes before weights) i == S.index before i) =
    final_writes_cell before weights i
  in
  FStar.Classical.forall_intro frame

#pop-options
