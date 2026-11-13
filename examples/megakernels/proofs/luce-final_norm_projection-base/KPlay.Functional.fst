module KPlay.Functional

#lang-pulse

#set-options "--fuel 4 --ifuel 2 --z3rlimit 40"

open Kuiper
module C = Kuiper.Float.Casts.Base
module Approx = Kuiper.Approximates.Base
module F = Kuiper.Floating.Base
module Shared = KPlay.SharedSpec
module Root = FStar.Math.Sqrt
module Sum = Kuiper.Sum
module Seq = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT

inline_for_extraction noextract
let epsilon () : f32 = F.div (one #f32) (F.of_int #f32 1000000L)

inline_for_extraction noextract
let inverse_width () : f32 = F.div (one #f32) (F.of_int #f32 1024L)

inline_for_extraction noextract
let constants_approx ()
  : Lemma
      (epsilon () %~ (Shared.reference_epsilon <: real) /\
       inverse_width () %~ (1.0R /. 1024.0R)) =
  let _ = Approx.of_int_approx #f32 1000000L in
  let _ = Approx.of_int_approx #f32 1024L in
  Approx.div_approx (one #f32) (F.of_int #f32 1000000L)
    1.0R 1000000.0R;
  Approx.div_approx (one #f32) (F.of_int #f32 1024L)
    1.0R 1024.0R

inline_for_extraction noextract
let rms_expand (x:Shared.vector real 1024)
  : Lemma (
      Shared.rms_argument #1024 Shared.reference_epsilon x ==
        Shared.real_sum #1024 (fun i -> x i *. x i) /.
          FStar.Real.of_int 1024 +. Shared.reference_epsilon) =
  assert (
    Shared.rms_argument #1024 Shared.reference_epsilon x ==
      Shared.real_sum #1024 (fun i -> x i *. x i) /.
        FStar.Real.of_int 1024 +. Shared.reference_epsilon) by (
    FStar.Tactics.V2.norm [zeta; iota; delta_only ["KPlay.SharedSpec.rms_argument"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let divide_by_width (s:real)
  : Lemma (s /. FStar.Real.of_int 1024 == s *. (1.0R /. 1024.0R)) =
  ()

inline_for_extraction noextract
let squares_sum (hidden:Shared.vector bf16 1024) : GTot real =
  Shared.real_sum #1024
    (fun i -> Approx.to_real (hidden i) *. Approx.to_real (hidden i))

inline_for_extraction noextract
let real_argument (hidden:Shared.vector bf16 1024) : GTot Root.rpos =
  Shared.rms_argument #1024 Shared.reference_epsilon
    (Shared.real_vector #1024 hidden)

inline_for_extraction noextract
let real_argument_expand (hidden:Shared.vector bf16 1024)
  : Lemma (
      real_argument hidden ==
        squares_sum hidden /. FStar.Real.of_int 1024 +.
          Shared.reference_epsilon) =
  assert (
    real_argument hidden ==
      squares_sum hidden /. FStar.Real.of_int 1024 +.
        Shared.reference_epsilon) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.real_argument";
        "KPlay.Functional.squares_sum";
        "KPlay.SharedSpec.rms_argument";
        "KPlay.SharedSpec.real_vector"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let scale_value (sum_squares:f32) : f32 =
  F.rsqrt (add (mul sum_squares (inverse_width ())) (epsilon ()))

inline_for_extraction noextract
let scale_approx
  (sum_squares:f32)
  (hidden:Shared.vector bf16 1024)
  : Lemma
      (requires sum_squares %~ squares_sum hidden)
      (ensures
        scale_value sum_squares %~
          (Root.rsqrt (real_argument hidden) <: real)) =
  constants_approx ();
  real_argument_expand hidden;
  divide_by_width (squares_sum hidden);
  Approx.a_mul sum_squares (inverse_width ())
    (squares_sum hidden) (1.0R /. 1024.0R);
  Approx.a_add (mul sum_squares (inverse_width ())) (epsilon ())
    (squares_sum hidden *. (1.0R /. 1024.0R)) Shared.reference_epsilon;
  Approx.rsqrt_approx
    (add (mul sum_squares (inverse_width ())) (epsilon ()))
    (real_argument hidden)

inline_for_extraction noextract
let scale_from_mean (mean:f32) : f32 =
  F.rsqrt (add mean (epsilon ()))

inline_for_extraction noextract
let scale_from_mean_approx (mean:f32) (hidden:Shared.vector bf16 1024)
  : Lemma
      (requires mean %~ (squares_sum hidden /. 1024.0R))
      (ensures
        scale_from_mean mean %~ (Root.rsqrt (real_argument hidden) <: real)) =
  constants_approx ();
  real_argument_expand hidden;
  Approx.a_add mean (epsilon ()) (squares_sum hidden /. 1024.0R)
    Shared.reference_epsilon;
  Approx.rsqrt_approx (add mean (epsilon ())) (real_argument hidden)

inline_for_extraction noextract
fn normalize_cell
  (hidden weight:bf16)
  (rms_scale:f32)
  (#scale_real:erased real)
  preserves gpu
  requires pure (rms_scale %~ (reveal scale_real))
  returns result:Kuiper.BFloat16.t
  ensures pure (
    result %~
      ((Approx.to_real hidden *. reveal scale_real) *.
        (1.0R +. Approx.to_real weight)))
{
  Approx.to_real_ok hidden;
  Approx.to_real_ok weight;
  C.cast_bf16_to_f32_ok hidden (Approx.to_real hidden);
  C.cast_bf16_to_f32_ok weight (Approx.to_real weight);
  let x = C.cast_bf16_to_f32 hidden;
  let w = C.cast_bf16_to_f32 weight;
  let scaled = mul x rms_scale;
  let gain = add (one #f32) w;
  let normalized = mul scaled gain;
  Approx.a_mul x rms_scale (Approx.to_real hidden) (reveal scale_real);
  Approx.a_add (one #f32) w 1.0R (Approx.to_real weight);
  Approx.a_mul scaled gain (Approx.to_real hidden *. reveal scale_real)
    (1.0R +. Approx.to_real weight);
  C.cast_f32_to_bf16_ok normalized
    ((Approx.to_real hidden *. reveal scale_real) *.
      (1.0R +. Approx.to_real weight));
  C.cast_f32_to_bf16 normalized
}

inline_for_extraction noextract
let square_real (x:bf16) : GTot real =
  Approx.to_real x *. Approx.to_real x

inline_for_extraction noextract
let square_value (x:bf16) : f32 =
  mul (C.cast_bf16_to_f32 x) (C.cast_bf16_to_f32 x)

inline_for_extraction noextract
let square_approx (x:bf16)
  : Lemma (square_value x %~ square_real x) =
  Approx.to_real_ok x;
  C.cast_bf16_to_f32_ok x (Approx.to_real x);
  Approx.a_mul (C.cast_bf16_to_f32 x) (C.cast_bf16_to_f32 x)
    (Approx.to_real x) (Approx.to_real x)

inline_for_extraction noextract
let add_four (a b c d:f32) : f32 =
  add (add (add (add (zero #f32) a) b) c) d

inline_for_extraction noextract
let add_four_approx (a b c d:f32) (ar br cr dr:real)
  : Lemma
      (requires a %~ ar /\ b %~ br /\ c %~ cr /\ d %~ dr)
      (ensures add_four a b c d %~ (((ar +. br) +. cr) +. dr)) =
  Approx.a_add (zero #f32) a 0.0R ar;
  Approx.a_add (add (zero #f32) a) b ar br;
  Approx.a_add (add (add (zero #f32) a) b) c (ar +. br) cr;
  Approx.a_add (add (add (add (zero #f32) a) b) c) d ((ar +. br) +. cr) dr

inline_for_extraction noextract
let four_value (a b c d:bf16) : f32 =
  add (add (add (add (zero #f32) (square_value a))
    (square_value b)) (square_value c)) (square_value d)

inline_for_extraction noextract
unfold
let four_real (a b c d:bf16) : GTot real =
  ((square_real a +. square_real b) +. square_real c) +. square_real d

inline_for_extraction noextract
let four_approx (a b c d:bf16)
  : Lemma (four_value a b c d %~ four_real a b c d) =
  square_approx a;
  square_approx b;
  square_approx c;
  square_approx d;
  Approx.a_add (zero #f32) (square_value a) 0.0R (square_real a);
  Approx.a_add (add (zero #f32) (square_value a)) (square_value b)
    (square_real a) (square_real b);
  Approx.a_add (add (add (zero #f32) (square_value a)) (square_value b))
    (square_value c) (square_real a +. square_real b) (square_real c);
  Approx.a_add
    (add (add (add (zero #f32) (square_value a)) (square_value b))
      (square_value c))
    (square_value d)
    ((square_real a +. square_real b) +. square_real c) (square_real d)

inline_for_extraction noextract
unfold
let group_real (hidden:lseq bf16 1024) (i:natlt 256) : GTot real =
  four_real (Seq.index hidden (4 * i)) (Seq.index hidden (4 * i + 1))
    (Seq.index hidden (4 * i + 2)) (Seq.index hidden (4 * i + 3))

inline_for_extraction noextract
let initial_reals (hidden:lseq bf16 1024) : GTot (lseq real 256) =
  Seq.init_ghost 256 (group_real hidden)

[@@ "opaque_to_smt"]
inline_for_extraction noextract
let active_sum (n:nat{n <= 256}) (values:lseq real 256) : GTot real =
  Sum.sum #real #Sum.real_add_semigroup 0 n
    (fun (i:natlt 256) -> Seq.index values i)

inline_for_extraction noextract
let active_sum_expand (n:nat{n <= 256}) (values:lseq real 256)
  : Lemma (
      active_sum n values ==
      Sum.sum #real #Sum.real_add_semigroup 0 n
        (fun (i:natlt 256) -> Seq.index values i)) =
  reveal_opaque (`%active_sum) (active_sum n values)

inline_for_extraction noextract
let active_sum_one (values:lseq real 256)
  : Lemma (active_sum 1 values == Seq.index values 0) =
  active_sum_expand 1 values

inline_for_extraction noextract
let paired_reals (n:pos{n <= 128}) (values:lseq real 256)
  : GTot (lseq real 256) =
  Seq.init_ghost 256 (fun i ->
    if i < n then Seq.index values i +. Seq.index values (n + i)
    else Seq.index values i)

#push-options "--fuel 1 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let real_add_op (x y:real)
  : Lemma (Sum.op #real #Sum.real_add_semigroup x y == x +. y) =
  ()

inline_for_extraction noextract
let sum_head
  (#n:nat) (first last:nat{first < last /\ last <= n})
  (f:natlt n -> GTot real)
  : Lemma (
      Sum.sum #real #Sum.real_add_semigroup first last f ==
      f first +. Sum.sum #real #Sum.real_add_semigroup (first + 1) last f) =
  Sum.sum_pop_left #real #Sum.real_add_semigroup first last f;
  real_add_op (f first) (Sum.sum #real #Sum.real_add_semigroup (first + 1) last f)

inline_for_extraction noextract
let rec sum_ext
  (#n:nat) (first last:nat{first <= last /\ last <= n})
  (f g:natlt n -> GTot real)
  : Lemma
      (requires forall (i:nat{first <= i /\ i < last}). f i == g i)
      (ensures
        Sum.sum #real #Sum.real_add_semigroup first last f ==
        Sum.sum #real #Sum.real_add_semigroup first last g)
      (decreases (last - first)) =
  if first < last then (
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last f;
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last g;
    sum_ext #n (first + 1) last f g
  )

inline_for_extraction noextract
let rec sum_add
  (#n:nat) (first last:nat{first <= last /\ last <= n})
  (f g:natlt n -> GTot real)
  : Lemma
      (Sum.sum #real #Sum.real_add_semigroup first last (fun (i:natlt n) -> f i +. g i) ==
       Sum.sum #real #Sum.real_add_semigroup first last f +.
       Sum.sum #real #Sum.real_add_semigroup first last g)
      (decreases (last - first)) =
  if first < last then (
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last (fun (i:natlt n) -> f i +. g i);
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last f;
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last g;
    sum_add #n (first + 1) last f g;
    real_add_op (f first +. g first)
      (Sum.sum #real #Sum.real_add_semigroup (first + 1) last (fun (i:natlt n) -> f i +. g i));
    real_add_op (f first) (Sum.sum #real #Sum.real_add_semigroup (first + 1) last f);
    real_add_op (g first) (Sum.sum #real #Sum.real_add_semigroup (first + 1) last g)
  )

inline_for_extraction noextract
let rec sum_shift
  (#n:nat) (first last offset:nat{first <= last /\ last + offset <= n})
  (f:natlt n -> GTot real)
  : Lemma
      (Sum.sum #real #Sum.real_add_semigroup first last (fun (i:natlt (n - offset)) -> f (offset + i)) ==
       Sum.sum #real #Sum.real_add_semigroup (offset + first) (offset + last) f)
      (decreases (last - first)) =
  if first < last then (
    Sum.sum_pop_left #real #Sum.real_add_semigroup first last (fun (i:natlt (n - offset)) -> f (offset + i));
    Sum.sum_pop_left #real #Sum.real_add_semigroup (offset + first) (offset + last) f;
    sum_shift #n (first + 1) last offset f
  )

#pop-options

[@@ "opaque_to_smt"]
inline_for_extraction noextract
let normalized_vector
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704)
  : GTot (Shared.vector real 1024) =
  Shared.normalized_for_logits #1024 Shared.reference_epsilon
    (Shared.real_vector #1024 (fun i -> Seq.index hidden i))
    (Shared.real_vector #1024 (fun i -> Seq.index packed (254279680 + i)))

inline_for_extraction noextract
let normalized_formula
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704) (i:natlt 1024)
  : Lemma (
      normalized_vector hidden packed i ==
      (Approx.to_real (Seq.index hidden i) *.
        Root.rsqrt (real_argument (fun j -> Seq.index hidden j))) *.
      (1.0R +. Approx.to_real (Seq.index packed (254279680 + i)))) =
  assert (
    normalized_vector hidden packed i ==
    (Approx.to_real (Seq.index hidden i) *.
      Root.rsqrt (real_argument (fun j -> Seq.index hidden j))) *.
    (1.0R +. Approx.to_real (Seq.index packed (254279680 + i)))) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.normalized_vector";
        "KPlay.Functional.real_argument";
        "KPlay.SharedSpec.normalized_for_logits";
        "KPlay.SharedSpec.real_vector"]];
    FStar.Tactics.V2.trefl ())

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let paired_sum (n:pos{n <= 128}) (values:lseq real 256)
  : Lemma (active_sum n (paired_reals n values) == active_sum (2 * n) values) =
  active_sum_expand n (paired_reals n values);
  active_sum_expand (2 * n) values;
  let f : natlt 256 -> GTot real = fun i -> Seq.index values i in
  let g : natlt n -> GTot real = fun i -> Seq.index values (n + i) in
  sum_ext #n 0 n
    (fun (i:natlt 256) -> Seq.index (paired_reals n values) i)
    (fun (i:natlt n) -> f i +. g i);
  sum_add #n 0 n f g;
  sum_ext #n 0 n g (fun (i:natlt (256 - n)) -> f (n + i));
  sum_shift #256 0 n n f;
  Sum.sum_split #real #Sum.real_add_semigroup 0 (2 * n) f n;
  real_add_op
    (Sum.sum #real #Sum.real_add_semigroup 0 n f)
    (Sum.sum #real #Sum.real_add_semigroup n (2 * n) f);
  assert (Sum.sum #real #Sum.real_add_semigroup 0 n
    (fun (i:natlt 256) -> Seq.index (paired_reals n values) i) ==
    Sum.sum #real #Sum.real_add_semigroup 0 (2 * n) f)

#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let rec groups_sum (hidden:lseq bf16 1024) (first:nat{first <= 256})
  : Lemma (
      Sum.sum #real #Sum.real_add_semigroup first 256 (group_real hidden) ==
      Sum.sum #real #Sum.real_add_semigroup (4 * first) 1024
        (fun (i:natlt 1024) -> square_real (Seq.index hidden i)))
      (decreases (256 - first)) =
  if first < 256 then (
    let f = fun (i:natlt 1024) -> square_real (Seq.index hidden i) in
    sum_head #256 first 256 (group_real hidden);
    sum_head #1024 (4 * first) 1024 f;
    sum_head #1024 (4 * first + 1) 1024 f;
    sum_head #1024 (4 * first + 2) 1024 f;
    sum_head #1024 (4 * first + 3) 1024 f;
    groups_sum hidden (first + 1)
  )

#pop-options

inline_for_extraction noextract
let squares_sum_seq (hidden:lseq bf16 1024)
  : Lemma (
      squares_sum (fun i -> Seq.index hidden i) ==
        Sum.sum #real #Sum.real_add_semigroup 0 1024
          (fun (i:natlt 1024) -> square_real (Seq.index hidden i))) =
  assert (
    squares_sum (fun i -> Seq.index hidden i) ==
      Sum.sum #real #Sum.real_add_semigroup 0 1024
        (fun (i:natlt 1024) -> square_real (Seq.index hidden i))) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.squares_sum";
        "KPlay.Functional.square_real";
        "KPlay.SharedSpec.real_sum"]];
    FStar.Tactics.V2.trefl ())

#push-options "--fuel 1 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let initial_sum (hidden:lseq bf16 1024)
  : Lemma (
      active_sum 256 (initial_reals hidden) ==
        squares_sum (fun i -> Seq.index hidden i)) =
  active_sum_expand 256 (initial_reals hidden);
  sum_ext #256 0 256
    (fun (i:natlt 256) -> Seq.index (initial_reals hidden) i) (group_real hidden);
  groups_sum hidden 0;
  squares_sum_seq hidden

#pop-options

inline_for_extraction noextract
fn initialize_partials
  (squares:larray f32 1024)
  (partial:larray f32 256)
  (#fraction:perm)
  (#hv:erased (lseq bf16 1024))
  (#sv:erased (lseq f32 1024))
  (#pv:erased (lseq f32 256))
  preserves gpu ** squares |-> Frac fraction sv
  requires partial |-> pv **
    pure (forall (i:natlt 1024). Seq.index sv i %~ square_real (Seq.index hv i))
  ensures exists* (values:lseq f32 256).
    partial |-> values **
    pure (forall (i:natlt 256).
      Seq.index values i %~ Seq.index (initial_reals hv) i)
{
  let mut index = 0sz;
  while (!index <^ 256sz)
    invariant exists* (i:sz{i <= 256}). index |-> i
    invariant exists* (values:lseq f32 256).
      partial |-> values **
      pure (forall (j:nat{j < !index}).
        Seq.index values j %~ Seq.index (initial_reals hv) j)
    decreases (256 - !index)
  {
    let i = !index;
    let a = squares.(4sz *^ i);
    let b = squares.(4sz *^ i +^ 1sz);
    let c = squares.(4sz *^ i +^ 2sz);
    let d = squares.(4sz *^ i +^ 3sz);
    add_four_approx a b c d
      (square_real (Seq.index hv (4 * i)))
      (square_real (Seq.index hv (4 * i + 1)))
      (square_real (Seq.index hv (4 * i + 2)))
      (square_real (Seq.index hv (4 * i + 3)));
    A.pts_to_len partial;
    partial.(i) <- add_four a b c d;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn reduce_stage
  (partial:larray f32 256)
  (n:szp{n <= 128})
  (#rv:erased (lseq real 256))
  (#pv:erased (lseq f32 256))
  preserves gpu
  requires partial |-> pv **
    pure (forall (i:nat{i < 2 * n}).
      Seq.index pv i %~ Seq.index rv i)
  ensures exists* (values:lseq f32 256).
    partial |-> values **
    pure (forall (i:nat{i < n}).
      Seq.index values i %~ Seq.index (paired_reals n rv) i)
{
  let mut index = 0sz;
  while (!index <^ n)
    invariant exists* (i:sz{i <= n}). index |-> i
    invariant exists* (values:lseq f32 256).
      partial |-> values **
      pure (forall (j:nat{j < 2 * n}).
        Seq.index values j %~
          (if j < !index then Seq.index rv j +. Seq.index rv (n + j)
           else Seq.index rv j))
    decreases (n - !index)
  {
    let i = !index;
    let left = partial.(i);
    let right = partial.(i +^ n);
    Approx.a_add left right (Seq.index rv i) (Seq.index rv (n + i));
    partial.(i) <- add left right;
    index := i +^ 1sz;
  }
}

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
fn rms_sum
  (squares:larray f32 1024)
  (#fraction:perm)
  (#hv:erased (lseq bf16 1024))
  (#sv:erased (lseq f32 1024))
  preserves gpu ** squares |-> Frac fraction sv
  requires
    pure (forall (i:natlt 1024). Seq.index sv i %~ square_real (Seq.index hv i))
  returns result:f32
  ensures pure (result %~ squares_sum (fun i -> Seq.index hv i))
{
  let mut partial : A.array f32 = [| zero #f32; 256sz |];
  initialize_partials squares partial #fraction #hv #sv;
  let rv0 : erased (lseq real 256) = initial_reals hv;
  initial_sum hv;
  reduce_stage partial 128sz #rv0;
  let rv1 : erased (lseq real 256) = paired_reals 128 rv0;
  paired_sum 128 rv0;
  assert pure (active_sum 128 rv1 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 64sz #rv1;
  let rv2 : erased (lseq real 256) = paired_reals 64 rv1;
  paired_sum 64 rv1;
  assert pure (active_sum 64 rv2 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 32sz #rv2;
  let rv3 : erased (lseq real 256) = paired_reals 32 rv2;
  paired_sum 32 rv2;
  assert pure (active_sum 32 rv3 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 16sz #rv3;
  let rv4 : erased (lseq real 256) = paired_reals 16 rv3;
  paired_sum 16 rv3;
  assert pure (active_sum 16 rv4 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 8sz #rv4;
  let rv5 : erased (lseq real 256) = paired_reals 8 rv4;
  paired_sum 8 rv4;
  assert pure (active_sum 8 rv5 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 4sz #rv5;
  let rv6 : erased (lseq real 256) = paired_reals 4 rv5;
  paired_sum 4 rv5;
  assert pure (active_sum 4 rv6 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 2sz #rv6;
  let rv7 : erased (lseq real 256) = paired_reals 2 rv6;
  paired_sum 2 rv6;
  assert pure (active_sum 2 rv7 == squares_sum (fun i -> Seq.index hv i));
  reduce_stage partial 1sz #rv7;
  let rv8 : erased (lseq real 256) = paired_reals 1 rv7;
  paired_sum 1 rv7;
  assert pure (active_sum 1 rv8 == squares_sum (fun i -> Seq.index hv i));
  active_sum_one rv8;
  partial.(0sz)
}

#pop-options

(* Separate launches materialize the products and mean in FP32. This prevents
   CUDA's default contraction from replacing the accepted multiply/add steps. *)
inline_for_extraction noextract
fn compute_squares
  (hidden:larray bf16 1024)
  (squares:larray f32 1024)
  (#fraction:perm)
  (#hv:erased (lseq bf16 1024))
  (#sv:erased (seq f32))
  preserves gpu ** hidden |-> Frac fraction hv
  requires squares |-> sv
  ensures exists* (values:lseq f32 1024).
    squares |-> values **
    pure (forall (i:natlt 1024).
      Seq.index values i %~ square_real (Seq.index hv i))
{
  A.pts_to_len squares;
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i:sz{i <= 1024}). index |-> i
    invariant exists* (values:lseq f32 1024).
      squares |-> values **
      pure (forall (j:nat{j < !index}).
        Seq.index values j %~ square_real (Seq.index hv j))
    decreases (1024 - !index)
  {
    let i = !index;
    let x = hidden.(i);
    square_approx x;
    squares.(i) <- square_value x;
    index := i +^ 1sz;
  }
}

inline_for_extraction noextract
fn compute_mean
  (squares:larray f32 1024)
  (mean:larray f32 1)
  (#hv:erased (lseq bf16 1024))
  (#sv:erased (lseq f32 1024))
  (#mv:erased (seq f32))
  preserves gpu ** squares |-> sv
  requires mean |-> mv **
    pure (forall (i:natlt 1024). Seq.index sv i %~ square_real (Seq.index hv i))
  ensures exists* (values:lseq f32 1).
    mean |-> values **
    pure (Seq.index values 0 %~ (squares_sum (fun i -> Seq.index hv i) /. 1024.0R))
{
  let sum = rms_sum squares #1.0R #hv #sv;
  constants_approx ();
  Approx.a_mul sum (inverse_width ())
    (squares_sum (fun i -> Seq.index hv i)) (1.0R /. 1024.0R);
  divide_by_width (squares_sum (fun i -> Seq.index hv i));
  A.pts_to_len mean;
  mean.(0sz) <- mul sum (inverse_width ());
}

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
fn normalize_and_replicate
  (packed:larray bf16 254280704)
  (hidden:larray bf16 1024)
  (mean:larray f32 1)
  (normalized:larray bf16 8192)
  (#packed_fraction #hidden_fraction:perm)
  (#pv:erased (lseq bf16 254280704))
  (#hv:erased (lseq bf16 1024))
  (#mv:erased (lseq f32 1))
  (#nv:erased (seq bf16))
  preserves
    gpu ** packed |-> Frac packed_fraction pv ** hidden |-> Frac hidden_fraction hv **
    mean |-> mv
  requires normalized |-> nv **
    pure (Seq.index mv 0 %~ (squares_sum (fun i -> Seq.index hv i) /. 1024.0R))
  ensures exists* (values:lseq bf16 8192).
    normalized |-> values **
    pure (forall (i:natlt 8192).
      Seq.index values i %~ normalized_vector hv pv (i % 1024))
{
  let mean_value = mean.(0sz);
  scale_from_mean_approx mean_value (fun i -> Seq.index hv i);
  let scale = scale_from_mean mean_value;
  A.pts_to_len normalized;
  let mut index = 0sz;
  while (!index <^ 8192sz)
    invariant exists* (i:sz{i <= 8192}). index |-> i
    invariant exists* (values:lseq bf16 8192).
      normalized |-> values **
      pure (forall (j:nat{j < !index}).
        Seq.index values j %~ normalized_vector hv pv (j % 1024))
    decreases (8192 - !index)
  {
    let i = !index;
    if (i <^ 1024sz)
      requires exists* (values:lseq bf16 8192).
        normalized |-> values **
        pure (forall (j:nat{j < i}).
          Seq.index values j %~ normalized_vector hv pv (j % 1024))
      ensures exists* (values:lseq bf16 8192).
        normalized |-> values **
        pure (forall (j:nat{j <= i}).
          Seq.index values j %~ normalized_vector hv pv (j % 1024))
    {
      let x = hidden.(i);
      let w = packed.(SZ.uint32_to_sizet 254279680ul +^ i);
      let result = normalize_cell x w scale
        #(Root.rsqrt (real_argument (fun j -> Seq.index hv j)));
      normalized_formula hv pv i;
      normalized.(i) <- result;
    } else {
      let copied = normalized.(i %^ 1024sz);
      normalized.(i) <- copied;
    };
    index := i +^ 1sz;
  }
}

#pop-options

inline_for_extraction noextract
instance send_global_array_contents
  (#a:Type0)
  (arr:A.array a{is_global_array arr})
  (#fraction:perm)
  (contents:seq a)
  : is_send_across gpu_of (A.pts_to arr #fraction contents) =
  Kuiper.Array.Core.is_send_pts_to arr #fraction contents

inline_for_extraction noextract
fn normalize_host
  (packed:larray bf16 254280704{is_global_array packed})
  (hidden:larray bf16 1024{is_global_array hidden})
  (normalized:larray bf16 8192{is_global_array normalized})
  (#packed_fraction #hidden_fraction:perm)
  (#pv:erased (lseq bf16 254280704))
  (#hv:erased (lseq bf16 1024))
  (#nv:erased (seq bf16))
  preserves
    cpu **
    on gpu_loc (packed |-> Frac packed_fraction pv) **
    on gpu_loc (hidden |-> Frac hidden_fraction hv)
  requires on gpu_loc (normalized |-> nv)
  ensures exists* (values:lseq bf16 8192).
    on gpu_loc (normalized |-> values) **
    pure (forall (i:natlt 8192).
      Seq.index values i %~ normalized_vector hv pv (i % 1024))
{
  let squares = gpu_array_alloc #f32 1024sz;
  with sv0. assert on gpu_loc (squares |-> sv0);
  launch_kernel_1
    (fun () -> compute_squares hidden squares #hidden_fraction #hv #sv0);
  with sv. assert on gpu_loc (squares |-> sv);
  let mean = gpu_array_alloc #f32 1sz;
  with mv0. assert on gpu_loc (mean |-> mv0);
  launch_kernel_1
    (fun () -> compute_mean squares mean #hv #sv #mv0);
  with mv. assert on gpu_loc (mean |-> mv);
  launch_kernel_1
    (fun () -> normalize_and_replicate packed hidden mean normalized
      #packed_fraction #hidden_fraction #pv #hv #mv #nv);
  gpu_array_free squares;
  gpu_array_free mean;
}

module T = Kuiper.Tensor
module Ch = Kuiper.Chest
module Tile = Kuiper.Tensor.Tiling
module Layout = Kuiper.Tensor.Layout.Alg
module Stride = Kuiper.Array2.Strided
module Inj = Kuiper.Injection
module MMA = Kuiper.TensorCore.MMA
module MS = Kuiper.Spec.GEMM

inline_for_extraction noextract
let embedding_layout : T.layout2 248320 1024 = {
  ulen = 254280704;
  imap = Inj.mk_injection
    #(T.abs (T.ICons 248320 (T.ICons 1024 T.INil)))
    #(natlt 254280704)
    (fun (row, (column, ())) -> row * 1024 + column)
    (fun _ _ -> ());
}

inline_for_extraction noextract
instance embedding_concrete :
  T.ctlayout #2 #(T.ICons 248320 (T.ICons 1024 T.INil)) embedding_layout = {
  ulen_fits = ();
  all_fit = ();
  cimap = (fun (index:T.conc (T.ICons 248320 (T.ICons 1024 T.INil))) ->
    let row, (column, ()) = index in row *^ 1024sz +^ column);
}

inline_for_extraction noextract
instance embedding_row_major : Stride.strided_row_major embedding_layout = {
  offset = 0sz;
  stride = 1024sz;
  pf = (fun _ _ -> ());
}

inline_for_extraction noextract
instance size_16 : concrete_sz 16 = { x = 16sz; }

inline_for_extraction noextract
instance size_8 : concrete_sz 8 = { x = 8sz; }

inline_for_extraction noextract
instance size_1024 : concrete_sz 1024 = { x = 1024sz; }

inline_for_extraction noextract
instance accumulator_concrete :
  T.ctlayout (Layout.l2_row_major 248320 8) =
  Layout.c_l2_row_major 248320 8sz

inline_for_extraction noextract
instance normalized_concrete :
  T.ctlayout (Layout.l2_col_major 1024 8) =
  Layout.c_l2_col_major 1024sz 8

inline_for_extraction noextract
let embedding_tensor (packed:larray bf16 254280704)
  : T.array2 bf16 embedding_layout =
  T.from_array embedding_layout packed

inline_for_extraction noextract
let embedding_matrix (packed:lseq bf16 254280704)
  : Ch.chest2 bf16 248320 1024 =
  Ch.mk2 (fun row column -> Seq.index packed (row * 1024 + column))

inline_for_extraction noextract
let norm_suffix (packed:lseq bf16 254280704) : lseq bf16 1024 =
  Seq.slice packed 254279680 254280704

ghost
fn embedding_open
  (packed:larray bf16 254280704)
  (#fraction:perm)
  (#pv:erased (lseq bf16 254280704))
  requires packed |-> Frac fraction pv
  ensures
    T.tensor_pts_to (embedding_tensor packed) #fraction (embedding_matrix pv) **
    pts_to_slice packed #fraction 254279680 254280704 (norm_suffix pv) **
    is_full_slice packed 254280704
{
  array_to_slice packed;
  let prefix = Seq.slice pv 0 254279680;
  let suffix = norm_suffix pv;
  assert pure (Seq.equal pv (Seq.append prefix suffix));
  slice_split packed #fraction #prefix #suffix 0 254279680 254280704;
  slice_to_cells packed #fraction 0 254279680 #prefix;
  forevery_ext
    (fun (i:natlt 254279680) -> pts_to_cell packed #fraction (0 + i) (Seq.index prefix i))
    (fun (i:natlt 254279680) -> pts_to_cell packed #fraction i (Seq.index pv i));
  forevery_factor 254279680 248320 1024
    (fun i -> pts_to_cell packed #fraction i (Seq.index pv i));
  let tensor = embedding_tensor packed;
  assert rewrites_to tensor (embedding_tensor packed);
  forevery_map_2
    (fun (row:natlt 248320) (column:natlt 1024) ->
      pts_to_cell packed #fraction (row * 1024 + column)
        (Seq.index pv (row * 1024 + column)))
    (fun (row:natlt 248320) (column:natlt 1024) ->
      T.tensor_pts_to_cell tensor #fraction (T.idx2 row column)
        (Ch.acc2 (embedding_matrix pv) row column))
    fn row column {
      assert pure (T.core tensor == packed);
      assert pure (Ch.acc2 (embedding_matrix pv) row column ==
        Seq.index pv (row * 1024 + column));
      T.tensor_pts_to_cell_eq tensor (T.idx2 row column) fraction
        (Seq.index pv (row * 1024 + column));
      rewrite pts_to_cell packed #fraction (row * 1024 + column)
        (Seq.index pv (row * 1024 + column))
      as T.tensor_pts_to_cell tensor #fraction (T.idx2 row column)
        (Ch.acc2 (embedding_matrix pv) row column);
    };
  T.tensor_iraise2 tensor;
}

ghost
fn embedding_close
  (packed:larray bf16 254280704)
  (#fraction:perm)
  (#pv:erased (lseq bf16 254280704))
  requires
    T.tensor_pts_to (embedding_tensor packed) #fraction (embedding_matrix pv) **
    pts_to_slice packed #fraction 254279680 254280704 (norm_suffix pv) **
    is_full_slice packed 254280704
  ensures packed |-> Frac fraction pv
{
  let tensor = embedding_tensor packed;
  assert rewrites_to tensor (embedding_tensor packed);
  T.tensor_ilower2 tensor;
  forevery_map_2
    (fun (row:natlt 248320) (column:natlt 1024) ->
      T.tensor_pts_to_cell tensor #fraction (T.idx2 row column)
        (Ch.acc2 (embedding_matrix pv) row column))
    (fun (row:natlt 248320) (column:natlt 1024) ->
      pts_to_cell packed #fraction (row * 1024 + column)
        (Seq.index pv (row * 1024 + column)))
    fn row column {
      assert pure (T.core tensor == packed);
      assert pure (Ch.acc2 (embedding_matrix pv) row column ==
        Seq.index pv (row * 1024 + column));
      T.tensor_pts_to_cell_eq tensor (T.idx2 row column) fraction
        (Seq.index pv (row * 1024 + column));
      rewrite T.tensor_pts_to_cell tensor #fraction (T.idx2 row column)
        (Ch.acc2 (embedding_matrix pv) row column)
      as pts_to_cell packed #fraction (row * 1024 + column)
        (Seq.index pv (row * 1024 + column));
    };
  forevery_unfactor 254279680 248320 1024
    (fun i -> pts_to_cell packed #fraction i (Seq.index pv i));
  let prefix = Seq.slice pv 0 254279680;
  forevery_ext
    (fun (i:natlt 254279680) -> pts_to_cell packed #fraction i (Seq.index pv i))
    (fun (i:natlt 254279680) -> pts_to_cell packed #fraction (0 + i) (Seq.index prefix i));
  cells_to_nonempty_slice packed #fraction 0 254279680 #prefix;
  slice_concat packed #fraction 0 254279680 254280704;
  assert pure (Seq.equal pv (Seq.append prefix (norm_suffix pv)));
  slice_to_array packed;
}

inline_for_extraction noextract
let mma_zero : Ch.chest2 f32 16 8 =
  Ch.const (T.ICons 16 (T.ICons 8 T.INil)) (zero #f32)

[@@ "opaque_to_smt"]
inline_for_extraction noextract
let rec mma_prefix
  (a:Ch.chest2 bf16 248320 1024)
  (b:Ch.chest2 bf16 1024 8)
  (row:natlt 15520)
  (k:nat{k <= 64})
  : GTot (Ch.chest2 f32 16 8) (decreases k) =
  if k = 0 then mma_zero
  else MMA.emma (mma_prefix a b row (k - 1))
    (Tile.ematrix_subtile a 16 16 row (k - 1))
    (Tile.ematrix_subtile b 16 8 (k - 1) 0)

inline_for_extraction noextract
let mma_prefix_zero
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  (row:natlt 15520)
  : Lemma (mma_prefix a b row 0 == mma_zero) =
  reveal_opaque (`%mma_prefix) (mma_prefix a b row 0)

inline_for_extraction noextract
let mma_prefix_step
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  (row:natlt 15520) (k:natlt 64)
  : Lemma (
      mma_prefix a b row (k + 1) ==
      MMA.emma (mma_prefix a b row k)
        (Tile.ematrix_subtile a 16 16 row k)
        (Tile.ematrix_subtile b 16 8 k 0)) =
  reveal_opaque (`%mma_prefix) (mma_prefix a b row (k + 1))

inline_for_extraction noextract
let mma_result
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  : Ch.chest2 f32 248320 8 =
  Ch.mk2 (fun row column ->
    Ch.acc2 (mma_prefix a b (row / 16) 64) (row % 16) column)

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
fn project_row
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (#lc:T.layout2 16 8) {| Stride.strided_row_major lc |}
  (c:T.array2 f32 lc)
  (row:szlt 15520)
  (#fa #fb:perm)
  (#ea:Ch.chest2 bf16 248320 1024)
  (#eb:Ch.chest2 bf16 1024 8)
  (#ec:Ch.chest2 f32 16 8)
  preserves
    gpu **
    T.tensor_pts_to a #fa ea **
    T.tensor_pts_to b #fb eb
  requires T.tensor_pts_to c #(1.0R /. 32.0R) ec
  ensures T.tensor_pts_to c #(1.0R /. 32.0R) (mma_prefix ea eb row 64)
{
  let fragment = MMA.alloc_fragment ();
  MMA.fill fragment (zero #f32);
  mma_prefix_zero ea eb row;
  let mut index = 0sz;
  while (!index <^ 64sz)
    invariant exists* (k:sz{k <= 64}). index |-> k
    invariant MMA.fragment_pts_to fragment (mma_prefix ea eb row !index)
    decreases (64 - !index)
  {
    let k = !index;
    let atile = Tile.array2_extract_tile_ro' a 16 16 (SZ.v row) (SZ.v k);
    let btile = Tile.array2_extract_tile_ro' b 16 8 (SZ.v k) 0;
    MMA.mma_sync atile btile fragment;
    mma_prefix_step ea eb row k;
    ambig_trade_elim ();
    ambig_trade_elim ();
    index := k +^ 1sz;
  };
  MMA.store fragment c;
  drop_ (MMA.fragment_pts_to fragment (mma_prefix ea eb row 64));
}

inline_for_extraction noextract
let updated_mma_prefix
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  (values:Ch.chest2 f32 248320 8) (tile:natlt 15520)
  : Lemma
      (requires forall (row:nat{row < 16 * tile}) (column:natlt 8).
        Ch.acc2 values row column == Ch.acc2 (mma_result a b) row column)
      (ensures forall (row:nat{row < 16 * (tile + 1)}) (column:natlt 8).
        Ch.acc2
          (Tile.update_tile values 16 8 tile 0 (mma_prefix a b tile 64))
          row column ==
        Ch.acc2 (mma_result a b) row column) =
  ()

inline_for_extraction noextract
fn project_all
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (#fa #fb:perm)
  (#ea:Ch.chest2 bf16 248320 1024)
  (#eb:Ch.chest2 bf16 1024 8)
  (#ec:Ch.chest2 f32 248320 8)
  preserves
    gpu ** T.tensor_pts_to a #fa ea ** T.tensor_pts_to b #fb eb
  requires T.tensor_pts_to c #(1.0R /. 32.0R) ec
  ensures T.tensor_pts_to c #(1.0R /. 32.0R) (mma_result ea eb)
{
  let mut row = 0sz;
  while (!row <^ 15520sz)
    invariant exists* (i:sz{i <= 15520}). row |-> i
    invariant exists* (values:Ch.chest2 f32 248320 8).
      T.tensor_pts_to c #(1.0R /. 32.0R) values **
      pure (forall (r:nat{r < 16 * !row}) (column:natlt 8).
        Ch.acc2 values r column == Ch.acc2 (mma_result ea eb) r column)
    decreases (15520 - !row)
  {
    let r = !row;
    with old. assert T.tensor_pts_to c #(1.0R /. 32.0R) old;
    Tile.array2_extract_tile c 16 8 (SZ.v r) 0;
    let ctile = Tile.array2_subtile c 16 8 (SZ.v r) 0;
    assert rewrites_to ctile (Tile.array2_subtile c 16 8 (SZ.v r) 0);
    project_row a b ctile r;
    Pulse.Lib.Forall.elim_forall (mma_prefix ea eb r 64);
    ambig_trade_elim ();
    updated_mma_prefix ea eb old r;
    row := r +^ 1sz;
  };
  with values. assert T.tensor_pts_to c #(1.0R /. 32.0R) values;
  assert pure (Ch.equal values (mma_result ea eb));
  Ch.ext values (mma_result ea eb);
}

#pop-options

inline_for_extraction noextract
let projection_pre
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (fa fb:perm)
  (ea:Ch.chest2 bf16 248320 1024)
  (eb:Ch.chest2 bf16 1024 8)
  (ec:Ch.chest2 f32 248320 8)
  : slprop =
  T.tensor_pts_to a #fa ea **
  T.tensor_pts_to b #fb eb **
  T.tensor_pts_to c ec

inline_for_extraction noextract
let projection_lane_pre
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (fa fb:perm)
  (ea:Ch.chest2 bf16 248320 1024)
  (eb:Ch.chest2 bf16 1024 8)
  (ec:Ch.chest2 f32 248320 8)
  (_:natlt 32)
  : slprop =
  T.tensor_pts_to a #(fa /. 32.0R) ea **
  T.tensor_pts_to b #(fb /. 32.0R) eb **
  T.tensor_pts_to c #(1.0R /. 32.0R) ec

ghost
fn projection_setup
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (fa fb:perm)
  (ea:Ch.chest2 bf16 248320 1024)
  (eb:Ch.chest2 bf16 1024 8)
  (ec:Ch.chest2 f32 248320 8)
  ()
  norewrite
  requires projection_pre a b c fa fb ea eb ec
  ensures
    (forall+ (tid:natlt 32). projection_lane_pre a b c fa fb ea eb ec tid) ** emp
{
  unfold projection_pre a b c fa fb ea eb ec;
  T.tensor_share_n a 32 #fa #ea;
  T.tensor_share_n b 32 #fb #eb;
  T.tensor_share_n c 32 #1.0R #ec;
  forevery_zip3
    (fun (_:natlt 32) -> T.tensor_pts_to a #(fa /. 32.0R) ea)
    (fun (_:natlt 32) -> T.tensor_pts_to b #(fb /. 32.0R) eb)
    (fun (_:natlt 32) -> T.tensor_pts_to c #(1.0R /. 32.0R) ec);
  forevery_ext
    (fun (_:natlt 32) ->
      T.tensor_pts_to a #(fa /. 32.0R) ea **
      T.tensor_pts_to b #(fb /. 32.0R) eb **
      T.tensor_pts_to c #(1.0R /. 32.0R) ec)
    (projection_lane_pre a b c fa fb ea eb ec);
}

ghost
fn projection_teardown
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (fa fb:perm)
  (ea:Ch.chest2 bf16 248320 1024)
  (eb:Ch.chest2 bf16 1024 8)
  (ec:Ch.chest2 f32 248320 8)
  ()
  norewrite
  requires
    (forall+ (tid:natlt 32). projection_lane_pre a b c fa fb ea eb ec tid) ** emp
  ensures projection_pre a b c fa fb ea eb ec
{
  forevery_ext (projection_lane_pre a b c fa fb ea eb ec)
    (fun (_:natlt 32) ->
      T.tensor_pts_to a #(fa /. 32.0R) ea **
      T.tensor_pts_to b #(fb /. 32.0R) eb **
      T.tensor_pts_to c #(1.0R /. 32.0R) ec);
  forevery_unzip3
    (fun (_:natlt 32) -> T.tensor_pts_to a #(fa /. 32.0R) ea)
    (fun (_:natlt 32) -> T.tensor_pts_to b #(fb /. 32.0R) eb)
    (fun (_:natlt 32) -> T.tensor_pts_to c #(1.0R /. 32.0R) ec);
  T.tensor_gather_n a 32 #fa #ea;
  T.tensor_gather_n b 32 #fb #eb;
  T.tensor_gather_n c 32 #1.0R #ec;
  fold projection_pre a b c fa fb ea eb ec;
}

(* Every one of the 32 lanes follows the same row/K loops. The normalized
   input is published by a completed launch before collective MMA execution. *)
inline_for_extraction noextract
fn project_lane
  (a:T.array2 bf16 embedding_layout)
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8))
  (c:T.array2 f32 (Layout.l2_row_major 248320 8))
  (#fa #fb:perm)
  (#ea:Ch.chest2 bf16 248320 1024)
  (#eb:Ch.chest2 bf16 1024 8)
  (#ec:Ch.chest2 f32 248320 8)
  (tid:szlt 32)
  ()
  norewrite
  requires gpu ** projection_lane_pre a b c fa fb ea eb ec tid ** thread_id 32 tid
  ensures
    gpu ** projection_lane_pre a b c fa fb ea eb (mma_result ea eb) tid **
    thread_id 32 tid
{
  unfold projection_lane_pre a b c fa fb ea eb ec tid;
  project_all a b c #(fa /. 32.0R) #(fb /. 32.0R) #ea #eb #ec;
  fold projection_lane_pre a b c fa fb ea eb (mma_result ea eb) tid;
}

inline_for_extraction noextract
let projection_kernel
  (a:T.array2 bf16 embedding_layout{T.is_global a})
  (b:T.array2 bf16 (Layout.l2_col_major 1024 8){T.is_global b})
  (c:T.array2 f32 (Layout.l2_row_major 248320 8){T.is_global c})
  (#fa #fb:perm)
  (#ea:Ch.chest2 bf16 248320 1024)
  (#eb:Ch.chest2 bf16 1024 8)
  (#ec:Ch.chest2 f32 248320 8)
  : kernel_desc
      (projection_pre a b c fa fb ea eb ec)
      (projection_pre a b c fa fb ea eb (mma_result ea eb)) =
  {
    nthr = 32sz;
    frame = emp;
    kpre = projection_lane_pre a b c fa fb ea eb ec;
    kpost = projection_lane_pre a b c fa fb ea eb (mma_result ea eb);
    block_setup = projection_setup a b c fa fb ea eb ec;
    block_teardown = projection_teardown a b c fa fb ea eb (mma_result ea eb);
    f = project_lane a b c #fa #fb #ea #eb #ec;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve;
  } <: kernel_desc_1_n _ _

inline_for_extraction noextract
let real_zero : Ch.chest2 real 16 8 =
  Ch.const (T.ICons 16 (T.ICons 8 T.INil)) 0.0R

inline_for_extraction noextract
let mma_zero_approximates () : Lemma (mma_zero %~ real_zero) =
  ()

[@@ "opaque_to_smt"]
inline_for_extraction noextract
let real_mma_prefix
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 15520) (k:nat{k <= 64})
  : GTot (Ch.chest2 real 16 8) =
  MS.__gmatmul_single real_zero (MS.matmul #real) (MS.matplus #real)
    (Tile.ematrix_tiled a 16 16) (Tile.ematrix_tiled b 16 8) row 0 k

inline_for_extraction noextract
let real_mma_prefix_expand
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 15520) (k:nat{k <= 64})
  : Lemma (
      real_mma_prefix a b row k ==
      MS.__gmatmul_single real_zero (MS.matmul #real) (MS.matplus #real)
        (Tile.ematrix_tiled a 16 16) (Tile.ematrix_tiled b 16 8) row 0 k) =
  reveal_opaque (`%real_mma_prefix) (real_mma_prefix a b row k)

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let real_mma_prefix_zero
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 15520)
  : Lemma (real_mma_prefix a b row 0 == real_zero) =
  real_mma_prefix_expand a b row 0;
  MS.__gmatmul_single_zero_lemma real_zero (MS.matmul #real) (MS.matplus #real)
    (Tile.ematrix_tiled a 16 16) (Tile.ematrix_tiled b 16 8) row 0

inline_for_extraction noextract
let real_mma_prefix_step
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 15520) (k:natlt 64)
  : Lemma (
      real_mma_prefix a b row (k + 1) ==
      MS.matplus (real_mma_prefix a b row k)
        (MS.matmul (Tile.ematrix_subtile a 16 16 row k)
          (Tile.ematrix_subtile b 16 8 k 0))) =
  real_mma_prefix_expand a b row (k + 1);
  real_mma_prefix_expand a b row k;
  MS.__gmatmul_single_lemma real_zero (MS.matmul #real) (MS.matplus #real)
    (Tile.ematrix_tiled a 16 16) (Tile.ematrix_tiled b 16 8) row 0 (k + 1);
  Tile.macc_ematrix_tiled a 16 16 row k;
  Tile.macc_ematrix_tiled b 16 8 k 0

inline_for_extraction noextract
let real_mma_prefix_complete
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 15520)
  : Lemma (
      real_mma_prefix a b row 64 ==
      Tile.ematrix_subtile (MS.matmul a b) 16 8 row 0) =
  real_mma_prefix_expand a b row 64;
  MS.matmul_tiles_lemma #real (fun _ -> ()) (fun _ _ _ -> ())
    16 8 16 real_zero a b row 0;
  MS.matmul_decompose_lemma a b 16 8 row 0;
  let result = Tile.ematrix_subtile (MS.matmul a b) 16 8 row 0 in
  assert (Ch.equal (MS.matplus real_zero result) result);
  Ch.ext (MS.matplus real_zero result) result

inline_for_extraction noextract
let subtile_approximates
  (#rows #columns:pos)
  (a:Ch.chest2 bf16 rows columns)
  (r:Ch.chest2 real rows columns)
  (height:pos{height /? rows}) (width:pos{width /? columns})
  (row:natlt (rows / height)) (column:natlt (columns / width))
  : Lemma
      (requires a %~ r)
      (ensures
        Tile.ematrix_subtile a height width row column %~
        Tile.ematrix_subtile r height width row column) =
  ()

inline_for_extraction noextract
let rec mma_prefix_approximates
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  (ra:Ch.chest2 real 248320 1024) (rb:Ch.chest2 real 1024 8)
  (row:natlt 15520) (k:nat{k <= 64})
  : Lemma
      (requires a %~ ra /\ b %~ rb)
      (ensures mma_prefix a b row k %~ real_mma_prefix ra rb row k)
      (decreases k) =
  if k = 0 then (
    mma_prefix_zero a b row;
    real_mma_prefix_zero ra rb row;
    mma_zero_approximates ()
  ) else (
    mma_prefix_approximates a b ra rb row (k - 1);
    subtile_approximates a ra 16 16 row (k - 1);
    subtile_approximates b rb 16 8 (k - 1) 0;
    MMA.emma_approx_lemma
      (mma_prefix a b row (k - 1))
      (Tile.ematrix_subtile a 16 16 row (k - 1))
      (Tile.ematrix_subtile b 16 8 (k - 1) 0)
      (real_mma_prefix ra rb row (k - 1))
      (Tile.ematrix_subtile ra 16 16 row (k - 1))
      (Tile.ematrix_subtile rb 16 8 (k - 1) 0);
    mma_prefix_step a b row (k - 1);
    real_mma_prefix_step ra rb row (k - 1)
  )

inline_for_extraction noextract
let mma_result_approximates
  (a:Ch.chest2 bf16 248320 1024) (b:Ch.chest2 bf16 1024 8)
  (ra:Ch.chest2 real 248320 1024) (rb:Ch.chest2 real 1024 8)
  : Lemma
      (requires a %~ ra /\ b %~ rb)
      (ensures mma_result a b %~ MS.matmul ra rb) =
  let cell (index:T.abs (T.ICons 248320 (T.ICons 8 T.INil)))
    : Lemma (
        Ch.acc (mma_result a b) index %~ Ch.acc (MS.matmul ra rb) index) =
    let row, (column, ()) = index in
    mma_prefix_approximates a b ra rb (row / 16) 64;
    real_mma_prefix_complete ra rb (row / 16);
    Tile.subtile_acc2 (MS.matmul ra rb) 16 8 (row / 16) 0 (row % 16) column
  in
  FStar.Classical.forall_intro cell

#pop-options

inline_for_extraction noextract
let real_embedding (packed:lseq bf16 254280704) : Ch.chest2 real 248320 1024 =
  Ch.to_real_chest (embedding_matrix packed)

inline_for_extraction noextract
let real_normalized
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704)
  : Ch.chest2 real 1024 8 =
  Ch.mk2 (fun row _ -> normalized_vector hidden packed row)

inline_for_extraction noextract
let normalized_matrix (values:lseq bf16 8192) : Ch.chest2 bf16 1024 8 =
  T.from_seq (Layout.l2_col_major 1024 8) values

inline_for_extraction noextract
let normalized_matrix_to_seq (values:lseq bf16 8192)
  : Lemma (
      T.to_seq (Layout.l2_col_major 1024 8) (normalized_matrix values) == values) =
  T.to_from (Layout.l2_col_major 1024 8) values

inline_for_extraction noextract
let normalized_matrix_approximates
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704)
  (values:lseq bf16 8192)
  : Lemma
      (requires forall (i:natlt 8192).
        Seq.index values i %~ normalized_vector hidden packed (i % 1024))
      (ensures normalized_matrix values %~ real_normalized hidden packed) =
  ()

inline_for_extraction noextract
let real_products
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 248320) (column:natlt 8)
  : natlt 1024 -> GTot real =
  fun i -> Ch.acc2 a row i *. Ch.acc2 b i column

inline_for_extraction noextract
let real_scalar_ops (x y:real)
  : Lemma (add x y == x +. y /\ mul x y == x *. y /\ zero #real == 0.0R) =
  ()

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let rec real_matmul_sum
  (a:Ch.chest2 real 248320 1024) (b:Ch.chest2 real 1024 8)
  (row:natlt 248320) (column:natlt 8) (n:nat{n <= 1024})
  : Lemma (
      MS.__matmul_single a b row column n ==
      Sum.sum #real #Sum.real_add_semigroup 0 n (real_products a b row column))
      (decreases n) =
  if n = 0 then (
    MS.matmul_zero_lemma a b row column;
    real_scalar_ops 0.0R 0.0R
  ) else (
    real_matmul_sum a b row column (n - 1);
    MS.matmul_single_lemma a b row column n;
    Sum.sum_pop_right #real #Sum.real_add_semigroup 0 n (real_products a b row column);
    real_scalar_ops (Ch.acc2 a row (n - 1)) (Ch.acc2 b (n - 1) column);
    real_scalar_ops
      (MS.__matmul_single a b row column (n - 1))
      (mul (Ch.acc2 a row (n - 1)) (Ch.acc2 b (n - 1) column));
    real_add_op
      (Sum.sum #real #Sum.real_add_semigroup 0 (n - 1) (real_products a b row column))
      (real_products a b row column (n - 1))
  )

#pop-options

inline_for_extraction noextract
let shared_scores (hidden:lseq bf16 1024) (packed:lseq bf16 254280704)
  : GTot (Shared.vector real 248320) =
  Shared.logical_logits #1024 #248320 Shared.reference_epsilon
    (fun i -> Seq.index hidden i)
    (fun i -> Seq.index packed (254279680 + i))
    (fun row i -> Seq.index packed (row * 1024 + i))

inline_for_extraction noextract
let shared_score_sum
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704) (token:natlt 248320)
  : Lemma (
      shared_scores hidden packed token ==
      Sum.sum #real #Sum.real_add_semigroup 0 1024
        (fun (i:natlt 1024) ->
          normalized_vector hidden packed i *.
          Approx.to_real (Seq.index packed (token * 1024 + i)))) =
  assert (
    shared_scores hidden packed token ==
    Sum.sum #real #Sum.real_add_semigroup 0 1024
      (fun (i:natlt 1024) ->
        normalized_vector hidden packed i *.
        Approx.to_real (Seq.index packed (token * 1024 + i)))) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.shared_scores";
        "KPlay.Functional.normalized_vector";
        "KPlay.SharedSpec.logical_logits";
        "KPlay.SharedSpec.project";
        "KPlay.SharedSpec.real_sum";
        "KPlay.SharedSpec.real_vector";
        "KPlay.SharedSpec.real_matrix"]];
    FStar.Tactics.V2.trefl ())

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
let real_matmul_shared_score
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704) (token:natlt 248320)
  : Lemma (
      Ch.acc2 (MS.matmul (real_embedding packed) (real_normalized hidden packed)) token 0 ==
      shared_scores hidden packed token) =
  let a = real_embedding packed in
  let b = real_normalized hidden packed in
  MS.lemma_matmul_index a b token 0;
  real_matmul_sum a b token 0 1024;
  sum_ext #1024 0 1024 (real_products a b token 0)
    (fun (i:natlt 1024) ->
      normalized_vector hidden packed i *.
      Approx.to_real (Seq.index packed (token * 1024 + i)));
  shared_score_sum hidden packed token

inline_for_extraction noextract
let mma_shared_scores
  (hidden:lseq bf16 1024) (packed:lseq bf16 254280704)
  (values:lseq bf16 8192)
  : Lemma
      (requires forall (i:natlt 8192).
        Seq.index values i %~ normalized_vector hidden packed (i % 1024))
      (ensures forall (token:natlt 248320).
        Ch.acc2 (mma_result (embedding_matrix packed) (normalized_matrix values)) token 0 %~
        shared_scores hidden packed token) =
  Ch.lemma_to_real_chest_approximates (embedding_matrix packed);
  normalized_matrix_approximates hidden packed values;
  mma_result_approximates
    (embedding_matrix packed) (normalized_matrix values)
    (real_embedding packed) (real_normalized hidden packed);
  let cell (token:natlt 248320)
    : Lemma (
        Ch.acc2 (mma_result (embedding_matrix packed) (normalized_matrix values)) token 0 %~
        shared_scores hidden packed token) =
    real_matmul_shared_score hidden packed token
  in
  FStar.Classical.forall_intro cell

#pop-options

inline_for_extraction noextract
let bf16_score_value (value:f32) : bf16 =
  C.cast_f32_to_bf16 (add (zero #f32) value)

inline_for_extraction noextract
let bf16_score_approx (value:f32) (r:real)
  : Lemma
      (requires value %~ r)
      (ensures bf16_score_value value %~ r) =
  let _ = Approx.a0 #f32 in
  Approx.a_add (zero #f32) value 0.0R r;
  C.cast_f32_to_bf16_ok (add (zero #f32) value) r

#push-options "--fuel 2 --ifuel 0 --z3rlimit 100"

inline_for_extraction noextract
fn store_scores
  (accumulator:T.array2 f32 (Layout.l2_row_major 248320 8))
  (scores:larray bf16 248320)
  (#av:Ch.chest2 f32 248320 8)
  (#hv:erased (lseq bf16 1024))
  (#pv:erased (lseq bf16 254280704))
  (#sv:erased (lseq bf16 248320))
  preserves gpu ** T.tensor_pts_to accumulator av
  requires scores |-> sv **
    pure (forall (token:natlt 248320).
      Ch.acc2 av token 0 %~ shared_scores hv pv token)
  ensures exists* (values:lseq bf16 248320).
    scores |-> values **
    pure (forall (token:natlt 248320).
      Seq.index values token %~ shared_scores hv pv token)
{
  let count = SZ.uint32_to_sizet 248320ul;
  let mut index = 0sz;
  while (!index <^ count)
    invariant exists* (i:sz{i <= 248320}). index |-> i
    invariant exists* (values:lseq bf16 248320).
      scores |-> values **
      pure (forall (token:nat{token < !index}).
        Seq.index values token %~ shared_scores hv pv token)
    decreases (248320 - !index)
  {
    let i = !index;
    let value = T.tensor_read accumulator (T.cidx2 i 0sz);
    bf16_score_approx value (shared_scores hv pv i);
    scores.(i) <- bf16_score_value value;
    index := i +^ 1sz;
  }
}

#pop-options

inline_for_extraction noextract
let device_buffer_owns
  (#size:nat) (ptr:bf16_buffer size) (fraction:perm) (values:lseq bf16 size)
  : slprop =
  if size = 0 then emp else A.pts_to ptr #fraction values

inline_for_extraction noextract
let device_buffer_empty
  (ptr:bf16_buffer 0) (fraction:perm) (values:lseq bf16 0)
  : Lemma (device_buffer_owns ptr fraction values == emp) =
  ()

inline_for_extraction noextract
let device_buffer_nonempty
  (size:pos) (ptr:bf16_buffer size) (fraction:perm) (values:lseq bf16 size)
  : Lemma (device_buffer_owns ptr fraction values == A.pts_to ptr #fraction values) =
  ()

inline_for_extraction noextract
instance send_device_buffer
  (#size:nat) (ptr:bf16_buffer size) (fraction:perm) (values:lseq bf16 size)
  : is_send_across gpu_of (device_buffer_owns ptr fraction values) =
  if size = 0 then (
    device_buffer_empty ptr fraction values;
    let result : is_send_across gpu_of emp = solve in result
  ) else (
    device_buffer_nonempty size ptr fraction values;
    send_global_array_contents ptr #fraction values
  )

ghost
fn locate_buffer
  (#size:nat) (ptr:bf16_buffer size) (fraction:perm) (values:lseq bf16 size)
  requires buffer_owns ptr fraction values
  ensures on gpu_loc (device_buffer_owns ptr fraction values)
{
  if (size = 0) {
    rewrite buffer_owns ptr fraction values as emp;
    ghost_impersonate gpu_loc emp (on gpu_loc emp)
      fn () { on_intro emp; };
    rewrite on gpu_loc emp as on gpu_loc (device_buffer_owns ptr fraction values);
  } else {
    rewrite buffer_owns ptr fraction values
      as on gpu_loc (device_buffer_owns ptr fraction values);
  }
}

ghost
fn unlocate_buffer
  (#size:nat) (ptr:bf16_buffer size) (fraction:perm) (values:lseq bf16 size)
  requires on gpu_loc (device_buffer_owns ptr fraction values)
  ensures buffer_owns ptr fraction values
{
  if (size = 0) {
    drop_ (on gpu_loc (device_buffer_owns ptr fraction values));
    rewrite emp as buffer_owns ptr fraction values;
  } else {
    rewrite on gpu_loc (device_buffer_owns ptr fraction values)
      as buffer_owns ptr fraction values;
  }
}

inline_for_extraction noextract
fn emit_history
  (count:sz{count <= 2096128})
  (previous:bf16_buffer count)
  (hidden:larray bf16 1024)
  (history:larray bf16 (count + 1024))
  (#previous_fraction #hidden_fraction:perm)
  (#previous_values:erased (lseq bf16 count))
  (#hidden_values:erased (lseq bf16 1024))
  (#initial_history:erased (lseq bf16 (count + 1024)))
  preserves
    gpu **
    device_buffer_owns previous previous_fraction previous_values **
    hidden |-> Frac hidden_fraction hidden_values
  requires history |-> initial_history
  ensures history |-> Seq.append previous_values hidden_values
{
  if (0sz <^ count)
    requires
      device_buffer_owns previous previous_fraction previous_values **
      history |-> initial_history
    ensures
      device_buffer_owns previous previous_fraction previous_values **
      (exists* (values:lseq bf16 (count + 1024)).
        history |-> values **
        pure (forall (i:nat{i < count}).
          Seq.index values i == Seq.index previous_values i))
  {
    device_buffer_nonempty count previous previous_fraction previous_values;
    rewrite device_buffer_owns previous previous_fraction previous_values
      as A.pts_to previous #previous_fraction previous_values;
    let mut index = 0sz;
    while (!index <^ count)
      invariant exists* (i:sz{i <= count}). index |-> i
      invariant exists* (values:lseq bf16 (count + 1024)).
        history |-> values **
        pure (forall (j:nat{j < !index}).
          Seq.index values j == Seq.index previous_values j)
      decreases (count - !index)
    {
      let i = !index;
      let value = previous.(i);
      history.(i) <- value;
      index := i +^ 1sz;
    };
    rewrite A.pts_to previous #previous_fraction previous_values
      as device_buffer_owns previous previous_fraction previous_values;
  } else {
  };
  let mut index = 0sz;
  while (!index <^ 1024sz)
    invariant exists* (i:sz{i <= 1024}). index |-> i
    invariant exists* (values:lseq bf16 (count + 1024)).
      history |-> values **
      pure (forall (j:nat{j < count + !index}).
        Seq.index values j == Seq.index (Seq.append previous_values hidden_values) j)
    decreases (1024 - !index)
  {
    let i = !index;
    let value = hidden.(i);
    history.(count +^ i) <- value;
    index := i +^ 1sz;
  };
  with values. assert history |-> values;
  assert pure (Seq.equal values (Seq.append previous_values hidden_values));
}

module I64 = FStar.Int64
module IC = FStar.Int.Cast

inline_for_extraction noextract
let history_count
  (position:position_t)
  (prompt_length:prompt_length_t)
  (_:squash (emits_observation position prompt_length))
  : result:sz{
      SZ.v result == previous_elements position prompt_length /\
      SZ.v result <= 2096128} =
  extent_refinement position prompt_length;
  let rows = I64.add (I64.sub position prompt_length) 1L in
  let count = I64.mul rows 1024L in
  SZ.uint32_to_sizet (IC.int64_to_uint32 count)

inline_for_extraction noextract
let shared_scores_mapping
  (hidden:lseq bf16 hidden_size) (packed:lseq bf16 packed_elements)
  (token:Shared.index vocab_size)
  : Lemma (
      shared_scores hidden packed token ==
      Shared.logical_logits Shared.reference_epsilon
        (vector_view hidden) (norm_weight_view packed) (embedding_view packed) token) =
  assert (
    shared_scores hidden packed token ==
    Shared.logical_logits Shared.reference_epsilon
      (vector_view hidden) (norm_weight_view packed) (embedding_view packed) token) by (
    FStar.Tactics.V2.norm [
      zeta; iota; delta_only [
        "KPlay.Functional.shared_scores";
        "KPlay.Functional.vector_view";
        "KPlay.Functional.norm_weight_view";
        "KPlay.Functional.embedding_view";
        "KPlay.Functional.embedding_elements";
        "KPlay.Functional.hidden_size";
        "KPlay.Functional.vocab_size";
        "KPlay.SharedSpec.reference_hidden_size";
        "KPlay.SharedSpec.reference_vocab_size"]];
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
let all_shared_scores_mapping
  (hidden:lseq bf16 hidden_size) (packed:lseq bf16 packed_elements)
  (scores:lseq bf16 vocab_size)
  : Lemma
      (requires forall (token:Shared.index vocab_size).
        Seq.index scores token %~ shared_scores hidden packed token)
      (ensures forall (token:Shared.index vocab_size).
        Seq.index scores token %~
          Shared.logical_logits Shared.reference_epsilon
            (vector_view hidden) (norm_weight_view packed) (embedding_view packed) token) =
  let cell (token:Shared.index vocab_size)
    : Lemma (
        Seq.index scores token %~
          Shared.logical_logits Shared.reference_epsilon
            (vector_view hidden) (norm_weight_view packed) (embedding_view packed) token) =
    shared_scores_mapping hidden packed token
  in
  FStar.Classical.forall_intro cell

inline_for_extraction noextract
fn emit_history_host
  (position:position_t)
  (prompt_length:prompt_length_t)
  (previous:bf16_buffer (previous_elements position prompt_length))
  (hidden:bf16_buffer hidden_size)
  (history:bf16_buffer (emitted_elements position prompt_length))
  (#previous_fraction #hidden_fraction:perm)
  (#previous_values:erased (lseq bf16 (previous_elements position prompt_length)))
  (#hidden_values:erased (lseq bf16 hidden_size))
  (#initial_history:erased (lseq bf16 (emitted_elements position prompt_length)))
  preserves
    cpu **
    buffer_owns previous previous_fraction previous_values **
    on gpu_loc (A.pts_to hidden #hidden_fraction hidden_values)
  requires buffer_owns history 1.0R initial_history
  ensures buffer_owns history 1.0R
    (emitted_history position prompt_length previous_values hidden_values)
{
  extent_refinement position prompt_length;
  if (I64.gte position (I64.sub prompt_length 1L)) {
    let count = history_count position prompt_length ();
    (* Repackage erased sequences at proved-equal extents; no runtime mapping. *)
    let counted_previous : erased (lseq bf16 (SZ.v count)) =
      hide (reveal previous_values);
    let counted_initial : erased (lseq bf16 (SZ.v count + 1024)) =
      hide (reveal initial_history);
    rewrite buffer_owns history 1.0R initial_history
      as on gpu_loc (A.pts_to history counted_initial);
    locate_buffer previous previous_fraction previous_values;
    rewrite on gpu_loc
      (device_buffer_owns #(previous_elements position prompt_length)
        previous previous_fraction previous_values)
      as on gpu_loc
        (device_buffer_owns #(SZ.v count) previous previous_fraction counted_previous);
    launch_kernel_1 (fun () -> emit_history count previous hidden history
      #previous_fraction #hidden_fraction #counted_previous #hidden_values #counted_initial);
    rewrite on gpu_loc
      (device_buffer_owns #(SZ.v count) previous previous_fraction counted_previous)
      as on gpu_loc
        (device_buffer_owns #(previous_elements position prompt_length)
          previous previous_fraction previous_values);
    unlocate_buffer previous previous_fraction previous_values;
    rewrite on gpu_loc (A.pts_to history (Seq.append counted_previous hidden_values))
      as buffer_owns history 1.0R
        (emitted_history position prompt_length previous_values hidden_values);
  } else {
    rewrite buffer_owns history 1.0R initial_history
      as buffer_owns history 1.0R
        (emitted_history position prompt_length previous_values hidden_values);
  }
}

fn kplay_run
  (#state:Type0)
  (#state_ownership:state -> slprop)
  (#before:erased state)
  (#packed_fraction #hidden_fraction #previous_fraction:perm)
  (position:position_t)
  (prompt_length:prompt_length_t)
  (#packed_values:erased (lseq bf16 packed_elements))
  (#hidden_values:erased (lseq bf16 hidden_size))
  (#previous_values:erased (lseq bf16 (previous_elements position prompt_length)))
  (packed:bf16_buffer packed_elements)
  (hidden:bf16_buffer hidden_size)
  (previous:bf16_buffer (previous_elements position prompt_length))
  (scores:bf16_buffer vocab_size)
  (history:bf16_buffer (emitted_elements position prompt_length))
  preserves
    cpu **
    buffer_owns packed packed_fraction packed_values **
    buffer_owns hidden hidden_fraction hidden_values **
    buffer_owns previous previous_fraction previous_values
  requires
    state_ownership before **
    (exists* (initial_scores:lseq bf16 vocab_size).
      buffer_owns scores 1.0R initial_scores) **
    (exists* (initial_history:lseq bf16 (emitted_elements position prompt_length)).
      buffer_owns history 1.0R initial_history)
  ensures
    exists* (score_values:lseq bf16 vocab_size)
      (history_values:lseq bf16 (emitted_elements position prompt_length))
      (after:state).
      buffer_owns scores 1.0R score_values **
      buffer_owns history 1.0R history_values **
      state_ownership after **
      pure (semantic_postcondition position prompt_length
        packed_values hidden_values previous_values before after score_values history_values)
{
  with initial_scores. assert buffer_owns scores 1.0R initial_scores;
  with initial_history. assert buffer_owns history 1.0R initial_history;
  rewrite buffer_owns packed packed_fraction packed_values
    as on gpu_loc (A.pts_to packed #packed_fraction packed_values);
  rewrite buffer_owns hidden hidden_fraction hidden_values
    as on gpu_loc (A.pts_to hidden #hidden_fraction hidden_values);
  rewrite buffer_owns scores 1.0R initial_scores
    as on gpu_loc (A.pts_to scores initial_scores);
  let normalized = gpu_array_alloc #bf16 8192sz;
  with nv0. assert on gpu_loc (normalized |-> nv0);
  normalize_host packed hidden normalized
    #packed_fraction #hidden_fraction #packed_values #hidden_values #nv0;
  with nv. assert on gpu_loc (normalized |-> nv);

  let a = embedding_tensor packed;
  assert rewrites_to a (embedding_tensor packed);
  let b = T.from_array (Layout.l2_col_major 1024 8) normalized;
  assert rewrites_to b (T.from_array (Layout.l2_col_major 1024 8) normalized);
  let c = T.alloc0 #f32 (SZ.uint32_to_sizet 1986560ul) (Layout.l2_row_major 248320 8);
  with cv. assert on gpu_loc (T.tensor_pts_to c cv);
  map_loc gpu_loc
    #(A.pts_to packed #packed_fraction packed_values)
    #(T.tensor_pts_to a #packed_fraction (embedding_matrix packed_values) **
      pts_to_slice packed #packed_fraction 254279680 254280704 (norm_suffix packed_values) **
      is_full_slice packed 254280704)
    fn () { embedding_open packed #packed_fraction #packed_values; };
  map_loc gpu_loc
    #(normalized |-> nv)
    #(T.tensor_pts_to b (normalized_matrix nv))
    fn () { T.tensor_abs' (Layout.l2_col_major 1024 8) normalized; };
  map_loc gpu_loc
    #(T.tensor_pts_to a #packed_fraction (embedding_matrix packed_values) **
      T.tensor_pts_to b (normalized_matrix nv) ** T.tensor_pts_to c cv)
    #(projection_pre a b c packed_fraction 1.0R
      (embedding_matrix packed_values) (normalized_matrix nv) cv)
    fn () { fold projection_pre a b c packed_fraction 1.0R
      (embedding_matrix packed_values) (normalized_matrix nv) cv; };
  launch_sync (projection_kernel a b c #packed_fraction #1.0R
    #(embedding_matrix packed_values) #(normalized_matrix nv) #cv);
  map_loc gpu_loc
    #(projection_pre a b c packed_fraction 1.0R
      (embedding_matrix packed_values) (normalized_matrix nv)
      (mma_result (embedding_matrix packed_values) (normalized_matrix nv)))
    #(T.tensor_pts_to a #packed_fraction (embedding_matrix packed_values) **
      T.tensor_pts_to b (normalized_matrix nv) **
      T.tensor_pts_to c (mma_result (embedding_matrix packed_values) (normalized_matrix nv)))
    fn () { unfold projection_pre a b c packed_fraction 1.0R
      (embedding_matrix packed_values) (normalized_matrix nv)
      (mma_result (embedding_matrix packed_values) (normalized_matrix nv)); };
  mma_shared_scores hidden_values packed_values nv;
  launch_kernel_1 (fun () -> store_scores c scores
    #(mma_result (embedding_matrix packed_values) (normalized_matrix nv))
    #hidden_values #packed_values #initial_scores);
  with score_values. assert on gpu_loc (A.pts_to scores score_values);
  T.free c;
  map_loc gpu_loc
    #(T.tensor_pts_to b (normalized_matrix nv))
    #(normalized |-> nv)
    fn () {
      T.tensor_concr b;
      normalized_matrix_to_seq nv;
      rewrite each T.core b as normalized;
      rewrite A.pts_to normalized
        (T.to_seq (Layout.l2_col_major 1024 8) (normalized_matrix nv))
        as A.pts_to normalized nv;
    };
  gpu_array_free normalized;
  map_loc gpu_loc
    #(T.tensor_pts_to a #packed_fraction (embedding_matrix packed_values) **
      pts_to_slice packed #packed_fraction 254279680 254280704 (norm_suffix packed_values) **
      is_full_slice packed 254280704)
    #(A.pts_to packed #packed_fraction packed_values)
    fn () { embedding_close packed #packed_fraction #packed_values; };

  emit_history_host position prompt_length previous hidden history
    #previous_fraction #hidden_fraction #previous_values #hidden_values #initial_history;
  let history_values : erased (lseq bf16 (emitted_elements position prompt_length)) =
    emitted_history position prompt_length previous_values hidden_values;
  all_shared_scores_mapping hidden_values packed_values score_values;
  establish_semantic_postcondition position prompt_length
    packed_values hidden_values previous_values before before score_values history_values;
  rewrite on gpu_loc (A.pts_to packed #packed_fraction packed_values)
    as buffer_owns packed packed_fraction packed_values;
  rewrite on gpu_loc (A.pts_to hidden #hidden_fraction hidden_values)
    as buffer_owns hidden hidden_fraction hidden_values;
  rewrite on gpu_loc (A.pts_to scores score_values) as buffer_owns scores 1.0R score_values;
}
