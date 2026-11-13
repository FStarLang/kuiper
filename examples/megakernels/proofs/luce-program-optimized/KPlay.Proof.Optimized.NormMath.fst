module KPlay.Proof.Optimized.NormMath

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open KPlay.Proof.Optimized.Sum
module S = FStar.Seq
module C = FStar.Classical
module FP = Kuiper.Float32
module F = Kuiper.Floating.Base
module Approx = Kuiper.Approximates.Base
module Lit = Kuiper.Float32.Literals
module Root = FStar.Math.Sqrt
module P = KPlay.Proof.Shared.Projection
module R = KPlay.Proof.Shared.Recurrent

inline_for_extraction noextract
let epsilon:f32 = F.of_literal "0.000001"

let epsilon_approx ()
  : Lemma (epsilon %~ 0.000001R) =
  assert_norm (Lit.common_normal_cell "0.000001" 1 1000000 1 1000000 8796093 (-43));
  FP.literal_interval_approx "0.000001" 1 1000000 1 1000000 8796093 (-43) 0.000001R

inline_for_extraction noextract
let rec sum_squares (#n:nat) (x:lseq real n) (count:nat{count <= n})
  : GTot Root.rnonneg (decreases count) =
  if count = 0 then 0.0R
  else sum_squares x (count - 1) +. S.index x (count - 1) *. S.index x (count - 1)

inline_for_extraction noextract
let square_cell (#n:nat) (x:lseq real n) (i:nat) : GTot real =
  if i < n then S.index x i *. S.index x i else 0.0R

inline_for_extraction noextract
let scale (#n:pos) (x:lseq real n) : GTot Root.rpos =
  Root.rsqrt (sum_squares x n /. FStar.Real.of_int n +. 0.000001R)

let rec squares_prefix (#n:nat) (x:lseq real n) (count:nat{count <= n})
  : Lemma
      (ensures sum_squares x count == prefix count (square_cell x))
      (decreases count) =
  if count > 0 then (
    squares_prefix x (count - 1);
    prefix_step (count - 1) (square_cell x)
  )

inline_for_extraction noextract
let source_view (#n:nat) (x:lseq real n) : P.vector real n =
  fun i -> S.index x i

let square_view (#n:nat) (x:lseq real n) (i:nat{i < n})
  : Lemma (square_cell x i == source_view x i *. source_view x i) = ()

inline_for_extraction noextract
let shared_squares (#n:nat) (x:lseq real n) : P.vector real n =
  fun i -> source_view x i *. source_view x i

let rec squares_shared (#n:nat) (x:lseq real n) (count:nat{count <= n})
  : Lemma
      (ensures sum_squares x count ==
        Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 count (shared_squares x))
      (decreases count) =
  if count = 0 then
    assert_norm (Kuiper.Sum.sum #real #Kuiper.Sum.real_add_semigroup 0 0 (shared_squares x) == 0.0R)
  else (
    squares_shared x (count - 1);
    Kuiper.Sum.sum_pop_right #real #Kuiper.Sum.real_add_semigroup 0 count (shared_squares x)
  )

let shared_scales_equal (#n:pos) (x:P.vector real n)
  : Lemma (R.rms_scale #n x ==
      Root.rsqrt (P.rms_argument #n P.reference_epsilon x)) =
  assert (R.rms_scale #n x ==
    Root.rsqrt (P.rms_argument #n P.reference_epsilon x))
  by (FStar.Tactics.V2.norm
        [delta_only [`%R.rms_scale; `%R.sum_squares; `%R.epsilon;
                     `%P.rms_argument; `%P.real_sum; `%P.reference_epsilon]; zeta; iota];
      FStar.Tactics.V2.trefl ())

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let scale_shared (#n:pos) (x:lseq real n)
  : Lemma (
      scale x == Root.rsqrt (P.rms_argument #n P.reference_epsilon (source_view x)) /\
      scale x == R.rms_scale #n (source_view x)) =
  squares_shared x n;
  FStar.Pervasives.norm_spec
    [delta_only [`%shared_squares]; zeta; iota] (shared_squares x);
  FStar.Pervasives.norm_spec
    [delta_only [`%scale]; zeta; iota] (scale x);
  FStar.Pervasives.norm_spec
    [delta_only [`%P.rms_argument; `%P.real_sum; `%P.reference_epsilon]; zeta; iota]
    (P.rms_argument #n P.reference_epsilon (source_view x));
  assert (P.rms_argument #n P.reference_epsilon (source_view x) ==
    sum_squares x n /. FStar.Real.of_int n +. 0.000001R);
  shared_scales_equal #n (source_view x)
#pop-options

inline_for_extraction noextract
let square_pair (x:lseq real 1024) (warp lane:nat) : GTot real =
  square_cell x (warp * 32 + lane) +. square_cell x (512 + (warp * 32 + lane))

inline_for_extraction noextract
let warp_square_sum (x:lseq real 1024) (warp:nat) : GTot real =
  if warp < 16 then prefix 32 (square_pair x warp) else 0.0R

let warp_squares_split (x:lseq real 1024) (warp:nat{warp < 16})
  : Lemma (
      warp_square_sum x warp ==
      prefix 32 (fun lane -> square_cell x (warp * 32 + lane)) +.
      prefix 32 (fun lane -> square_cell x (512 + (warp * 32 + lane)))) =
  prefix_ext 32 (square_pair x warp)
    (fun lane -> square_cell x (warp * 32 + lane) +.
      square_cell x (512 + (warp * 32 + lane)));
  prefix_add 32
    (fun lane -> square_cell x (warp * 32 + lane))
    (fun lane -> square_cell x (512 + (warp * 32 + lane)))

let all_squares (x:lseq real 1024)
  : Lemma (prefix 32 (warp_square_sum x) == sum_squares x 1024) =
  let first = fun (warp:nat) -> prefix 32 (fun lane -> square_cell x (warp * 32 + lane)) in
  let second = fun (warp:nat) -> prefix 32 (fun lane -> square_cell x (512 + (warp * 32 + lane))) in
  C.forall_intro (warp_squares_split x);
  prefix_ext 16 (warp_square_sum x) (fun warp -> first warp +. second warp);
  prefix_add 16 first second;
  prefix_blocks 16 32 (square_cell x);
  prefix_blocks 16 32 (fun i -> square_cell x (512 + i));
  prefix_split 512 512 (square_cell x);
  squares_prefix x 1024;
  prefix_split 16 16 (warp_square_sum x);
  prefix_ext 16 (fun i -> warp_square_sum x (16 + i)) (fun _ -> 0.0R);
  prefix_zero 16

inline_for_extraction noextract
let inverse_width (n:pos{n <= 1024}) : f32 =
  F.div (F.one #f32) (F.of_int #f32 (FStar.Int64.int_to_t n))

let inverse_width_approx (n:pos{n <= 1024})
  : Lemma (inverse_width n %~ (1.0R /. FStar.Real.of_int n)) =
  Approx.of_int_approx #f32 (FStar.Int64.int_to_t n);
  Approx.div_approx (F.one #f32) (F.of_int #f32 (FStar.Int64.int_to_t n))
    1.0R (FStar.Real.of_int n)

let rms_finish (#n:pos) (hidden:lseq bf16 n) (argument result:f32)
  : Lemma
      (requires
        (forall (model:lseq real n). hidden %~ model ==>
          argument %~ (sum_squares model n /. FStar.Real.of_int n +. 0.000001R <: real)) /\
        (forall (x:Root.rpos). argument %~ (x <: real) ==>
          result %~ (1.0R /. Root.sqrt x)))
      (ensures forall (model:lseq real n).
        hidden %~ model ==> result %~ (scale model <: real)) =
  let related (model:lseq real n{hidden %~ model})
    : Lemma (result %~ (scale model <: real)) = ()
  in C.forall_intro related
