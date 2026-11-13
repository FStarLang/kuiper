module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm_layout, l2_col_major as col_major }
open Kuiper.Array2.Strided
open Pulse.Lib.Array
module S = KPlay.SharedSpec
module A = Pulse.Lib.Array
module GA = Kuiper.Array
module Seq = FStar.Seq
module SZ = Kuiper.SizeT
module Sum = Kuiper.Sum
module MMA = Kuiper.TensorCore.MMA
module G = Kuiper.Spec.GEMM
module EM = Kuiper.EMatrix
module C = Kuiper.Float.Casts
module R = FStar.Math.Sqrt
module E = FStar.Math.Exp
module Fast = Kuiper.Float32.FastMath
module Softplus = Luce.Softplus
module TM = Kuiper.Kernel.TMap
module Shape = Kuiper.Shape

#set-options "--fuel 4 --ifuel 4 --z3rlimit 20"

#push-options "--fuel 2,2 --ifuel 1,1 --z3rlimit 20"
inline_for_extraction noextract
let rec native_sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures Sum.sum 0 n f == Sum.sum 0 n g)
      (decreases n) =
  if n > 0 then (
    native_sum_ext (n - 1) f g;
    Sum.sum_pop_right 0 n f;
    Sum.sum_pop_right 0 n g;
    assert (Sum.sum 0 n f == Sum.sum 0 (n - 1) f +. f (n - 1));
    assert (Sum.sum 0 n g == Sum.sum 0 (n - 1) g +. g (n - 1))
  )
#pop-options

[@@"opaque_to_smt"]
inline_for_extraction noextract
let sum_n (n:nat) (f:S.vector n) : GTot real = Sum.sum 0 n (fun i -> f i)

inline_for_extraction noextract
let sum_unfold (n:nat) (f:S.vector n)
  : Lemma (sum_n n f == Sum.sum 0 n f) =
  norm_spec [delta_once ["KPlay.Functional.sum_n"]] (sum_n n f);
  native_sum_ext n (fun (i:Kuiper.Common.between 0 n) -> f i) f

inline_for_extraction noextract
let sum_zero (f:S.vector 0)
  : Lemma (ensures sum_n 0 f == 0.0R) [SMTPat (sum_n 0 f)] =
  sum_unfold 0 f

inline_for_extraction noextract
let sum_pop (n:pos) (f:S.vector n)
  : Lemma (sum_n n f == sum_n (n - 1) f +. f (n - 1)) =
  sum_unfold n f;
  sum_unfold (n - 1) f;
  Sum.sum_pop_right 0 n f

inline_for_extraction noextract
let rec sum_ext (n:nat) (f g:S.vector n)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures sum_n n f == sum_n n g)
      (decreases n) =
  if n = 0 then (sum_zero f; sum_zero g)
  else (
    sum_ext (n - 1) f g;
    sum_pop n f;
    sum_pop n g
  )

inline_for_extraction noextract
let sum_step (n:pos) (f:S.vector n)
  : Lemma (sum_n n f == sum_n (n - 1) (fun i -> f i) +. f (n - 1)) =
  sum_pop n f;
  sum_ext (n - 1) f (fun i -> f i)

inline_for_extraction noextract
let sum_shared_ext (n:nat) (f:S.vector n)
  (g:Kuiper.Common.between 0 n -> GTot real)
  : Lemma
      (requires forall (i:S.fin n). f i == g i)
      (ensures sum_n n f == Sum.sum 0 n g) =
  sum_ext n f g;
  sum_unfold n g

inline_for_extraction noextract
let rec sum_plus (n:nat) (f g:S.vector n)
  : Lemma
      (sum_n n (fun i -> f i +. g i) == sum_n n f +. sum_n n g)
      (decreases n) =
  let combined:S.vector n = fun i -> f i +. g i in
  if n = 0 then (
    sum_zero f; sum_zero g; sum_zero combined
  ) else (
    sum_plus (n - 1) f g;
    sum_ext (n - 1) combined (fun i -> f i +. g i);
    sum_pop n f; sum_pop n g; sum_pop n combined
  )

inline_for_extraction noextract
let rec sum_split (n m:nat) (f:S.vector (n + m))
  : Lemma
      (sum_n (n + m) f ==
       sum_n n (fun i -> f i) +. sum_n m (fun i -> f (n + i)))
      (decreases m) =
  let prefix:S.vector n = fun i -> f i in
  let suffix:S.vector m = fun i -> f (n + i) in
  if m = 0 then (
    sum_zero suffix;
    sum_ext n f prefix
  ) else (
    sum_split n (m - 1) f;
    sum_ext (m - 1) suffix (fun i -> f (n + i));
    sum_pop (n + m) f;
    sum_pop m suffix
  )

inline_for_extraction noextract
let sum_halves (n:pos) (f:S.vector (2 * n))
  : Lemma
      (sum_n n (fun i -> f i +. f (i + n)) == sum_n (2 * n) f) =
  sum_split n n f;
  sum_plus n (fun i -> f i) (fun i -> f (i + n));
  sum_ext n (fun i -> f (i + n)) (fun i -> f (n + i))

inline_for_extraction noextract
let rec sum_groups (n:nat) (width:pos) (f:S.vector (width * n))
  : Lemma
      (sum_n n (fun g -> sum_n width (fun j -> f (width * g + j))) ==
       sum_n (width * n) f)
      (decreases n) =
  let groups (g:S.fin n)
    : GTot (r:real{r == sum_n width (fun j -> f (width * g + j))}) =
    sum_n width (fun j -> f (width * g + j))
  in
  sum_ext n groups (fun g -> sum_n width (fun j -> f (width * g + j)));
  if n = 0 then (
    sum_zero groups;
    sum_zero f
  ) else (
    sum_groups (n - 1) width f;
    sum_ext (n - 1) groups (fun g -> sum_n width (fun j -> f (width * g + j)));
    sum_pop n groups;
    sum_split (width * (n - 1)) width f;
    sum_ext (width * (n - 1)) f (fun i -> f i);
    assert (width * (n - 1) + width == width * n);
    assert (groups (n - 1) == sum_n width (fun i -> f (width * (n - 1) + i)));
    assert (sum_n (width * n) f ==
      sum_n (width * (n - 1)) f +. groups (n - 1))
  )

inline_for_extraction noextract
let rec sum_all_zero (n:nat) (f:S.vector n)
  : Lemma
      (requires forall (i:S.fin n). f i == 0.0R)
      (ensures sum_n n f == 0.0R)
      (decreases n) =
  if n = 0 then sum_zero f
  else (sum_all_zero (n - 1) f; sum_pop n f)

inline_for_extraction noextract
let rec sum_swap (n m:nat) (f:S.matrix n m)
  : Lemma
      (sum_n n (fun i -> sum_n m (fun j -> f i j)) ==
       sum_n m (fun j -> sum_n n (fun i -> f i j)))
      (decreases n) =
  let rows (i:S.fin n)
    : GTot (v:real{v == sum_n m (fun j -> f i j)}) =
    sum_n m (fun j -> f i j) in
  let columns (j:S.fin m)
    : GTot (v:real{v == sum_n n (fun i -> f i j)}) =
    sum_n n (fun i -> f i j) in
  sum_ext n rows (fun i -> sum_n m (fun j -> f i j));
  sum_ext m columns (fun j -> sum_n n (fun i -> f i j));
  if n = 0 then (
    sum_zero rows;
    let point (j:S.fin m)
      : Lemma (columns j == 0.0R) =
      let column:S.vector n = fun i -> f i j in
      sum_zero column
    in
    FStar.Classical.forall_intro point;
    sum_all_zero m columns
  ) else (
    let lower (j:S.fin m)
      : GTot (v:real{v == sum_n (n - 1) (fun i -> f i j)}) =
      sum_n (n - 1) (fun i -> f i j)
    in
    sum_swap (n - 1) m f;
    sum_ext (n - 1) rows (fun i -> sum_n m (fun j -> f i j));
    sum_ext m lower (fun j -> sum_n (n - 1) (fun i -> f i j));
    sum_pop n rows;
    let point (j:S.fin m)
      : Lemma (columns j == lower j +. f (n - 1) j) =
      sum_step n (fun i -> f i j)
    in
    FStar.Classical.forall_intro point;
    sum_ext m columns (fun j -> lower j +. f (n - 1) j);
    sum_plus m lower (f (n - 1));
    sum_ext m (f (n - 1)) (fun j -> f (n - 1) j);
    assert (sum_n (n - 1) rows == sum_n m lower);
    assert (rows (n - 1) == sum_n m (f (n - 1)));
    assert (sum_n n rows == sum_n m columns)
  )

inline_for_extraction noextract
let bounded_sum (#n:nat) (count:nat{count <= n}) (terms:S.vector n) : GTot real =
  sum_n count (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_step (#n:nat) (count:nat{count < n}) (terms:S.vector n)
  : Lemma (bounded_sum (count + 1) terms == bounded_sum count terms +. terms count) =
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum (count + 1) terms);
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum count terms);
  sum_step (count + 1) (fun i -> terms i);
  sum_ext count (fun (i:S.fin (count + 1 - 1)) -> terms i) (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_full (#n:nat) (terms:S.vector n)
  : Lemma (bounded_sum n terms == sum_n n terms) =
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum n terms);
  sum_ext n (fun i -> terms i) terms

inline_for_extraction noextract
let bounded_zero (#n:nat) (terms:S.vector n)
  : Lemma (ensures bounded_sum 0 terms == 0.0R) [SMTPat (bounded_sum 0 terms)] =
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]] (bounded_sum 0 terms);
  sum_zero (fun i -> terms i)

inline_for_extraction noextract
let sum_squares_bridge (n:nat) (x:S.vector n)
  : Lemma (sum_n n (fun i -> x i *. x i) == S.sum_squares x) =
  sum_unfold n (fun i -> x i *. x i)

inline_for_extraction noextract
let real_family (d:Type0) = d -> GTot real

inline_for_extraction noextract
let vector_family (d:Type0) (n:nat) = d -> GTot (S.vector n)

inline_for_extraction noextract
let matrix_family (d:Type0) (m n:nat) = d -> GTot (S.matrix m n)

inline_for_extraction noextract
let family_vector_at (#d:Type0) (#n:nat)
  (x:erased (vector_family d n)) (c:d) (i:nat{i < n}) : GTot real =
  (reveal #(vector_family d n) x) c i

inline_for_extraction noextract
let ua (#d:Type0) (v:f32) (r:real_family d) : GTot prop =
  forall (c:d). v_approximates v (r c)

inline_for_extraction noextract
let uba (#d:Type0) (v:bf16) (r:real_family d) : GTot prop =
  forall (c:d). v_approximates v (r c)

inline_for_extraction noextract
let bf_rep (n:nat) (storage:seq bf16) (logical:S.vector n) : GTot prop =
  Seq.length storage == n /\
  (forall (i:nat{i < n /\ i < Seq.length storage}).
    v_approximates (Seq.index storage i) (logical i))

inline_for_extraction noextract
let fp_rep (n:nat) (storage:seq f32) (logical:S.vector n) : GTot prop =
  Seq.length storage == n /\
  (forall (i:nat{i < n /\ i < Seq.length storage}).
    v_approximates (Seq.index storage i) (logical i))

inline_for_extraction noextract
let ubf (#d:Type0) (n:nat) (s:seq bf16) (x:vector_family d n) : GTot prop =
  Seq.length s == n /\ (forall (c:d). bf_rep n s (x c))

inline_for_extraction noextract
let ubf_congr (#d:Type0) (n:nat) (s:seq bf16) (x y:vector_family d n)
  : Lemma
      (requires ubf n s x /\ (forall (c:d) (i:S.fin n). x c i == y c i))
      (ensures ubf n s y) =
  norm_spec [delta_only ["KPlay.Functional.ubf"; "KPlay.Functional.bf_rep"]] (ubf n s x);
  norm_spec [delta_only ["KPlay.Functional.ubf"; "KPlay.Functional.bf_rep"]] (ubf n s y)

inline_for_extraction noextract
let ufp (#d:Type0) (n:nat) (s:seq f32) (x:vector_family d n) : GTot prop =
  Seq.length s == n /\ (forall (c:d). fp_rep n s (x c))

inline_for_extraction noextract
let bounded_full_family (#d:Type0) (n:nat) (f:vector_family d n)
  : Lemma (forall (c:d). bounded_sum n (f c) == sum_n n (f c)) =
  let point (c:d) : Lemma (bounded_sum n (f c) == sum_n n (f c)) =
    bounded_sum_full (f c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let instantiate3 (p:real -> real -> real -> prop) (x y z:real)
  : Lemma (requires forall (a b c:real). p a b c) (ensures p x y z) =
  let first:squash (forall (b c:real). p x b c) =
    FStar.Classical.Sugar.forall_elim x () in
  let second:squash (forall (c:real). p x y c) =
    FStar.Classical.Sugar.forall_elim y first in
  let third:squash (p x y z) = FStar.Classical.Sugar.forall_elim z second in ()

inline_for_extraction noextract
let fma_family (#d:Type0) (x y z result:f32) (xr yr zr:real_family d)
  : Lemma
      (requires
        (forall (a b c:real).
          v_approximates x a /\ v_approximates y b /\ v_approximates z c ==>
          v_approximates result (a *. b +. c)) /\
        ua x xr /\ ua y yr /\ ua z zr)
      (ensures ua result (fun c -> xr c *. yr c +. zr c)) =
  let point (c:d)
    : Lemma (v_approximates result (xr c *. yr c +. zr c)) =
    instantiate3
      (fun a b c ->
        v_approximates x a /\ v_approximates y b /\ v_approximates z c ==>
        v_approximates result (a *. b +. c)) (xr c) (yr c) (zr c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let negative_zero () : Tot (v:f32{v_approximates v 0.0R}) =
  let minus_one = of_int #f32 (-1L) in
  of_int_approx #f32 (-1L);
  a_mul minus_one (zero #f32) (0.0R -. 1.0R) 0.0R;
  mul minus_one (zero #f32)

inline_for_extraction noextract
fn rounded_mul (#d:Type0) (x y:f32) (#xr #yr:erased (real_family d))
  preserves gpu
  requires pure (ua x xr /\ ua y yr)
  returns result:f32
  ensures pure (ua result (fun c -> (reveal #(real_family d) xr) c *. (reveal #(real_family d) yr) c))
{
  let nz = negative_zero ();
  let result = Fast.fma_rn x y nz;
  fma_family x y nz result xr yr (fun _ -> 0.0R);
  result
}

inline_for_extraction noextract
fn rounded_add (#d:Type0) (x y:f32) (#xr #yr:erased (real_family d))
  preserves gpu
  requires pure (ua x xr /\ ua y yr)
  returns result:f32
  ensures pure (ua result (fun c -> (reveal #(real_family d) xr) c +. (reveal #(real_family d) yr) c))
{
  let result = Fast.fma_rn x (one #f32) y;
  fma_family x (one #f32) y result xr (fun _ -> 1.0R) yr;
  result
}

inline_for_extraction noextract
fn rounded_fma (#d:Type0) (x y z:f32)
  (#xr #yr #zr:erased (real_family d))
  preserves gpu
  requires pure (ua x xr /\ ua y yr /\ ua z zr)
  returns result:f32
  ensures pure (ua result (fun c -> (reveal #(real_family d) xr) c *. (reveal #(real_family d) yr) c +. (reveal #(real_family d) zr) c))
{
  let result = Fast.fma_rn x y z;
  fma_family x y z result xr yr zr;
  result
}

inline_for_extraction noextract
fn rounded_sub (#d:Type0) (x y:f32) (#xr #yr:erased (real_family d))
  preserves gpu
  requires pure (ua x xr /\ ua y yr)
  returns result:f32
  ensures pure (ua result (fun c -> (reveal #(real_family d) xr) c -. (reveal #(real_family d) yr) c))
{
  let result = Fast.sub_rn x y;
  result
}

inline_for_extraction noextract
let widen_family (#d:Type0) (v:bf16) (r:real_family d)
  : Lemma (requires uba v r) (ensures ua (C.fcast #bf16 #f32 v) r) =
  let point (c:d) : Lemma (v_approximates (C.fcast #bf16 #f32 v) (r c)) =
    C.fcast_approx #bf16 #f32 v (r c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let narrow_family (#d:Type0) (v:f32) (r:real_family d)
  : Lemma (requires ua v r) (ensures uba (C.fcast #f32 #bf16 v) r) =
  let point (c:d) : Lemma (v_approximates (C.fcast #f32 #bf16 v) (r c)) =
    C.fcast_approx #f32 #bf16 v (r c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let round_bf (#d:Type0) (v:f32) (#r:erased (real_family d))
  : Pure f32 (requires ua v r)
      (ensures fun result -> ua result r) =
  narrow_family v r;
  let b = C.fcast #f32 #bf16 v in
  widen_family b r;
  C.fcast #bf16 #f32 b

inline_for_extraction noextract
let exp_family (#d:Type0) (v:f32) (r:real_family d)
  : Lemma (requires ua v r)
      (ensures ua (fexp v) (fun c -> E.exp (r c))) =
  let point (c:d) : Lemma (v_approximates (fexp v) (E.exp (r c))) =
    exp_approx v (r c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let neg_family (#d:Type0) (v:f32) (r:real_family d)
  : Lemma (requires ua v r)
      (ensures ua (mul v (of_int #f32 (-1L))) (fun c -> 0.0R -. r c)) =
  of_int_approx #f32 (-1L);
  let point (c:d)
    : Lemma (v_approximates (mul v (of_int #f32 (-1L))) (0.0R -. r c)) =
    a_mul v (of_int #f32 (-1L)) (r c) (0.0R -. 1.0R)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let divide_family (#d:Type0) (x y:f32) (xr yr:real_family d)
  : Lemma
      (requires ua x xr /\ ua y yr /\ (forall (c:d). yr c =!= 0.0R))
      (ensures ua (div x y) (fun c -> xr c /. yr c)) =
  let point (c:d) : Lemma (v_approximates (div x y) (xr c /. yr c)) =
    div_approx x y (xr c) (yr c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let multiply_reciprocal (x:real) (y:real{y =!= 0.0R})
  : Lemma (x *. (1.0R /. y) == x /. y) = ()

inline_for_extraction noextract
let reciprocal_family (#d:Type0) (v:f32) (n:pos) (x:real_family d)
  : Lemma
      (requires ua v (fun c -> x c *. (1.0R /. FStar.Real.of_int n)))
      (ensures ua v (fun c -> x c /. FStar.Real.of_int n)) =
  let point (c:d) : Lemma (v_approximates v (x c /. FStar.Real.of_int n)) =
    multiply_reciprocal (x c) (FStar.Real.of_int n)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let rsqrt_family (#d:Type0) (v:f32) (r:real_family d)
  : Lemma (requires ua v r /\ (forall (c:d). r c >. 0.0R))
      (ensures ua (rsqrt v) (fun c -> R.rsqrt (r c))) =
  let point (c:d) : Lemma (v_approximates (rsqrt v) (R.rsqrt (r c))) =
    rsqrt_approx v (r c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let softplus_family (#d:Type0) (v:f32) (r:real_family d)
  : Lemma (requires ua v r)
      (ensures ua (Softplus.softplus20 v) (fun c -> S.softplus (r c))) =
  let point (c:d)
    : Lemma (v_approximates (Softplus.softplus20 v) (S.softplus (r c))) =
    Softplus.softplus20_approx v (r c);
    assert (Softplus.model (r c) == S.softplus (r c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let reciprocal (n:FStar.Int64.t{FStar.Int64.v n > 0})
  : Tot (v:f32{v_approximates v (1.0R /. FStar.Real.of_int (FStar.Int64.v n))}) =
  of_int_approx #f32 n;
  div_approx (one #f32) (of_int #f32 n) 1.0R (FStar.Real.of_int (FStar.Int64.v n));
  div (one #f32) (of_int #f32 n)

inline_for_extraction noextract
let epsilon32 () : Tot (v:f32{v_approximates v S.epsilon}) = reciprocal 1000000L

inline_for_extraction noextract
let sigmoid_domain_family (#d:Type0) (r:real_family d)
  : Lemma (forall (c:d). 1.0R +. E.exp (0.0R -. r c) >. 0.0R) =
  let point (c:d) : Lemma (1.0R +. E.exp (0.0R -. r c) >. 0.0R) =
    E.exp_positive (0.0R -. r c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn activation32 (#d:Type0) (silu:bool) (v:bf16)
  (#r:erased (real_family d))
  preserves gpu
  requires pure (uba v r)
  returns result:f32
  ensures pure (ua result (fun c -> if silu then S.silu ((reveal #(real_family d) r) c) else S.sigmoid ((reveal #(real_family d) r) c)))
{
  widen_family v r;
  let x = C.fcast #bf16 #f32 v;
  neg_family x r;
  let negative = mul x (of_int #f32 (-1L));
  exp_family negative (fun c -> 0.0R -. (reveal #(real_family d) r) c);
  let ex = fexp negative;
  let denominator = rounded_add (one #f32) ex
    #(fun _ -> 1.0R) #(fun c -> E.exp (0.0R -. (reveal #(real_family d) r) c));
  let numerator:f32 = (if silu then x else one);
  let nr:erased (real_family d) = hide (fun c -> if silu then (reveal #(real_family d) r) c else 1.0R);
  sigmoid_domain_family r;
  divide_family numerator denominator nr (fun c -> 1.0R +. E.exp (0.0R -. (reveal #(real_family d) r) c));
  let result = div numerator denominator;
  result
}

instance send_initialized
  (#a:Type0) (p:A.array a{GA.is_global_array p}) (#f:perm) (s:seq a)
  : is_send_across gpu_of (A.pts_to p #f s) =
  GA.is_send_pts_to p #f s

instance send_star3 (p q r:slprop)
  {| sp:is_send_across gpu_of p, sq:is_send_across gpu_of q, sr:is_send_across gpu_of r |}
  : is_send_across gpu_of (p ** q ** r) =
  Pulse.Lib.Send.is_send_across_star p (q ** r) #sp
    #(Pulse.Lib.Send.is_send_across_star q r #sq #sr)

inline_for_extraction noextract
let pad16 (n:pos) : pos = ((n + 15) / 16) * 16

inline_for_extraction noextract
let pad16_bounds (n:pos) : Lemma (n <= pad16 n /\ pad16 n < n + 16 /\ pad16 n % 16 == 0) = ()

inline_for_extraction noextract
let padded_matrix (rows cols:pos) (w:S.matrix rows cols)
  : chest2 real (pad16 rows) (pad16 cols) =
  mk2 (fun r c -> if r < rows && c < cols then w r c else 0.0R)

inline_for_extraction noextract
let padded_input (cols:pos) (x:S.vector cols)
  : chest2 real (pad16 cols) 8 =
  mk2 (fun c _ -> if c < cols then x c else 0.0R)

inline_for_extraction noextract
let matrix_rep
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (storage:seq bf16) (w:S.matrix rows cols) : prop =
  if Seq.length storage = count then
    forall (r:S.fin rows) (c:S.fin cols).
      v_approximates (Seq.index storage (offset + r * rs + c * cs)) (w r c)
  else False

inline_for_extraction noextract
let umatrix (#d:Type0)
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (storage:seq bf16) (w:matrix_family d rows cols) : GTot prop =
  Seq.length storage == count /\
  (forall (c:d). matrix_rep rows cols count offset rs cs storage (w c))

inline_for_extraction noextract
let matrix_index_bounds
  (rows cols:pos) (count:nat)
  (offset rs:nat) (cs:nat{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (r:nat{r < rows}) (c:nat{c < cols})
  : Lemma (
      r * rs < count /\ c * cs < count /\
      offset + r * rs < count /\ offset + r * rs + c * cs < count) = ()

inline_for_extraction noextract
let row_index (nr nc:pos) (r:nat{r < nr}) (c:nat{c < nc})
  : Lemma (
      r * nc + c < nr * nc /\ (r * nc + c) / nc == r /\ (r * nc + c) % nc == c) = ()

inline_for_extraction noextract
let row_rep (nr nc:pos) (storage:seq bf16) (logical:chest2 real nr nc) : prop =
  if Seq.length storage = nr * nc then
    forall (r:nat{r < nr}) (c:nat{c < nc}).
      v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)
  else False

inline_for_extraction noextract
let flat_matrix_result
  (nr nc:pos) (storage:lseq bf16 (nr * nc)) (logical:chest2 real nr nc)
  : Lemma
      (requires forall (k:nat{k < nr * nc}).
        v_approximates (Seq.index storage k) (acc2 logical (k / nc) (k % nc)))
      (ensures row_rep nr nc storage logical) =
  let point (r:nat{r < nr}) (c:nat{c < nc})
    : Lemma (v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)) =
    row_index nr nc r c;
    let _ : squash (v_approximates (Seq.index storage (r * nc + c))
      (acc2 logical ((r * nc + c) / nc) ((r * nc + c) % nc))) =
      FStar.Classical.Sugar.forall_elim (r * nc + c) ()
    in ()
  in
  FStar.Classical.forall_intro_2 #(r:nat{r < nr}) #(fun _ -> c:nat{c < nc})
    #(fun r c -> v_approximates (Seq.index storage (r * nc + c)) (acc2 logical r c)) point

inline_for_extraction noextract
let flat_matrix_family (#d:Type0) (nr nc:pos) (storage:lseq bf16 (nr * nc))
  (logical:d -> GTot (chest2 real nr nc))
  : Lemma
      (requires forall (ctx:d) (k:nat{k < nr * nc}).
        v_approximates (Seq.index storage k) (acc2 (logical ctx) (k / nc) (k % nc)))
      (ensures forall (ctx:d). row_rep nr nc storage (logical ctx)) =
  let point (ctx:d) : Lemma (row_rep nr nc storage (logical ctx)) =
    flat_matrix_result nr nc storage (logical ctx)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn read_matrix_element
  (#d:Type0)
  (rows:szp{rows <= 6144}) (cols:szp{cols <= 3584}) (count:szp)
  (source:A.larray bf16 count)
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (r:sz{r < pad16 rows}) (c:sz{c < pad16 cols})
  (#fraction:perm) (#storage:erased (seq bf16))
  (#w:erased (matrix_family d rows cols))
  preserves gpu ** source |-> Frac fraction storage
  requires pure (umatrix rows cols count offset rs cs storage w)
  returns result:bf16
  ensures pure (uba result (fun ctx -> acc2 (padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) ctx)) r c))
{
  if (r <^ rows && c <^ cols) {
    matrix_index_bounds rows cols count offset rs cs r c;
    source.(offset +^ r *^ rs +^ c *^ cs)
  } else {
    (zero #bf16)
  }
}

instance packing_source_shareable
  (source:A.array bf16) (storage:erased (seq bf16))
  : Kuiper.Shareable.shareable (fun fr -> A.pts_to source #fr storage) = {
  _share_n = (fun (n:pos) (#fr:perm) ->
    Kuiper.Array.Extra.array_share source #storage #fr n);
  _gather_n = (fun (n:pos) (#fr:perm) ->
    Kuiper.Array.Extra.array_gather source #storage #fr n);
}

inline_for_extraction noextract
let packing_relation (#d:Type0) (rows cols:pos)
  (w:erased (matrix_family d rows cols))
  (i:Shape.abs (Shape.ICons (pad16 rows * pad16 cols) Shape.INil))
  (_old value:bf16) : GTot prop =
  uba value (fun ctx ->
    acc2 (padded_matrix rows cols ((reveal w) ctx))
      (fst i / pad16 cols) (fst i % pad16 cols))

inline_for_extraction noextract
fn read_packed_element
  (#d:Type0)
  (rows:szp{rows <= 6144})
  (cols:szp{cols <= 3584})
  (count:szp) (source:A.larray bf16 count)
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (nc:szp{SZ.v nc == pad16 cols})
  (#storage:erased (seq bf16))
  (#w:erased (matrix_family d rows cols))
  (#_:squash (umatrix rows cols count offset rs cs storage w))
  (#fr:perm)
  (i:Shape.conc (Shape.ICons (pad16 rows * pad16 cols) Shape.INil)) (old:bf16)
  norewrite
  preserves gpu ** (A.pts_to source #fr storage)
  returns value:bf16
  ensures pure (packing_relation rows cols w (Shape.up i) old value)
{
  let j, _ = i;
  let r = j /^ nc;
  let c = j %^ nc;
  read_matrix_element rows cols count source offset rs cs r c #fr #storage #w
}

inline_for_extraction noextract
let packed_tensor_family (#d:Type0) (rows cols:pos)
  (values:chest1 bf16 (pad16 rows * pad16 cols))
  (w:matrix_family d rows cols)
  : Lemma
      (requires forall (k:nat{k < pad16 rows * pad16 cols}).
        uba (acc1 values k) (fun ctx ->
          acc2 (padded_matrix rows cols (w ctx))
            (k / pad16 cols) (k % pad16 cols)))
      (ensures forall (ctx:d) (k:nat{k < pad16 rows * pad16 cols}).
        v_approximates
          (Seq.index (to_seq (Kuiper.Tensor.Layout.Alg.l1_forward
            (pad16 rows * pad16 cols)) values) k)
          (acc2 (padded_matrix rows cols (w ctx))
            (k / pad16 cols) (k % pad16 cols))) =
  let layout = Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols) in
  let point (ctx:d) (k:nat{k < pad16 rows * pad16 cols})
    : Lemma (
        v_approximates (Seq.index (to_seq layout values) k)
          (acc2 (padded_matrix rows cols (w ctx))
            (k / pad16 cols) (k % pad16 cols))) =
    let coordinate = (k, ()) in
    assert (layout.imap.f coordinate == k);
    Kuiper.Injection.lem_pat layout.imap coordinate
  in
  FStar.Classical.forall_intro_2 #d
    #(fun _ -> k:nat{k < pad16 rows * pad16 cols})
    #(fun ctx k ->
      v_approximates (Seq.index (to_seq layout values) k)
        (acc2 (padded_matrix rows cols (w ctx))
          (k / pad16 cols) (k % pad16 cols))) point

inline_for_extraction noextract
fn prepare_matrix
  (#d:Type0)
  (rows:szp{rows <= 6144})
  (cols:szp{cols <= 3584})
  (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source})
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (dest:A.larray bf16 (pad16 rows * pad16 cols){GA.is_global_array dest})
  (#fraction:perm)
  (#storage #before:erased (seq bf16))
  (#w:erased (matrix_family d rows cols))
  preserves cpu ** on gpu_loc (A.pts_to source #fraction storage)
  requires on gpu_loc (dest |-> before)
  requires pure (umatrix rows cols count offset rs cs storage w)
  ensures exists* (next:seq bf16).
    on gpu_loc (dest |-> next) **
    pure (Seq.length next == pad16 rows * pad16 cols /\
      (forall (c:d). row_rep (pad16 rows) (pad16 cols) next (padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) c))))
{
  pad16_bounds rows;
  pad16_bounds cols;
  let nr = ((rows +^ 15sz) /^ 16sz) *^ 16sz;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nr == pad16 rows /\ SZ.v nc == pad16 cols);
  let cells = nr *^ nc;
  map_loc gpu_loc #(dest |-> before)
    #(dest |-> before ** pure (Seq.length before == nr * nc))
    fn _ { A.pts_to_len dest; };
  let output = from_array (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) dest;
  assert rewrites_to output (from_array (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) dest);
  map_loc gpu_loc #(dest |-> before)
    #(output |-> from_seq (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) before)
    fn _ { tensor_abs' (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) dest; };
  launch_sync (TM.kmap
    (Shape.CCons cells Shape.CNil)
    (fun fr -> A.pts_to source #fr storage)
    #(packing_source_shareable source storage)
    #(fun fr -> GA.is_send_pts_to source #fr storage)
    (packing_relation rows cols w)
    (read_packed_element rows cols count source offset rs cs nc #storage #w #())
    #(Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols))
    #(Kuiper.Tensor.Layout.Alg.c_l1_forward (pad16 rows * pad16 cols))
    cells output
    #(from_seq (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) before) #_ #fraction);
  with values. assert on gpu_loc (output |-> values);
  packed_tensor_family rows cols values w;
  lem_from_array_core
    #bf16 #1 #(Shape.ICons (pad16 rows * pad16 cols) Shape.INil)
    #(Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) dest;
  assert rewrites_to (core output) dest;
  map_loc gpu_loc #(output |-> values)
    #(dest |-> to_seq (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) values)
    fn _ {
      tensor_concr output;
      rewrite each (core (from_array
        (Kuiper.Tensor.Layout.Alg.l1_forward (pad16 rows * pad16 cols)) dest))
        as dest;
    };
  with next. assert on gpu_loc (dest |-> next);
  flat_matrix_family nr nc next (fun ctx -> padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) ctx));
  assert pure (forall (ctx:d).
    row_rep (pad16 rows) (pad16 cols) next (padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) ctx)));
}

inline_for_extraction noextract
fn read_input_element
  (#d:Type0)
  (cols:szp{cols <= 3584}) (source:A.larray bf16 cols)
  (r:sz{r < pad16 cols}) (c:sz{c < 8})
  (#fraction:perm) (#storage:erased (seq bf16))
  (#x:erased (vector_family d cols))
  preserves gpu ** source |-> Frac fraction storage
  requires pure (ubf cols storage x)
  returns result:bf16
  ensures pure (uba result (fun ctx -> acc2 (padded_input cols ((reveal #(vector_family d cols) x) ctx)) r c))
{
  if (r <^ cols) { source.(r) } else { (zero #bf16) }
}

inline_for_extraction noextract
fn prepare_input
  (#d:Type0)
  (cols:szp{cols <= 3584})
  (source:A.larray bf16 cols)
  (dest:A.larray bf16 (pad16 cols * 8))
  (#fraction:perm)
  (#storage #before:erased (seq bf16))
  (#x:erased (vector_family d cols))
  preserves gpu ** source |-> Frac fraction storage
  requires dest |-> before
  requires pure (ubf cols storage x)
  ensures exists* (next:seq bf16).
    dest |-> next **
    pure (Seq.length next == pad16 cols * 8 /\
      (forall (c:d). row_rep 8 (pad16 cols) next (EM.mtranspose (padded_input cols ((reveal #(vector_family d cols) x) c)))))
{
  pad16_bounds cols;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nc == pad16 cols);
  A.pts_to_len dest;
  let mut i = 0sz;
  while (!i <^ nc *^ 8sz)
    invariant exists* (j:sz{j <= nc * 8}) (values:seq bf16).
      i |-> j ** dest |-> values **
      pure (Seq.length values == pad16 cols * 8 /\
        (forall (ctx:d) (k:nat{k < j}).
        v_approximates (Seq.index values k)
          (acc2 (padded_input cols ((reveal #(vector_family d cols) x) ctx)) (k % nc) (k / nc))))
    decreases (nc * 8 - !i)
  {
    let j = !i;
    let r = j %^ nc;
    let value = read_input_element cols source r (j /^ nc) #fraction #storage #x;
    dest.(j) <- value;
    i := j +^ 1sz;
  };
  with next. assert dest |-> next;
  flat_matrix_family 8 nc next (fun ctx -> EM.mtranspose (padded_input cols ((reveal #(vector_family d cols) x) ctx)));
  assert pure (forall (ctx:d).
    row_rep 8 (pad16 cols) next (EM.mtranspose (padded_input cols ((reveal #(vector_family d cols) x) ctx))));
}

inline_for_extraction noextract
let rec mma_accum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Tot (chest2 f32 16 8) (decreases (finish - start)) =
  if finish = start then const (16 @| 8 @| INil) (zero #f32)
  else
    MMA.emma (mma_accum va vb row start (finish - 1))
      (ematrix_subtile va 16 16 row (finish - 1))
      (ematrix_subtile vb 16 8 (finish - 1) 0)

inline_for_extraction noextract
let block_result
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : chest2 f32 nr 8 =
  mk2 (fun r c -> acc2 (mma_accum va vb (r / 16) start finish) (r % 16) c)

inline_for_extraction noextract
instance c8 : concrete_sz 8 = {x = 8sz}

inline_for_extraction noextract
fn tensor_kernel
  (nr:szp{nr <= 6144 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (rm_layout nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (rm_layout nr 8))
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc)
  (#vb:chest2 bf16 nc 8)
  (#vc:chest2 f32 nr 8)
  (tid:szlt 32)
  ()
  norewrite
  requires gpu **
    (a |-> Frac (1.0R /. 32.0R) va **
     b |-> Frac (1.0R /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) vc) ** thread_id 32 tid
  ensures gpu **
    (a |-> Frac (1.0R /. 32.0R) va **
     b |-> Frac (1.0R /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) (block_result va vb start finish)) ** thread_id 32 tid
{
  let mut row = 0sz;
  while (!row <^ nr /^ 16sz)
    invariant exists* (r:sz{r <= nr / 16}) (values:chest2 f32 nr 8).
      row |-> r ** c |-> Frac (1.0R /. 32.0R) values **
      pure (forall (i:nat{i < r * 16}) (j:nat{j < 8}).
        acc2 values i j == acc2 (block_result va vb start finish) i j)
    decreases (nr / 16 - !row)
  {
    let r = !row;
    array2_extract_tile c 16 8 r 0;
    let ct = array2_subtile c 16 8 (SZ.v r) 0;
    assert rewrites_to ct (array2_subtile c 16 8 (SZ.v r) 0);
    let fragment = MMA.alloc_fragment ();
    MMA.fill fragment (zero #f32);
    let mut k = start;
    while (!k <^ finish)
      invariant exists* (kk:sz{start <= kk /\ kk <= finish}).
        k |-> kk ** fragment |-> mma_accum va vb r start kk
      decreases (finish - !k)
    {
      let kk = !k;
      array2_extract_tile_ro a 16 16 r kk;
      let at = array2_subtile a 16 16 (SZ.v r) (SZ.v kk);
      assert rewrites_to at (array2_subtile a 16 16 (SZ.v r) (SZ.v kk));
      array2_extract_tile_ro b 16 8 kk 0;
      let bt = array2_subtile b 16 8 (SZ.v kk) 0;
      assert rewrites_to bt (array2_subtile b 16 8 (SZ.v kk) 0);
      MMA.mma_sync at bt fragment;
      with av. assert at |-> Frac (1.0R /. 32.0R) av;
      Pulse.Lib.Trade.elim_trade (at |-> Frac (1.0R /. 32.0R) av) (a |-> Frac (1.0R /. 32.0R) va);
      with bv. assert bt |-> Frac (1.0R /. 32.0R) bv;
      Pulse.Lib.Trade.elim_trade (bt |-> Frac (1.0R /. 32.0R) bv) (b |-> Frac (1.0R /. 32.0R) vb);
      norm_spec [delta_once ["KPlay.Functional.mma_accum"]]
        (mma_accum va vb r start (kk + 1));
      k := kk +^ 1sz;
    };
    MMA.store fragment ct;
    with result. assert fragment |-> result;
    drop_ (fragment |-> result);
    with cv. assert ct |-> Frac (1.0R /. 32.0R) cv;
    Pulse.Lib.Forall.elim_forall cv;
    Pulse.Lib.Trade.elim_trade (ct |-> Frac (1.0R /. 32.0R) cv) _;
    row := r +^ 1sz;
  };
  with result. assert c |-> Frac (1.0R /. 32.0R) result;
  EM.lemma_equal_intro result (block_result va vb start finish);
  Kuiper.Chest.ext result (block_result va vb start finish);
  assert pure (result == block_result va vb start finish);
}

inline_for_extraction noextract
let rec real_accum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Tot (chest2 real 16 8) (decreases (finish - start)) =
  if finish = start then const (16 @| 8 @| INil) 0.0R
  else
    G.matplus (real_accum ra rb row start (finish - 1))
      (G.matmul (ematrix_subtile ra 16 16 row (finish - 1))
        (ematrix_subtile rb 16 8 (finish - 1) 0))

inline_for_extraction noextract
let rec mma_accum_approx
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  : Lemma
      (requires va %~ ra /\ vb %~ rb)
      (ensures mma_accum va vb row start finish %~ real_accum ra rb row start finish)
      (decreases (finish - start)) =
  norm_spec [delta_once ["KPlay.Functional.mma_accum"]] (mma_accum va vb row start finish);
  norm_spec [delta_once ["KPlay.Functional.real_accum"]] (real_accum ra rb row start finish);
  if finish = start then ()
  else (
    mma_accum_approx va vb ra rb row start (finish - 1);
    MMA.emma_approx_lemma
      (mma_accum va vb row start (finish - 1))
      (ematrix_subtile va 16 16 row (finish - 1))
      (ematrix_subtile vb 16 8 (finish - 1) 0)
      (real_accum ra rb row start (finish - 1))
      (ematrix_subtile ra 16 16 row (finish - 1))
      (ematrix_subtile rb 16 8 (finish - 1) 0)
  )

inline_for_extraction noextract
let dot_terms (#nr #nc:nat)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (r:nat{r < nr}) (c:nat{c < 8}) : S.vector nc =
  fun k -> acc2 ra r k *. acc2 rb k c

inline_for_extraction noextract
let segment_sum (#n:nat) (start:nat) (finish:nat{start <= finish /\ finish <= n})
  (terms:S.vector n) : real =
  sum_n (finish - start) (fun i -> terms (start + i))

inline_for_extraction noextract
let segment_step (#n:nat) (start:nat) (middle:nat{start <= middle})
  (finish:nat{middle <= finish /\ finish <= n}) (terms:S.vector n)
  : Lemma (
      segment_sum start finish terms ==
      segment_sum start middle terms +. segment_sum middle finish terms) =
  sum_split (middle - start) (finish - middle) (fun i -> terms (start + i));
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum start finish terms);
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum start middle terms);
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum middle finish terms);
  sum_ext (finish - start)
    (fun (i:S.fin (middle - start + (finish - middle))) -> terms (start + i))
    (fun i -> terms (start + i));
  sum_ext (finish - middle)
    (fun i -> terms (start + (middle - start + i))) (fun i -> terms (middle + i))

inline_for_extraction noextract
let rec tile_dot
  (a:chest2 real 16 16) (b:chest2 real 16 8)
  (r:nat{r < 16}) (c:nat{c < 8}) (n:nat{n <= 16})
  : Lemma (
      G.__matmul_single a b r c n == bounded_sum n (dot_terms a b r c))
      (decreases n) =
  if n = 0 then (
    G.__gmatmul_single_zero_lemma 0.0R ( *. ) ( +. ) a b r c;
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum 0 (dot_terms a b r c))
  ) else (
    tile_dot a b r c (n - 1);
    G.matmul_single_lemma a b r c n;
    bounded_sum_step (n - 1) (dot_terms a b r c)
  )

inline_for_extraction noextract
let tile_sum
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (block:nat{block < nc / 16})
  (r:nat{r < 16}) (c:nat{c < 8})
  : Lemma (
      acc2 (G.matmul (ematrix_subtile ra 16 16 row block)
        (ematrix_subtile rb 16 8 block 0)) r c ==
      segment_sum (16 * block) (16 * (block + 1)) (dot_terms ra rb (16 * row + r) c)) =
  let a = ematrix_subtile ra 16 16 row block in
  let b = ematrix_subtile rb 16 8 block 0 in
  G.lemma_matmul_index a b r c;
  tile_dot a b r c 16;
  bounded_sum_full (dot_terms a b r c);
  sum_ext 16
    (dot_terms a b r c)
    (fun k -> dot_terms ra rb (16 * row + r) c (16 * block + k));
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
    (segment_sum (16 * block) (16 * (block + 1)) (dot_terms ra rb (16 * row + r) c));
  sum_ext 16
    (fun (i:S.fin (16 * (block + 1) - 16 * block)) ->
      dot_terms ra rb (16 * row + r) c (16 * block + i))
    (fun i -> dot_terms ra rb (16 * row + r) c (16 * block + i))

inline_for_extraction noextract
let rec real_accum_dot
  (#nr:nat{nr % 16 == 0}) (#nc:nat{nc % 16 == 0})
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (row:nat{row < nr / 16}) (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  (r:nat{r < 16}) (c:nat{c < 8})
  : Lemma (
      acc2 (real_accum ra rb row start finish) r c ==
      segment_sum (16 * start) (16 * finish) (dot_terms ra rb (16 * row + r) c))
      (decreases (finish - start)) =
  norm_spec [delta_once ["KPlay.Functional.real_accum"]] (real_accum ra rb row start finish);
  if finish = start then (
    norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
      (segment_sum (16 * start) (16 * finish) (dot_terms ra rb (16 * row + r) c))
  ) else (
    real_accum_dot ra rb row start (finish - 1) r c;
    tile_sum ra rb row (finish - 1) r c;
    EM.chest_comb_acc2 (add #real)
      (real_accum ra rb row start (finish - 1))
      (G.matmul (ematrix_subtile ra 16 16 row (finish - 1))
        (ematrix_subtile rb 16 8 (finish - 1) 0)) r c;
    segment_step (16 * start) (16 * (finish - 1)) (16 * finish)
      (dot_terms ra rb (16 * row + r) c)
  )

inline_for_extraction noextract
let block_approx
  (#nr:pos{nr % 16 == 0}) (#nc:pos{nc % 16 == 0})
  (va:chest2 bf16 nr nc) (vb:chest2 bf16 nc 8)
  (ra:chest2 real nr nc) (rb:chest2 real nc 8)
  (start:nat) (finish:nat{start <= finish /\ finish <= nc / 16})
  (r:nat{r < nr})
  : Lemma
      (requires va %~ ra /\ vb %~ rb)
      (ensures
        v_approximates (acc2 (block_result va vb start finish) r 0)
          (segment_sum (16 * start) (16 * finish) (dot_terms ra rb r 0))) =
  mma_accum_approx va vb ra rb (r / 16) start finish;
  real_accum_dot ra rb (r / 16) start finish (r % 16) 0;
  let actual = mma_accum va vb (r / 16) start finish in
  let logical = real_accum ra rb (r / 16) start finish in
  assert (chest_approximates actual logical);
  let _ : squash (
      acc actual (idx2 #16 #8 (r % 16) 0) %~ acc logical (idx2 #16 #8 (r % 16) 0)) =
    FStar.Classical.Sugar.forall_elim (idx2 #16 #8 (r % 16) 0) ()
  in ()

inline_for_extraction noextract
let rec zero_suffix_sum (n:nat) (count:nat{count <= n}) (terms:S.vector n)
  : Lemma
      (requires forall (i:nat{count <= i /\ i < n}). terms i == 0.0R)
      (ensures sum_n n terms == bounded_sum count terms)
      (decreases (n - count)) =
  if n = count then bounded_sum_full terms
  else (
    let lower:S.vector (n - 1) = fun i -> terms i in
    zero_suffix_sum (n - 1) count lower;
    sum_step n terms;
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum count lower);
    norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
      (bounded_sum count terms);
    sum_ext count (fun i -> lower i) (fun i -> terms i)
  )

inline_for_extraction noextract
let project_sum (#rows #cols:nat) (w:S.matrix rows cols) (x:S.vector cols) (r:S.fin rows)
  : Lemma (S.project w x r == sum_n cols (fun i -> w r i *. x i)) =
  norm_spec [delta_only ["KPlay.SharedSpec.project"]] (S.project w x r);
  sum_shared_ext cols (fun i -> w r i *. x i) (fun i -> w r i *. x i)

inline_for_extraction noextract
let full_linear_bridge
  (rows cols:pos) (w:S.matrix rows cols) (x:S.vector cols) (r:nat{r < rows})
  : Lemma (
      segment_sum 0 (pad16 cols) (dot_terms (padded_matrix rows cols w) (padded_input cols x) r 0) ==
      S.project #rows #cols w x r) =
  pad16_bounds rows;
  pad16_bounds cols;
  let terms = dot_terms (padded_matrix rows cols w) (padded_input cols x) r 0 in
  zero_suffix_sum (pad16 cols) cols terms;
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]]
    (segment_sum 0 (pad16 cols) terms);
  sum_ext (pad16 cols)
    (fun (i:S.fin (pad16 cols - 0)) -> terms (0 + i))
    (fun i -> terms (0 + i));
  sum_ext (pad16 cols) (fun i -> terms (0 + i)) terms;
  assert (segment_sum 0 (pad16 cols) terms == bounded_sum cols terms);
  norm_spec [delta_once ["KPlay.Functional.bounded_sum"]]
    (bounded_sum cols terms);
  sum_ext cols (fun i -> terms (i <: nat)) (fun i -> w r i *. x i);
  assert (bounded_sum cols terms == sum_n cols (fun i -> w r i *. x i));
  project_sum w x r

inline_for_extraction noextract
let segment_empty (#n:nat) (position:nat{position <= n}) (terms:S.vector n)
  : Lemma (segment_sum position position terms == 0.0R) =
  norm_spec [delta_once ["KPlay.Functional.segment_sum"]] (segment_sum position position terms);
  sum_zero (fun i -> terms (position + i))

inline_for_extraction noextract
let flat_tensor_index
  (nr nc:pos) (values:chest2 f32 nr nc)
  (r:nat{r < nr}) (c:nat{c < nc})
  : Lemma (
      Seq.index (to_seq (rm_layout nr nc) values) (r * nc + c) == acc2 values r c) =
  let layout = rm_layout nr nc in
  let coordinate = idx2 #nr #nc r c in
  assert (layout.imap.f coordinate == r * nc + c);
  Kuiper.Injection.lem_pat layout.imap coordinate

inline_for_extraction noextract
fn store_parts
  (#d:Type0)
  (result:Pulse.Lib.Reference.ref f32) (split:bool) (a b:f32)
  (#before:erased f32)
  (#ar #br:erased (real_family d))
  preserves gpu
  requires result |-> before
  requires pure (
    ua a ar /\ ua b br /\
    (forall (c:d). not split ==> (reveal #(real_family d) ar) c == 0.0R))
  ensures exists* (value:f32). result |-> value **
    pure (ua value (fun c -> (reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c))
{
  if split {
    let first = rounded_add (zero #f32) a #(fun _ -> 0.0R) #ar;
    let combined = rounded_add first b #ar #br;
    result := combined;
  } else {
    let combined = rounded_add (zero #f32) b #(fun _ -> 0.0R) #br;
    result := combined;
  }
}

inline_for_extraction noextract
fn combine_parts
  (#d:Type0)
  (split:bool) (a b:f32)
  (#ar #br:erased (real_family d))
  preserves gpu
  requires pure (
    ua a ar /\ ua b br /\
    (forall (c:d). not split ==> (reveal #(real_family d) ar) c == 0.0R))
  returns combined:f32
  ensures pure (ua combined (fun c -> (reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c))
{
  let mut result : f32 = zero;
  store_parts result split a b #_ #ar #br;
  !result
}

inline_for_extraction noextract
let block_approx_family (#d:Type0)
  (rows cols:pos) (w:matrix_family d rows cols) (x:vector_family d cols)
  (va:chest2 bf16 (pad16 rows) (pad16 cols))
  (vb:chest2 bf16 (pad16 cols) 8)
  (middle:nat{middle <= pad16 cols / 16}) (r:nat{r < rows})
  : Lemma
      (requires forall (c:d).
        va %~ padded_matrix rows cols (w c) /\ vb %~ padded_input cols (x c))
      (ensures
        ua (acc2 (block_result va vb 0 middle) r 0)
          (fun c -> segment_sum 0 (16 * middle)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0)) /\
        ua (acc2 (block_result va vb middle (pad16 cols / 16)) r 0)
          (fun c -> segment_sum (16 * middle) (pad16 cols)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0)) /\
        (forall (c:d).
          segment_sum 0 (16 * middle)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) +.
          segment_sum (16 * middle) (pad16 cols)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) ==
          S.project (w c) (x c) r) /\
        (forall (c:d). middle = 0 ==>
          segment_sum 0 (16 * middle)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) == 0.0R)) =
  pad16_bounds rows;
  pad16_bounds cols;
  let point (c:d)
    : Lemma (
        v_approximates (acc2 (block_result va vb 0 middle) r 0)
          (segment_sum 0 (16 * middle)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0)) /\
        v_approximates (acc2 (block_result va vb middle (pad16 cols / 16)) r 0)
          (segment_sum (16 * middle) (pad16 cols)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0)) /\
        segment_sum 0 (16 * middle)
          (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) +.
        segment_sum (16 * middle) (pad16 cols)
          (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) ==
          S.project (w c) (x c) r /\
        (middle = 0 ==>
          segment_sum 0 (16 * middle)
            (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0) == 0.0R)) =
    block_approx va vb (padded_matrix rows cols (w c)) (padded_input cols (x c)) 0 middle r;
    block_approx va vb (padded_matrix rows cols (w c)) (padded_input cols (x c))
      middle (pad16 cols / 16) r;
    segment_step 0 (16 * middle) (pad16 cols)
      (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0);
    full_linear_bridge rows cols (w c) (x c) r;
    segment_empty 0 (dot_terms (padded_matrix rows cols (w c)) (padded_input cols (x c)) r 0)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn finish_value (#d:Type0)
  (rows:szp{rows <= 6144}) (cols:szp{cols <= 3584})
  (middle:sz{middle <= pad16 cols / 16}) (row:sz{row < rows}) (a b:f32)
  (#va:chest2 bf16 (pad16 rows) (pad16 cols)) (#vb:chest2 bf16 (pad16 cols) 8)
  (#w:erased (matrix_family d rows cols)) (#x:erased (vector_family d cols))
  preserves gpu
  requires pure (
    a == acc2 (block_result va vb 0 middle) row 0 /\
    b == acc2 (block_result va vb middle (pad16 cols / 16)) row 0 /\
    (forall (c:d).
      va %~ padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) c) /\ vb %~ padded_input cols ((reveal #(vector_family d cols) x) c)))
  returns result:bf16
  ensures pure (uba result (fun c -> S.project ((reveal #(matrix_family d rows cols) w) c) ((reveal #(vector_family d cols) x) c) row))
{
  block_approx_family rows cols w x va vb middle row;
  let combined = combine_parts (middle <> 0sz) a b
    #(fun c -> segment_sum 0 (16 * middle)
      (dot_terms (padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) c)) (padded_input cols ((reveal #(vector_family d cols) x) c)) row 0))
    #(fun c -> segment_sum (16 * middle) (pad16 cols)
      (dot_terms (padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) c)) (padded_input cols ((reveal #(vector_family d cols) x) c)) row 0));
  narrow_family combined (fun c -> S.project ((reveal #(matrix_family d rows cols) w) c) ((reveal #(vector_family d cols) x) c) row);
  C.fcast #f32 #bf16 combined
}

inline_for_extraction noextract
fn finish_matvec
  (#d:Type0)
  (rows:szp{rows <= 6144}) (cols:szp{cols <= 3584})
  (middle:sz{middle <= pad16 cols / 16})
  (left right:array2 f32 (rm_layout (pad16 rows) 8))
  (output:A.larray bf16 rows)
  (#va:chest2 bf16 (pad16 rows) (pad16 cols))
  (#vb:chest2 bf16 (pad16 cols) 8)
  (#w:erased (matrix_family d rows cols)) (#x:erased (vector_family d cols))
  (#before:erased (seq bf16))
  preserves gpu
  preserves left |-> block_result va vb 0 middle
  preserves right |-> block_result va vb middle (pad16 cols / 16)
  requires output |-> before
  requires pure (forall (c:d).
    va %~ padded_matrix rows cols ((reveal #(matrix_family d rows cols) w) c) /\ vb %~ padded_input cols ((reveal #(vector_family d cols) x) c))
  ensures exists* (result:seq bf16).
    output |-> result ** pure (Seq.length result == rows /\
      ubf rows result (fun c -> S.project ((reveal #(matrix_family d rows cols) w) c) ((reveal #(vector_family d cols) x) c)))
{
  pad16_bounds rows;
  pad16_bounds cols;
  tensor_concr left;
  tensor_concr right;
  let lp = core left;
  let rp = core right;
  assert rewrites_to lp (core left);
  assert rewrites_to rp (core right);
  A.pts_to_len output;
  let mut i = 0sz;
  while (!i <^ rows)
    invariant exists* (j:sz{j <= rows}) (values:seq bf16).
      i |-> j ** output |-> values **
      pure (Seq.length values == rows /\ (forall (ctx:d) (k:nat{k < j}).
        v_approximates (Seq.index values k) (S.project ((reveal #(matrix_family d rows cols) w) ctx) ((reveal #(vector_family d cols) x) ctx) k)))
    decreases (rows - !i)
  {
    let j = !i;
    let a = lp.(8sz *^ j);
    let b = rp.(8sz *^ j);
    flat_tensor_index (pad16 rows) 8 (block_result va vb 0 middle) j 0;
    flat_tensor_index (pad16 rows) 8 (block_result va vb middle (pad16 cols / 16)) j 0;
    let value = finish_value rows cols middle j a b #va #vb #w #x;
    output.(j) <- value;
    i := j +^ 1sz;
  };
  tensor_abs (rm_layout (pad16 rows) 8) lp #1.0R #(block_result va vb 0 middle);
  tensor_abs (rm_layout (pad16 rows) 8) rp #1.0R #(block_result va vb middle (pad16 cols / 16));
  lem_core_from_array left;
  lem_core_from_array right;
  rewrite each (from_array (rm_layout (pad16 rows) 8) (core left)) as left;
  rewrite each (from_array (rm_layout (pad16 rows) 8) (core right)) as right;
}

ghost
fn tensor_setup
  (#nr #nc:erased nat)
  (a:array2 bf16 (rm_layout nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (rm_layout nr 8))
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  ()
  norewrite
  requires a |-> va ** b |-> vb ** c |-> vc
  ensures (forall+ (_:natlt 32).
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
{
  tensor_share_n a 32;
  tensor_share_n b 32;
  tensor_share_n c 32;
  forevery_zip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
}

ghost
fn tensor_gather
  (#nr #nc:erased nat)
  (a:array2 bf16 (rm_layout nr nc))
  (b:array2 bf16 (col_major nc 8))
  (c:array2 f32 (rm_layout nr 8))
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  ()
  norewrite
  requires (forall+ (_:natlt 32).
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
  ensures a |-> va ** b |-> vb ** c |-> vc
{
  forevery_unzip3
    (fun (_:natlt 32) -> a |-> Frac (1.0R /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (1.0R /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
  tensor_gather_n a 32;
  tensor_gather_n b 32;
  tensor_gather_n c 32;
}

inline_for_extraction noextract
let tensor_desc
  (nr:szp{nr <= 6144 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (rm_layout nr nc){is_global a})
  (b:array2 bf16 (col_major nc 8){is_global b})
  (c:array2 f32 (rm_layout nr 8){is_global c})
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  : kernel_desc_1_n
      (a |-> va ** b |-> vb ** c |-> vc)
      (a |-> va ** b |-> vb ** c |-> block_result va vb start finish) = {
  nthr = 32sz;
  frame = emp;
  kpre = (fun _ ->
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc);
  kpost = (fun _ ->
    a |-> Frac (1.0R /. 32.0R) va **
    b |-> Frac (1.0R /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) (block_result va vb start finish));
  f = tensor_kernel nr nc a b c start finish #va #vb #vc;
  block_setup = tensor_setup a b c #va #vb #vc;
  block_teardown = tensor_gather a b c #va #vb #(block_result va vb start finish);
  full_pre_sendable = solve;
  full_post_sendable = solve;
  kpre_sendable = solve;
  kpost_sendable = solve;
}

fn run_tensor
  (nr:szp{nr <= 6144 /\ nr % 16 == 0})
  (nc:szp{nc <= 3584 /\ nc % 16 == 0})
  (a:array2 bf16 (rm_layout nr nc){is_global a})
  (b:array2 bf16 (col_major nc 8){is_global b})
  (c:array2 f32 (rm_layout nr 8){is_global c})
  (start:sz) (finish:sz{start <= finish /\ finish <= nc / 16})
  (#va:chest2 bf16 nr nc) (#vb:chest2 bf16 nc 8) (#vc:chest2 f32 nr 8)
  preserves cpu ** on gpu_loc (a |-> va ** b |-> vb)
  requires on gpu_loc (c |-> vc)
  ensures on gpu_loc (c |-> block_result va vb start finish)
{
  launch_kernel_full_sync (tensor_desc nr nc a b c start finish #va #vb #vc);
}

inline_for_extraction noextract
let row_from_seq_approx
  (nr nc:pos) (storage:lseq bf16 (nr * nc)) (logical:chest2 real nr nc)
  : Lemma
      (requires row_rep nr nc storage logical)
      (ensures from_seq (rm_layout nr nc) storage %~ logical) =
  EM.lemma_approximates_intro (from_seq (rm_layout nr nc) storage) logical

inline_for_extraction noextract
let col_from_seq_approx
  (nr:pos) (storage:lseq bf16 (nr * 8)) (logical:chest2 real nr 8)
  : Lemma
      (requires row_rep 8 nr storage (EM.mtranspose logical))
      (ensures from_seq (col_major nr 8) storage %~ logical) =
  EM.lemma_approximates_intro (from_seq (col_major nr 8) storage) logical

inline_for_extraction noextract
let prepared_matrix_family (#d:Type0) (rows cols:pos)
  (s:lseq bf16 (pad16 rows * pad16 cols)) (w:matrix_family d rows cols)
  : Lemma
      (requires forall (c:d). row_rep (pad16 rows) (pad16 cols) s (padded_matrix rows cols (w c)))
      (ensures forall (c:d).
        from_seq (rm_layout (pad16 rows) (pad16 cols)) s %~ padded_matrix rows cols (w c)) =
  let point (c:d)
    : Lemma (from_seq (rm_layout (pad16 rows) (pad16 cols)) s %~
      padded_matrix rows cols (w c)) =
    row_from_seq_approx (pad16 rows) (pad16 cols) s (padded_matrix rows cols (w c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let prepared_input_family (#d:Type0) (cols:pos)
  (s:lseq bf16 (pad16 cols * 8)) (x:vector_family d cols)
  : Lemma
      (requires forall (c:d). row_rep 8 (pad16 cols) s (EM.mtranspose (padded_input cols (x c))))
      (ensures forall (c:d).
        from_seq (col_major (pad16 cols) 8) s %~ padded_input cols (x c)) =
  let point (c:d)
    : Lemma (from_seq (col_major (pad16 cols) 8) s %~ padded_input cols (x c)) =
    col_from_seq_approx (pad16 cols) s (padded_input cols (x c))
  in FStar.Classical.forall_intro point

fn matvec
  (#d:Type0)
  (rows:szp{rows <= 6144}) (cols:szp{cols <= 3584}) (count:szp)
  (source:A.larray bf16 count{GA.is_global_array source})
  (offset rs:sz) (cs:sz{offset + (rows - 1) * rs + (cols - 1) * cs < count})
  (input:A.larray bf16 cols{GA.is_global_array input})
  (output:A.larray bf16 rows{GA.is_global_array output})
  (split:bool{split ==> cols % 32 == 0})
  (#fw #fx:perm)
  (#sw #sx:erased (seq bf16))
  (#w:erased (matrix_family d rows cols)) (#x:erased (vector_family d cols))
  preserves cpu **
    on gpu_loc (source |-> Frac fw sw ** input |-> Frac fx sx)
  requires on gpu_loc (exists* (before:seq bf16). output |-> before)
  requires pure (umatrix rows cols count offset rs cs sw w /\ ubf cols sx x)
  ensures exists* (result:seq bf16).
    on gpu_loc (output |-> result) ** pure (Seq.length result == rows /\
      ubf rows result (fun c -> S.project ((reveal #(matrix_family d rows cols) w) c) ((reveal #(vector_family d cols) x) c)))
{
  with output_before. assert on gpu_loc (output |-> output_before);
  pad16_bounds rows;
  pad16_bounds cols;
  let nr = ((rows +^ 15sz) /^ 16sz) *^ 16sz;
  let nc = ((cols +^ 15sz) /^ 16sz) *^ 16sz;
  assert pure (SZ.v nr == pad16 rows /\ SZ.v nc == pad16 cols);

  let raw_a = GA.gpu_array_alloc #bf16 (nr *^ nc);
  let ar : (p:A.larray bf16 (pad16 rows * pad16 cols){GA.is_global_array p}) = raw_a;
  assert rewrites_to ar raw_a;
  with a0. assert on gpu_loc (ar |-> a0);
  let initial_a : erased (seq bf16) = hide (reveal a0 <: seq bf16);
  prepare_matrix rows cols count source offset rs cs ar #fw #sw #initial_a #w;
  with sa. assert on gpu_loc (ar |-> sa);
  prepared_matrix_family rows cols sa w;
  let a = from_array (rm_layout nr nc) ar;
  assert rewrites_to a (from_array (rm_layout nr nc) ar);
  map_loc gpu_loc #(ar |-> sa) #(a |-> from_seq (rm_layout nr nc) sa)
    fn _ { tensor_abs' (rm_layout nr nc) ar; };

  let raw_b = GA.gpu_array_alloc #bf16 (nc *^ 8sz);
  let br : (p:A.larray bf16 (pad16 cols * 8){GA.is_global_array p}) = raw_b;
  assert rewrites_to br raw_b;
  with b0. assert on gpu_loc (br |-> b0);
  let initial_b : erased (seq bf16) = hide (reveal b0 <: seq bf16);
  launch_kernel_1 (fun () -> prepare_input cols input br #fx #sx #initial_b #x);
  with sb. assert on gpu_loc (br |-> sb);
  prepared_input_family cols sb x;
  let b = from_array (col_major nc 8) br;
  assert rewrites_to b (from_array (col_major nc 8) br);
  map_loc gpu_loc #(br |-> sb) #(b |-> from_seq (col_major nc 8) sb)
    fn _ { tensor_abs' (col_major nc 8) br; };

  let left = Kuiper.Tensor.alloc0 #f32 (nr *^ 8sz) (rm_layout nr 8);
  let right = Kuiper.Tensor.alloc0 #f32 (nr *^ 8sz) (rm_layout nr 8);
  let middle : sz = (if split then nc /^ 32sz else 0sz);
  with la. assert on gpu_loc (left |-> la);
  run_tensor nr nc a b left 0sz middle #_ #_ #la;
  with rb. assert on gpu_loc (right |-> rb);
  run_tensor nr nc a b right middle (nc /^ 16sz) #_ #_ #rb;
  launch_kernel_1 (fun () -> finish_matvec rows cols middle left right output
    #(from_seq (rm_layout nr nc) sa) #(from_seq (col_major nc 8) sb) #w #x
    #(hide (reveal output_before <: seq bf16)));
  Kuiper.Tensor.free left;
  Kuiper.Tensor.free right;
  Kuiper.Tensor.free a;
  Kuiper.Tensor.free b;
}

inline_for_extraction noextract
fn allocate_bf (n:szp)
  preserves cpu
  returns result:(p:A.larray bf16 n{GA.is_global_array p /\ A.is_full_array p})
  ensures exists* (values:seq bf16).
    on gpu_loc (result |-> values) ** pure (Seq.length values == n)
{
  let a = GA.gpu_array_alloc #bf16 n;
  a
}

inline_for_extraction noextract
let segment_rep (#d:Type0) (n offset:nat) (width:nat{offset + width <= n})
  (s:seq bf16) (x:vector_family d width) : GTot prop =
  if Seq.length s = n then
    forall (c:d) (i:S.fin width).
      v_approximates (Seq.index s (offset + i)) (x c i)
  else False

inline_for_extraction noextract
fn read_segment (#d:Type0)
  (n width:szp) (source:A.larray bf16 n)
  (offset:sz{offset + width <= n}) (i:sz{i < width})
  (#f:perm) (#storage:erased (seq bf16)) (#x:erased (vector_family d width))
  preserves gpu ** source |-> Frac f storage
  requires pure (segment_rep n offset width storage x)
  returns result:bf16
  ensures pure (uba result (fun c -> (reveal #(vector_family d width) x) c i))
{
  source.(offset +^ i)
}

inline_for_extraction noextract
let partial_sum (n:nat{n <= 256}) (x:S.vector 256) : GTot real =
  sum_n n (fun i -> x i)

inline_for_extraction noextract
let partial_rep (#d:Type0) (n:nat{n <= 256})
  (s:seq f32) (x:vector_family d 256) : GTot prop =
  if Seq.length s = 256 then
    forall (c:d) (i:S.fin n). v_approximates (Seq.index s i) (x c i)
  else False

inline_for_extraction noextract
let partial_at (#d:Type0) (n:nat{n <= 256})
  (s:lseq f32 256) (x:vector_family d 256) (i:nat{i < n})
  : Lemma (requires partial_rep n s x)
      (ensures ua (Seq.index s i) (fun c -> x c i)) =
  let point (c:d) : Lemma (v_approximates (Seq.index s i) (x c i)) = ()
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let tree_width (n:nat) : prop =
  n = 1 \/ n = 2 \/ n = 4 \/ n = 8 \/ n = 16 \/ n = 32 \/
  n = 64 \/ n = 128 \/ n = 256

inline_for_extraction noextract
let merge_partial (half:pos{half <= 128}) (x:S.vector 256) : S.vector 256 =
  fun i -> if i < half then x i +. x (i + half) else x i

inline_for_extraction noextract
let merge_sum_family (#d:Type0) (half:pos{half <= 128}) (x:vector_family d 256)
  : Lemma (forall (c:d).
      partial_sum half (merge_partial half (x c)) == partial_sum (2 * half) (x c)) =
  let point (c:d) : Lemma (
      partial_sum half (merge_partial half (x c)) == partial_sum (2 * half) (x c)) =
    norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
      (partial_sum half (merge_partial half (x c)));
    norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
      (partial_sum (2 * half) (x c));
    sum_halves half (fun i -> x c i);
    sum_ext half (fun i -> merge_partial half (x c) i)
      (fun i -> x c i +. x c (i + half))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let partial_one_family (#d:Type0) (x:vector_family d 256)
  : Lemma (forall (c:d). partial_sum 1 (x c) == x c 0) =
  let point (c:d) : Lemma (partial_sum 1 (x c) == x c 0) =
    norm_spec [delta_once ["KPlay.Functional.partial_sum"]] (partial_sum 1 (x c));
    sum_step 1 (fun i -> x c i); sum_zero (fun i -> x c i)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn halve_partial (#d:Type0)
  (partial:A.larray f32 256) (half:szp{half <= 128})
  (#storage:erased (seq f32)) (#logical:erased (vector_family d 256))
  preserves gpu
  requires partial |-> storage ** pure (partial_rep (2 * half) storage logical)
  ensures exists* (next:seq f32). partial |-> next **
    pure (Seq.length next == 256 /\
      partial_rep half next (fun c -> merge_partial half ((reveal #(vector_family d 256) logical) c)))
{
  A.pts_to_len partial;
  let mut i = 0sz;
  while (!i <^ half)
    invariant exists* (j:sz{j <= half}) (values:seq f32).
      i |-> j ** partial |-> values **
      pure (Seq.length values == 256 /\ (forall (ctx:d) (k:nat{k < j}).
          v_approximates (Seq.index values k) (merge_partial half ((reveal #(vector_family d 256) logical) ctx) k)) /\
        (forall (k:nat{j <= k /\ k < 256}). Seq.index values k == Seq.index storage k))
    decreases (half - !i)
  {
    let j = !i;
    let x = partial.(j);
    let y = partial.(j +^ half);
    partial_at (2 * half) storage logical j;
    partial_at (2 * half) storage logical (j + half);
    let next = rounded_add x y #(fun c -> (reveal #(vector_family d 256) logical) c j) #(fun c -> (reveal #(vector_family d 256) logical) c (j + half));
    partial.(j) <- next;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn tree_reduce (#d:Type0)
  (partial:A.larray f32 256) (width:szp{tree_width width})
  (#storage:erased (seq f32)) (#logical:erased (vector_family d 256))
  preserves gpu
  requires partial |-> storage ** pure (partial_rep width storage logical)
  returns result:f32
  ensures exists* (next:seq f32). partial |-> next **
    pure (Seq.length next == 256 /\ ua result (fun c -> partial_sum width ((reveal #(vector_family d 256) logical) c)))
{
  A.pts_to_len partial;
  let mut active = width;
  let mut current:erased (vector_family d 256) = logical;
  while (!active >^ 1sz)
    invariant exists* (n:szp{tree_width n /\ n <= width})
      (r:erased (vector_family d 256)) (values:seq f32).
      active |-> n ** current |-> r ** partial |-> values **
      pure (Seq.length values == 256 /\ partial_rep n values r /\
        (forall (c:d). partial_sum n ((reveal #(vector_family d 256) r) c) == partial_sum width ((reveal #(vector_family d 256) logical) c)))
    decreases (SZ.v (!active))
  {
    let n = !active;
    let r = !current;
    assert pure (n % 2 == 0 /\ tree_width (n / 2));
    let half = n /^ 2sz;
    let next:erased (vector_family d 256) = hide (fun c -> merge_partial half ((reveal #(vector_family d 256) r) c));
    with values. assert partial |-> values;
    halve_partial partial half #(hide (reveal values <: seq f32)) #r;
    merge_sum_family half r;
    current := next;
    active := half;
  };
  let r = !current;
  partial_one_family r;
  partial.(0sz)
}

inline_for_extraction noextract
let square_terms (width chunk:pos) (group:nat{(group + 1) * chunk <= width})
  (x:S.vector width) : S.vector chunk =
  fun j -> x (chunk * group + j) *. x (chunk * group + j)

inline_for_extraction noextract
let squares_step_family (#d:Type0) (width chunk:pos)
  (group:nat{(group + 1) * chunk <= width}) (k:nat{k < chunk})
  (x:vector_family d width)
  : Lemma (forall (c:d).
      bounded_sum (k + 1) (square_terms width chunk group (x c)) ==
      bounded_sum k (square_terms width chunk group (x c)) +.
        x c (chunk * group + k) *. x c (chunk * group + k)) =
  let point (c:d) : Lemma (
      bounded_sum (k + 1) (square_terms width chunk group (x c)) ==
      bounded_sum k (square_terms width chunk group (x c)) +.
        x c (chunk * group + k) *. x c (chunk * group + k)) =
    bounded_sum_step k (square_terms width chunk group (x c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn square_group (#d:Type0)
  (n width:szp) (chunk:szp{SZ.v chunk = 4 \/ SZ.v chunk = 8})
  (input:A.larray bf16 n) (offset:sz{offset + width <= n})
  (group:sz{(group + 1) * chunk <= width}) (rounded:bool)
  (#fraction:perm) (#storage:erased (seq bf16))
  (#logical:erased (vector_family d width))
  preserves gpu ** input |-> Frac fraction storage
  requires pure (segment_rep n offset width storage logical)
  returns result:f32
  ensures pure (ua result (fun c -> sum_n chunk (square_terms width chunk group ((reveal #(vector_family d width) logical) c))))
{
  let mut j = 0sz;
  let mut accum:f32 = zero;
  while (!j <^ chunk)
    invariant exists* (k:sz{k <= chunk}) (acc:f32).
      j |-> k ** accum |-> acc **
      pure (ua acc (fun c -> bounded_sum k (square_terms width chunk group ((reveal #(vector_family d width) logical) c))))
    decreases (chunk - !j)
  {
    let k = !j;
    let index = offset +^ chunk *^ group +^ k;
    let value = input.(index);
    widen_family value (fun c -> (reveal #(vector_family d width) logical) c (chunk * group + k));
    let wide = C.fcast #bf16 #f32 value;
    let squared = rounded_mul wide wide
      #(fun c -> (reveal #(vector_family d width) logical) c (chunk * group + k))
      #(fun c -> (reveal #(vector_family d width) logical) c (chunk * group + k));
    let term = (if rounded then round_bf squared
      #(fun c -> (reveal #(vector_family d width) logical) c (chunk * group + k) *. (reveal #(vector_family d width) logical) c (chunk * group + k))
      else squared);
    let acc = !accum;
    let next = rounded_add acc term
      #(fun c -> bounded_sum k (square_terms width chunk group ((reveal #(vector_family d width) logical) c)))
      #(fun c -> (reveal #(vector_family d width) logical) c (chunk * group + k) *. (reveal #(vector_family d width) logical) c (chunk * group + k));
    squares_step_family width chunk group k logical;
    accum := next;
    j := k +^ 1sz;
  };
  bounded_full_family chunk (fun c -> square_terms width chunk group ((reveal #(vector_family d width) logical) c));
  !accum
}

inline_for_extraction noextract
let grouped_squares (width:pos) (chunk:pos{width % chunk == 0 /\ width / chunk <= 256})
  (x:S.vector width) : S.vector 256 =
  fun i -> if i < width / chunk then sum_n chunk (square_terms width chunk i x) else 0.0R

inline_for_extraction noextract
let grouped_squares_family (#d:Type0)
  (width:pos) (chunk:pos{width % chunk == 0 /\ width / chunk <= 256})
  (x:vector_family d width)
  : Lemma (forall (c:d).
      partial_sum (width / chunk) (grouped_squares width chunk (x c)) == S.sum_squares (x c)) =
  let point (c:d) : Lemma (
      partial_sum (width / chunk) (grouped_squares width chunk (x c)) == S.sum_squares (x c)) =
    let group (g:S.fin (width / chunk))
      : Lemma (grouped_squares width chunk (x c) g ==
        sum_n chunk (fun j -> x c (chunk * g + j) *. x c (chunk * g + j))) =
      norm_spec [delta_once ["KPlay.Functional.grouped_squares"]]
        (grouped_squares width chunk (x c) g);
      sum_ext chunk (square_terms width chunk g (x c))
        (fun j -> x c (chunk * g + j) *. x c (chunk * g + j))
    in
    FStar.Classical.forall_intro group;
    norm_spec [delta_once ["KPlay.Functional.partial_sum"]]
      (partial_sum (width / chunk) (grouped_squares width chunk (x c)));
    sum_ext (width / chunk)
      (fun g -> grouped_squares width chunk (x c) g)
      (fun g -> sum_n chunk (fun j -> x c (chunk * g + j) *. x c (chunk * g + j)));
    sum_groups (width / chunk) chunk (fun i -> x c i *. x c i);
    assert (chunk * (width / chunk) == width);
    sum_ext width
      (fun (i:S.fin (chunk * (width / chunk))) -> x c (i <: nat) *. x c (i <: nat))
      (fun i -> x c i *. x c i);
    sum_squares_bridge width (x c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let scale_model (l2:bool) (width:pos) (x:S.vector width) : GTot real =
  if l2 then R.rsqrt (S.sum_squares x +. S.epsilon) else S.rms_scale x

inline_for_extraction noextract
fn finish_scale (#d:Type0)
  (width:szp{SZ.v width = 128 \/ SZ.v width = 1024}) (l2:bool) (sum_value:f32)
  (#logical:erased (vector_family d width))
  preserves gpu
  requires pure (ua sum_value (fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c)))
  returns result:f32
  ensures pure (ua result (fun c -> scale_model l2 width ((reveal #(vector_family d width) logical) c)))
{
  let epsilon = epsilon32 ();
  if l2 {
    let rounded = round_bf sum_value #(fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c));
    let shifted = rounded_add rounded epsilon
      #(fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c)) #(fun _ -> S.epsilon);
    let shifted_bf = round_bf shifted #(fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c) +. S.epsilon);
    rsqrt_family shifted_bf (fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c) +. S.epsilon);
    let inverse = rsqrt shifted_bf;
    round_bf inverse #(fun c -> R.rsqrt (S.sum_squares ((reveal #(vector_family d width) logical) c) +. S.epsilon))
  } else {
    let reciprocal_width:f32 = (if width = 128sz then reciprocal 128L else reciprocal 1024L);
    let mean = rounded_mul sum_value reciprocal_width
      #(fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c)) #(fun _ -> 1.0R /. FStar.Real.of_int width);
    reciprocal_family mean width (fun c -> (S.sum_squares ((reveal #(vector_family d width) logical) c) <: real));
    let shifted = rounded_add mean epsilon
      #(fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c) /. FStar.Real.of_int width) #(fun _ -> S.epsilon);
    rsqrt_family shifted (fun c -> S.sum_squares ((reveal #(vector_family d width) logical) c) /. FStar.Real.of_int width +. S.epsilon);
    rsqrt shifted
  }
}

inline_for_extraction noextract
fn norm_scale (#d:Type0)
  (n:szp) (width:szp{SZ.v width = 128 \/ SZ.v width = 1024})
  (input:A.larray bf16 n) (offset:sz{offset + width <= n})
  (partial:A.larray f32 256) (l2:bool{l2 ==> SZ.v width = 128})
  (#fraction:perm) (#storage:erased (seq bf16)) (#before:erased (seq f32))
  (#logical:erased (vector_family d width))
  preserves gpu ** input |-> Frac fraction storage
  requires partial |-> before ** pure (segment_rep n offset width storage logical)
  returns result:f32
  ensures exists* (next:seq f32). partial |-> next **
    pure (Seq.length next == 256 /\ ua result (fun c -> scale_model l2 width ((reveal #(vector_family d width) logical) c)))
{
  A.pts_to_len partial;
  let chunk:szp = (if l2 then 8sz else 4sz);
  let groups = width /^ chunk;
  let mut i = 0sz;
  while (!i <^ groups)
    invariant exists* (j:sz{j <= groups}) (values:seq f32).
      i |-> j ** partial |-> values **
      pure (Seq.length values == 256 /\
        partial_rep j values (fun c -> grouped_squares width chunk ((reveal #(vector_family d width) logical) c)))
    decreases (groups - !i)
  {
    let j = !i;
    let group = square_group n width chunk input offset j l2 #fraction #storage #logical;
    partial.(j) <- group;
    i := j +^ 1sz;
  };
  with values. assert partial |-> values;
  let sum_value = tree_reduce partial groups #(hide (reveal values <: seq f32))
    #(fun c -> grouped_squares width chunk ((reveal #(vector_family d width) logical) c));
  grouped_squares_family width chunk logical;
  finish_scale width l2 sum_value #logical
}

inline_for_extraction noextract
fn rms_kernel (#d:Type0)
  (input:A.larray bf16 1024) (weight:A.larray bf16 weight_count)
  (offset:sz{offset + 1024 <= weight_count})
  (output:A.larray bf16 1024) (partial:A.larray f32 256)
  (#fi #fw:perm) (#si #sw:erased (seq bf16)) (#sp:erased (seq f32))
  (#xr #wr:erased (vector_family d 1024))
  (#before:erased (seq bf16))
  preserves gpu ** input |-> Frac fi si ** weight |-> Frac fw sw
  requires partial |-> sp ** output |-> before
  requires pure (ubf 1024 si xr /\ segment_rep weight_count offset 1024 sw wr)
  ensures exists* (so:seq bf16) (next:seq f32).
    output |-> so ** partial |-> next **
    pure (Seq.length so == 1024 /\ Seq.length next == 256 /\
      ubf 1024 so (fun c -> S.centered_rms_norm ((reveal #(vector_family d 1024) wr) c) ((reveal #(vector_family d 1024) xr) c)))
{
  let scale = norm_scale 1024sz 1024sz input 0sz partial false #fi #si #sp #xr;
  A.pts_to_len output;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (j:sz{j <= 1024}) (values:seq bf16).
      i |-> j ** output |-> values **
      pure (Seq.length values == 1024 /\ (forall (c:d) (k:nat{k < j}).
        v_approximates (Seq.index values k) (S.centered_rms_norm ((reveal #(vector_family d 1024) wr) c) ((reveal #(vector_family d 1024) xr) c) k)))
    decreases (1024 - !i)
  {
    let j = !i;
    let x = input.(j);
    let w = weight.(offset +^ j);
    widen_family x (fun c -> (reveal #(vector_family d 1024) xr) c j);
    widen_family w (fun c -> (reveal #(vector_family d 1024) wr) c j);
    let scaled = rounded_mul (C.fcast #bf16 #f32 x) scale
      #(fun c -> (reveal #(vector_family d 1024) xr) c j) #(fun c -> S.rms_scale ((reveal #(vector_family d 1024) xr) c));
    let scale_weight = rounded_add (one #f32) (C.fcast #bf16 #f32 w)
      #(fun _ -> 1.0R) #(fun c -> (reveal #(vector_family d 1024) wr) c j);
    let y = rounded_mul scaled scale_weight
      #(fun c -> (reveal #(vector_family d 1024) xr) c j *. S.rms_scale ((reveal #(vector_family d 1024) xr) c)) #(fun c -> 1.0R +. (reveal #(vector_family d 1024) wr) c j);
    narrow_family y (fun c -> S.centered_rms_norm ((reveal #(vector_family d 1024) wr) c) ((reveal #(vector_family d 1024) xr) c) j);
    output.(j) <- C.fcast #f32 #bf16 y;
    i := j +^ 1sz;
  }
}

fn execute_rms (#d:Type0)
  (input:A.larray bf16 1024{GA.is_global_array input})
  (weight:A.larray bf16 weight_count{GA.is_global_array weight})
  (offset:sz{offset + 1024 <= weight_count})
  (output:A.larray bf16 1024{GA.is_global_array output})
  (#fi #fw:perm) (#si #sw:erased (seq bf16))
  (#xr #wr:erased (vector_family d 1024))
  preserves cpu ** on gpu_loc (input |-> Frac fi si ** weight |-> Frac fw sw)
  requires on gpu_loc (exists* (before:seq bf16). output |-> before)
  requires pure (ubf 1024 si xr /\ segment_rep weight_count offset 1024 sw wr)
  ensures exists* (so:seq bf16). on gpu_loc (output |-> so) **
    pure (Seq.length so == 1024 /\ ubf 1024 so (fun c -> S.centered_rms_norm ((reveal #(vector_family d 1024) wr) c) ((reveal #(vector_family d 1024) xr) c)))
{
  with output_before. assert on gpu_loc (output |-> output_before);
  let partial = GA.gpu_array_alloc #f32 256sz;
  with sp. assert on gpu_loc (partial |-> sp);
  launch_kernel_1 (fun () -> rms_kernel input weight offset output partial
    #fi #fw #si #sw #(hide (reveal sp <: seq f32)) #xr #wr
    #(hide (reveal output_before <: seq bf16)));
  GA.gpu_array_free partial;
}

inline_for_extraction noextract
fn pointwise_value (#d:Type0) (product:bool) (a b:bf16)
  (#ar #br:erased (real_family d))
  preserves gpu
  requires pure (uba a ar /\ uba b br)
  returns result:bf16
  ensures pure (uba result (fun c ->
    if product then S.silu ((reveal #(real_family d) ar) c) *. (reveal #(real_family d) br) c
    else (reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c))
{
  widen_family b br;
  if product {
    let activated = activation32 true a #ar;
    let gate = round_bf activated #(fun c -> S.silu ((reveal #(real_family d) ar) c));
    let value = rounded_mul gate (C.fcast #bf16 #f32 b)
      #(fun c -> S.silu ((reveal #(real_family d) ar) c)) #br;
    narrow_family value
      (fun c -> S.silu ((reveal #(real_family d) ar) c) *. (reveal #(real_family d) br) c);
    C.fcast #f32 #bf16 value
  } else {
    widen_family a ar;
    let value = rounded_add (C.fcast #bf16 #f32 a) (C.fcast #bf16 #f32 b) #ar #br;
    narrow_family value
      (fun c -> (reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c);
    C.fcast #f32 #bf16 value
  }
}

inline_for_extraction noextract
fn pointwise_kernel (#d:Type0) (n:szp{n <= 3584}) (product:bool)
  (left right output:A.larray bf16 n)
  (#fl #fr:perm) (#sl #sr:erased (seq bf16))
  (#xl #xr:erased (vector_family d n))
  (#before:erased (seq bf16))
  preserves gpu ** left |-> Frac fl sl ** right |-> Frac fr sr
  requires output |-> before ** pure (ubf n sl xl /\ ubf n sr xr)
  ensures exists* (result:seq bf16). output |-> result **
    pure (Seq.length result == n /\ ubf n result
      (fun c i -> if product then S.silu ((reveal #(vector_family d n) xl) c i) *. (reveal #(vector_family d n) xr) c i else (reveal #(vector_family d n) xl) c i +. (reveal #(vector_family d n) xr) c i))
{
  A.pts_to_len output;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j:sz{j <= n}) (values:seq bf16).
      i |-> j ** output |-> values **
      pure (Seq.length values == n /\ (forall (c:d) (k:nat{k < j}).
        v_approximates (Seq.index values k)
          (if product then S.silu ((reveal #(vector_family d n) xl) c k) *. (reveal #(vector_family d n) xr) c k else (reveal #(vector_family d n) xl) c k +. (reveal #(vector_family d n) xr) c k)))
    decreases (n - !i)
  {
    let j = !i;
    let a = left.(j);
    let b = right.(j);
    let value = pointwise_value product a b
      #(fun c -> (reveal #(vector_family d n) xl) c j)
      #(fun c -> (reveal #(vector_family d n) xr) c j);
    output.(j) <- value;
    i := j +^ 1sz;
  }
}

fn pointwise (#d:Type0) (n:szp{n <= 3584}) (product:bool)
  (left:A.larray bf16 n{GA.is_global_array left})
  (right:A.larray bf16 n{GA.is_global_array right})
  (output:A.larray bf16 n{GA.is_global_array output})
  (#fl #fr:perm) (#sl #sr:erased (seq bf16))
  (#xl #xr:erased (vector_family d n))
  preserves cpu ** on gpu_loc (left |-> Frac fl sl ** right |-> Frac fr sr)
  requires on gpu_loc (exists* (before:seq bf16). output |-> before) ** pure (ubf n sl xl /\ ubf n sr xr)
  ensures exists* (result:seq bf16). on gpu_loc (output |-> result) **
    pure (Seq.length result == n /\ ubf n result
      (fun c i -> if product then S.silu ((reveal #(vector_family d n) xl) c i) *. (reveal #(vector_family d n) xr) c i else (reveal #(vector_family d n) xl) c i +. (reveal #(vector_family d n) xr) c i))
{
  with output_before. assert on gpu_loc (output |-> output_before);
  launch_kernel_1 (fun () -> pointwise_kernel n product left right output #fl #fr #sl #sr #xl #xr
    #(hide (reveal output_before <: seq bf16)));
}

noextract
noeq type head_model = {
  qinput:S.vector 128;
  kinput:S.vector 128;
  value:S.vector 128;
  z:S.vector 128;
  weight:S.vector 128;
  old:S.matrix 128 128;
  b:real;
  a:real;
  a_log:real;
  dt_bias:real;
  query:S.vector 128;
  key:S.vector 128;
  beta:real;
  decay:real;
  decayed:S.matrix 128 128;
  delta:S.vector 128;
  updated:S.matrix 128 128;
  core:S.vector 128;
  gated:S.vector 128;
}

[@@"opaque_to_smt"]
inline_for_extraction noextract
let valid_head (p:head_model) : GTot prop =
  (forall (i:S.fin 128). p.query i == S.l2_normalize p.qinput i *. R.rsqrt 128.0R) /\
  p.key == S.l2_normalize p.kinput /\
  p.beta == S.sigmoid p.b /\
  p.decay == E.exp ((0.0R -. E.exp p.a_log) *. S.softplus (p.a +. p.dt_bias)) /\
  (forall (i j:S.fin 128). p.decayed i j == p.old i j *. p.decay) /\
  (forall (j:S.fin 128). p.delta j ==
    (p.value j -. sum_n 128 (fun i -> p.decayed i j *. p.key i)) *. p.beta) /\
  (forall (i j:S.fin 128). p.updated i j == p.decayed i j +. p.key i *. p.delta j) /\
  (forall (j:S.fin 128). p.core j == sum_n 128 (fun i -> p.updated i j *. p.query i)) /\
  (forall (i:S.fin 128). p.gated i ==
    (p.weight i *. (p.core i *. S.rms_scale p.core)) *. S.silu (p.z i))

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
[@@"opaque_to_smt"]
noextract
let make_head (w:S.weights) (qkv:S.vector 6144) (z:S.vector 2048)
  (b a:S.vector 16) (old:S.recurrent_state) (h:S.fin 16)
  : GTot (p:head_model{
      valid_head p /\
      p.qinput == (fun (i:S.fin 128) -> qkv (h * 128 + i)) /\
      p.kinput == (fun (i:S.fin 128) -> qkv (2048 + h * 128 + i)) /\
      p.value == (fun (i:S.fin 128) -> qkv (4096 + h * 128 + i)) /\
      p.z == (fun (i:S.fin 128) -> z (h * 128 + i)) /\
      p.weight == w.gated_norm /\ p.old == old h /\
      p.b == b h /\ p.a == a h /\ p.a_log == w.a_log h /\ p.dt_bias == w.dt_bias h}) =
  let qinput:S.vector 128 = fun i -> qkv (h * 128 + i) in
  let kinput:S.vector 128 = fun i -> qkv (2048 + h * 128 + i) in
  let value:S.vector 128 = fun i -> qkv (4096 + h * 128 + i) in
  let query:S.vector 128 = fun i -> S.l2_normalize qinput i *. R.rsqrt 128.0R in
  let key = S.l2_normalize kinput in
  let beta = S.sigmoid (b h) in
  let decay = E.exp ((0.0R -. E.exp (w.a_log h)) *. S.softplus (a h +. w.dt_bias h)) in
  let decayed:S.matrix 128 128 = fun i j -> old h i j *. decay in
  let delta (j:S.fin 128)
    : GTot (r:real{r == (value j -. sum_n 128 (fun i -> decayed i j *. key i)) *. beta}) =
    (value j -. sum_n 128 (fun i -> decayed i j *. key i)) *. beta in
  let updated:S.matrix 128 128 = fun i j -> decayed i j +. key i *. delta j in
  let core (j:S.fin 128)
    : GTot (r:real{r == sum_n 128 (fun i -> updated i j *. query i)}) =
    sum_n 128 (fun i -> updated i j *. query i) in
  let gated:S.vector 128 = fun i ->
    (w.gated_norm i *. (core i *. S.rms_scale core)) *. S.silu (z (h * 128 + i)) in
  let p = {qinput; kinput; value; z=(fun (i:S.fin 128) -> z (h * 128 + i)); weight=w.gated_norm;
   old=old h; b=b h; a=a h; a_log=w.a_log h; dt_bias=w.dt_bias h;
   query; key; beta; decay; decayed; delta; updated; core; gated} in
  let delta_point (j:S.fin 128)
    : Lemma (p.delta j ==
      (p.value j -. sum_n 128 (fun i -> p.decayed i j *. p.key i)) *. p.beta) =
    sum_ext 128 (fun i -> p.decayed i j *. p.key i) (fun i -> decayed i j *. key i)
  in
  FStar.Classical.forall_intro delta_point;
  let core_point (j:S.fin 128)
    : Lemma (p.core j == sum_n 128 (fun i -> p.updated i j *. p.query i)) =
    sum_ext 128 (fun i -> p.updated i j *. p.query i) (fun i -> updated i j *. query i)
  in
  FStar.Classical.forall_intro core_point;
  norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head p);
  p
#pop-options

inline_for_extraction noextract
let head_normalization_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d) (i:S.fin 128).
        (p c).query i == S.l2_normalize (p c).qinput i *. R.rsqrt 128.0R /\
        (p c).key i == S.l2_normalize (p c).kinput i) =
  let point (c:d) : Lemma (forall (i:S.fin 128).
      (p c).query i == S.l2_normalize (p c).qinput i *. R.rsqrt 128.0R /\
      (p c).key i == S.l2_normalize (p c).kinput i) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_parameter_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d).
        (p c).beta == S.sigmoid (p c).b /\
        (p c).decay == E.exp ((0.0R -. E.exp (p c).a_log) *.
          S.softplus ((p c).a +. (p c).dt_bias))) =
  let point (c:d) : Lemma (
      (p c).beta == S.sigmoid (p c).b /\
      (p c).decay == E.exp ((0.0R -. E.exp (p c).a_log) *.
        S.softplus ((p c).a +. (p c).dt_bias))) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_decay_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d) (i j:S.fin 128).
        (p c).decayed i j == (p c).old i j *. (p c).decay) =
  let point (c:d) : Lemma (forall (i j:S.fin 128).
      (p c).decayed i j == (p c).old i j *. (p c).decay) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_update_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d) (i j:S.fin 128).
        (p c).updated i j == (p c).decayed i j +. (p c).key i *. (p c).delta j) =
  let point (c:d) : Lemma (forall (i j:S.fin 128).
      (p c).updated i j == (p c).decayed i j +. (p c).key i *. (p c).delta j) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_dot_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d) (j:S.fin 128).
        (p c).delta j ==
          ((p c).value j -. sum_n 128 (fun i -> (p c).decayed i j *. (p c).key i)) *. (p c).beta /\
        (p c).core j == sum_n 128 (fun i -> (p c).updated i j *. (p c).query i)) =
  let point (c:d) : Lemma (forall (j:S.fin 128).
      (p c).delta j ==
        ((p c).value j -. sum_n 128 (fun i -> (p c).decayed i j *. (p c).key i)) *. (p c).beta /\
      (p c).core j == sum_n 128 (fun i -> (p c).updated i j *. (p c).query i)) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_gating_laws (#d:Type0) (p:d -> GTot head_model)
  : Lemma
      (requires forall (c:d). valid_head (p c))
      (ensures forall (c:d) (i:S.fin 128).
        (p c).gated i ==
          ((p c).weight i *. ((p c).core i *. S.rms_scale (p c).core)) *. S.silu ((p c).z i)) =
  let point (c:d) : Lemma (forall (i:S.fin 128).
      (p c).gated i ==
        ((p c).weight i *. ((p c).core i *. S.rms_scale (p c).core)) *. S.silu ((p c).z i)) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_delta_result (#d:Type0) (p:d -> GTot head_model) (j:S.fin 128) (v:f32)
  : Lemma
      (requires
        (forall (c:d). valid_head (p c)) /\
        ua v (fun c -> ((p c).value j -.
          sum_n 128 (fun i -> (p c).decayed i j *. (p c).key i)) *. (p c).beta))
      (ensures ua v (fun c -> (p c).delta j)) =
  let point (c:d) : Lemma (v_approximates v ((p c).delta j)) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c));
    assert ((p c).delta j == ((p c).value j -.
      sum_n 128 (fun i -> (p c).decayed i j *. (p c).key i)) *. (p c).beta)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_core_result (#d:Type0) (p:d -> GTot head_model) (j:S.fin 128) (v:f32)
  : Lemma
      (requires
        (forall (c:d). valid_head (p c)) /\
        ua v (fun c -> sum_n 128 (fun i -> (p c).updated i j *. (p c).query i)))
      (ensures ua v (fun c -> (p c).core j)) =
  let point (c:d) : Lemma (v_approximates v ((p c).core j)) =
    norm_spec [delta_once ["KPlay.Functional.valid_head"]] (valid_head (p c));
    assert ((p c).core j == sum_n 128 (fun i -> (p c).updated i j *. (p c).query i))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn compute_delta (#d:Type0) (column:sz{column < 128}) (value:bf16) (memory beta:f32)
  (#p:erased (d -> GTot head_model))
  preserves gpu
  requires pure (
    (forall (c:d). valid_head ((reveal p) c)) /\
    uba value (fun c -> ((reveal p) c).value column) /\
    ua memory (fun c -> sum_n 128 (fun i -> ((reveal p) c).decayed i column *. ((reveal p) c).key i)) /\
    ua beta (fun c -> ((reveal p) c).beta))
  returns result:f32
  ensures pure (ua result (fun c -> ((reveal p) c).delta column))
{
  widen_family value (fun c -> ((reveal p) c).value column);
  let difference = rounded_sub (C.fcast #bf16 #f32 value) memory
    #(fun c -> ((reveal p) c).value column)
    #(fun c -> sum_n 128 (fun i -> ((reveal p) c).decayed i column *. ((reveal p) c).key i));
  let result = rounded_mul difference beta
    #(fun c -> ((reveal p) c).value column -.
      sum_n 128 (fun i -> ((reveal p) c).decayed i column *. ((reveal p) c).key i))
    #(fun c -> ((reveal p) c).beta);
  head_delta_result p column result;
  result
}

inline_for_extraction noextract
fn round_readout (#d:Type0) (column:sz{column < 128}) (value:f32)
  (#p:erased (d -> GTot head_model))
  preserves gpu
  requires pure (
    (forall (c:d). valid_head ((reveal p) c)) /\
    ua value (fun c -> sum_n 128 (fun i -> ((reveal p) c).updated i column *. ((reveal p) c).query i)))
  returns result:bf16
  ensures pure (uba result (fun c -> ((reveal p) c).core column))
{
  head_core_result p column value;
  narrow_family value (fun c -> ((reveal p) c).core column);
  C.fcast #f32 #bf16 value
}

inline_for_extraction noextract
let sum_four (f:S.vector 4)
  : Lemma (sum_n 4 f == ((f 0 +. f 1) +. f 2) +. f 3) =
  sum_pop 4 f; sum_pop 3 f; sum_pop 2 f; sum_pop 1 f; sum_zero f

inline_for_extraction noextract
let sum_interleaved (rows cols:pos) (f:S.vector (rows * cols))
  : Lemma (
      sum_n rows (fun r -> sum_n cols (fun c -> f (r + rows * c))) ==
      sum_n (rows * cols) f) =
  sum_swap rows cols (fun r c -> f (r + rows * c));
  let point (c:S.fin cols)
    : Lemma (sum_n rows (fun r -> f (r + rows * c)) ==
      sum_n rows (fun r -> f (rows * c + r))) =
    sum_ext rows (fun r -> f (r + rows * c)) (fun r -> f (rows * c + r))
  in
  FStar.Classical.forall_intro point;
  sum_ext cols (fun c -> sum_n rows (fun r -> f (r + rows * c)))
    (fun c -> sum_n rows (fun r -> f (rows * c + r)));
  sum_groups cols rows f

inline_for_extraction noextract
let stripe_terms (f:S.vector 128) (y r:S.fin 4) : S.vector 8 =
  fun t -> f (y + 4 * r + 16 * t)

inline_for_extraction noextract
let dot_partial (f:S.vector 128) (y:S.fin 4) : GTot real =
  sum_n 4 (fun r -> sum_n 8 (stripe_terms f y r))

inline_for_extraction noextract
let recurrent_tree_sum (f:S.vector 128)
  : Lemma (
      (dot_partial f 0 +. dot_partial f 2) +.
      (dot_partial f 1 +. dot_partial f 3) == sum_n 128 f) =
  let point (y:S.fin 4)
    : Lemma (dot_partial f y == sum_n 32 (fun i -> f (y + 4 * i))) =
    let row (r:S.fin 4)
      : Lemma (sum_n 8 (stripe_terms f y r) ==
        sum_n 8 (fun t -> f (y + 4 * (r + 4 * t)))) =
      sum_ext 8 (stripe_terms f y r) (fun t -> f (y + 4 * (r + 4 * t)))
    in
    FStar.Classical.forall_intro row;
    sum_interleaved 4 8 (fun i -> f (y + 4 * i));
    norm_spec [delta_once ["KPlay.Functional.dot_partial"]] (dot_partial f y);
    sum_ext 4
      (fun r -> sum_n 8 (stripe_terms f y r))
      (fun r -> sum_n 8 (fun t -> f (y + 4 * (r + 4 * t))))
  in
  FStar.Classical.forall_intro point;
  sum_four (dot_partial f);
  sum_ext 4 (dot_partial f) (fun y -> sum_n 32 (fun i -> f (y + 4 * i)));
  sum_interleaved 4 32 f

inline_for_extraction noextract
let stripe_step_family (#d:Type0) (f:vector_family d 128) (y r:S.fin 4) (t:S.fin 8)
  : Lemma (forall (c:d).
      bounded_sum (t + 1) (stripe_terms (f c) y r) ==
      bounded_sum t (stripe_terms (f c) y r) +. f c (y + 4 * r + 16 * t)) =
  let point (c:d) : Lemma (
      bounded_sum (t + 1) (stripe_terms (f c) y r) ==
      bounded_sum t (stripe_terms (f c) y r) +. f c (y + 4 * r + 16 * t)) =
    bounded_sum_step t (stripe_terms (f c) y r)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let column_rep (#d:Type0) (h:S.fin 16) (j:S.fin 128)
  (s:seq f32) (x:vector_family d 128) : GTot prop =
  if Seq.length s = 262144 then
    forall (c:d) (i:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i)
  else False

inline_for_extraction noextract
let column_at (#d:Type0) (h:S.fin 16) (j:S.fin 128) (s:lseq f32 262144)
  (x:vector_family d 128) (i:nat{i < 128})
  : Lemma (requires column_rep h j s x)
      (ensures ua (Seq.index s ((h * 128 + i) * 128 + j)) (fun c -> x c i)) =
  let point (c:d)
    : Lemma (v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i)) =
    let all_values:squash (forall (ctx:d) (k:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + k) * 128 + j)) (x ctx k)) = () in
    let this_head:squash (forall (k:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + k) * 128 + j)) (x c k)) =
      FStar.Classical.Sugar.forall_elim c all_values in
    let this_value:squash (v_approximates
      (Seq.index s ((h * 128 + i) * 128 + j)) (x c i)) =
      FStar.Classical.Sugar.forall_elim i this_head in ()
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let column_intro (#d:Type0) (h:S.fin 16) (j:S.fin 128) (s:lseq f32 262144)
  (x:vector_family d 128)
  : Lemma
      (requires forall (c:d) (i:S.fin 128).
        v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i))
      (ensures column_rep h j s x) = ()

inline_for_extraction noextract
let product_vector (x y:S.vector 128) : S.vector 128 =
  fun i -> x i *. y i

inline_for_extraction noextract
let product_sum_family (#d:Type0) (x y:vector_family d 128)
  : Lemma (forall (c:d).
      sum_n 128 (product_vector (x c) (y c)) == sum_n 128 (fun i -> x c i *. y c i)) =
  let point (c:d) : Lemma (
      sum_n 128 (product_vector (x c) (y c)) == sum_n 128 (fun i -> x c i *. y c i)) =
    sum_ext 128 (product_vector (x c) (y c)) (fun i -> x c i *. y c i)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn dot_stripe (#d:Type0)
  (state:A.larray f32 262144) (vector:A.larray f32 128)
  (head:sz{head < 16}) (column:sz{column < 128})
  (y:sz{y < 4}) (r:sz{r < 4})
  (#fs #fv:perm) (#ss #sv:erased (seq f32))
  (#xs #xv:erased (vector_family d 128))
  preserves gpu ** state |-> Frac fs ss ** vector |-> Frac fv sv
  requires pure (column_rep head column ss xs /\ ufp 128 sv xv)
  returns result:f32
  ensures pure (ua result (fun c -> sum_n 8
    (stripe_terms (product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c)) y r)))
{
  let mut t = 0sz;
  let mut accum:f32 = zero;
  while (!t <^ 8sz)
    invariant exists* (k:sz{k <= 8}) (value:f32).
      t |-> k ** accum |-> value **
      pure (ua value (fun c -> bounded_sum k
        (stripe_terms (product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c)) y r)))
    decreases (8 - !t)
  {
    let k = !t;
    let i = y +^ 4sz *^ r +^ 16sz *^ k;
    let a = state.((head *^ 128sz +^ i) *^ 128sz +^ column);
    let b = vector.(i);
    let product = rounded_mul a b #(fun c -> (reveal #(vector_family d 128) xs) c i) #(fun c -> (reveal #(vector_family d 128) xv) c i);
    let value = !accum;
    let next = rounded_add value product
      #(fun c -> bounded_sum k (stripe_terms (product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c)) y r))
      #(fun c -> (reveal #(vector_family d 128) xs) c i *. (reveal #(vector_family d 128) xv) c i);
    stripe_step_family (fun c -> product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c)) y r k;
    accum := next;
    t := k +^ 1sz;
  };
  bounded_full_family 8 (fun c -> stripe_terms (product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c)) y r);
  !accum
}

inline_for_extraction noextract
let partial_four_family (#d:Type0) (f:vector_family d 128) (y:S.fin 4)
  : Lemma (forall (c:d).
      ((sum_n 8 (stripe_terms (f c) y 0) +. sum_n 8 (stripe_terms (f c) y 1)) +.
        sum_n 8 (stripe_terms (f c) y 2)) +. sum_n 8 (stripe_terms (f c) y 3) ==
      dot_partial (f c) y) =
  let point (c:d) : Lemma (
      ((sum_n 8 (stripe_terms (f c) y 0) +. sum_n 8 (stripe_terms (f c) y 1)) +.
        sum_n 8 (stripe_terms (f c) y 2)) +. sum_n 8 (stripe_terms (f c) y 3) ==
      dot_partial (f c) y) =
    norm_spec [delta_once ["KPlay.Functional.dot_partial"]] (dot_partial (f c) y);
    sum_four (fun r -> sum_n 8 (stripe_terms (f c) y r))
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let recurrent_tree_family (#d:Type0) (f:vector_family d 128)
  : Lemma (forall (c:d).
      (dot_partial (f c) 0 +. dot_partial (f c) 2) +.
      (dot_partial (f c) 1 +. dot_partial (f c) 3) == sum_n 128 (f c)) =
  let point (c:d) : Lemma (
      (dot_partial (f c) 0 +. dot_partial (f c) 2) +.
      (dot_partial (f c) 1 +. dot_partial (f c) 3) == sum_n 128 (f c)) =
    recurrent_tree_sum (f c)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn recurrent_dot (#d:Type0)
  (state:A.larray f32 262144) (vector:A.larray f32 128) (partial:A.larray f32 256)
  (head:sz{head < 16}) (column:sz{column < 128})
  (#fs #fv:perm) (#ss #sv #sp:erased (seq f32))
  (#xs #xv:erased (vector_family d 128))
  preserves gpu ** state |-> Frac fs ss ** vector |-> Frac fv sv
  requires partial |-> sp ** pure (column_rep head column ss xs /\ ufp 128 sv xv)
  returns result:f32
  ensures exists* (after:seq f32). partial |-> after **
    pure (Seq.length after == 256 /\ ua result (fun c -> sum_n 128 (fun i -> (reveal #(vector_family d 128) xs) c i *. (reveal #(vector_family d 128) xv) c i)))
{
  A.pts_to_len partial;
  let terms:erased (vector_family d 128) =
    hide (fun c -> product_vector ((reveal #(vector_family d 128) xs) c) ((reveal #(vector_family d 128) xv) c));
  let mut y = 0sz;
  while (!y <^ 4sz)
    invariant exists* (j:sz{j <= 4}) (values:seq f32).
      y |-> j ** partial |-> values **
      pure (Seq.length values == 256 /\ (forall (c:d) (i:nat{i < j}).
        v_approximates (Seq.index values i) (dot_partial ((reveal #(vector_family d 128) terms) c) i)))
    decreases (4 - !y)
  {
    let j = !y;
    let a = dot_stripe state vector head column j 0sz #fs #fv #ss #sv #xs #xv;
    let b = dot_stripe state vector head column j 1sz #fs #fv #ss #sv #xs #xv;
    let c = dot_stripe state vector head column j 2sz #fs #fv #ss #sv #xs #xv;
    let d0 = dot_stripe state vector head column j 3sz #fs #fv #ss #sv #xs #xv;
    let ab = rounded_add a b
      #(fun ctx -> sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 0))
      #(fun ctx -> sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 1));
    let abc = rounded_add ab c
      #(fun ctx -> sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 0) +. sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 1))
      #(fun ctx -> sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 2));
    let sum_value = rounded_add abc d0
      #(fun ctx -> (sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 0) +.
        sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 1)) +. sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 2))
      #(fun ctx -> sum_n 8 (stripe_terms ((reveal #(vector_family d 128) terms) ctx) j 3));
    partial_four_family terms j;
    partial.(j) <- sum_value;
    y := j +^ 1sz;
  };
  let a = partial.(0sz);
  let b = partial.(2sz);
  let c = partial.(1sz);
  let d0 = partial.(3sz);
  let ab = rounded_add a b #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 0) #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 2);
  let cd = rounded_add c d0 #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 1) #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 3);
  let sum_value = rounded_add ab cd
    #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 0 +. dot_partial ((reveal #(vector_family d 128) terms) ctx) 2)
    #(fun ctx -> dot_partial ((reveal #(vector_family d 128) terms) ctx) 1 +. dot_partial ((reveal #(vector_family d 128) terms) ctx) 3);
  recurrent_tree_family terms;
  product_sum_family xs xv;
  sum_value
}

inline_for_extraction noextract
let query_scaling () : Tot (v:f32{v_approximates v (R.rsqrt 128.0R)}) =
  let width = of_int #f32 128L in
  of_int_approx #f32 128L;
  rsqrt_approx width 128.0R;
  rsqrt width

inline_for_extraction noextract
fn prepare_vectors (#d:Type0)
  (qkv:A.larray bf16 6144) (head:sz{head < 16})
  (query key:A.larray f32 128) (partial:A.larray f32 256)
  (#f:perm) (#s:erased (seq bf16)) (#sq #sk #sp:erased (seq f32))
  (#p:erased (d -> GTot head_model))
  preserves gpu ** qkv |-> Frac f s
  requires query |-> sq ** key |-> sk ** partial |-> sp **
    pure ((forall (c:d). valid_head ((reveal p) c)) /\
      segment_rep 6144 (head * 128) 128 s (fun c -> ((reveal p) c).qinput) /\
      segment_rep 6144 (2048 + head * 128) 128 s (fun c -> ((reveal p) c).kinput))
  ensures exists* (nq nk np:seq f32).
    query |-> nq ** key |-> nk ** partial |-> np **
    pure (Seq.length nq == 128 /\ Seq.length nk == 128 /\ Seq.length np == 256 /\
      ufp 128 nq (fun c -> ((reveal p) c).query) /\ ufp 128 nk (fun c -> ((reveal p) c).key))
{
  head_normalization_laws p;
  A.pts_to_len query;
  A.pts_to_len key;
  let qscale = norm_scale 6144sz 128sz qkv (head *^ 128sz) partial true
    #f #s #sp #(fun c -> ((reveal p) c).qinput);
  with pp. assert partial |-> pp;
  let kscale = norm_scale 6144sz 128sz qkv (2048sz +^ head *^ 128sz) partial true
    #f #s #(hide (reveal pp <: seq f32)) #(fun c -> ((reveal p) c).kinput);
  let scale = query_scaling ();
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{j <= 128}) (qv kv:seq f32).
      i |-> j ** query |-> qv ** key |-> kv **
      pure (Seq.length qv == 128 /\ Seq.length kv == 128 /\ (forall (c:d) (k:nat{k < j}).
        v_approximates (Seq.index qv k) (((reveal p) c).query k) /\
        v_approximates (Seq.index kv k) (((reveal p) c).key k)))
    decreases (128 - !i)
  {
    let j = !i;
    let q = read_segment 6144sz 128sz qkv (head *^ 128sz) j #f #s #(fun c -> ((reveal p) c).qinput);
    let k = read_segment 6144sz 128sz qkv (2048sz +^ head *^ 128sz) j #f #s #(fun c -> ((reveal p) c).kinput);
    widen_family q (fun c -> ((reveal p) c).qinput j);
    widen_family k (fun c -> ((reveal p) c).kinput j);
    let normalized_q = rounded_mul (C.fcast #bf16 #f32 q) qscale
      #(fun c -> ((reveal p) c).qinput j)
      #(fun c -> R.rsqrt (S.sum_squares ((reveal p) c).qinput +. S.epsilon));
    let normalized_k = rounded_mul (C.fcast #bf16 #f32 k) kscale
      #(fun c -> ((reveal p) c).kinput j)
      #(fun c -> R.rsqrt (S.sum_squares ((reveal p) c).kinput +. S.epsilon));
    let q16 = round_bf normalized_q #(fun c -> S.l2_normalize ((reveal p) c).qinput j);
    let k16 = round_bf normalized_k #(fun c -> ((reveal p) c).key j);
    let scaled_q = rounded_mul q16 scale
      #(fun c -> S.l2_normalize ((reveal p) c).qinput j) #(fun _ -> R.rsqrt 128.0R);
    query.(j) <- scaled_q;
    key.(j) <- k16;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
fn decay_value (#d:Type0) (a bias alog:bf16)
  (#ar #br #lr:erased (real_family d))
  preserves gpu
  requires pure (uba a ar /\ uba bias br /\ uba alog lr)
  returns result:f32
  ensures pure (ua result
    (fun c -> E.exp ((0.0R -. E.exp ((reveal #(real_family d) lr) c)) *. S.softplus ((reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c))))
{
  widen_family a ar;
  widen_family bias br;
  widen_family alog lr;
  let shifted = rounded_add (C.fcast #bf16 #f32 a) (C.fcast #bf16 #f32 bias) #ar #br;
  softplus_family shifted (fun c -> (reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c);
  let softplus = Softplus.softplus20 shifted;
  let log = C.fcast #bf16 #f32 alog;
  exp_family log lr;
  let exponential = fexp log;
  neg_family exponential (fun c -> E.exp ((reveal #(real_family d) lr) c));
  let negative = mul exponential (of_int #f32 (-1L));
  let argument = rounded_mul negative softplus
    #(fun c -> 0.0R -. E.exp ((reveal #(real_family d) lr) c)) #(fun c -> S.softplus ((reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c));
  exp_family argument (fun c -> (0.0R -. E.exp ((reveal #(real_family d) lr) c)) *. S.softplus ((reveal #(real_family d) ar) c +. (reveal #(real_family d) br) c));
  fexp argument
}

inline_for_extraction noextract
let head_rep (#d:Type0) (h:S.fin 16) (s:seq f32)
  (x:matrix_family d 128 128) : GTot prop =
  if Seq.length s = 262144 then
    forall (c:d) (i j:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i j)
  else False

inline_for_extraction noextract
let head_flat_at (#d:Type0) (h:S.fin 16) (s:lseq f32 262144)
  (x:matrix_family d 128 128) (k:nat{k < 16384})
  : Lemma (requires head_rep h s x)
      (ensures ua (Seq.index s (h * 16384 + k)) (fun c -> x c (k / 128) (k % 128))) =
  let point (c:d)
    : Lemma (v_approximates (Seq.index s (h * 16384 + k)) (x c (k / 128) (k % 128))) =
    let all_heads:squash (forall (ctx:d) (i j:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x ctx i j)) = () in
    let this_head:squash (forall (i j:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i j)) =
      FStar.Classical.Sugar.forall_elim c all_heads in
    let this_row:squash (forall (j:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + k / 128) * 128 + j)) (x c (k / 128) j)) =
      FStar.Classical.Sugar.forall_elim (k / 128) this_head in
    let this_value:squash (v_approximates
      (Seq.index s ((h * 128 + k / 128) * 128 + k % 128)) (x c (k / 128) (k % 128))) =
      FStar.Classical.Sugar.forall_elim (k % 128) this_row in
    assert (h * 16384 + k == (h * 128 + k / 128) * 128 + k % 128)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let head_from_flat (#d:Type0) (h:S.fin 16) (s:lseq f32 262144) (x:matrix_family d 128 128)
  : Lemma
      (requires forall (c:d) (k:nat{k < 16384}).
        v_approximates (Seq.index s (h * 16384 + k)) (x c (k / 128) (k % 128)))
      (ensures head_rep h s x) =
  let point (c:d) (i j:S.fin 128)
    : Lemma (v_approximates (Seq.index s ((h * 128 + i) * 128 + j)) (x c i j)) =
    let all_values:squash (forall (ctx:d) (k:nat{k < 16384}).
      v_approximates (Seq.index s (h * 16384 + k)) (x ctx (k / 128) (k % 128))) = () in
    let this_head:squash (forall (k:nat{k < 16384}).
      v_approximates (Seq.index s (h * 16384 + k)) (x c (k / 128) (k % 128))) =
      FStar.Classical.Sugar.forall_elim c all_values in
    let this_value:squash (v_approximates (Seq.index s (h * 16384 + (i * 128 + j)))
      (x c ((i * 128 + j) / 128) ((i * 128 + j) % 128))) =
      FStar.Classical.Sugar.forall_elim (i * 128 + j) this_head in
    assert ((i * 128 + j) / 128 == i /\ (i * 128 + j) % 128 == j);
    assert ((h * 128 + i) * 128 + j == h * 16384 + (i * 128 + j))
  in FStar.Classical.forall_intro_3 point

inline_for_extraction noextract
let outside_head (h:S.fin 16) (before after:seq f32) : prop =
  if Seq.length before = 262144 && Seq.length after = 262144 then
    forall (i:S.fin 262144). i / 16384 =!= h ==> Seq.index after i == Seq.index before i
  else False

inline_for_extraction noextract
let outside_column (h:S.fin 16) (j:S.fin 128) (before after:seq f32) : prop =
  if Seq.length before = 262144 && Seq.length after = 262144 then
    forall (i:S.fin 262144).
      (i / 16384 =!= h \/ i % 128 =!= j) ==> Seq.index after i == Seq.index before i
  else False

inline_for_extraction noextract
fn decay_head (#d:Type0)
  (previous state:A.larray f32 262144) (head:sz{head < 16}) (decay:f32)
  (#f:perm) (#old #before:erased (seq f32)) (#p:erased (d -> GTot head_model))
  preserves gpu ** previous |-> Frac f old
  requires state |-> before **
    pure ((forall (c:d). valid_head ((reveal p) c)) /\
      head_rep head old (fun c -> ((reveal p) c).old) /\ ua decay (fun c -> ((reveal p) c).decay))
  ensures exists* (after:seq f32). state |-> after **
    pure (Seq.length after == 262144 /\ outside_head head before after /\
      head_rep head after (fun c -> ((reveal p) c).decayed))
{
  head_decay_laws p;
  A.pts_to_len state;
  let offset = head *^ 16384sz;
  let mut i = 0sz;
  while (!i <^ 16384sz)
    invariant exists* (j:sz{j <= 16384}) (values:seq f32).
      i |-> j ** state |-> values **
      pure (Seq.length values == 262144 /\ (forall (k:S.fin 262144).
        (k < offset \/ offset + j <= k) ==> Seq.index values k == Seq.index before k) /\
        (forall (c:d) (k:nat{k < j}).
          v_approximates (Seq.index values (offset + k)) (((reveal p) c).decayed (k / 128) (k % 128))))
    decreases (16384 - !i)
  {
    let j = !i;
    let old_value = previous.(offset +^ j);
    head_flat_at head old (fun c -> ((reveal p) c).old) j;
    let next = rounded_mul old_value decay
      #(fun c -> ((reveal p) c).old (j / 128) (j % 128)) #(fun c -> ((reveal p) c).decay);
    state.(offset +^ j) <- next;
    i := j +^ 1sz;
  };
  with after. assert state |-> after;
  head_from_flat head after (fun c -> ((reveal p) c).decayed);
  assert pure (head_rep head after (fun c -> ((reveal p) c).decayed));
}

inline_for_extraction noextract
fn update_column (#d:Type0)
  (state:A.larray f32 262144) (key:A.larray f32 128)
  (head:sz{head < 16}) (column:sz{column < 128}) (delta:f32)
  (#f:perm) (#before #ks:erased (seq f32)) (#p:erased (d -> GTot head_model))
  preserves gpu ** key |-> Frac f ks
  requires state |-> before **
    pure ((forall (c:d). valid_head ((reveal p) c)) /\
      column_rep head column before (fun c i -> ((reveal p) c).decayed i column) /\
      ufp 128 ks (fun c -> ((reveal p) c).key) /\ ua delta (fun c -> ((reveal p) c).delta column))
  ensures exists* (after:seq f32). state |-> after **
    pure (Seq.length after == 262144 /\ outside_column head column before after /\
      column_rep head column after (fun c i -> ((reveal p) c).updated i column))
{
  head_update_laws p;
  A.pts_to_len state;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{j <= 128}) (values:seq f32).
      i |-> j ** state |-> values **
      pure (Seq.length values == 262144 /\ outside_column head column before values /\
        (forall (c:d) (k:nat{k < j}).
          v_approximates (Seq.index values ((head * 128 + k) * 128 + column))
            (((reveal p) c).updated k column)) /\
        (forall (k:nat{j <= k /\ k < 128}).
          Seq.index values ((head * 128 + k) * 128 + column) ==
          Seq.index before ((head * 128 + k) * 128 + column)))
    decreases (128 - !i)
  {
    let j = !i;
    let index = (head *^ 128sz +^ j) *^ 128sz +^ column;
    let old_value = state.(index);
    let k = key.(j);
    column_at head column before (fun c i -> ((reveal p) c).decayed i column) j;
    let change = rounded_mul k delta #(fun c -> ((reveal p) c).key j) #(fun c -> ((reveal p) c).delta column);
    let next = rounded_add old_value change
      #(fun c -> ((reveal p) c).decayed j column) #(fun c -> ((reveal p) c).key j *. ((reveal p) c).delta column);
    state.(index) <- next;
    i := j +^ 1sz;
  };
  with after. assert state |-> after;
  column_intro head column after (fun c i -> ((reveal p) c).updated i column);
  assert pure (column_rep head column after (fun c i -> ((reveal p) c).updated i column));
}

inline_for_extraction noextract
let head_progress (#d:Type0) (h:S.fin 16) (j:nat{j <= 128})
  (s:seq f32) (p:d -> GTot head_model) : GTot prop =
  if Seq.length s = 262144 then
    forall (c:d) (i k:S.fin 128).
      v_approximates (Seq.index s ((h * 128 + i) * 128 + k))
        (if k < j then (p c).updated i k else (p c).decayed i k)
  else False

inline_for_extraction noextract
let head_progress_start (#d:Type0) (h:S.fin 16) (s:lseq f32 262144)
  (p:d -> GTot head_model)
  : Lemma (requires head_rep h s (fun c -> (p c).decayed))
      (ensures head_progress h 0 s p) =
  norm_spec [delta_only ["KPlay.Functional.head_rep"]] (head_rep h s (fun c -> (p c).decayed));
  norm_spec [delta_only ["KPlay.Functional.head_progress"]] (head_progress h 0 s p)

inline_for_extraction noextract
let head_progress_step (#d:Type0) (h:S.fin 16) (j:sz{j < 128})
  (before after:lseq f32 262144) (p:d -> GTot head_model)
  : Lemma
      (requires head_progress h j before p /\ outside_column h j before after /\
        column_rep h j after (fun c i -> (p c).updated i j))
      (ensures head_progress h (j + 1) after p) =
  let point (c:d) (i k:S.fin 128)
    : Lemma (v_approximates (Seq.index after ((h * 128 + i) * 128 + k))
      (if k < j + 1 then (p c).updated i k else (p c).decayed i k)) =
    if k = SZ.v j then
      column_at h j after (fun ctx i -> (p ctx).updated i j) i
    else (
      assert (((h * 128 + i) * 128 + k) % 128 == k);
      assert (Seq.index after ((h * 128 + i) * 128 + k) ==
        Seq.index before ((h * 128 + i) * 128 + k));
      assert ((k < j + 1) == (k < j))
    )
  in FStar.Classical.forall_intro_3 point

inline_for_extraction noextract
let head_progress_done (#d:Type0) (h:S.fin 16) (s:lseq f32 262144)
  (p:d -> GTot head_model)
  : Lemma (requires head_progress h 128 s p)
      (ensures head_rep h s (fun c -> (p c).updated)) =
  norm_spec [delta_only ["KPlay.Functional.head_rep"]] (head_rep h s (fun c -> (p c).updated));
  norm_spec [delta_only ["KPlay.Functional.head_progress"]] (head_progress h 128 s p)

inline_for_extraction noextract
fn recurrent_head (#d:Type0)
  (qkv:A.larray bf16 6144) (previous state:A.larray f32 262144)
  (query key:A.larray f32 128) (core:A.larray bf16 128) (partial:A.larray f32 256)
  (head:sz{head < 16}) (beta decay:f32)
  (#fc #fr:perm) (#cs:erased (seq bf16))
  (#old #before #sq #sk #sp:erased (seq f32)) (#co:erased (seq bf16))
  (#p:erased (d -> GTot head_model))
  preserves gpu ** qkv |-> Frac fc cs ** previous |-> Frac fr old
  requires state |-> before ** query |-> sq ** key |-> sk ** core |-> co ** partial |-> sp **
    pure ((forall (c:d). valid_head ((reveal p) c)) /\
      segment_rep 6144 (head * 128) 128 cs (fun c -> ((reveal p) c).qinput) /\
      segment_rep 6144 (2048 + head * 128) 128 cs (fun c -> ((reveal p) c).kinput) /\
      segment_rep 6144 (4096 + head * 128) 128 cs (fun c -> ((reveal p) c).value) /\
      head_rep head old (fun c -> ((reveal p) c).old) /\
      ua beta (fun c -> ((reveal p) c).beta) /\ ua decay (fun c -> ((reveal p) c).decay))
  ensures exists* (after nq nk np:seq f32) (nc:seq bf16).
    state |-> after ** query |-> nq ** key |-> nk ** partial |-> np ** core |-> nc **
    pure (Seq.length after == 262144 /\ Seq.length nq == 128 /\ Seq.length nk == 128 /\
      Seq.length np == 256 /\ Seq.length nc == 128 /\
      outside_head head before after /\ head_rep head after (fun c -> ((reveal p) c).updated) /\
      ubf 128 nc (fun c -> ((reveal p) c).core))
{
  A.pts_to_len core;
  prepare_vectors qkv head query key partial #fc #cs #sq #sk #sp #p;
  with qvalues. assert query |-> qvalues;
  with kvalues. assert key |-> kvalues;
  decay_head previous state head decay #fr #old #before #p;
  with decayed_values. assert state |-> decayed_values;
  head_progress_start head decayed_values p;
  let mut column = 0sz;
  while (!column <^ 128sz)
    invariant exists* (j:sz{j <= 128}) (current:seq f32) (values:seq bf16) (scratch:seq f32).
      column |-> j ** state |-> current ** core |-> values ** partial |-> scratch **
      pure (Seq.length current == 262144 /\ Seq.length values == 128 /\
        Seq.length scratch == 256 /\ outside_head head before current /\
        head_progress head j current p /\
        (forall (c:d) (k:nat{k < j}).
          v_approximates (Seq.index values k) (((reveal p) c).core k)))
    decreases (128 - !column)
  {
    let j = !column;
    with logical_column. assert column |-> logical_column;
    with current. assert state |-> current;
    with scratch. assert partial |-> scratch;
    let memory = recurrent_dot state key partial head j #1.0R #1.0R
      #(hide (reveal current <: seq f32)) #(hide (reveal kvalues <: seq f32))
      #(hide (reveal scratch <: seq f32))
      #(fun c i -> ((reveal p) c).decayed i j) #(fun c -> ((reveal p) c).key);
    let value = read_segment 6144sz 128sz qkv (4096sz +^ head *^ 128sz) j
      #fc #cs #(fun c -> ((reveal p) c).value);
    let delta = compute_delta j value memory beta #p;
    update_column state key head j delta #1.0R
      #(hide (reveal current <: seq f32)) #(hide (reveal kvalues <: seq f32)) #p;
    with next. assert state |-> next;
    head_progress_step head logical_column current next p;
    with scratch2. assert partial |-> scratch2;
    let readout = recurrent_dot state query partial head j #1.0R #1.0R
      #(hide (reveal next <: seq f32)) #(hide (reveal qvalues <: seq f32))
      #(hide (reveal scratch2 <: seq f32))
      #(fun c i -> ((reveal p) c).updated i j) #(fun c -> ((reveal p) c).query);
    let rounded = round_readout j readout #p;
    core.(j) <- rounded;
    column := j +^ 1sz;
  };
  with after. assert state |-> after;
  head_progress_done head after p;
  assert pure (head_rep head after (fun c -> ((reveal p) c).updated));
}

inline_for_extraction noextract
fn gate_head (#d:Type0)
  (core:A.larray bf16 128) (z:A.larray bf16 2048)
  (weight:A.larray bf16 weight_count) (output:A.larray bf16 2048)
  (partial:A.larray f32 256) (head:sz{head < 16})
  (#fc #fz #fw:perm) (#sc #z_values #sw #so:erased (seq bf16)) (#sp:erased (seq f32))
  (#p:erased (d -> GTot head_model))
  preserves gpu ** core |-> Frac fc sc ** z |-> Frac fz z_values ** weight |-> Frac fw sw
  requires output |-> so ** partial |-> sp **
    pure (Seq.length so == 2048 /\ (forall (c:d). valid_head ((reveal p) c)) /\
      ubf 128 sc (fun c -> ((reveal p) c).core) /\
      segment_rep 2048 (head * 128) 128 z_values (fun c -> ((reveal p) c).z) /\
      segment_rep weight_count gated_norm_offset 128 sw (fun c -> ((reveal p) c).weight))
  ensures exists* (after:seq bf16) (np:seq f32). output |-> after ** partial |-> np **
    pure (Seq.length so == 2048 /\ Seq.length after == 2048 /\ Seq.length np == 256 /\
      segment_rep 2048 (head * 128) 128 after (fun c -> ((reveal p) c).gated) /\
      (forall (i:S.fin 2048). i / 128 =!= head ==> Seq.index after i == Seq.index so i))
{
  head_gating_laws p;
  A.pts_to_len output;
  let scale = norm_scale 128sz 128sz core 0sz partial false #fc #sc #sp #(fun c -> ((reveal p) c).core);
  let weight_offset = SZ.uint32_to_sizet 19458080ul;
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant exists* (j:sz{j <= 128}) (values:seq bf16).
      i |-> j ** output |-> values **
      pure (Seq.length values == 2048 /\ (forall (k:S.fin 2048).
        (k < head * 128 \/ head * 128 + j <= k) ==> Seq.index values k == Seq.index so k) /\
        (forall (c:d) (k:nat{k < j}).
          v_approximates (Seq.index values (head * 128 + k)) (((reveal p) c).gated k)))
    decreases (128 - !i)
  {
    let j = !i;
    let x = read_segment 128sz 128sz core 0sz j #fc #sc #(fun c -> ((reveal p) c).core);
    let w = read_segment (SZ.uint32_to_sizet 21555360ul) 128sz weight weight_offset j
      #fw #sw #(fun c -> ((reveal p) c).weight);
    let z0 = read_segment 2048sz 128sz z (head *^ 128sz) j #fz #z_values #(fun c -> ((reveal p) c).z);
    widen_family x (fun c -> ((reveal p) c).core j);
    widen_family w (fun c -> ((reveal p) c).weight j);
    let normalized = rounded_mul (C.fcast #bf16 #f32 x) scale
      #(fun c -> ((reveal p) c).core j) #(fun c -> S.rms_scale ((reveal p) c).core);
    let rounded = round_bf normalized #(fun c -> ((reveal p) c).core j *. S.rms_scale ((reveal p) c).core);
    let weighted = rounded_mul rounded (C.fcast #bf16 #f32 w)
      #(fun c -> ((reveal p) c).core j *. S.rms_scale ((reveal p) c).core) #(fun c -> ((reveal p) c).weight j);
    let weighted_bf = round_bf weighted
      #(fun c -> ((reveal p) c).weight j *. (((reveal p) c).core j *. S.rms_scale ((reveal p) c).core));
    let gate = activation32 true z0 #(fun c -> ((reveal p) c).z j);
    let value = rounded_mul weighted_bf gate
      #(fun c -> ((reveal p) c).weight j *. (((reveal p) c).core j *. S.rms_scale ((reveal p) c).core))
      #(fun c -> S.silu (((reveal p) c).z j));
    narrow_family value (fun c -> ((reveal p) c).gated j);
    output.(head *^ 128sz +^ j) <- C.fcast #f32 #bf16 value;
    i := j +^ 1sz;
  }
}

inline_for_extraction noextract
let flat_convolution (x:S.convolution_state) : S.vector 24576 =
  fun i -> x (i / 4) (i % 4)

inline_for_extraction noextract
let convolution_terms (w:S.matrix 6144 4) (next:S.convolution_state)
  (channel:S.fin 6144) : S.vector 4 =
  fun tap -> w channel tap *. next channel tap

inline_for_extraction noextract
let convolve_point (w:S.matrix 6144 4) (next:S.convolution_state) (channel:S.fin 6144)
  : Lemma (S.convolve w next channel == S.silu (sum_n 4 (convolution_terms w next channel))) =
  norm_spec [delta_only ["KPlay.SharedSpec.convolve"]] (S.convolve w next channel);
  sum_shared_ext 4 (convolution_terms w next channel) (fun tap -> w channel tap *. next channel tap)

inline_for_extraction noextract
let convolve_family (#d:Type0) (w:matrix_family d 6144 4) (next:d -> GTot S.convolution_state)
  (channel:S.fin 6144)
  : Lemma (forall (c:d).
      S.convolve (w c) (next c) channel == S.silu (sum_n 4 (convolution_terms (w c) (next c) channel))) =
  let point (c:d) : Lemma (
      S.convolve (w c) (next c) channel == S.silu (sum_n 4 (convolution_terms (w c) (next c) channel))) =
    convolve_point (w c) (next c) channel
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let convolution_step_family (#d:Type0)
  (w:matrix_family d 6144 4) (next:d -> GTot S.convolution_state)
  (channel:S.fin 6144) (tap:S.fin 4)
  : Lemma (forall (c:d).
      bounded_sum (tap + 1) (convolution_terms (w c) (next c) channel) ==
      (next c channel tap *. w c channel tap) +.
        bounded_sum tap (convolution_terms (w c) (next c) channel)) =
  let point (c:d) : Lemma (
      bounded_sum (tap + 1) (convolution_terms (w c) (next c) channel) ==
      (next c channel tap *. w c channel tap) +.
        bounded_sum tap (convolution_terms (w c) (next c) channel)) =
    bounded_sum_step tap (convolution_terms (w c) (next c) channel)
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn convolution_tap (#d:Type0)
  (old:A.larray bf16 24576) (raw:A.larray bf16 6144)
  (channel:sz{channel < 6144}) (tap:sz{tap < 4})
  (#fo #fr:perm) (#so #sr:erased (seq bf16))
  (#xo:erased (d -> GTot S.convolution_state)) (#xr:erased (vector_family d 6144))
  preserves gpu ** old |-> Frac fo so ** raw |-> Frac fr sr
  requires pure (ubf 24576 so (fun c -> flat_convolution ((reveal xo) c)) /\ ubf 6144 sr xr)
  returns result:bf16
  ensures pure (uba result (fun c -> S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c) channel tap))
{
  if (tap <^ 3sz) {
    old.(4sz *^ channel +^ tap +^ 1sz)
  } else {
    raw.(channel)
  }
}

inline_for_extraction noextract
fn convolution_kernel (#d:Type0)
  (weight:A.larray bf16 weight_count) (old:A.larray bf16 24576)
  (raw:A.larray bf16 6144) (next:A.larray bf16 24576) (activated:A.larray bf16 6144)
  (#fw #fo #fr:perm) (#sw #so #sr #before:erased (seq bf16))
  (#w:erased (matrix_family d 6144 4))
  (#xo:erased (d -> GTot S.convolution_state)) (#xr:erased (vector_family d 6144))
  (#initial:erased (seq bf16))
  preserves gpu ** weight |-> Frac fw sw ** old |-> Frac fo so ** raw |-> Frac fr sr
  requires next |-> before ** activated |-> initial **
    pure (umatrix 6144 4 weight_count conv_kernel_offset 4 1 sw w /\
      ubf 24576 so (fun c -> flat_convolution ((reveal xo) c)) /\ ubf 6144 sr xr)
  ensures exists* (nc nq:seq bf16). next |-> nc ** activated |-> nq **
    pure (Seq.length nc == 24576 /\ Seq.length nq == 6144 /\
      ubf 24576 nc (fun c -> flat_convolution (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c))) /\
      ubf 6144 nq (fun c -> S.convolve ((reveal #(matrix_family d 6144 4) w) c) (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c))))
{
  A.pts_to_len next;
  A.pts_to_len activated;
  let offset = SZ.uint32_to_sizet 19433472ul;
  let mut channel = 0sz;
  while (!channel <^ 6144sz)
    invariant exists* (j:sz{j <= 6144}) (values acts:seq bf16).
      channel |-> j ** next |-> values ** activated |-> acts **
      pure (Seq.length values == 24576 /\ Seq.length acts == 6144 /\
        (forall (c:d) (k:nat{k < 4 * j}).
          v_approximates (Seq.index values k)
            (flat_convolution (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) k)) /\
        (forall (c:d) (k:nat{k < j}).
          v_approximates (Seq.index acts k) (S.convolve ((reveal #(matrix_family d 6144 4) w) c)
            (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) k)))
    decreases (6144 - !channel)
  {
    let j = !channel;
    with logical_channel. assert channel |-> logical_channel;
    let mut tap = 0sz;
    let mut accum:f32 = zero;
    while (!tap <^ 4sz)
      invariant exists* (k:sz{k <= 4}) (value:f32) (values:seq bf16).
        tap |-> k ** accum |-> value ** next |-> values **
        pure (Seq.length values == 24576 /\
          ua value (fun c -> bounded_sum k (convolution_terms ((reveal #(matrix_family d 6144 4) w) c)
            (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j)) /\
          (forall (c:d) (i:nat{i < 4 * j + k}).
            v_approximates (Seq.index values i)
              (flat_convolution (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) i)))
      decreases (4 - !tap)
    {
      let k = !tap;
      let value = convolution_tap old raw j k #fo #fr #so #sr #xo #xr;
      next.(4sz *^ j +^ k) <- value;
      let coefficient = weight.(offset +^ 4sz *^ j +^ k);
      widen_family value (fun c -> S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c) j k);
      widen_family coefficient (fun c -> (reveal #(matrix_family d 6144 4) w) c j k);
      let previous = !accum;
      let sum_value = rounded_fma (C.fcast #bf16 #f32 value) (C.fcast #bf16 #f32 coefficient) previous
        #(fun c -> S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c) j k)
        #(fun c -> (reveal #(matrix_family d 6144 4) w) c j k)
        #(fun c -> bounded_sum k (convolution_terms ((reveal #(matrix_family d 6144 4) w) c)
          (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j));
      convolution_step_family w
        (fun c -> S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j k;
      accum := sum_value;
      tap := k +^ 1sz;
    };
    let sum_value = !accum;
    bounded_full_family 4 (fun c -> convolution_terms
      ((reveal #(matrix_family d 6144 4) w) c)
      (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) logical_channel);
    narrow_family sum_value (fun c -> sum_n 4 (convolution_terms ((reveal #(matrix_family d 6144 4) w) c)
      (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j));
    let rounded = C.fcast #f32 #bf16 sum_value;
    let result = activation32 true rounded
      #(fun c -> sum_n 4 (convolution_terms ((reveal #(matrix_family d 6144 4) w) c)
        (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j));
    convolve_family w
      (fun c -> S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) logical_channel;
    narrow_family result (fun c -> S.convolve ((reveal #(matrix_family d 6144 4) w) c)
      (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c)) j);
    activated.(j) <- C.fcast #f32 #bf16 result;
    channel := j +^ 1sz;
  }
}

fn execute_convolution (#d:Type0)
  (weight:A.larray bf16 weight_count{GA.is_global_array weight})
  (old:A.larray bf16 24576{GA.is_global_array old})
  (raw:A.larray bf16 6144{GA.is_global_array raw})
  (next:A.larray bf16 24576{GA.is_global_array next})
  (activated:A.larray bf16 6144{GA.is_global_array activated})
  (#fw #fo #fr:perm) (#sw #so #sr #before:erased (seq bf16))
  (#w:erased (matrix_family d 6144 4))
  (#xo:erased (d -> GTot S.convolution_state)) (#xr:erased (vector_family d 6144))
  preserves cpu ** on gpu_loc (weight |-> Frac fw sw ** old |-> Frac fo so ** raw |-> Frac fr sr)
  requires on gpu_loc (next |-> before ** (exists* (initial:seq bf16). activated |-> initial)) **
    pure (umatrix 6144 4 weight_count conv_kernel_offset 4 1 sw w /\
      ubf 24576 so (fun c -> flat_convolution ((reveal xo) c)) /\ ubf 6144 sr xr)
  ensures exists* (nc nq:seq bf16). on gpu_loc (next |-> nc ** activated |-> nq) **
    pure (Seq.length nc == 24576 /\ Seq.length nq == 6144 /\
      ubf 24576 nc (fun c -> flat_convolution (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c))) /\
      ubf 6144 nq (fun c -> S.convolve ((reveal #(matrix_family d 6144 4) w) c) (S.shift_convolution ((reveal xo) c) ((reveal #(vector_family d 6144) xr) c))))
{
  with initial. assert on gpu_loc (activated |-> initial);
  launch_kernel_1 (fun () -> convolution_kernel weight old raw next activated
    #fw #fo #fr #sw #so #sr #before #w #xo #xr #(hide (reveal initial <: seq bf16)));
}

[@@"opaque_to_smt"]
inline_for_extraction noextract
let mixer_inputs (#d:Type0)
  (sw sq z_values sb sa:seq bf16) (sr:seq f32) (p:d -> GTot (S.fin 16 -> GTot head_model))
  : GTot prop =
  ubf 16 sb (fun c h -> (p c h).b) /\ ubf 16 sa (fun c h -> (p c h).a) /\
  segment_rep weight_count a_log_offset 16 sw (fun c h -> (p c h).a_log) /\
  segment_rep weight_count dt_bias_offset 16 sw (fun c h -> (p c h).dt_bias) /\
  (forall (h:S.fin 16).
    (forall (c:d). valid_head (p c h)) /\
    segment_rep 6144 (h * 128) 128 sq (fun c -> (p c h).qinput) /\
    segment_rep 6144 (2048 + h * 128) 128 sq (fun c -> (p c h).kinput) /\
    segment_rep 6144 (4096 + h * 128) 128 sq (fun c -> (p c h).value) /\
    segment_rep 2048 (h * 128) 128 z_values (fun c -> (p c h).z) /\
    segment_rep weight_count gated_norm_offset 128 sw (fun c -> (p c h).weight) /\
    head_rep h sr (fun c -> (p c h).old))

inline_for_extraction noextract
let mixer_outputs (#d:Type0) (sr:seq f32) (sy:seq bf16)
  (p:d -> GTot (S.fin 16 -> GTot head_model)) : GTot prop =
  (forall (h:S.fin 16). head_rep h sr (fun c -> (p c h).updated)) /\
  ubf 2048 sy (fun c i -> (p c (i / 128)).gated (i % 128))

inline_for_extraction noextract
let mixer_outputs_intro (#d:Type0) (sr:lseq f32 262144) (sy:lseq bf16 2048)
  (p:d -> GTot (S.fin 16 -> GTot head_model))
  : Lemma
      (requires
        (forall (h:S.fin 16). head_rep h sr (fun c -> (p c h).updated)) /\
        (forall (c:d) (i:S.fin 2048).
          v_approximates (Seq.index sy i) ((p c (i / 128)).gated (i % 128))))
      (ensures mixer_outputs sr sy p) =
  norm_spec [delta_only ["KPlay.Functional.mixer_outputs"; "KPlay.Functional.ubf"; "KPlay.Functional.bf_rep"]]
    (mixer_outputs sr sy p)

inline_for_extraction noextract
let mixer_head_inputs (#d:Type0) (sw sq z_values sb sa:seq bf16) (sr:seq f32)
  (p:d -> GTot (S.fin 16 -> GTot head_model)) (h:S.fin 16)
  : Lemma (requires mixer_inputs sw sq z_values sb sa sr p)
      (ensures
        (forall (c:d). valid_head (p c h)) /\
        segment_rep 6144 (h * 128) 128 sq (fun c -> (p c h).qinput) /\
        segment_rep 6144 (2048 + h * 128) 128 sq (fun c -> (p c h).kinput) /\
        segment_rep 6144 (4096 + h * 128) 128 sq (fun c -> (p c h).value) /\
        segment_rep 2048 (h * 128) 128 z_values (fun c -> (p c h).z) /\
        segment_rep weight_count gated_norm_offset 128 sw (fun c -> (p c h).weight) /\
        head_rep h sr (fun c -> (p c h).old)) =
  norm_spec [delta_once ["KPlay.Functional.mixer_inputs"]] (mixer_inputs sw sq z_values sb sa sr p);
  let all_heads:squash (forall (head:S.fin 16).
    (forall (c:d). valid_head (p c head)) /\
    segment_rep 6144 (head * 128) 128 sq (fun c -> (p c head).qinput) /\
    segment_rep 6144 (2048 + head * 128) 128 sq (fun c -> (p c head).kinput) /\
    segment_rep 6144 (4096 + head * 128) 128 sq (fun c -> (p c head).value) /\
    segment_rep 2048 (head * 128) 128 z_values (fun c -> (p c head).z) /\
    segment_rep weight_count gated_norm_offset 128 sw (fun c -> (p c head).weight) /\
    head_rep head sr (fun c -> (p c head).old)) = () in
  let selected:squash (
    (forall (c:d). valid_head (p c h)) /\
    segment_rep 6144 (h * 128) 128 sq (fun c -> (p c h).qinput) /\
    segment_rep 6144 (2048 + h * 128) 128 sq (fun c -> (p c h).kinput) /\
    segment_rep 6144 (4096 + h * 128) 128 sq (fun c -> (p c h).value) /\
    segment_rep 2048 (h * 128) 128 z_values (fun c -> (p c h).z) /\
    segment_rep weight_count gated_norm_offset 128 sw (fun c -> (p c h).weight) /\
    head_rep h sr (fun c -> (p c h).old)) =
    FStar.Classical.Sugar.forall_elim h all_heads in ()

inline_for_extraction noextract
let gated_progress_step (#d:Type0) (h:S.fin 16)
  (before after:lseq bf16 2048) (p:d -> GTot (S.fin 16 -> GTot head_model))
  : Lemma
      (requires
        (forall (c:d) (i:nat{i < h * 128}).
          v_approximates (Seq.index before i) ((p c (i / 128)).gated (i % 128))) /\
        segment_rep 2048 (h * 128) 128 after (fun c -> (p c h).gated) /\
        (forall (i:S.fin 2048). i / 128 =!= h ==> Seq.index after i == Seq.index before i))
      (ensures forall (c:d) (i:nat{i < (h + 1) * 128}).
        v_approximates (Seq.index after i) ((p c (i / 128)).gated (i % 128))) =
  norm_spec [delta_once ["KPlay.Functional.segment_rep"]]
    (segment_rep 2048 (h * 128) 128 after (fun c -> (p c h).gated));
  let point (c:d) (i:nat{i < (h + 1) * 128})
    : Lemma (v_approximates (Seq.index after i) ((p c (i / 128)).gated (i % 128))) =
    if i / 128 = h then (
      let all_values:squash (forall (ctx:d) (k:S.fin 128).
        v_approximates (Seq.index after (h * 128 + k)) ((p ctx h).gated k)) = () in
      let this_head:squash (forall (k:S.fin 128).
        v_approximates (Seq.index after (h * 128 + k)) ((p c h).gated k)) =
        FStar.Classical.Sugar.forall_elim c all_values in
      let this_value:squash (v_approximates
        (Seq.index after (h * 128 + i % 128)) ((p c h).gated (i % 128))) =
        FStar.Classical.Sugar.forall_elim (i % 128) this_head in
      assert (h * 128 + i % 128 == i)
    ) else (
      assert (i < h * 128);
      assert (Seq.index after i == Seq.index before i)
    )
  in FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
let head_frame_rep (#d:Type0) (changed:S.fin 16) (h:S.fin 16{h =!= changed})
  (before after:lseq f32 262144) (x:matrix_family d 128 128)
  : Lemma (requires outside_head changed before after /\ head_rep h before x)
      (ensures head_rep h after x) =
  let point (c:d) (i j:S.fin 128)
    : Lemma (v_approximates (Seq.index after ((h * 128 + i) * 128 + j)) (x c i j)) =
    assert (((h * 128 + i) * 128 + j) / 16384 == h);
    assert (Seq.index after ((h * 128 + i) * 128 + j) ==
      Seq.index before ((h * 128 + i) * 128 + j))
  in FStar.Classical.forall_intro_3 point

inline_for_extraction noextract
let state_progress_step (#d:Type0) (h:S.fin 16)
  (before after:lseq f32 262144) (p:d -> GTot (S.fin 16 -> GTot head_model))
  : Lemma
      (requires
        (forall (head:nat{head < h}). head_rep head before (fun c -> (p c head).updated)) /\
        outside_head h before after /\ head_rep h after (fun c -> (p c h).updated))
      (ensures forall (head:nat{head < h + 1}). head_rep head after (fun c -> (p c head).updated)) =
  let point (head:nat{head < h + 1})
    : Lemma (head_rep head after (fun c -> (p c head).updated)) =
    if head <> h then (
      let previous:squash (head_rep head before (fun c -> (p c head).updated)) =
        FStar.Classical.Sugar.forall_elim head () in
      head_frame_rep h head before after (fun c -> (p c head).updated)
    )
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
fn mixer_kernel (#d:Type0)
  (weight:A.larray bf16 weight_count) (qkv:A.larray bf16 6144)
  (z:A.larray bf16 2048) (b a:A.larray bf16 16)
  (previous state:A.larray f32 262144) (output:A.larray bf16 2048)
  (query key:A.larray f32 128) (core:A.larray bf16 128) (partial:A.larray f32 256)
  (#fw #fq #fz #fb #fa #fr:perm)
  (#sw #sq #z_values #sb #sa #so #sc:erased (seq bf16))
  (#sr #before #squery #skey #sp:erased (seq f32))
  (#p:erased (d -> GTot (S.fin 16 -> GTot head_model)))
  preserves gpu ** weight |-> Frac fw sw ** qkv |-> Frac fq sq **
    z |-> Frac fz z_values ** b |-> Frac fb sb ** a |-> Frac fa sa ** previous |-> Frac fr sr
  requires state |-> before ** output |-> so ** query |-> squery ** key |-> skey **
    core |-> sc ** partial |-> sp ** pure (mixer_inputs sw sq z_values sb sa sr p)
  ensures exists* (nr nq nk np:seq f32) (ny nc:seq bf16).
    state |-> nr ** output |-> ny ** query |-> nq ** key |-> nk ** core |-> nc ** partial |-> np **
    pure (Seq.length nr == 262144 /\ Seq.length ny == 2048 /\
      Seq.length nq == 128 /\ Seq.length nk == 128 /\ Seq.length nc == 128 /\ Seq.length np == 256 /\
      mixer_outputs nr ny p)
{
  physical_extents ();
  norm_spec [delta_once ["KPlay.Functional.mixer_inputs"]] (mixer_inputs sw sq z_values sb sa sr p);
  A.pts_to_len state;
  A.pts_to_len output;
  A.pts_to_len query;
  A.pts_to_len key;
  A.pts_to_len core;
  A.pts_to_len partial;
  let log_offset = SZ.uint32_to_sizet 19458048ul;
  let bias_offset = SZ.uint32_to_sizet 19458064ul;
  let mut h = 0sz;
  while (!h <^ 16sz)
    invariant exists* (j:sz{j <= 16}) (rv:seq f32) (yv:seq bf16)
      (qv kv:seq f32) (cv:seq bf16) (pv:seq f32).
      h |-> j ** state |-> rv ** output |-> yv ** query |-> qv ** key |-> kv ** core |-> cv ** partial |-> pv **
      pure (Seq.length rv == 262144 /\ Seq.length yv == 2048 /\
        Seq.length qv == 128 /\ Seq.length kv == 128 /\ Seq.length cv == 128 /\ Seq.length pv == 256 /\
        (forall (head:nat{head < j}). head_rep head rv (fun c -> ((reveal p) c head).updated)) /\
        (forall (c:d) (i:nat{i < j * 128}).
          v_approximates (Seq.index yv i) (((reveal p) c (i / 128)).gated (i % 128))))
    decreases (16 - !h)
  {
    let j = !h;
    with logical_head. assert h |-> logical_head;
    head_parameter_laws (fun c -> (reveal p) c logical_head);
    mixer_head_inputs sw sq z_values sb sa sr p logical_head;
    let beta0 = read_segment 16sz 16sz b 0sz j #fb #sb #(fun c h -> ((reveal p) c h).b);
    let beta32 = activation32 false beta0 #(fun c -> ((reveal p) c j).b);
    let beta = round_bf beta32 #(fun c -> ((reveal p) c j).beta);
    let a0 = read_segment 16sz 16sz a 0sz j #fa #sa #(fun c h -> ((reveal p) c h).a);
    let bias = read_segment (SZ.uint32_to_sizet 21555360ul) 16sz weight bias_offset j
      #fw #sw #(fun c h -> ((reveal p) c h).dt_bias);
    let log = read_segment (SZ.uint32_to_sizet 21555360ul) 16sz weight log_offset j
      #fw #sw #(fun c h -> ((reveal p) c h).a_log);
    let decay = decay_value a0 bias log
      #(fun c -> ((reveal p) c j).a) #(fun c -> ((reveal p) c j).dt_bias) #(fun c -> ((reveal p) c j).a_log);
    with rv. assert state |-> rv;
    with qv. assert query |-> qv;
    with kv. assert key |-> kv;
    with cv. assert core |-> cv;
    with pv. assert partial |-> pv;
    recurrent_head qkv previous state query key core partial j beta decay #fq #fr #sq #sr
      #(hide (reveal rv <: seq f32)) #(hide (reveal qv <: seq f32))
      #(hide (reveal kv <: seq f32)) #(hide (reveal pv <: seq f32))
      #(hide (reveal cv <: seq bf16)) #(fun c -> (reveal p) c j);
    with state_values. assert state |-> state_values;
    state_progress_step logical_head rv state_values p;
    with nc. assert core |-> nc;
    with np. assert partial |-> np;
    with yv. assert output |-> yv;
    gate_head core z weight output partial j #1.0R #fz #fw
      #(hide (reveal nc <: seq bf16)) #z_values #sw #(hide (reveal yv <: seq bf16))
      #(hide (reveal np <: seq f32)) #(fun c -> (reveal p) c j);
    with gated_values. assert output |-> gated_values;
    gated_progress_step logical_head yv gated_values p;
    h := j +^ 1sz;
  };
  with final_state. assert state |-> final_state;
  with final_values. assert output |-> final_values;
  mixer_outputs_intro final_state final_values p;
  assert pure (mixer_outputs final_state final_values p);
}

fn execute_mixer (#d:Type0)
  (weight:A.larray bf16 weight_count{GA.is_global_array weight})
  (qkv:A.larray bf16 6144{GA.is_global_array qkv})
  (z:A.larray bf16 2048{GA.is_global_array z})
  (b:A.larray bf16 16{GA.is_global_array b})
  (a:A.larray bf16 16{GA.is_global_array a})
  (previous:A.larray f32 262144{GA.is_global_array previous})
  (state:A.larray f32 262144{GA.is_global_array state})
  (output:A.larray bf16 2048{GA.is_global_array output})
  (#fw #fq #fz #fb #fa #fr:perm)
  (#sw #sq #z_values #sb #sa #so:erased (seq bf16)) (#sr #before:erased (seq f32))
  (#p:erased (d -> GTot (S.fin 16 -> GTot head_model)))
  preserves cpu ** on gpu_loc (
    weight |-> Frac fw sw ** qkv |-> Frac fq sq ** z |-> Frac fz z_values **
    b |-> Frac fb sb ** a |-> Frac fa sa ** previous |-> Frac fr sr)
  requires on gpu_loc (state |-> before ** output |-> so) **
    pure (mixer_inputs sw sq z_values sb sa sr p)
  ensures exists* (nr:seq f32) (ny:seq bf16). on gpu_loc (state |-> nr ** output |-> ny) **
    pure (Seq.length nr == 262144 /\ Seq.length ny == 2048 /\ mixer_outputs nr ny p)
{
  let query = GA.gpu_array_alloc #f32 128sz;
  let key = GA.gpu_array_alloc #f32 128sz;
  let core = GA.gpu_array_alloc #bf16 128sz;
  let partial = GA.gpu_array_alloc #f32 256sz;
  with q0. assert on gpu_loc (query |-> q0);
  with k0. assert on gpu_loc (key |-> k0);
  with c0. assert on gpu_loc (core |-> c0);
  with p0. assert on gpu_loc (partial |-> p0);
  launch_kernel_1 (fun () -> mixer_kernel weight qkv z b a previous state output query key core partial
    #fw #fq #fz #fb #fa #fr #sw #sq #z_values #sb #sa #so #(hide (reveal c0 <: seq bf16)) #sr #before
    #(hide (reveal q0 <: seq f32)) #(hide (reveal k0 <: seq f32)) #(hide (reveal p0 <: seq f32)) #p);
  GA.gpu_array_free query;
  GA.gpu_array_free key;
  GA.gpu_array_free core;
  GA.gpu_array_free partial;
}

noextract
noeq type logical_call = {
  call:S.invocation;
  weight:S.weights;
  input:S.vector 1024;
  before:cache:S.model_cache{S.cache_domain call cache};
}

// Avoid expanding every input tensor relation inside each host-stage query.
[@@"opaque_to_smt"]
inline_for_extraction noextract
let logical_input_matches
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576) (c:logical_call) : GTot prop =
  input_representation c.call c.weight c.input c.before w x r conv

inline_for_extraction noextract
let interpretations
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576) =
  c:logical_call{logical_input_matches w x r conv c}

inline_for_extraction noextract
let interpretation_intro
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576) (c:logical_call)
  : Lemma
      (requires input_representation c.call c.weight c.input c.before w x r conv)
      (ensures logical_input_matches w x r conv c) =
  norm_spec [delta_once ["KPlay.Functional.logical_input_matches"]] (logical_input_matches w x r conv c)

inline_for_extraction noextract
let interpretation_laws
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576) (c:interpretations w x r conv)
  : Lemma (input_representation c.call c.weight c.input c.before w x r conv) =
  norm_spec [delta_once ["KPlay.Functional.logical_input_matches"]] (logical_input_matches w x r conv c)

noextract
noeq type decoder_stages = {
  normalized:S.vector 1024;
  raw:S.vector 6144;
  projected_z:S.vector 2048;
  projected_b:S.vector 16;
  projected_a:S.vector 16;
  next_conv:S.convolution_state;
  convolved:S.vector 6144;
  head_plans:S.fin 16 -> GTot head_model;
  gated_values:S.vector 2048;
  mixed:S.vector 1024;
  residual:S.vector 1024;
  postnorm:S.vector 1024;
  mlp_gate:S.vector 3584;
  mlp_up:S.vector 3584;
  product:S.vector 3584;
  down:S.vector 1024;
  final_hidden:S.vector 1024;
}

#push-options "--fuel 1,1 --ifuel 1,1 --z3rlimit 40"
[@@"opaque_to_smt"]
noextract
let build_stages (c:logical_call)
  : GTot (s:decoder_stages{
      s.normalized == S.centered_rms_norm c.weight.input_norm c.input /\
      s.raw == S.project c.weight.in_qkv s.normalized /\
      s.projected_z == S.project c.weight.in_z s.normalized /\
      s.projected_b == S.project c.weight.in_beta s.normalized /\
      s.projected_a == S.project c.weight.in_a s.normalized /\
      s.next_conv == S.shift_convolution (old_convolution c.call c.before) s.raw /\
      s.convolved == S.convolve c.weight.conv_kernel s.next_conv /\
      (forall (h:S.fin 16). s.head_plans h == make_head c.weight s.convolved s.projected_z
        s.projected_b s.projected_a (old_recurrent c.call c.before) h) /\
      (forall (i:S.fin 2048). s.gated_values i == (s.head_plans (i / 128)).gated (i % 128)) /\
      s.mixed == S.project c.weight.out_projection s.gated_values /\
      s.residual == S.add_vectors c.input s.mixed /\
      s.postnorm == S.centered_rms_norm c.weight.post_attention_norm s.residual /\
      s.mlp_gate == S.project c.weight.mlp_gate s.postnorm /\
      s.mlp_up == S.project c.weight.mlp_up s.postnorm /\
      (forall (i:S.fin 3584). s.product i == S.silu (s.mlp_gate i) *. s.mlp_up i) /\
      s.down == S.project c.weight.mlp_down s.product /\
      s.final_hidden == S.add_vectors s.residual s.down}) =
  let w = c.weight in
  let normalized = S.centered_rms_norm w.input_norm c.input in
  let raw = S.project w.in_qkv normalized in
  let projected_z = S.project w.in_z normalized in
  let projected_b = S.project w.in_beta normalized in
  let projected_a = S.project w.in_a normalized in
  let next_conv = S.shift_convolution (old_convolution c.call c.before) raw in
  let convolved = S.convolve w.conv_kernel next_conv in
  let head_plans:S.fin 16 -> GTot head_model =
    fun h -> (make_head c.weight convolved projected_z projected_b projected_a
      (old_recurrent c.call c.before) h <: head_model) in
  let gated_values:S.vector 2048 = fun i -> (head_plans (i / 128)).gated (i % 128) in
  let mixed = S.project w.out_projection gated_values in
  let residual = S.add_vectors c.input mixed in
  let postnorm = S.centered_rms_norm w.post_attention_norm residual in
  let mlp_gate = S.project w.mlp_gate postnorm in
  let mlp_up = S.project w.mlp_up postnorm in
  let product:S.vector 3584 = fun i -> S.silu (mlp_gate i) *. mlp_up i in
  let down = S.project w.mlp_down product in
  let final_hidden = S.add_vectors residual down in
  {normalized; raw; projected_z; projected_b; projected_a; next_conv; convolved;
   head_plans; gated_values; mixed; residual; postnorm; mlp_gate; mlp_up; product; down; final_hidden}
#pop-options

[@@"opaque_to_smt"]
inline_for_extraction noextract
let decoder_head_models
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576)
  : GTot (interpretations w x r conv -> GTot (S.fin 16 -> GTot head_model)) =
  fun c -> (build_stages c).head_plans

inline_for_extraction noextract
let stage_head_view (c:logical_call) (h:S.fin 16) : GTot prop =
  let s = build_stages c in
  let p = s.head_plans h in
  valid_head p /\
  (forall (i:S.fin 128). p.qinput i == s.convolved (h * 128 + i)) /\
  (forall (i:S.fin 128). p.kinput i == s.convolved (2048 + h * 128 + i)) /\
  (forall (i:S.fin 128). p.value i == s.convolved (4096 + h * 128 + i)) /\
  (forall (i:S.fin 128). p.z i == s.projected_z (h * 128 + i)) /\
  p.weight == c.weight.gated_norm /\ p.old == old_recurrent c.call c.before h /\
  p.b == s.projected_b h /\ p.a == s.projected_a h /\
  p.a_log == c.weight.a_log h /\ p.dt_bias == c.weight.dt_bias h

#push-options "--fuel 1,1 --ifuel 1,1 --z3rlimit 40"
inline_for_extraction noextract
let stage_head_view_lemma (c:logical_call) (h:S.fin 16) : Lemma (stage_head_view c h) =
  let s = build_stages c in
  let p = make_head c.weight s.convolved s.projected_z s.projected_b s.projected_a
    (old_recurrent c.call c.before) h in
  assert (s.head_plans h == p);
  norm_spec [delta_once ["KPlay.Functional.stage_head_view"]] (stage_head_view c h)
#pop-options

inline_for_extraction noextract
let weight_matrix_view (rows cols:pos) (offset:nat{offset + rows * cols <= weight_count})
  (sw:lseq bf16 weight_count) (w:S.matrix rows cols)
  : Lemma (requires weight_matrix_representation rows cols offset sw w)
      (ensures matrix_rep rows cols weight_count offset cols 1 sw w) =
  norm_spec [delta_only ["KPlay.Functional.weight_matrix_representation"; "KPlay.Functional.row_major"]]
    (weight_matrix_representation rows cols offset sw w);
  norm_spec [delta_once ["KPlay.Functional.matrix_rep"]]
    (matrix_rep rows cols weight_count offset cols 1 sw w)

inline_for_extraction noextract
let weight_matrix_family (#d:Type0) (rows cols:pos)
  (offset:nat{offset + rows * cols <= weight_count})
  (sw:lseq bf16 weight_count) (w:matrix_family d rows cols)
  : Lemma (requires forall (c:d). weight_matrix_representation rows cols offset sw (w c))
      (ensures umatrix rows cols weight_count offset cols 1 sw w) =
  let point (c:d) : Lemma (matrix_rep rows cols weight_count offset cols 1 sw (w c)) =
    weight_matrix_view rows cols offset sw (w c)
  in
  FStar.Classical.forall_intro point;
  norm_spec [delta_once ["KPlay.Functional.umatrix"]]
    (umatrix rows cols weight_count offset cols 1 sw w)

inline_for_extraction noextract
let weight_vector_family (#d:Type0) (width:pos)
  (offset:nat{offset + width <= weight_count})
  (sw:lseq bf16 weight_count) (w:vector_family d width)
  : Lemma (requires forall (c:d). weight_vector_representation width offset sw (w c))
      (ensures segment_rep weight_count offset width sw w) =
  let point (c:d) (i:S.fin width)
    : Lemma (v_approximates (Seq.index sw (offset + i)) (w c i)) =
    norm_spec [delta_once ["KPlay.Functional.weight_vector_representation"]]
      (weight_vector_representation width offset sw (w c))
  in
  FStar.Classical.forall_intro_2 point;
  norm_spec [delta_once ["KPlay.Functional.segment_rep"]]
    (segment_rep weight_count offset width sw w)

inline_for_extraction noextract
let weight_views (#d:Type0) (sw:lseq bf16 weight_count) (w:d -> GTot S.weights)
  : Lemma
      (requires forall (c:d). weights_representation sw (w c))
      (ensures
        segment_rep weight_count input_norm_offset 1024 sw (fun c -> (w c).input_norm) /\
        segment_rep weight_count post_norm_offset 1024 sw (fun c -> (w c).post_attention_norm) /\
        umatrix 6144 1024 weight_count qkv_offset 1024 1 sw (fun c -> (w c).in_qkv) /\
        umatrix 2048 1024 weight_count z_offset 1024 1 sw (fun c -> (w c).in_z) /\
        umatrix 16 1024 weight_count beta_offset 1024 1 sw (fun c -> (w c).in_beta) /\
        umatrix 16 1024 weight_count a_offset 1024 1 sw (fun c -> (w c).in_a) /\
        umatrix 6144 4 weight_count conv_kernel_offset 4 1 sw (fun c -> (w c).conv_kernel) /\
        umatrix 1024 2048 weight_count out_projection_offset 2048 1 sw (fun c -> (w c).out_projection) /\
        umatrix 3584 1024 weight_count mlp_gate_offset 1024 1 sw (fun c -> (w c).mlp_gate) /\
        umatrix 3584 1024 weight_count mlp_up_offset 1024 1 sw (fun c -> (w c).mlp_up) /\
        umatrix 1024 3584 weight_count mlp_down_offset 3584 1 sw (fun c -> (w c).mlp_down)) =
  physical_extents ();
  weight_vector_family 1024 input_norm_offset sw (fun c -> (w c).input_norm);
  weight_vector_family 1024 post_norm_offset sw (fun c -> (w c).post_attention_norm);
  weight_matrix_family 6144 1024 qkv_offset sw (fun c -> (w c).in_qkv);
  weight_matrix_family 2048 1024 z_offset sw (fun c -> (w c).in_z);
  weight_matrix_family 16 1024 beta_offset sw (fun c -> (w c).in_beta);
  weight_matrix_family 16 1024 a_offset sw (fun c -> (w c).in_a);
  weight_matrix_family 6144 4 conv_kernel_offset sw (fun c -> (w c).conv_kernel);
  weight_matrix_family 1024 2048 out_projection_offset sw (fun c -> (w c).out_projection);
  weight_matrix_family 3584 1024 mlp_gate_offset sw (fun c -> (w c).mlp_gate);
  weight_matrix_family 3584 1024 mlp_up_offset sw (fun c -> (w c).mlp_up);
  weight_matrix_family 1024 3584 mlp_down_offset sw (fun c -> (w c).mlp_down)

inline_for_extraction noextract
let convolution_flat_rep (physical:lseq bf16 24576) (logical:S.convolution_state)
  : Lemma (requires convolution_representation physical logical)
      (ensures bf_rep 24576 physical (flat_convolution logical)) =
  let point (i:S.fin 24576)
    : Lemma (v_approximates (Seq.index physical i) (flat_convolution logical i)) =
    convolution_index_covers i;
    let all_channels:squash (forall (c:S.fin qkv_width) (tap:S.fin conv_width).
      v_approximates (Seq.index physical (convolution_index c tap)) (logical c tap)) = () in
    let channel:squash (forall (tap:S.fin conv_width).
      v_approximates (Seq.index physical (convolution_index (i / 4) tap)) (logical (i / 4) tap)) =
      FStar.Classical.Sugar.forall_elim (i / 4) all_channels in
    let value:squash (v_approximates (Seq.index physical (convolution_index (i / 4) (i % 4)))
      (logical (i / 4) (i % 4))) =
      FStar.Classical.Sugar.forall_elim (i % 4) channel in ()
  in FStar.Classical.forall_intro point

inline_for_extraction noextract
let input_views
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576)
  : Lemma (
      ubf 1024 x (fun (c:interpretations w x r conv) -> c.input) /\
      ubf 24576 conv
        (fun (c:interpretations w x r conv) -> flat_convolution (old_convolution c.call c.before)) /\
      (forall (c:interpretations w x r conv). weights_representation w c.weight) /\
      (forall (h:S.fin 16).
        head_rep h r (fun (c:interpretations w x r conv) -> old_recurrent c.call c.before h))) =
  physical_extents ();
  let point (c:interpretations w x r conv)
    : Lemma (
      bf_rep 1024 x c.input /\
      bf_rep 24576 conv (flat_convolution (old_convolution c.call c.before)) /\
      weights_representation w c.weight) =
    interpretation_laws w x r conv c;
    norm_spec [delta_once ["KPlay.Functional.input_representation"]]
      (input_representation c.call c.weight c.input c.before w x r conv);
    convolution_flat_rep conv (old_convolution c.call c.before)
  in
  FStar.Classical.forall_intro point;
  let head (h:S.fin 16)
    : Lemma (head_rep h r (fun (c:interpretations w x r conv) -> old_recurrent c.call c.before h)) =
    let element (c:interpretations w x r conv) (i j:S.fin 128)
      : Lemma (v_approximates (Seq.index r ((h * 128 + i) * 128 + j))
        (old_recurrent c.call c.before h i j)) =
      interpretation_laws w x r conv c;
      norm_spec [delta_only ["KPlay.Functional.input_representation";
        "KPlay.Functional.recurrent_representation"; "KPlay.Functional.recurrent_index"]]
        (input_representation c.call c.weight c.input c.before w x r conv)
    in
    FStar.Classical.forall_intro_3 element;
    norm_spec [delta_once ["KPlay.Functional.head_rep"]]
      (head_rep h r (fun (c:interpretations w x r conv) -> old_recurrent c.call c.before h))
  in
  FStar.Classical.forall_intro head;
  norm_spec [delta_once ["KPlay.Functional.ubf"]]
    (ubf 1024 x (fun (c:interpretations w x r conv) -> c.input));
  norm_spec [delta_once ["KPlay.Functional.ubf"]]
    (ubf 24576 conv (fun (c:interpretations w x r conv) -> flat_convolution (old_convolution c.call c.before)))

inline_for_extraction noextract
let segment_from_vector (#d:Type0) (n offset:nat) (width:nat{offset + width <= n})
  (s:seq bf16) (x:vector_family d n) (y:vector_family d width)
  : Lemma
      (requires ubf n s x /\ (forall (c:d) (i:S.fin width). x c (offset + i) == y c i))
      (ensures segment_rep n offset width s y) =
  norm_spec [delta_only ["KPlay.Functional.ubf"; "KPlay.Functional.bf_rep"]] (ubf n s x);
  norm_spec [delta_once ["KPlay.Functional.segment_rep"]] (segment_rep n offset width s y)

inline_for_extraction noextract
let segment_congr (#d:Type0) (n offset:nat) (width:nat{offset + width <= n})
  (s:seq bf16) (x y:vector_family d width)
  : Lemma
      (requires segment_rep n offset width s x /\ (forall (c:d) (i:S.fin width). x c i == y c i))
      (ensures segment_rep n offset width s y) =
  norm_spec [delta_once ["KPlay.Functional.segment_rep"]] (segment_rep n offset width s x);
  norm_spec [delta_once ["KPlay.Functional.segment_rep"]] (segment_rep n offset width s y)

inline_for_extraction noextract
let head_rep_congr (#d:Type0) (h:S.fin 16) (s:seq f32) (x y:matrix_family d 128 128)
  : Lemma
      (requires head_rep h s x /\ (forall (c:d) (i j:S.fin 128). x c i j == y c i j))
      (ensures head_rep h s y) =
  norm_spec [delta_once ["KPlay.Functional.head_rep"]] (head_rep h s x);
  norm_spec [delta_once ["KPlay.Functional.head_rep"]] (head_rep h s y)

inline_for_extraction noextract
let mixer_ready
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576)
  (q z b a:seq bf16)
  : Lemma
      (requires
        ubf 6144 q (fun (c:interpretations w x r conv) -> (build_stages c).convolved) /\
        ubf 2048 z (fun (c:interpretations w x r conv) -> (build_stages c).projected_z) /\
        ubf 16 b (fun (c:interpretations w x r conv) -> (build_stages c).projected_b) /\
        ubf 16 a (fun (c:interpretations w x r conv) -> (build_stages c).projected_a))
      (ensures mixer_inputs w q z b a r (decoder_head_models w x r conv)) =
  input_views w x r conv;
  let point (c:interpretations w x r conv)
    : Lemma (input_representation c.call c.weight c.input c.before w x r conv) =
    interpretation_laws w x r conv c
  in
  FStar.Classical.forall_intro point;
  let heads (c:interpretations w x r conv) (h:S.fin 16)
    : Lemma (stage_head_view c h) = stage_head_view_lemma c h in
  FStar.Classical.forall_intro_2 heads;
  ubf_congr 16 b
    (fun (c:interpretations w x r conv) -> (build_stages c).projected_b)
    (fun (c:interpretations w x r conv) h -> ((build_stages c).head_plans h).b);
  ubf_congr 16 a
    (fun (c:interpretations w x r conv) -> (build_stages c).projected_a)
    (fun (c:interpretations w x r conv) h -> ((build_stages c).head_plans h).a);
  weight_vector_family 16 a_log_offset w (fun (c:interpretations w x r conv) -> c.weight.a_log);
  weight_vector_family 16 dt_bias_offset w (fun (c:interpretations w x r conv) -> c.weight.dt_bias);
  weight_vector_family 128 gated_norm_offset w (fun (c:interpretations w x r conv) -> c.weight.gated_norm);
  segment_congr weight_count a_log_offset 16 w
    (fun (c:interpretations w x r conv) -> c.weight.a_log)
    (fun (c:interpretations w x r conv) h -> ((build_stages c).head_plans h).a_log);
  segment_congr weight_count dt_bias_offset 16 w
    (fun (c:interpretations w x r conv) -> c.weight.dt_bias)
    (fun (c:interpretations w x r conv) h -> ((build_stages c).head_plans h).dt_bias);
  let head (h:S.fin 16)
    : Lemma (
      (forall (c:interpretations w x r conv). valid_head ((build_stages c).head_plans h)) /\
      segment_rep 6144 (h * 128) 128 q
        (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).qinput) /\
      segment_rep 6144 (2048 + h * 128) 128 q
        (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).kinput) /\
      segment_rep 6144 (4096 + h * 128) 128 q
        (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).value) /\
      segment_rep 2048 (h * 128) 128 z
        (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).z) /\
      segment_rep weight_count gated_norm_offset 128 w
        (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).weight) /\
      head_rep h r (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).old)) =
    segment_from_vector 6144 (h * 128) 128 q
      (fun (c:interpretations w x r conv) -> (build_stages c).convolved)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).qinput);
    segment_from_vector 6144 (2048 + h * 128) 128 q
      (fun (c:interpretations w x r conv) -> (build_stages c).convolved)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).kinput);
    segment_from_vector 6144 (4096 + h * 128) 128 q
      (fun (c:interpretations w x r conv) -> (build_stages c).convolved)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).value);
    segment_from_vector 2048 (h * 128) 128 z
      (fun (c:interpretations w x r conv) -> (build_stages c).projected_z)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).z);
    segment_congr weight_count gated_norm_offset 128 w
      (fun (c:interpretations w x r conv) -> c.weight.gated_norm)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).weight);
    head_rep_congr h r
      (fun (c:interpretations w x r conv) -> old_recurrent c.call c.before h)
      (fun (c:interpretations w x r conv) -> ((build_stages c).head_plans h).old)
  in
  FStar.Classical.forall_intro head;
  norm_spec [delta_only ["KPlay.Functional.mixer_inputs"; "KPlay.Functional.decoder_head_models"]]
    (mixer_inputs w q z b a r (decoder_head_models w x r conv))

noextract
let models_match (c:logical_call)
  : Lemma (
      (forall (i:S.fin 1024).
        (build_stages c).final_hidden i ==
        (S.decoder_values c.call c.weight c.input c.before).hidden_value i) /\
      (forall (h:S.fin 16) (i j:S.fin 128).
        ((build_stages c).head_plans h).updated i j ==
        (S.decoder_values c.call c.weight c.input c.before).recurrent_value h i j) /\
      (forall (channel:S.fin 6144) (tap:S.fin 4).
        (build_stages c).next_conv channel tap ==
        (S.decoder_values c.call c.weight c.input c.before).convolution_value channel tap)) =
  assert (
      (forall (i:S.fin 1024).
        (build_stages c).final_hidden i ==
        (S.decoder_values c.call c.weight c.input c.before).hidden_value i) /\
      (forall (h:S.fin 16) (i j:S.fin 128).
        ((build_stages c).head_plans h).updated i j ==
        (S.decoder_values c.call c.weight c.input c.before).recurrent_value h i j) /\
      (forall (channel:S.fin 6144) (tap:S.fin 4).
        (build_stages c).next_conv channel tap ==
        (S.decoder_values c.call c.weight c.input c.before).convolution_value channel tap))
    by (FStar.Tactics.V2.norm [iota; reduce_projections; unascribe; primops; simplify; delta_only [
      "KPlay.Functional.build_stages"; "KPlay.Functional.make_head";
      "KPlay.Functional.sum_n"; "KPlay.Functional.old_convolution"; "KPlay.Functional.old_recurrent";
      "KPlay.SharedSpec.decoder_values"; "KPlay.SharedSpec.token_mixer"; "KPlay.SharedSpec.feed_forward"
    ]];
    FStar.Tactics.V2.trivial ())

noextract
let output_model_at (#d:Type0)
  (ny:lseq bf16 1024) (nr:lseq f32 262144) (nc:lseq bf16 24576)
  (p:d -> GTot decoder_stages) (c:d) (expected:S.layer_values)
  : Lemma
      (requires
        ubf 1024 ny (fun ctx -> (p ctx).final_hidden) /\
        (forall (h:S.fin 16). head_rep h nr (fun ctx -> ((p ctx).head_plans h).updated)) /\
        ubf 24576 nc (fun ctx -> flat_convolution (p ctx).next_conv) /\
        (forall (i:S.fin 1024). (p c).final_hidden i == expected.hidden_value i) /\
        (forall (h:S.fin 16) (i j:S.fin 128).
          ((p c).head_plans h).updated i j == expected.recurrent_value h i j) /\
        (forall (channel:S.fin 6144) (tap:S.fin 4).
          (p c).next_conv channel tap == expected.convolution_value channel tap))
      (ensures output_representation ny nr nc expected) =
  let hidden_models:squash (forall (ctx:d). bf_rep 1024 ny (p ctx).final_hidden) = () in
  let hidden_model:squash (bf_rep 1024 ny (p c).final_hidden) =
    FStar.Classical.Sugar.forall_elim c hidden_models in
  let conv_models:squash (forall (ctx:d). bf_rep 24576 nc (flat_convolution (p ctx).next_conv)) = () in
  let conv_model:squash (bf_rep 24576 nc (flat_convolution (p c).next_conv)) =
    FStar.Classical.Sugar.forall_elim c conv_models in
  let hidden_point (i:S.fin 1024)
    : Lemma (v_approximates (Seq.index ny i) (expected.hidden_value i)) = () in
  FStar.Classical.forall_intro hidden_point;
  let conv_point (channel:S.fin 6144) (tap:S.fin 4)
    : Lemma (v_approximates (Seq.index nc (convolution_index channel tap))
      (expected.convolution_value channel tap)) =
    assert (convolution_index channel tap == channel * 4 + tap);
    assert ((channel * 4 + tap) / 4 == channel /\ (channel * 4 + tap) % 4 == tap)
  in
  FStar.Classical.forall_intro_2 conv_point;
  let state_point (h:S.fin 16) (i j:S.fin 128)
    : Lemma (v_approximates (Seq.index nr (recurrent_index h i j))
      (expected.recurrent_value h i j)) =
    let this_head:squash (head_rep h nr (fun ctx -> ((p ctx).head_plans h).updated)) =
      FStar.Classical.Sugar.forall_elim h () in
    head_flat_at h nr (fun ctx -> ((p ctx).head_plans h).updated) (i * 128 + j);
    row_index 128 128 i j;
    assert (recurrent_index h i j == h * 16384 + (i * 128 + j))
  in
  FStar.Classical.forall_intro_3 state_point;
  norm_spec [delta_only ["KPlay.Functional.output_representation";
    "KPlay.Functional.bf16_vector_representation"; "KPlay.Functional.recurrent_representation";
    "KPlay.Functional.convolution_representation"]] (output_representation ny nr nc expected)

noextract
let establish_postcondition
  (w:lseq bf16 weight_count) (x:lseq bf16 1024)
  (r:lseq f32 262144) (conv:lseq bf16 24576)
  (ny:lseq bf16 1024) (nr:lseq f32 262144) (nc:lseq bf16 24576)
  (gated:lseq bf16 2048)
  : Lemma
      (requires
        ubf 1024 ny (fun (c:interpretations w x r conv) -> (build_stages c).final_hidden) /\
        mixer_outputs nr gated (decoder_head_models w x r conv) /\
        ubf 24576 nc (fun (c:interpretations w x r conv) -> flat_convolution (build_stages c).next_conv))
      (ensures semantic_postcondition w x r conv ny nr nc) =
  norm_spec [delta_only ["KPlay.Functional.mixer_outputs"; "KPlay.Functional.decoder_head_models"]]
    (mixer_outputs nr gated (decoder_head_models w x r conv));
  let point (call:S.invocation) (weight:S.weights) (input:S.vector 1024)
    (before:S.model_cache{S.cache_domain call before /\
      input_representation call weight input before w x r conv})
    : Lemma (call_postcondition call weight input before ny nr nc) =
    let candidate:logical_call = {call; weight; input; before} in
    interpretation_intro w x r conv candidate;
    let c:interpretations w x r conv = candidate in
    models_match c;
    output_model_at ny nr nc
      (fun (ctx:interpretations w x r conv) -> (build_stages ctx <: decoder_stages))
      c (S.decoder_values call weight input before);
    decoder_values_establish_call_postcondition call weight input before ny nr nc
  in
  FStar.Classical.forall_intro_4 point

#push-options "--fuel 1,1 --ifuel 1,1 --z3rlimit 40"
fn kplay_run
  (#weights:erased (lseq bf16 weight_count))
  (#hidden:erased (lseq bf16 hidden_width))
  (#recurrent:erased (lseq f32 recurrent_count))
  (#convolution:erased (lseq bf16 convolution_count))
  (#weights_perm #hidden_perm #recurrent_perm #convolution_perm:perm)
  (input_weights:larray bf16 weight_count{is_global_array input_weights})
  (input_hidden:larray bf16 hidden_width{is_global_array input_hidden})
  (input_recurrent:larray f32 recurrent_count{is_global_array input_recurrent})
  (input_convolution:larray bf16 convolution_count{is_global_array input_convolution})
  (output_hidden:larray bf16 hidden_width{is_global_array output_hidden})
  (output_recurrent:larray f32 recurrent_count{is_global_array output_recurrent})
  (output_convolution:larray bf16 convolution_count{is_global_array output_convolution})
  preserves cpu **
    on gpu_loc (
      input_weights |-> Frac weights_perm weights **
      input_hidden |-> Frac hidden_perm hidden **
      input_recurrent |-> Frac recurrent_perm recurrent **
      input_convolution |-> Frac convolution_perm convolution)
  requires
    exists* (hidden_before:lseq bf16 hidden_width)
            (recurrent_before:lseq f32 recurrent_count)
            (convolution_before:lseq bf16 convolution_count).
      on gpu_loc (
        output_hidden |-> hidden_before **
        output_recurrent |-> recurrent_before **
        output_convolution |-> convolution_before)
  returns unit
  ensures
    exists* (next_hidden:lseq bf16 hidden_width)
            (next_recurrent:lseq f32 recurrent_count)
            (next_convolution:lseq bf16 convolution_count).
      on gpu_loc (
        output_hidden |-> next_hidden **
        output_recurrent |-> next_recurrent **
        output_convolution |-> next_convolution) **
      pure (semantic_postcondition weights hidden recurrent convolution
              next_hidden next_recurrent next_convolution)
{
  physical_extents ();
  input_views weights hidden recurrent convolution;
  weight_views weights (fun (c:interpretations weights hidden recurrent convolution) -> c.weight);
  let stages:(f:erased (interpretations weights hidden recurrent convolution -> GTot decoder_stages){
      forall (c:interpretations weights hidden recurrent convolution).
        (reveal f) c == build_stages c}) =
    hide (fun (c:interpretations weights hidden recurrent convolution) -> (build_stages c <: decoder_stages));
  let w:(f:erased (interpretations weights hidden recurrent convolution -> GTot S.weights){
      forall (c:interpretations weights hidden recurrent convolution).
        (reveal f) c == c.weight}) =
    hide (fun (c:interpretations weights hidden recurrent convolution) -> c.weight);
  let count = SZ.uint32_to_sizet 21555360ul;
  let qoff = SZ.uint32_to_sizet 11012096ul;
  let zoff = SZ.uint32_to_sizet 17303552ul;
  let boff = SZ.uint32_to_sizet 19400704ul;
  let aoff = SZ.uint32_to_sizet 19417088ul;
  let outoff = SZ.uint32_to_sizet 19458208ul;
  let upoff = SZ.uint32_to_sizet 3672064ul;
  let downoff = SZ.uint32_to_sizet 7342080ul;
  let pw:erased (seq bf16) = hide (reveal weights <: seq bf16);
  let px:erased (seq bf16) = hide (reveal hidden <: seq bf16);
  let pr:erased (seq f32) = hide (reveal recurrent <: seq f32);
  let pc:erased (seq bf16) = hide (reveal convolution <: seq bf16);
  assert pure (reveal #(seq bf16) pw == (reveal #(lseq bf16 weight_count) weights <: seq bf16));
  assert pure (reveal #(seq f32) pr == (reveal #(lseq f32 recurrent_count) recurrent <: seq f32));
  with h0 r0 c0. assert on gpu_loc (
    output_hidden |-> h0 ** output_recurrent |-> r0 ** output_convolution |-> c0);
  let initial_recurrent:erased (seq f32) =
    hide #(seq f32) (reveal #(lseq f32 recurrent_count) r0);
  let initial_convolution:erased (seq bf16) =
    hide #(seq bf16) (reveal #(lseq bf16 convolution_count) c0);

  let normalized = allocate_bf 1024sz;
  execute_rms input_hidden input_weights 0sz normalized
    #hidden_perm #weights_perm #px #pw #(fun (c:interpretations weights hidden recurrent convolution) -> c.input) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).input_norm);
  with sn. assert on gpu_loc (normalized |-> sn);
  ubf_congr 1024 sn
    (fun (c:interpretations weights hidden recurrent convolution) -> S.centered_rms_norm ((reveal w) c).input_norm c.input)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).normalized);
  let raw = allocate_bf 6144sz;
  let z = allocate_bf 2048sz;
  let b = allocate_bf 16sz;
  let a = allocate_bf 16sz;
  matvec 6144sz 1024sz count input_weights qoff 1024sz 1sz normalized raw false
    #weights_perm #1.0R #pw #sn #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).in_qkv) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).normalized);
  matvec 2048sz 1024sz count input_weights zoff 1024sz 1sz normalized z false
    #weights_perm #1.0R #pw #sn #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).in_z) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).normalized);
  matvec 16sz 1024sz count input_weights boff 1024sz 1sz normalized b false
    #weights_perm #1.0R #pw #sn #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).in_beta) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).normalized);
  matvec 16sz 1024sz count input_weights aoff 1024sz 1sz normalized a false
    #weights_perm #1.0R #pw #sn #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).in_a) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).normalized);
  with sr. assert on gpu_loc (raw |-> sr);
  with z_values. assert on gpu_loc (z |-> z_values);
  with sb. assert on gpu_loc (b |-> sb);
  with sa. assert on gpu_loc (a |-> sa);
  ubf_congr 6144 sr (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).in_qkv ((reveal stages) c).normalized)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).raw);
  ubf_congr 2048 z_values (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).in_z ((reveal stages) c).normalized)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).projected_z);
  ubf_congr 16 sb (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).in_beta ((reveal stages) c).normalized)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).projected_b);
  ubf_congr 16 sa (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).in_a ((reveal stages) c).normalized)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).projected_a);

  let convolved = allocate_bf 6144sz;
  execute_convolution input_weights input_convolution raw output_convolution convolved
    #weights_perm #convolution_perm #1.0R #pw #pc #sr #initial_convolution
    #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).conv_kernel) #(fun (c:interpretations weights hidden recurrent convolution) -> old_convolution c.call c.before)
    #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).raw);
  with sq. assert on gpu_loc (convolved |-> sq);
  with nc. assert on gpu_loc (output_convolution |-> nc);
  ubf_congr 6144 sq
    (fun (c:interpretations weights hidden recurrent convolution) -> S.convolve ((reveal w) c).conv_kernel
      (S.shift_convolution (old_convolution c.call c.before) ((reveal stages) c).raw))
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).convolved);
  ubf_congr 24576 nc
    (fun (c:interpretations weights hidden recurrent convolution) -> flat_convolution (S.shift_convolution (old_convolution c.call c.before) ((reveal stages) c).raw))
    (fun (c:interpretations weights hidden recurrent convolution) -> flat_convolution (build_stages c).next_conv);
  let gated = GA.gpu_array_alloc #bf16 2048sz;
  with g0. assert on gpu_loc (gated |-> g0);
  mixer_ready weights hidden recurrent convolution sq z_values sb sa;
  execute_mixer #(interpretations weights hidden recurrent convolution)
    input_weights convolved z b a input_recurrent output_recurrent gated
    #weights_perm #1.0R #1.0R #1.0R #1.0R #recurrent_perm
    #pw #sq #z_values #sb #sa #(hide (reveal g0 <: seq bf16)) #pr #initial_recurrent
    #(decoder_head_models weights hidden recurrent convolution);
  with sgated. assert on gpu_loc (gated |-> sgated);
  with nr. assert on gpu_loc (output_recurrent |-> nr);
  norm_spec [delta_once ["KPlay.Functional.decoder_head_models"]]
    (mixer_outputs nr sgated (decoder_head_models weights hidden recurrent convolution));
  ubf_congr 2048 sgated
    (fun (c:interpretations weights hidden recurrent convolution) i -> ((build_stages c).head_plans (i / 128)).gated (i % 128))
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).gated_values);

  let mixed = allocate_bf 1024sz;
  matvec 1024sz 2048sz count input_weights outoff 2048sz 1sz gated mixed false
    #weights_perm #1.0R #pw #sgated #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).out_projection) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).gated_values);
  with sm. assert on gpu_loc (mixed |-> sm);
  ubf_congr 1024 sm
    (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).out_projection ((reveal stages) c).gated_values)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mixed);
  let residual = allocate_bf 1024sz;
  pointwise 1024sz false input_hidden mixed residual #hidden_perm #1.0R #px #sm
    #(fun (c:interpretations weights hidden recurrent convolution) -> c.input) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mixed);
  with sres. assert on gpu_loc (residual |-> sres);
  ubf_congr 1024 sres (fun (c:interpretations weights hidden recurrent convolution) i -> c.input i +. ((reveal stages) c).mixed i)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).residual);
  let postnorm = allocate_bf 1024sz;
  execute_rms residual input_weights 1024sz postnorm #1.0R #weights_perm #sres #pw
    #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).residual) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).post_attention_norm);
  with spost. assert on gpu_loc (postnorm |-> spost);
  ubf_congr 1024 spost
    (fun (c:interpretations weights hidden recurrent convolution) -> S.centered_rms_norm ((reveal w) c).post_attention_norm ((reveal stages) c).residual)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).postnorm);
  let gate = allocate_bf 3584sz;
  let up = allocate_bf 3584sz;
  matvec 3584sz 1024sz count input_weights 2048sz 1024sz 1sz postnorm gate false
    #weights_perm #1.0R #pw #spost #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).mlp_gate) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).postnorm);
  matvec 3584sz 1024sz count input_weights upoff 1024sz 1sz postnorm up false
    #weights_perm #1.0R #pw #spost #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).mlp_up) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).postnorm);
  with sgate. assert on gpu_loc (gate |-> sgate);
  with sup. assert on gpu_loc (up |-> sup);
  ubf_congr 3584 sgate (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).mlp_gate ((reveal stages) c).postnorm)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mlp_gate);
  ubf_congr 3584 sup (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).mlp_up ((reveal stages) c).postnorm)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mlp_up);
  let product = allocate_bf 3584sz;
  pointwise 3584sz true gate up product #1.0R #1.0R #sgate #sup
    #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mlp_gate) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).mlp_up);
  with sproduct. assert on gpu_loc (product |-> sproduct);
  ubf_congr 3584 sproduct
    (fun (c:interpretations weights hidden recurrent convolution) i -> S.silu (((reveal stages) c).mlp_gate i) *. ((reveal stages) c).mlp_up i)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).product);
  let down = allocate_bf 1024sz;
  matvec 1024sz 3584sz count input_weights downoff 3584sz 1sz product down true
    #weights_perm #1.0R #pw #sproduct #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal w) c).mlp_down) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).product);
  with sdown. assert on gpu_loc (down |-> sdown);
  ubf_congr 1024 sdown (fun (c:interpretations weights hidden recurrent convolution) -> S.project ((reveal w) c).mlp_down ((reveal stages) c).product)
    (fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).down);
  pointwise 1024sz false residual down output_hidden #1.0R #1.0R #sres #sdown
    #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).residual) #(fun (c:interpretations weights hidden recurrent convolution) -> ((reveal stages) c).down);
  with ny. assert on gpu_loc (output_hidden |-> ny);
  ubf_congr 1024 ny (fun (c:interpretations weights hidden recurrent convolution) i -> ((reveal stages) c).residual i +. ((reveal stages) c).down i)
    (fun (c:interpretations weights hidden recurrent convolution) -> (build_stages c).final_hidden);
  establish_postcondition weights hidden recurrent convolution ny nr nc sgated;

  GA.gpu_array_free normalized;
  GA.gpu_array_free raw;
  GA.gpu_array_free z;
  GA.gpu_array_free b;
  GA.gpu_array_free a;
  GA.gpu_array_free convolved;
  GA.gpu_array_free gated;
  GA.gpu_array_free mixed;
  GA.gpu_array_free residual;
  GA.gpu_array_free postnorm;
  GA.gpu_array_free gate;
  GA.gpu_array_free up;
  GA.gpu_array_free product;
  GA.gpu_array_free down;
}
#pop-options
