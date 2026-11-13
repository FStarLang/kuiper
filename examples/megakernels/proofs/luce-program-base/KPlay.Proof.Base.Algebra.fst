module KPlay.Proof.Base.Algebra

open FStar.Real
module C = Kuiper.Chest
module G = Kuiper.Spec.GEMM
module Sum = Kuiper.Sum
module P = KPlay.Proof.Shared.Projection

inline_for_extraction noextract
let vector (n:nat) = P.vector real n

inline_for_extraction noextract
let rec sum_ext (#n:nat) (f g:vector n) (count:nat{count <= n})
  : Lemma
      (requires forall (i:nat{i < count}). f i == g i)
      (ensures Sum.sum 0 count f == Sum.sum 0 count g)
      (decreases count) =
  if count > 0 then (
    sum_ext #n f g (count - 1);
    Sum.sum_pop_right 0 count f;
    Sum.sum_pop_right 0 count g
  )

inline_for_extraction noextract
let shifted (#n:nat) (f:vector n) (start:nat{start <= n}) : vector (n - start) =
  fun i -> f (start + i)

inline_for_extraction noextract
let rec shift_sum (#n:nat) (f:vector n) (start:nat{start <= n})
  (count:nat{start + count <= n})
  : Lemma
      (ensures Sum.sum start (start + count) f ==
        Sum.sum 0 count (shifted #n f start))
      (decreases count) =
  if count > 0 then (
    shift_sum #n f start (count - 1);
    Sum.sum_pop_right start (start + count) f;
    Sum.sum_pop_right 0 count (shifted #n f start)
  )

inline_for_extraction noextract
let dot_terms (a:C.chest2 real 16 16) (b:C.chest2 real 16 8)
  (row:nat{row < 16}) (col:nat{col < 8}) : vector 16 =
  fun i -> C.acc2 a row i *. C.acc2 b i col

inline_for_extraction noextract
let rec matmul_sum (a:C.chest2 real 16 16) (b:C.chest2 real 16 8)
  (row:nat{row < 16}) (col:nat{col < 8}) (count:nat{count <= 16})
  : Lemma
      (ensures G.__matmul_single a b row col count ==
        Sum.sum 0 count (dot_terms a b row col))
      (decreases count) =
  if count = 0 then
    G.__gmatmul_single_zero_lemma 0.0R ( *. ) ( +. ) a b row col
  else (
    matmul_sum a b row col (count - 1);
    G.matmul_single_lemma a b row col count;
    Sum.sum_pop_right 0 count (dot_terms a b row col)
  )

inline_for_extraction noextract
let products (#n:nat) (x y:vector n) : vector n =
  fun i -> x i *. y i

inline_for_extraction noextract
let product_comm_sum (#n:nat) (x y:vector n)
  : Lemma (
      Sum.sum 0 n (products x y) == Sum.sum 0 n (products y x)) =
  sum_ext #n (products x y) (products y x) n

#push-options "--fuel 1 --ifuel 1"

inline_for_extraction noextract
let project_products (#h #v:pos)
  (normalized:vector h) (embedding:P.matrix real v h) (token:P.index v)
  : Lemma (
      P.project #h #v normalized embedding token ==
      Sum.sum #real #Sum.real_add_semigroup 0 h (products #h (embedding token) normalized)) =
  sum_ext #h (fun (i:P.index h) -> normalized i *. embedding token i)
    (products #h (embedding token) normalized) h;
  assert_norm (P.project #h #v normalized embedding token ==
    Sum.sum #real #Sum.real_add_semigroup 0 h
      (fun (i:P.index h) -> normalized i *. embedding token i))

#pop-options
