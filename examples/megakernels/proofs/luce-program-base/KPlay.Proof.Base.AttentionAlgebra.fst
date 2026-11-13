module KPlay.Proof.Base.AttentionAlgebra

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open FStar.Real
open KPlay.Proof.Base.ReductionMath
module S = KPlay.Proof.Shared.Attention
module R = FStar.Math.Sqrt
module SQ = FStar.Math.Sqrt
module E = FStar.Math.Exp
module Proof = FStar.Tactics.V2

inline_for_extraction noextract
let bounded_sum (#n:nat) (count:nat{count <= n}) (terms:S.vector n) : real =
  S.sum count (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_step (#n:nat) (count:nat{count < n}) (terms:S.vector n)
  : Lemma (
      bounded_sum (count + 1) terms == bounded_sum count terms +. terms count) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]] (bounded_sum (count + 1) terms);
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]] (bounded_sum count terms);
  sum_step (count + 1) (fun i -> terms i);
  sum_ext count (fun (i:S.index (count + 1 - 1)) -> terms i) (fun i -> terms i)

inline_for_extraction noextract
let bounded_sum_full (#n:nat) (terms:S.vector n)
  : Lemma (bounded_sum n terms == S.sum n terms) =
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]] (bounded_sum n terms);
  sum_ext n (fun i -> terms i) terms

inline_for_extraction noextract
let partial_sum (n:nat{n <= 256}) (logical:S.vector 256) : real =
  S.sum n (fun i -> logical i)

inline_for_extraction noextract
let partial_sum_one (logical:S.vector 256)
  : Lemma (partial_sum 1 logical == logical 0) =
  sum_step 1 (fun i -> logical i);
  sum_zero (fun i -> logical i)

inline_for_extraction noextract
let row_index (nr nc:pos) (r:nat{r < nr}) (c:nat{c < nc})
  : Lemma (
      r * nc + c < nr * nc /\ (r * nc + c) / nc == r /\ (r * nc + c) % nc == c) = ()

inline_for_extraction noextract
let rec zero_suffix_sum (n:nat) (count:nat{count <= n}) (terms:S.vector n)
  : Lemma
      (requires forall (i:nat{count <= i /\ i < n}). terms i == 0.0R)
      (ensures S.sum n terms == bounded_sum count terms)
      (decreases (n - count)) =
  if n = count then bounded_sum_full terms
  else (
    let lower:S.vector (n - 1) = fun i -> terms i in
    zero_suffix_sum (n - 1) count lower;
    sum_step n terms;
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]]
      (bounded_sum count lower);
    norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]]
      (bounded_sum count terms);
    sum_ext count (fun i -> lower i) (fun i -> terms i)
  )

inline_for_extraction noextract
let rec sum_all_zero (n:nat) (f:S.vector n)
  : Lemma (requires forall (i:S.index n). f i == 0.0R)
          (ensures S.sum n f == 0.0R)
          (decreases n) =
  if n = 0 then sum_zero f
  else (
    sum_all_zero (n - 1) (fun i -> f i);
    sum_step n f
  )

inline_for_extraction noextract
let rec sum_swap (n m:nat) (f:S.matrix n m)
  : Lemma (
      S.sum n (fun i -> S.sum m (fun j -> f i j)) ==
      S.sum m (fun j -> S.sum n (fun i -> f i j)))
      (decreases n) =
  if n = 0 then (
    sum_zero (fun i -> S.sum m (fun j -> f i j));
    sum_all_zero m (fun j -> S.sum n (fun i -> f i j))
  ) else (
    sum_swap (n - 1) m (fun i j -> f i j);
    sum_step n (fun i -> S.sum m (fun j -> f i j));
    sum_plus m
      (fun j -> S.sum (n - 1) (fun i -> f i j))
      (fun j -> f (n - 1) j);
    let point (j:S.index m)
      : Lemma (
          S.sum n (fun i -> f i j) ==
          S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j) =
      sum_step n (fun i -> f i j)
    in
    FStar.Classical.forall_intro #(S.index m) #(fun j ->
      S.sum n (fun i -> f i j) ==
      S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j) point;
    sum_ext m
      (fun j -> S.sum n (fun i -> f i j))
      (fun j -> S.sum (n - 1) (fun i -> f i j) +. f (n - 1) j)
  )

inline_for_extraction noextract
let rec sum_grid (rows:nat) (cols:pos) (f:S.vector (rows * cols))
  : Lemma (
      S.sum rows (fun r -> S.sum cols (fun c -> f (r * cols + c))) ==
      S.sum (rows * cols) f)
      (decreases rows) =
  if rows = 0 then (
    sum_zero (fun r -> S.sum cols (fun c -> f (r * cols + c)));
    sum_zero f
  ) else (
    sum_grid (rows - 1) cols (fun i -> f i);
    sum_step rows (fun r -> S.sum cols (fun c -> f (r * cols + c)));
    sum_split ((rows - 1) * cols) cols f
  )

inline_for_extraction noextract
let slots (n width:pos) : pos = (n + width - 1) / width

inline_for_extraction noextract
let slots_bounds (n width:pos)
  : Lemma (n <= slots n width * width /\ slots n width * width < n + width) = ()

inline_for_extraction noextract
let padded_terms (n:pos) (extent:pos{n <= extent}) (terms:S.vector n) : S.vector extent =
  fun i -> if i < n then terms i else 0.0R

inline_for_extraction noextract
let stripe_terms
  (n width:pos) (group:nat{group < width}) (terms:S.vector n)
  : S.vector (slots n width) =
  fun q -> if q * width + group < n then terms (q * width + group) else 0.0R

inline_for_extraction noextract
let group_sums
  (n:pos) (width:pos{width <= 32}) (terms:S.vector n) : S.vector 256 =
  fun group ->
    if group < width then S.sum (slots n width) (stripe_terms n width group terms)
    else 0.0R

inline_for_extraction noextract
let stripes_sum
  (n:pos) (width:pos{width <= 32}) (terms:S.vector n)
  : Lemma (partial_sum width (group_sums n width terms) == S.sum n terms) =
  slots_bounds n width;
  let padded = padded_terms n (slots n width * width) terms in
  let stripe (group:S.index width)
    : Lemma (
        group_sums n width terms (group <: nat) ==
        S.sum (slots n width) (fun q -> padded (q * width + group))) =
    sum_ext (slots n width) (stripe_terms n width group terms)
      (fun q -> padded (q * width + group))
  in
  FStar.Classical.forall_intro #(S.index width) #(fun group ->
    group_sums n width terms (group <: nat) ==
    S.sum (slots n width) (fun q -> padded (q * width + group))) stripe;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.partial_sum"]]
    (partial_sum width (group_sums n width terms));
  sum_ext width
    (fun group -> group_sums n width terms (group <: nat))
    (fun group -> S.sum (slots n width) (fun q -> padded (q * width + group)));
  sum_swap width (slots n width) (fun group q -> padded (q * width + group));
  sum_grid (slots n width) width padded;
  zero_suffix_sum (slots n width * width) n padded;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]]
    (bounded_sum n padded);
  sum_ext n (fun i -> padded (i <: nat)) terms

inline_for_extraction noextract
let chunk_work (n groups chunk:pos) : pos = slots n (groups * chunk) * chunk

inline_for_extraction noextract
let chunk_index (groups chunk:pos) (group:nat{group < groups}) (q:nat) : nat =
  (q / chunk) * (groups * chunk) + group * chunk + q % chunk

inline_for_extraction noextract
let chunk_terms (n groups chunk:pos) (group:nat{group < groups}) (terms:S.vector n)
  : S.vector (chunk_work n groups chunk) =
  fun q ->
    let i = chunk_index groups chunk group q in
    if i < n then terms i else 0.0R

inline_for_extraction noextract
let chunk_group_sums
  (n:pos) (groups:pos{groups <= 32}) (chunk:pos) (terms:S.vector n)
  : S.vector 256 =
  fun group ->
    if group < groups then S.sum (chunk_work n groups chunk) (chunk_terms n groups chunk group terms)
    else 0.0R

inline_for_extraction noextract
let chunk_coordinate
  (blocks groups chunk:pos) (group:nat{group < groups})
  (q:nat{q < blocks}) (o:nat{o < chunk})
  : Lemma (
      chunk_index groups chunk group (q * chunk + o) ==
      q * (groups * chunk) + group * chunk + o /\
      chunk_index groups chunk group (q * chunk + o) < blocks * (groups * chunk)) =
  row_index blocks chunk q o

inline_for_extraction noextract
let grid_index (rows cols:pos) (r:nat{r < rows}) (c:nat{c < cols})
  : i:nat{i < rows * cols /\ i == r * cols + c} =
  row_index rows cols r c;
  r * cols + c

inline_for_extraction noextract
let block_values (blocks width:pos) (padded:S.vector (blocks * width))
  (q:nat{q < blocks}) : S.vector width =
  fun i -> padded (q * width + i)

inline_for_extraction noextract
let block_values_sum (blocks width:pos) (padded:S.vector (blocks * width))
  (q:nat{q < blocks})
  : Lemma (
      S.sum width (block_values blocks width padded q) ==
      S.sum width (fun i -> padded (q * width + i))) =
  sum_ext width (block_values blocks width padded q) (fun i -> padded (q * width + i))

inline_for_extraction noextract
let block_rowsum (blocks groups chunk:pos)
  (padded:S.vector (blocks * (groups * chunk)))
  (q:nat{q < blocks}) (g:nat{g < groups}) : real =
  S.sum chunk (fun o ->
    padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))

inline_for_extraction noextract
let block_rowsums_total (blocks groups chunk:pos)
  (padded:S.vector (blocks * (groups * chunk))) (q:nat{q < blocks})
  : Lemma (
      S.sum groups (fun g -> block_rowsum blocks groups chunk padded q g) ==
      S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) =
  sum_grid groups chunk (block_values blocks (groups * chunk) padded q);
  let same (g:S.index groups)
    : Lemma (
        block_rowsum blocks groups chunk padded q g ==
        S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))) =
    sum_ext chunk
      (fun o -> padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))
      (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))
  in
  FStar.Classical.forall_intro #(S.index groups) #(fun g ->
    block_rowsum blocks groups chunk padded q g ==
    S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o))) same;
  sum_ext groups
    (fun g -> block_rowsum blocks groups chunk padded q g)
    (fun g -> S.sum chunk (fun o -> block_values blocks (groups * chunk) padded q (g * chunk + o)))

inline_for_extraction noextract
let chunk_groups_sum
  (n:pos) (groups:pos{groups <= 32}) (chunk:pos) (terms:S.vector n)
  : Lemma (partial_sum groups (chunk_group_sums n groups chunk terms) == S.sum n terms) =
  slots_bounds n (groups * chunk);
  let blocks = slots n (groups * chunk) in
  let padded = padded_terms n (blocks * (groups * chunk)) terms in
  let rowsums (q:S.index blocks) (g:S.index groups) : real =
    block_rowsum blocks groups chunk padded q g
  in
  let one_group (g:S.index groups)
    : Lemma (
        chunk_group_sums n groups chunk terms (g <: nat) ==
        S.sum blocks (fun q -> rowsums q g)) =
    let expanded : S.vector (blocks * chunk) = chunk_terms n groups chunk g terms in
    sum_grid blocks chunk expanded;
    let one_block (q:S.index blocks)
      : Lemma (
          S.sum chunk (fun o -> expanded (grid_index blocks chunk q o)) == rowsums q g) =
      let point (o:S.index chunk)
        : Lemma (
            expanded (grid_index blocks chunk q o) ==
            padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o))) =
        chunk_coordinate blocks groups chunk g q o
      in
      FStar.Classical.forall_intro #(S.index chunk) #(fun o ->
        expanded (grid_index blocks chunk q o) ==
        padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o))) point;
      sum_ext chunk (fun o -> expanded (grid_index blocks chunk q o))
        (fun o -> padded (grid_index blocks (groups * chunk) q (grid_index groups chunk g o)))
    in
    FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
      S.sum chunk (fun o -> expanded (grid_index blocks chunk q o)) == rowsums q g) one_block;
    let grid_block (q:S.index blocks)
      : Lemma (
          S.sum chunk (fun o -> expanded (q * chunk + o)) ==
          S.sum chunk (fun o -> expanded (grid_index blocks chunk q o))) =
      sum_ext chunk (fun o -> expanded (q * chunk + o))
        (fun o -> expanded (grid_index blocks chunk q o))
    in
    FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
      S.sum chunk (fun o -> expanded (q * chunk + o)) ==
      S.sum chunk (fun o -> expanded (grid_index blocks chunk q o))) grid_block;
    sum_ext blocks
      (fun q -> S.sum chunk (fun o -> expanded (q * chunk + o)))
      (fun q -> rowsums q g)
  in
  FStar.Classical.forall_intro #(S.index groups) #(fun g ->
    chunk_group_sums n groups chunk terms (g <: nat) ==
    S.sum blocks (fun q -> rowsums q g)) one_group;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.partial_sum"]]
    (partial_sum groups (chunk_group_sums n groups chunk terms));
  sum_ext groups
    (fun g -> chunk_group_sums n groups chunk terms (g <: nat))
    (fun g -> S.sum blocks (fun q -> rowsums q g));
  sum_swap groups blocks (fun g q -> rowsums q g);
  let block_sum (q:S.index blocks)
    : Lemma (
        S.sum groups (fun g -> rowsums q g) ==
        S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) =
    block_rowsums_total blocks groups chunk padded q;
    sum_ext groups (fun g -> rowsums q g) (fun g -> block_rowsum blocks groups chunk padded q g)
  in
  FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
    S.sum groups (fun g -> rowsums q g) ==
    S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q)) block_sum;
  sum_ext blocks
    (fun q -> S.sum groups (fun g -> rowsums q g))
    (fun q -> S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q));
  sum_grid blocks (groups * chunk) padded;
  let flat_block (q:S.index blocks)
    : Lemma (
        S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q) ==
        S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i))) =
    block_values_sum blocks (groups * chunk) padded q
  in
  FStar.Classical.forall_intro #(S.index blocks) #(fun q ->
    S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q) ==
    S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i))) flat_block;
  sum_ext blocks
    (fun q -> S.sum (groups * chunk) (block_values blocks (groups * chunk) padded q))
    (fun q -> S.sum (groups * chunk) (fun i -> padded (q * (groups * chunk) + i)));
  zero_suffix_sum (blocks * (groups * chunk)) n padded;
  norm_spec [delta_once ["KPlay.Proof.Base.AttentionAlgebra.bounded_sum"]] (bounded_sum n padded);
  sum_ext n (fun i -> padded (i <: nat)) terms

inline_for_extraction noextract
let linear_terms (rows cols:pos) (w:S.matrix rows cols) (x:S.vector cols)
  (row:nat{row < rows}) : S.vector cols =
  fun c -> w row c *. x c

inline_for_extraction noextract
let chunk_runtime_bounds
  (n:pos{n <= 647}) (groups:pos{groups = 16 \/ groups = 32})
  (chunk:pos{chunk <= 64}) (group:nat{group < groups})
  (q:nat{q <= chunk_work n groups chunk})
  : Lemma (
      groups * chunk <= 2048 /\
      chunk_work n groups chunk <= 2695 /\
      q / chunk * (groups * chunk) < 10000 /\
      q / chunk * (groups * chunk) + group * chunk < 10000 /\
      chunk_index groups chunk group q < 10000) =
  slots_bounds n (groups * chunk)

inline_for_extraction noextract
let partial_bounded (n:nat{n <= 256}) (x:S.vector 256)
  : Lemma (partial_sum n x == bounded_sum #256 n x) = ()

inline_for_extraction noextract
let rec sum_scale (n:nat) (f:S.vector n) (c:real)
  : Lemma (S.sum n (fun i -> c *. f i) == c *. S.sum n f)
      (decreases n) =
  if n = 0 then (
    sum_zero f;
    sum_zero (fun i -> c *. f i)
  ) else (
    sum_scale (n - 1) (fun i -> f i) c;
    sum_step n f;
    sum_step n (fun i -> c *. f i)
  )

inline_for_extraction noextract
let rec positive_sum_bridge (n:pos) (f:S.index n -> R.rpos)
  : Lemma (S.sum n f == S.sum_positive n f)
      (decreases n) =
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_positive"]] (S.sum_positive n f);
  sum_step n f;
  if n = 1 then sum_zero (fun i -> f i)
  else (
    let lower : S.index (n - 1) -> R.rpos = fun i -> f i in
    positive_sum_bridge (n - 1) lower;
    sum_ext (n - 1) lower (fun i -> f i)
  )

inline_for_extraction noextract
let multiply_reciprocal (x:real) (y:real{y =!= 0.0R})
  : Lemma (x *. (1.0R /. y) == x /. y) = ()

inline_for_extraction noextract
let scaled_scores (n:nat) (x:S.vector n) : S.vector n =
  fun i -> x i *. 0.0625R

inline_for_extraction noextract
let exponentials (n:nat) (scores:S.vector n) (shift:real)
  : S.index n -> SQ.rpos =
  fun i -> E.exp_positive (scores i -. shift); E.exp (scores i -. shift)

inline_for_extraction noextract
let exponential_sum (n:pos) (scores:S.vector n) (shift:real) : SQ.rpos =
  S.sum_positive n (exponentials n scores shift)

inline_for_extraction noextract
let exponential_sum_bridge (n:pos) (scores:S.vector n) (shift:real)
  : Lemma (
      S.sum n (exponentials n scores shift) == exponential_sum n scores shift) =
  positive_sum_bridge n (exponentials n scores shift)

inline_for_extraction noextract
let exponential_shift (n:pos) (scores:S.vector n) (shift:real)
  : Lemma (
      exponential_sum n scores shift == exponential_sum n scores 0.0R /. E.exp shift) =
  E.exp_positive shift;
  let point (i:S.index n)
    : Lemma (
        exponentials n scores shift i ==
        (1.0R /. E.exp shift) *. exponentials n scores 0.0R i) =
    E.exp_sub shift (scores i)
  in
  FStar.Classical.forall_intro #(S.index n) #(fun i ->
    exponentials n scores shift i ==
    (1.0R /. E.exp shift) *. exponentials n scores 0.0R i) point;
  sum_ext n (exponentials n scores shift)
    (fun i -> (1.0R /. E.exp shift) *. exponentials n scores 0.0R i);
  sum_scale n (exponentials n scores 0.0R) (1.0R /. E.exp shift);
  exponential_sum_bridge n scores shift;
  exponential_sum_bridge n scores 0.0R;
  multiply_reciprocal (exponential_sum n scores 0.0R) (E.exp shift)

inline_for_extraction noextract
let softmax_denominator (n:pos) (scores:S.vector n) (i:S.index n)
  : Lemma (
      S.softmax n scores i == E.exp (scores i) /. exponential_sum n scores 0.0R) =
  let original : S.index n -> SQ.rpos =
    fun j -> E.exp_positive (scores j); E.exp (scores j)
  in
  positive_sum_bridge n original;
  sum_ext n original (exponentials n scores 0.0R);
  exponential_sum_bridge n scores 0.0R;
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.softmax"]] (S.softmax n scores i)

inline_for_extraction noextract
let cancel_division (a:real) (b c:SQ.rpos)
  : Lemma ((a /. c) /. (b /. c) == a /. b) = ()

inline_for_extraction noextract
let softmax_shift_value (n:pos) (scores:S.vector n) (shift:real) (i:S.index n)
  : Lemma (
      E.exp (scores i -. shift) /. exponential_sum n scores shift ==
      S.softmax n scores i) =
  E.exp_positive shift;
  E.exp_sub shift (scores i);
  exponential_shift n scores shift;
  softmax_denominator n scores i;
  cancel_division (E.exp (scores i)) (exponential_sum n scores 0.0R) (E.exp shift)

inline_for_extraction noextract
let stripe_tail_zero
  (n width:pos) (group:nat{group < width})
  (q:nat{q <= slots n width}) (terms:S.vector n)
  : Lemma
      (requires q == slots n width \/ n <= q * width + group)
      (ensures forall (i:nat{q <= i /\ i < slots n width}).
        stripe_terms n width group terms (i <: nat) == 0.0R) = ()

inline_for_extraction noextract
let rec sum_squares_bridge (n:nat) (x:S.vector n)
  : Lemma (S.sum n (fun i -> x i *. x i) == S.sum_squares n x)
      (decreases n) =
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.sum_squares"]] (S.sum_squares n x);
  if n = 0 then sum_zero (fun i -> x i *. x i)
  else (
    sum_squares_bridge (n - 1) (fun i -> x i);
    sum_step n (fun i -> x i *. x i)
  )

inline_for_extraction noextract
let activation_value (silu:bool) (x:real) : real =
  if silu then S.silu x else S.sigmoid x

inline_for_extraction noextract
let activated_product (silu:bool) (gate other:real) : real =
  if silu then S.silu gate *. other else other *. S.sigmoid gate

inline_for_extraction noextract
let activation_vector (n:nat) (silu:bool) (gate other:S.vector n) : S.vector n =
  fun i -> activated_product silu (gate i) (other i)

inline_for_extraction noextract
let head_vector (heads:pos) (values:S.matrix heads 256) (h:nat{h < heads}) : S.vector 256 =
  values h

inline_for_extraction noextract
let normalized_heads (heads:pos) (weight:S.vector 256) (values:S.matrix heads 256)
  : S.vector (heads * 256) =
  fun i -> S.rms_norm 256 weight (head_vector heads values (i / 256)) (i % 256)

inline_for_extraction noextract
let rotated_matrix (heads:pos) (position:nat) (weight:S.vector 256)
  (inputs:S.matrix heads 256) : S.matrix heads 256 =
  fun h -> S.rotate_partial position (S.rms_norm 256 weight (head_vector heads inputs h))

inline_for_extraction noextract
let rotation_congr (position:nat) (f g:S.vector 256) (d:S.index 256)
  : Lemma (requires forall (i:S.index 256). f i == g i)
          (ensures S.rotate_partial position f d == S.rotate_partial position g d) = ()

inline_for_extraction noextract
let flat_heads (heads:pos) (values:S.matrix heads 256) : S.vector (heads * 256) =
  fun i -> values (i / 256) (i % 256)

inline_for_extraction noextract
let score_matrix (n:pos) (history:S.kv_history{history.length == n}) (head:S.index 8)
  : S.matrix n 256 =
  fun t d -> history.keys (S.kv_head head) t d

inline_for_extraction noextract
let value_matrix (n:pos) (history:S.kv_history{history.length == n}) (head:S.index 8)
  : S.matrix 256 n =
  fun d t -> history.values (S.kv_head head) t d

inline_for_extraction noextract
let score_dots (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  fun t -> S.sum 256 (fun d -> queries head d *. history.keys (S.kv_head head) t d)

inline_for_extraction noextract
let attention_scores (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  fun t -> score_dots position queries history head t *. (1.0R /. 16.0R) +. S.causal_bias position t

inline_for_extraction noextract
let attention_probabilities (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8)
  : S.vector (position + 1) =
  S.softmax (position + 1) (attention_scores position queries history head)

inline_for_extraction noextract
let softmax_congr (n:pos) (f g:S.vector n) (i:S.index n)
  : Lemma
      (requires forall (j:S.index n). f j == g j)
      (ensures S.softmax n f i == S.softmax n g i) =
  sum_ext n (exponentials n f 0.0R) (exponentials n g 0.0R);
  exponential_sum_bridge n f 0.0R;
  exponential_sum_bridge n g 0.0R;
  softmax_denominator n f i;
  softmax_denominator n g i

inline_for_extraction noextract
let score_linear (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (t:S.index (position + 1))
  : Lemma (
      S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t ==
      score_dots position queries history head t) =
  sum_ext 256
    (fun d -> score_matrix (position + 1) history head t d *. queries head d)
    (fun d -> queries head d *. history.keys (S.kv_head head) t d);
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.linear"]]
    (S.linear (position + 1) 256 (score_matrix (position + 1) history head) (queries head) t)

inline_for_extraction noextract
let attention_unfold (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (d:S.index 256)
  : Lemma (
      S.attention position queries history head d ==
      S.sum (position + 1) (fun t ->
        attention_probabilities position queries history head t *. history.values (S.kv_head head) t d)) =
  assert (
      S.attention position queries history head d ==
      S.sum (position + 1) (fun t ->
        attention_probabilities position queries history head t *. history.values (S.kv_head head) t d))
    by (
      FStar.Tactics.V2.norm [delta_only [
        "KPlay.Proof.Shared.Attention.attention";
        "KPlay.Proof.Base.AttentionAlgebra.attention_probabilities";
        "KPlay.Proof.Base.AttentionAlgebra.attention_scores";
        "KPlay.Proof.Base.AttentionAlgebra.score_dots"
      ]];
      FStar.Tactics.V2.trefl ()
    )

inline_for_extraction noextract
let context_linear (position:nat) (queries:S.matrix 8 256)
  (history:S.kv_history{history.length == position + 1}) (head:S.index 8) (d:S.index 256)
  : Lemma (
      S.linear 256 (position + 1) (value_matrix (position + 1) history head)
        (attention_probabilities position queries history head) d ==
      S.attention position queries history head d) =
  attention_unfold position queries history head d;
  sum_ext (position + 1)
    (fun t -> value_matrix (position + 1) history head d t *. attention_probabilities position queries history head t)
    (fun t -> attention_probabilities position queries history head t *. history.values (S.kv_head head) t d);
  norm_spec [delta_once ["KPlay.Proof.Shared.Attention.linear"]]
    (S.linear 256 (position + 1) (value_matrix (position + 1) history head)
      (attention_probabilities position queries history head) d)

inline_for_extraction noextract
let shared_decoder_cache
  (call:S.invocation) (w:S.layer_weights) (x:S.vector 1024)
  (before:S.hybrid_cache{S.domain call before})
  : Lemma (
      (S.full_attention_decoder call w x before).cache ==
      S.replace_kv before call.layer
        (S.append_kv call.position (before.kv call.layer)
          (rotated_matrix 2 call.position w.key_norm
            (S.reshape_kv (S.linear 512 1024 w.key_proj (S.rms_norm 1024 w.input_norm x))))
          (S.reshape_kv (S.linear 512 1024 w.value_proj (S.rms_norm 1024 w.input_norm x))))) =
  assert (
      (S.full_attention_decoder call w x before).cache ==
      S.replace_kv before call.layer
        (S.append_kv call.position (before.kv call.layer)
          (rotated_matrix 2 call.position w.key_norm
            (S.reshape_kv (S.linear 512 1024 w.key_proj (S.rms_norm 1024 w.input_norm x))))
          (S.reshape_kv (S.linear 512 1024 w.value_proj (S.rms_norm 1024 w.input_norm x)))))
    by (
      Proof.norm [delta_only [
        "KPlay.Proof.Shared.Attention.full_attention_decoder";
        "KPlay.Proof.Base.AttentionAlgebra.rotated_matrix"; "KPlay.Proof.Base.AttentionAlgebra.head_vector"
      ]];
      Proof.trefl ()
    )

inline_for_extraction noextract
let shared_decoder_hidden
  (call:S.invocation) (w:S.layer_weights) (x:S.vector 1024)
  (before:S.hybrid_cache{S.domain call before})
  : Lemma (
      (let u = S.rms_norm 1024 w.input_norm x in
       let qg = S.linear 4096 1024 w.query_gate_proj u in
       let q = rotated_matrix 8 call.position w.query_norm (S.split_queries qg) in
       let k = rotated_matrix 2 call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
       let v = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
       let next = S.append_kv call.position (before.kv call.layer) k v in
       let gated = activation_vector 2048 false (flat_heads 8 (S.split_gates qg))
         (flat_heads 8 (S.attention call.position q next)) in
       let mixed = S.linear 1024 2048 w.output_proj gated in
       let residual = S.vector_add 1024 x mixed in
       let mlp_input = S.rms_norm 1024 w.post_attention_norm residual in
       let gate = S.linear 3584 1024 w.mlp_gate_proj mlp_input in
       let up = S.linear 3584 1024 w.mlp_up_proj mlp_input in
       let product = activation_vector 3584 true gate up in
       S.vector_add 1024 residual (S.linear 1024 3584 w.mlp_down_proj product))
      == (S.full_attention_decoder call w x before).hidden) =
  assert (
      (let u = S.rms_norm 1024 w.input_norm x in
       let qg = S.linear 4096 1024 w.query_gate_proj u in
       let q = rotated_matrix 8 call.position w.query_norm (S.split_queries qg) in
       let k = rotated_matrix 2 call.position w.key_norm (S.reshape_kv (S.linear 512 1024 w.key_proj u)) in
       let v = S.reshape_kv (S.linear 512 1024 w.value_proj u) in
       let next = S.append_kv call.position (before.kv call.layer) k v in
       let gated = activation_vector 2048 false (flat_heads 8 (S.split_gates qg))
         (flat_heads 8 (S.attention call.position q next)) in
       let mixed = S.linear 1024 2048 w.output_proj gated in
       let residual = S.vector_add 1024 x mixed in
       let mlp_input = S.rms_norm 1024 w.post_attention_norm residual in
       let gate = S.linear 3584 1024 w.mlp_gate_proj mlp_input in
       let up = S.linear 3584 1024 w.mlp_up_proj mlp_input in
       let product = activation_vector 3584 true gate up in
       S.vector_add 1024 residual (S.linear 1024 3584 w.mlp_down_proj product))
      == (S.full_attention_decoder call w x before).hidden)
    by (
      Proof.norm [nbe; unmeta; unascribe; delta_only [
        "KPlay.Proof.Shared.Attention.full_attention_decoder";
        "KPlay.Proof.Base.AttentionAlgebra.rotated_matrix"; "KPlay.Proof.Base.AttentionAlgebra.head_vector";
        "KPlay.Proof.Base.AttentionAlgebra.activation_vector"; "KPlay.Proof.Base.AttentionAlgebra.activated_product";
        "KPlay.Proof.Base.AttentionAlgebra.flat_heads"
      ]];
      Proof.trefl ()
    )
