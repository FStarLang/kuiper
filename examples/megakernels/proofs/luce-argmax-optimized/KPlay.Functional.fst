module KPlay.Functional

#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module Cast = FStar.Int.Cast
module Math = FStar.Math.Lemmas
module Shared = KPlay.SharedSpec

#set-options "--fuel 1 --ifuel 1 --z3rlimit 100"

inline_for_extraction noextract
let minus_infinity : f32 = negative_infinity

inline_for_extraction noextract
type index = i:int{-1 <= i /\ i < vocabulary_size}

inline_for_extraction noextract
type region = natlt vocabulary_size -> prop

inline_for_extraction noextract
let index_nat (i : index) : natlt vocabulary_size =
  if i < 0 then 0 else i

inline_for_extraction noextract
let candidate (scores : score_vector) (value : f32) (i : index) : prop =
  if i < 0 then value == minus_infinity
  else value == Seq.index scores (index_nat i)

inline_for_extraction noextract
let best (scores : score_vector) (r : region) (value : f32) (i : index) : prop =
  candidate scores value i /\
  (i >= 0 ==> r (index_nat i)) /\
  (forall (j : natlt vocabulary_size).
    r j ==> i >= 0 /\ priority_le (Seq.index scores j) value)

inline_for_extraction noextract
let union (a b : region) : region = fun j -> a j \/ b j

inline_for_extraction noextract
let empty_best (scores : score_vector) (r : region)
  : Lemma
      (requires forall (j : natlt vocabulary_size). ~(r j))
      (ensures best scores r minus_infinity (-1))
  = ()

inline_for_extraction noextract
let singleton_best (scores : score_vector) (j : natlt vocabulary_size)
  : Lemma
      (ensures
        best scores (fun k -> k == j) (Seq.index scores j) j)
  = priority_reflexive (Seq.index scores j)

inline_for_extraction noextract
let merge_point
  (scores : score_vector) (a b : region)
  (av bv : f32) (ai bi : index)
  (j : natlt vocabulary_size)
  : Lemma
      (requires best scores a av ai /\ best scores b bv bi)
      (ensures
        union a b j ==>
          (if better_candidate bv bi av ai then bi else ai) >= 0 /\
          priority_le (Seq.index scores j)
            (if better_candidate bv bi av ai then bv else av))
  = priority_total av bv;
    if union a b j then (
      if better_candidate bv bi av ai then (
        if a j then priority_transitive (Seq.index scores j) av bv
      ) else (
        if b j then priority_transitive (Seq.index scores j) bv av
      )
    )

inline_for_extraction noextract
let merge_best
  (scores : score_vector) (a b : region)
  (av bv : f32) (ai bi : index)
  : Lemma
      (requires best scores a av ai /\ best scores b bv bi)
      (ensures
        best scores (union a b)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = FStar.Classical.forall_intro (merge_point scores a b av bv ai bi)

inline_for_extraction noextract
let add_best
  (scores : score_vector) (r : region)
  (value : f32) (i : index) (j : natlt vocabulary_size)
  : Lemma
      (requires best scores r value i)
      (ensures
        best scores (union r (fun k -> k == j))
          (if better_candidate (Seq.index scores j) j value i then Seq.index scores j else value)
          (if better_candidate (Seq.index scores j) j value i then j else i))
  = singleton_best scores j;
    merge_best scores r (fun k -> k == j) value (Seq.index scores j) i j

inline_for_extraction noextract
let best_ext
  (scores : score_vector) (a b : region) (value : f32) (i : index)
  : Lemma
      (requires best scores a value i /\
                (forall (j : natlt vocabulary_size). a j <==> b j))
      (ensures best scores b value i)
  = ()

inline_for_extraction noextract
let best_member
  (scores:score_vector) (r:region) (value:f32) (i:index) (j:natlt vocabulary_size)
  : Lemma
      (requires best scores r value i /\ r j)
      (ensures i >= 0 /\ priority_le (Seq.index scores j) value) =
  ()

inline_for_extraction noextract
let best_selected (scores : score_vector) (value : f32) (token : i64)
  : Lemma
      (requires -1 <= I64.v token /\ I64.v token < vocabulary_size /\
                best scores (fun _ -> True) value (I64.v token))
      (ensures selected_token scores token)
  = best_member scores (fun _ -> True) value (I64.v token) 0;
    let i = index_nat (I64.v token) in
    assert (native_maximum #vocabulary_size scores i);
    logical_scores_represent #vocabulary_size scores;
    maximum_representation #vocabulary_size scores (logical_scores #vocabulary_size scores) i

inline_for_extraction noextract
let half_stride (d : nat) : prop =
  d == 1 \/ d == 2 \/ d == 4 \/ d == 8 \/ d == 16 \/
  d == 32 \/ d == 64 \/ d == 128 \/ d == 256

inline_for_extraction noextract
let lane_region (block : nat) (stride : pos) (lane : nat) : region =
  fun j -> j / 485 == block /\ ((j % 485) % 8) % stride == lane

inline_for_extraction noextract
let scan_region (block warp limit : nat) : region =
  fun j -> j / 485 == block /\ (j % 485) % 8 == warp /\ j < limit

inline_for_extraction noextract
let block_region (stride : pos) (lane : nat) : region =
  fun j -> (j / 485) % stride == lane

inline_for_extraction noextract
let remainder_split (x : nat) (d : pos{half_stride d}) (lane : natlt d)
  : Lemma
      ((x % (2 * d) == lane \/ x % (2 * d) == lane + d) <==>
        x % d == lane)
  = Math.modulo_modulo_lemma x d 2

inline_for_extraction noextract
let lane_merge
  (scores : score_vector) (block : natlt 512)
  (d : pos{d == 1 \/ d == 2 \/ d == 4 \/ d == 8 \/ d == 16})
  (lane : natlt d) (av bv : f32) (ai bi : index)
  : Lemma
      (requires
        best scores (lane_region block (2 * d) lane) av ai /\
        best scores (lane_region block (2 * d) (lane + d)) bv bi)
      (ensures
        best scores (lane_region block d lane)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = merge_best scores
      (lane_region block (2 * d) lane) (lane_region block (2 * d) (lane + d))
      av bv ai bi;
    let point (j : natlt vocabulary_size)
      : Lemma
          (union (lane_region block (2 * d) lane)
                 (lane_region block (2 * d) (lane + d)) j <==>
           lane_region block d lane j)
      = remainder_split ((j % 485) % 8) d lane
    in
    FStar.Classical.forall_intro point;
    best_ext scores
      (union (lane_region block (2 * d) lane) (lane_region block (2 * d) (lane + d)))
      (lane_region block d lane)
      (if better_candidate bv bi av ai then bv else av)
      (if better_candidate bv bi av ai then bi else ai)

inline_for_extraction noextract
let block_merge
  (scores : score_vector) (d : pos{half_stride d}) (lane : natlt d)
  (av bv : f32) (ai bi : index)
  : Lemma
      (requires
        best scores (block_region (2 * d) lane) av ai /\
        best scores (block_region (2 * d) (lane + d)) bv bi)
      (ensures
        best scores (block_region d lane)
          (if better_candidate bv bi av ai then bv else av)
          (if better_candidate bv bi av ai then bi else ai))
  = merge_best scores (block_region (2 * d) lane) (block_region (2 * d) (lane + d))
      av bv ai bi;
    let point (j : natlt vocabulary_size)
      : Lemma
          (union (block_region (2 * d) lane) (block_region (2 * d) (lane + d)) j <==>
           block_region d lane j)
      = remainder_split (j / 485) d lane
    in
    FStar.Classical.forall_intro point;
    best_ext scores
      (union (block_region (2 * d) lane) (block_region (2 * d) (lane + d)))
      (block_region d lane)
      (if better_candidate bv bi av ai then bv else av)
      (if better_candidate bv bi av ai then bi else ai)

inline_for_extraction noextract
let scan_step
  (scores : score_vector) (block : natlt 512) (warp : natlt 8)
  (row : natlt vocabulary_size{
    block * 485 + warp <= row /\ row < (block + 1) * 485 /\
    (row - block * 485 - warp) % 8 == 0})
  (value : f32) (i : index)
  : Lemma
      (requires best scores (scan_region block warp row) value i)
      (ensures
        best scores (scan_region block warp (row + 8))
          (if better_candidate (Seq.index scores row) row value i then Seq.index scores row else value)
          (if better_candidate (Seq.index scores row) row value i then row else i))
  = assert (row / 485 == block /\ (row % 485) % 8 == warp);
    add_best scores (scan_region block warp row) value i row;
    assert (forall (j : natlt vocabulary_size).
      union (scan_region block warp row) (fun k -> k == row) j <==>
      scan_region block warp (row + 8) j);
    best_ext scores
      (union (scan_region block warp row) (fun k -> k == row))
      (scan_region block warp (row + 8))
      (if better_candidate (Seq.index scores row) row value i then Seq.index scores row else value)
      (if better_candidate (Seq.index scores row) row value i then row else i)

inline_for_extraction noextract
let array_best
  (scores : score_vector) (regions : nat -> region)
  (values : seq f32) (tokens : lseq i64 (Seq.length values)) (lo hi : nat)
  : prop =
  forall (k : natlt (Seq.length values)).
    lo <= k /\ k < hi ==>
      -1 <= I64.v (Seq.index tokens k) /\ I64.v (Seq.index tokens k) < vocabulary_size /\
      best scores (regions k) (Seq.index values k) (I64.v (Seq.index tokens k))

inline_for_extraction noextract
let prefix_extend
  (scores : score_vector) (regions : nat -> region)
  (values : seq f32) (tokens : lseq i64 (Seq.length values))
  (k : natlt (Seq.length values)) (value : f32) (token : i64)
  : Lemma
      (requires
        array_best scores regions values tokens 0 k /\
        -1 <= I64.v token /\ I64.v token < vocabulary_size /\
        best scores (regions k) value (I64.v token))
      (ensures array_best scores regions
        (Seq.upd values k value) (Seq.upd tokens k token) 0 (k + 1))
  = ()

inline_for_extraction noextract
let preserve_range
  (scores : score_vector) (regions : nat -> region)
  (values : seq f32) (tokens : lseq i64 (Seq.length values)) (lo hi : nat)
  (k : natlt (Seq.length values)) (value : f32) (token : i64)
  : Lemma
      (requires array_best scores regions values tokens lo hi /\ (k < lo \/ hi <= k))
      (ensures array_best scores regions
        (Seq.upd values k value) (Seq.upd tokens k token) lo hi)
  = ()

inline_for_extraction noextract
let reduction_arrays
  (scores : score_vector) (old_regions new_regions : nat -> region)
  (values : seq f32) (tokens : lseq i64 (Seq.length values))
  (k : natlt (Seq.length values)) (width : nat)
  (av bv : f32) (ai bi : i64)
  : Lemma
      (requires
        array_best scores new_regions values tokens 0 k /\
        array_best scores old_regions values tokens k width /\
        av == Seq.index values k /\ ai == Seq.index tokens k /\
        -1 <= I64.v ai /\ I64.v ai < vocabulary_size /\
        -1 <= I64.v bi /\ I64.v bi < vocabulary_size /\
        best scores (new_regions k)
          (if better_candidate bv (I64.v bi) av (I64.v ai) then bv else av)
          (if better_candidate bv (I64.v bi) av (I64.v ai) then I64.v bi else I64.v ai))
      (ensures (
        let vs = if better_candidate bv (I64.v bi) av (I64.v ai) then Seq.upd values k bv else values in
        let ts = if better_candidate bv (I64.v bi) av (I64.v ai) then Seq.upd tokens k bi else tokens in
        array_best scores new_regions vs ts 0 (k + 1) /\
        array_best scores old_regions vs ts (k + 1) width))
  = if better_candidate bv (I64.v bi) av (I64.v ai) then (
      prefix_extend scores new_regions values tokens k bv bi;
      preserve_range scores old_regions values tokens (k + 1) width k bv bi
    )

inline_for_extraction noextract
fn write_if_better
  (n : sz) (values : larray f32 n) (tokens : larray i64 n)
  (k : szlt n) (av bv : f32) (ai bi : i64)
  (#vs : erased (lseq f32 n)) (#ts : erased (lseq i64 n))
  requires values |-> vs ** tokens |-> ts
  ensures
    values |-> (if better_candidate bv (I64.v bi) av (I64.v ai)
                then Seq.upd (reveal vs) k bv else reveal vs) **
    tokens |-> (if better_candidate bv (I64.v bi) av (I64.v ai)
                then Seq.upd (reveal ts) k bi else reveal ts)
{
  if (replace_best bv bi av ai) {
    values.(k) <- bv;
    tokens.(k) <- bi;
  } else {
    ();
  }
}

inline_for_extraction noextract
let tree_region (n block : nat) (stride : pos) : nat -> region =
  if n = 32 then lane_region block stride else block_region stride

inline_for_extraction noextract
let halve_stride (stride : pos{stride == 512 \/ half_stride stride})
  : Lemma
      (requires stride > 1)
      (ensures
        stride / 2 > 0 /\ half_stride (stride / 2) /\
        2 * (stride / 2) == stride)
  = ()

inline_for_extraction noextract
fn scan_warp
  (inputs : larray f32 vocabulary_size)
  (block : szlt 512) (warp : szlt 32)
  (#scores : erased score_vector)
  preserves gpu ** inputs |-> scores
  returns result : (f32 & i64)
  ensures pure (
    -1 <= I64.v result._2 /\ I64.v result._2 < vocabulary_size /\
    best scores (lane_region block 32 warp) result._1 (I64.v result._2))
{
  let mut maximum = minus_infinity;
  let mut token = -1L;
  let start = block *^ 485sz +^ warp;
  let finish = (block +^ 1sz) *^ 485sz;
  let mut row = start;
  if (warp <^ 8sz) {
    empty_best scores (scan_region block warp start);
    while (!row <^ finish)
      invariant exists* (r : sz) (v : f32) (t : i64).
        row |-> r ** maximum |-> v ** token |-> t **
        pure (
          start <= r /\ r < finish + 8 /\
          (r - start) % 8 == 0 /\
          -1 <= I64.v t /\ I64.v t < vocabulary_size /\
          best scores (scan_region block warp r) v (I64.v t))
      decreases (248328 - !row)
    {
      let r = !row;
      let value = inputs.(r);
      let old_value = !maximum;
      let old_token = !token;
      let incoming_token = Cast.uint32_to_int64 (SZ.sizet_to_u32 r);
      scan_step scores block warp r old_value (I64.v old_token);
      if (replace_best value incoming_token old_value old_token) {
        maximum := value;
        token := incoming_token;
      };
      row := !row +^ 8sz;
    };
    let end_row = !row;
    let v = !maximum;
    let t = !token;
    best_ext scores (scan_region block warp end_row) (lane_region block 32 warp)
      v (I64.v t);
    (v, t)
  } else {
    empty_best scores (lane_region block 32 warp);
    (minus_infinity, -1L)
  }
}

inline_for_extraction noextract
fn reduce_tree
  (n : sz{SZ.v n == 32 \/ SZ.v n == 512}) (block : szlt 512)
  (values : larray f32 n) (tokens : larray i64 n)
  (#scores : erased score_vector)
  (#initial_values : erased (lseq f32 n))
  (#initial_tokens : erased (lseq i64 n))
  preserves gpu
  requires
    values |-> initial_values ** tokens |-> initial_tokens **
    pure (array_best scores (tree_region n block n) initial_values initial_tokens 0 n)
  returns result : (f32 & i64)
  ensures
    (exists* (vs : lseq f32 n) (ts : lseq i64 n).
      values |-> vs ** tokens |-> ts) **
    pure (
      -1 <= I64.v result._2 /\ I64.v result._2 < vocabulary_size /\
      best scores (tree_region n block 1 0) result._1 (I64.v result._2))
{
  let mut stride = n;
  while (!stride >^ 1sz)
    invariant exists* (width : sz) (vs : lseq f32 n) (ts : lseq i64 n).
      stride |-> width ** values |-> vs ** tokens |-> ts **
      pure (
        1 <= width /\ width <= n /\ (SZ.v width == 512 \/ half_stride width) /\
        array_best scores (tree_region n block width) vs ts 0 width)
    decreases (SZ.v !stride)
  {
    let width = !stride;
    halve_stride width;
    let distance = width /^ 2sz;
    let mut lane = 0sz;
    while (!lane <^ distance)
      invariant exists* (k : sz) (vs : lseq f32 n) (ts : lseq i64 n).
        lane |-> k ** values |-> vs ** tokens |-> ts **
        pure (
          k <= distance /\
          array_best scores (tree_region n block distance) vs ts 0 k /\
          array_best scores (tree_region n block width) vs ts k width)
      decreases (distance - !lane)
    {
      let k = !lane;
      with vs. assert (values |-> vs);
      with ts. assert (tokens |-> ts);
      let av = values.(k);
      let bv = values.(k +^ distance);
      let ai = tokens.(k);
      let bi = tokens.(k +^ distance);
      if (n <^ 33sz) {
        lane_merge scores block distance k av bv (I64.v ai) (I64.v bi);
        ();
      } else {
        block_merge scores distance k av bv (I64.v ai) (I64.v bi);
        ();
      };
      reduction_arrays scores (tree_region n block width) (tree_region n block distance)
        vs ts k width av bv ai bi;
      write_if_better n values tokens k av bv ai bi;
      lane := !lane +^ 1sz;
    };
    stride := !stride /^ 2sz;
  };
  let v = values.(0sz);
  let t = tokens.(0sz);
  (v, t)
}

inline_for_extraction noextract
fn block_argmax
  (inputs : larray f32 vocabulary_size) (block : szlt 512)
  (#scores : erased score_vector)
  preserves gpu ** inputs |-> scores
  returns result : (f32 & i64)
  ensures pure (
    -1 <= I64.v result._2 /\ I64.v result._2 < vocabulary_size /\
    best scores (block_region 512 block) result._1 (I64.v result._2))
{
  let mut values = [| minus_infinity; 32sz |];
  let mut tokens = [| -1L; 32sz |];
  let mut warp = 0sz;
  while (!warp <^ 32sz)
    invariant exists* (w : sz) (vs : lseq f32 32) (ts : lseq i64 32).
      warp |-> w ** values |-> vs ** tokens |-> ts **
      pure (w <= 32 /\ array_best scores (lane_region block 32) vs ts 0 w)
    decreases (32 - !warp)
  {
    let w = !warp;
    let result = scan_warp inputs block w;
    with vs. assert (values |-> vs);
    with ts. assert (tokens |-> ts);
    prefix_extend scores (lane_region block 32) vs ts w result._1 result._2;
    values.(w) <- result._1;
    tokens.(w) <- result._2;
    warp := !warp +^ 1sz;
  };
  let result = reduce_tree 32sz block values tokens #scores;
  best_ext scores (lane_region block 1 0) (block_region 512 block)
    result._1 (I64.v result._2);
  result
}

inline_for_extraction noextract
fn ordered_argmax
  (inputs : larray f32 vocabulary_size)
  (#scores : erased score_vector)
  preserves gpu ** inputs |-> scores
  returns token : i64
  ensures pure (selected_token scores token)
{
  let mut values = [| minus_infinity; 512sz |];
  let mut tokens = [| -1L; 512sz |];
  let mut block = 0sz;
  while (!block <^ 512sz)
    invariant exists* (b : sz) (vs : lseq f32 512) (ts : lseq i64 512).
      block |-> b ** values |-> vs ** tokens |-> ts **
      pure (b <= 512 /\ array_best scores (block_region 512) vs ts 0 b)
    decreases (512 - !block)
  {
    let b = !block;
    let result = block_argmax inputs b;
    with vs. assert (values |-> vs);
    with ts. assert (tokens |-> ts);
    prefix_extend scores (block_region 512) vs ts b result._1 result._2;
    values.(b) <- result._1;
    tokens.(b) <- result._2;
    block := !block +^ 1sz;
  };
  let result = reduce_tree 512sz 0sz values tokens #scores;
  best_ext scores (block_region 1 0) (fun _ -> True) result._1 (I64.v result._2);
  best_selected scores result._1 result._2;
  result._2
}

inline_for_extraction noextract
let history_index (i : i64{0 <= I64.v i /\ I64.v i < 2048})
  : r:sz{SZ.v r == I64.v i}
  = SZ.uint32_to_sizet (Cast.int64_to_uint32 i)

inline_for_extraction noextract
fn copy_append
  (count : i64{0 < I64.v count /\ I64.v count < 2048})
  (previous_pointer : larray i64 (I64.v count))
  (output_pointer : larray i64 (I64.v count + 1))
  (token : i64)
  (#previous : erased (lseq i64 (I64.v count)))
  (#before_values : erased (lseq i64 (I64.v count + 1)))
  preserves gpu ** previous_pointer |-> previous
  requires output_pointer |-> before_values
  ensures output_pointer |-> Seq.append previous (Seq.cons token Seq.empty)
{
  let mut i = 0L;
  while (I64.lt !i count)
    invariant exists* (j : i64) (values : lseq i64 (I64.v count + 1)).
      i |-> j ** output_pointer |-> values **
      pure (
        0 <= I64.v j /\ I64.v j <= I64.v count /\
        (forall (k : natlt (I64.v count)).
          k < I64.v j ==> Seq.index values k == Seq.index previous k))
    decreases (I64.v count - I64.v !i)
  {
    let j = !i;
    let k = history_index j;
    let value = previous_pointer.(k);
    output_pointer.(k) <- value;
    i := I64.add !i 1L;
  };
  let last = history_index count;
  output_pointer.(last) <- token;
  with after_values. assert (output_pointer |-> after_values);
  assert (pure (Seq.equal after_values (Seq.append previous (Seq.cons token Seq.empty))));
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a : Type0)
  (pointer : array a{is_global_array pointer})
  (#f : perm) (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to pointer #f contents)
  = Kuiper.Array.Core.is_send_pts_to pointer #f contents

inline_for_extraction noextract
instance placeless_history (pointer : array i64) (contents : seq i64)
  : Pulse.Lib.Send.placeless (history_ownership pointer contents)
  = if Seq.length contents = 0 then
      (let p : Pulse.Lib.Send.placeless emp = solve in p)
    else
      (let p : Pulse.Lib.Send.placeless
        (on gpu_loc (pointer |-> contents) ** pure (is_global_array pointer)) = solve in p)

inline_for_extraction noextract
let history_length
  (previous : seq i64) (position prompt_length token : i64)
  : Lemma
      (requires valid_position position prompt_length /\
                Seq.length previous == previous_count position prompt_length)
      (ensures Seq.length (updated_history previous position prompt_length token) ==
                output_count position prompt_length)
  = ()

inline_for_extraction noextract
fn write_history
  (previous_pointer output_pointer : array i64)
  (position prompt_length token : i64)
  (#previous : erased (lseq i64 (previous_count position prompt_length)))
  (#before_values : erased (lseq i64 (output_count position prompt_length)))
  preserves gpu ** history_ownership previous_pointer previous
  requires
    pure (valid_position position prompt_length) **
    history_ownership output_pointer before_values
  ensures
    history_ownership output_pointer (updated_history previous position prompt_length token)
{
  let last_prompt = I64.sub prompt_length 1L;
  history_length previous position prompt_length token;
  if (I64.lt position last_prompt) {
    assert (pure (Seq.length before_values == 0 /\ Seq.length previous == 0));
    rewrite (history_ownership output_pointer before_values) as emp;
    rewrite emp as (history_ownership output_pointer
      (updated_history previous position prompt_length token));
  } else if (I64.lt position prompt_length) {
    assert (pure (Seq.length before_values == 1 /\ Seq.length previous == 0));
    rewrite (history_ownership output_pointer before_values) as
      (on gpu_loc (output_pointer |-> before_values) ** pure (is_global_array output_pointer));
    elim_gpu (output_pointer |-> before_values) ();
    pts_to_len output_pointer;
    output_pointer.(0sz) <- token;
    with values. assert (output_pointer |-> values);
    assert (pure (Seq.equal previous Seq.empty));
    assert (pure (Seq.equal values (Seq.cons token Seq.empty)));
    intro_gpu (output_pointer |-> values) ();
    rewrite (on gpu_loc (output_pointer |-> values) ** pure (is_global_array output_pointer))
      as (history_ownership output_pointer
        (updated_history previous position prompt_length token));
  } else {
    let count = I64.add (I64.sub position prompt_length) 1L;
    assert (pure (Seq.length previous > 0 /\ Seq.length before_values > 0));
    rewrite (history_ownership previous_pointer previous) as
      (on gpu_loc (previous_pointer |-> previous) ** pure (is_global_array previous_pointer));
    rewrite (history_ownership output_pointer before_values) as
      (on gpu_loc (output_pointer |-> before_values) ** pure (is_global_array output_pointer));
    elim_gpu (previous_pointer |-> previous) ();
    elim_gpu (output_pointer |-> before_values) ();
    pts_to_len previous_pointer;
    pts_to_len output_pointer;
    copy_append count previous_pointer output_pointer token;
    intro_gpu (previous_pointer |-> previous) ();
    intro_gpu (output_pointer |-> Seq.append previous (Seq.cons token Seq.empty)) ();
    rewrite (on gpu_loc (previous_pointer |-> previous) ** pure (is_global_array previous_pointer))
      as (history_ownership previous_pointer previous);
    rewrite (on gpu_loc (output_pointer |-> Seq.append previous (Seq.cons token Seq.empty)) **
      pure (is_global_array output_pointer))
      as (history_ownership output_pointer
        (updated_history previous position prompt_length token));
  }
}

inline_for_extraction noextract
fn kernel_body
  (input0 : larray f32 vocabulary_size)
  (input1 : array i64)
  (input2 input3 : i64)
  (output0 : larray i64 1)
  (output1 : array i64)
  (#scores : erased score_vector)
  (#previous : erased (lseq i64 (previous_count input2 input3)))
  (#before_token : erased (lseq i64 1))
  (#before_history : erased (lseq i64 (output_count input2 input3)))
  preserves
    gpu ** input0 |-> scores ** history_ownership input1 previous
  requires
    pure (valid_position input2 input3) **
    output0 |-> before_token ** history_ownership output1 before_history
  ensures
    exists* (token_values : lseq i64 1)
            (history_values : lseq i64 (output_count input2 input3)).
      output0 |-> token_values **
      history_ownership output1 history_values **
      pure (semantic_post scores previous input2 input3 token_values history_values)
{
  let token = ordered_argmax input0;
  output0.(0sz) <- token;
  write_history input1 output1 input2 input3 token #previous #before_history;
  history_length previous input2 input3 token;
}

fn kplay_run
  (input0 : larray f32 vocabulary_size{is_global_array input0})
  (input1 : array i64)
  (input2 input3 : i64)
  (output0 : larray i64 1{is_global_array output0})
  (output1 : array i64)
  (#scores : erased score_vector)
  (#previous : erased (lseq i64 (previous_count input2 input3)))
  preserves
    cpu **
    on gpu_loc (input0 |-> scores) **
    history_ownership input1 previous
  requires
    pure (valid_position input2 input3) **
    (exists* (before_token : lseq i64 1).
      on gpu_loc (output0 |-> before_token)) **
    (exists* (before_history : lseq i64 (output_count input2 input3)).
      history_ownership output1 before_history)
  ensures
    exists* (token_values : lseq i64 1)
            (history_values : lseq i64 (output_count input2 input3)).
      on gpu_loc (output0 |-> token_values) **
      history_ownership output1 history_values **
      pure (semantic_post scores previous input2 input3 token_values history_values)
{
  with before_token. assert (on gpu_loc (output0 |-> before_token));
  with before_history. assert (history_ownership output1 before_history);
  Pulse.Lib.Send.placeless_on_intro (history_ownership input1 previous) gpu_loc;
  Pulse.Lib.Send.placeless_on_intro (history_ownership output1 before_history) gpu_loc;
  launch_kernel_1
    (fun () -> kernel_body input0 input1 input2 input3 output0 output1
      #scores #previous #before_token #before_history);
  Pulse.Lib.Send.placeless_on_elim (history_ownership input1 previous) gpu_loc;
  with history_values. assert (on gpu_loc (history_ownership output1 history_values));
  Pulse.Lib.Send.placeless_on_elim (history_ownership output1 history_values) gpu_loc;
}
