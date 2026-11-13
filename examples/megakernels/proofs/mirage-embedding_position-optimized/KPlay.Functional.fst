module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module SZ = Kuiper.SizeT
module FC = Kuiper.Float.Casts
module IC = FStar.Int.Cast
module C = FStar.Classical
module Shared = KPlay.SharedSpec
module F32 = Kuiper.Float32
module R = FStar.Real
module Pow = FStar.Math.Pow
module Trig = Kuiper.Real.Trigonometry

#set-options "--fuel 4 --ifuel 2 --z3rlimit 80"

(* Private native-expression certificates from the existing candidate witness.
   They are intermediates, not the authority for the public postcondition. *)
inline_for_extraction noextract
let valid_prompt (vocab tokens:nat) (prompt:lseq I64.t tokens) : prop =
  forall (i:nat{i < tokens}).
    0 <= I64.v (S.index prompt i) /\ I64.v (S.index prompt i) < vocab

inline_for_extraction noextract
let padded_token (#tokens:nat) (prompt:lseq I64.t tokens) (row:nat) : int =
  if row < tokens then I64.v (S.index prompt row) else 0

inline_for_extraction noextract
let embedding_post
  (vocab width tokens rows:nat)
  (weights:lseq bf16 (vocab * width))
  (prompt:lseq I64.t tokens)
  (embedding:lseq bf16 (rows * width)) : prop =
  forall (row:nat{row < rows}) (column:nat{column < width}).
    let token = padded_token prompt row in
    if 0 <= token && token < vocab then
      S.index embedding (row * width + column) ==
        S.index weights (token * width + column)
    else False

inline_for_extraction noextract
let coordinate_f32 (i:nat{i < 0x100000000}) : f32 =
  of_int #f32 (I64.int_to_t i)

inline_for_extraction noextract
let inverse_frequency
  (half_width:pos{half_width < 0x100000000})
  (frequency:nat{frequency < half_width}) : f32 =
  div (one #f32)
    (pow (of_int #f32 1000000L)
      (div (coordinate_f32 frequency) (coordinate_f32 half_width)))

inline_for_extraction noextract
let rotary_phase
  (half_width:pos{half_width < 0x100000000})
  (position:nat{position < 0x100000000})
  (frequency:nat{frequency < half_width}) : f32 =
  mul (inverse_frequency half_width frequency) (coordinate_f32 position)

inline_for_extraction noextract
let rotary_value
  (half_width:pos{half_width < 0x100000000})
  (position:nat{position < 0x100000000})
  (component:nat{component < 2})
  (column:nat{column < 2 * half_width}) : bf16 =
  let phase = rotary_phase half_width position (column % half_width) in
  FC.fcast #f32 #bf16 (if component = 0 then cos phase else sin phase)

inline_for_extraction noextract
let rope_post
  (positions:nat{positions < 0x100000000})
  (half_width:pos{half_width < 0x100000000})
  (table:lseq bf16 (4 * positions * half_width)) : prop =
  forall
    (component:nat{component < 2})
    (position:nat{position < positions})
    (column:nat{column < 2 * half_width}).
    S.index table ((component * positions + position) * (2 * half_width) + column)
      == rotary_value half_width position component column

inline_for_extraction noextract
let row_major_unique
  (width : pos)
  (row other_row : nat)
  (column other_column : nat{other_column < width})
  : Lemma
    (requires column < width)
    (ensures
      (row * width + column == other_row * width + other_column)
      <==> (row == other_row /\ column == other_column))
  =
  Kuiper.Math.div_mod_of_mul_add width row column;
  Kuiper.Math.div_mod_of_mul_add width other_row other_column

inline_for_extraction noextract
let token_range
  (vocab : pos)
  (tokens : nat)
  (prompt : lseq I64.t tokens)
  (row : nat)
  : Lemma
    (requires valid_prompt vocab tokens prompt)
    (ensures 0 <= padded_token prompt row /\ padded_token prompt row < vocab)
  = if row < tokens then () else ()

inline_for_extraction noextract
let embedding_cell
  (vocab width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq I64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (row : nat{row < rows})
  (column : nat{column < width})
  : prop =
  let token = padded_token prompt row in
  if 0 <= token && token < vocab then
    S.index embedding (row * width + column) ==
      S.index weights (token * width + column)
  else False

inline_for_extraction noextract
let embedding_ready
  (vocab width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq I64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (done_rows done_columns : nat)
  : prop =
  forall (r : nat{r < rows}) (c : nat{c < width}).
    (r < done_rows \/ (r == done_rows /\ c < done_columns)) ==>
      embedding_cell vocab width tokens rows weights prompt embedding r c

inline_for_extraction noextract
let embedding_step
  (vocab : pos) (width : pos) (tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq I64.t tokens)
  (embedding : lseq bf16 (rows * width))
  (row : nat{row < rows})
  (column : nat{column < width})
  (value : bf16)
  : Lemma
    (requires
      valid_prompt vocab tokens prompt /\
      embedding_ready vocab width tokens rows weights prompt embedding row column /\
      (let token = padded_token prompt row in
       if 0 <= token && token < vocab then
         value == S.index weights (token * width + column)
       else False))
    (ensures
      embedding_ready vocab width tokens rows weights prompt
        (S.upd embedding (row * width + column) value) row (column + 1))
  =
  let after = S.upd embedding (row * width + column) value in
  let at_cell (r : nat{r < rows}) (c : nat{c < width})
    : Lemma
      ((r < row \/ (r == row /\ c < column + 1)) ==>
        embedding_cell vocab width tokens rows weights prompt after r c)
    =
    row_major_unique width row r column c;
    token_range vocab tokens prompt r;
    if r < row || (r = row && c < column + 1) then (
      if r = row && c = column then
        S.lemma_index_upd1 embedding (row * width + column) value
      else (
        assert (embedding_cell vocab width tokens rows weights prompt embedding r c);
        S.lemma_index_upd2 embedding (row * width + column) value (r * width + c)
      )
    )
  in
  C.forall_intro_2 at_cell

inline_for_extraction noextract
let embedding_complete
  (vocab : pos) (width tokens rows : nat)
  (weights : lseq bf16 (vocab * width))
  (prompt : lseq I64.t tokens)
  (embedding : lseq bf16 (rows * width))
  : Lemma
    (requires valid_prompt vocab tokens prompt /\
      embedding_ready vocab width tokens rows weights prompt embedding rows 0)
    (ensures embedding_post vocab width tokens rows weights prompt embedding)
  =
  let valid_row (r : nat)
    : Lemma (0 <= padded_token prompt r /\ padded_token prompt r < vocab)
    = token_range vocab tokens prompt r
  in
  C.forall_intro valid_row;
  let at_cell (r : nat{r < rows}) (c : nat{c < width})
    : Lemma (
      let token = padded_token prompt r in
      if 0 <= token && token < vocab then
        S.index embedding (r * width + c) == S.index weights (token * width + c)
      else False)
    = assert (embedding_cell vocab width tokens rows weights prompt embedding r c)
  in
  C.forall_intro_2 at_cell;
  assert (embedding_post vocab width tokens rows weights prompt embedding) by (
    FStar.Tactics.V2.norm [delta_only [`%embedding_post]; iota; zeta];
    FStar.Tactics.V2.smt ()
  )

inline_for_extraction noextract
let rope_ready
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (done_frequencies done_positions : nat)
  : prop =
  forall
    (component : nat{component < 2})
    (position : nat{position < positions})
    (column : nat{column < 2 * half_width}).
    (column % half_width < done_frequencies \/
      (column % half_width == done_frequencies /\ position < done_positions)) ==>
    S.index table ((component * positions + position) * (2 * half_width) + column)
      == rotary_value half_width position component column

inline_for_extraction noextract
let rope_next_frequency
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (frequency : nat{frequency < half_width})
  : Lemma
    (requires rope_ready positions half_width table frequency positions)
    (ensures rope_ready positions half_width table (frequency + 1) 0)
  = ()

inline_for_extraction noextract
let rope_complete
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  : Lemma
    (requires rope_ready positions half_width table half_width 0)
    (ensures rope_post positions half_width table)
  = ()

inline_for_extraction noextract
let table_index_unique
  (positions half_width : pos)
  (component other_component : nat{other_component < 2})
  (position other_position : nat{other_position < positions})
  (column other_column : nat{other_column < 2 * half_width})
  : Lemma
    (requires component < 2 /\ position < positions /\ column < 2 * half_width)
    (ensures
      (((component * positions + position) * (2 * half_width) + column ==
        (other_component * positions + other_position) * (2 * half_width) + other_column)
       <==>
       (component == other_component /\ position == other_position /\ column == other_column)))
  =
  row_major_unique (2 * half_width)
    (component * positions + position) (other_component * positions + other_position)
    column other_column;
  row_major_unique positions component other_component position other_position

inline_for_extraction noextract
let updated_index
  (values : seq bf16)
  (at : nat{at < S.length values})
  (value : bf16)
  (index : nat{index < S.length values})
  : Lemma
    (S.index (S.upd values at value) index ==
      (if index = at then value else S.index values index))
  =
  if index = at then S.lemma_index_upd1 values at value
  else S.lemma_index_upd2 values at value index

inline_for_extraction noextract
let rope_step
  (positions : pos{positions < 0x100000000})
  (half_width : pos{half_width < 0x100000000})
  (table : lseq bf16 (4 * positions * half_width))
  (frequency : nat{frequency < half_width})
  (position : nat{position < positions})
  (cosine sine : bf16)
  : Lemma
    (requires
      rope_ready positions half_width table frequency position /\
      cosine == rotary_value half_width position 0 frequency /\
      sine == rotary_value half_width position 1 frequency)
    (ensures (
      let a = position * (2 * half_width) + frequency in
      let b = positions * (2 * half_width) + a in
      let after = S.upd (S.upd (S.upd (S.upd table
        a cosine) (a + half_width) cosine) b sine) (b + half_width) sine in
      rope_ready positions half_width after frequency (position + 1)))
  =
  let a = position * (2 * half_width) + frequency in
  let b = positions * (2 * half_width) + a in
  let first : lseq bf16 (4 * positions * half_width) = S.upd table a cosine in
  let second : lseq bf16 (4 * positions * half_width) =
    S.upd first (a + half_width) cosine in
  let third : lseq bf16 (4 * positions * half_width) = S.upd second b sine in
  let after : lseq bf16 (4 * positions * half_width) =
    S.upd third (b + half_width) sine in
  Kuiper.Math.div_mod_of_mul_add half_width 0 frequency;
  Kuiper.Math.div_mod_of_mul_add half_width 1 frequency;
  let at_cell
    (k : nat{k < 2})
    (p : nat{p < positions})
    (c : nat{c < 2 * half_width})
    : Lemma
      ((c % half_width < frequency \/
        (c % half_width == frequency /\ p < position + 1)) ==>
       S.index after ((k * positions + p) * (2 * half_width) + c)
         == rotary_value half_width p k c)
    =
    table_index_unique positions half_width k 0 p position c frequency;
    table_index_unique positions half_width k 0 p position c (frequency + half_width);
    table_index_unique positions half_width k 1 p position c frequency;
    table_index_unique positions half_width k 1 p position c (frequency + half_width);
    (if c < half_width then
      Kuiper.Math.div_mod_of_mul_add half_width 0 c
    else
      Kuiper.Math.div_mod_of_mul_add half_width 1 (c - half_width));
    let index = (k * positions + p) * (2 * half_width) + c in
    updated_index table a cosine index;
    updated_index first (a + half_width) cosine index;
    updated_index second b sine index;
    updated_index third (b + half_width) sine index;
    if c % half_width < frequency ||
      (c % half_width = frequency && p < position + 1) then (
      if c % half_width = frequency && p = position then (
        assert (c == frequency \/ c == frequency + half_width);
        if k = 0 then (
          assert (index == a \/ index == a + half_width);
          assert (S.index after index == cosine);
          assert (rotary_value half_width p k c == rotary_value half_width position 0 frequency)
        ) else (
          assert (index == b \/ index == b + half_width);
          assert (S.index after index == sine);
          assert (rotary_value half_width p k c == rotary_value half_width position 1 frequency)
        )
      ) else (
        assert (c % half_width < frequency \/
          (c % half_width == frequency /\ p < position));
        assert (S.index table index == rotary_value half_width p k c);
        assert (index <> a /\ index <> a + half_width /\ index <> b /\ index <> b + half_width);
        assert (S.index after index == S.index table index)
      )
    )
  in
  C.forall_intro_3 at_cell

inline_for_extraction noextract
let frequency_refinement (d:Shared.rotary_index)
  : Lemma (
      d % 64 == Shared.frequency_of_coordinate d /\
      R.of_int (d % 64) /. 64.0R ==
        R.of_int (2 * Shared.frequency_of_coordinate d) /. R.of_int Shared.head_dim) =
  if d < 64 then Kuiper.Math.div_mod_of_mul_add 64 0 d
  else Kuiper.Math.div_mod_of_mul_add 64 1 (d - 64)

inline_for_extraction noextract
let coordinate_approx (i:nat{i < 0x100000000})
  : Lemma (v_approximates (coordinate_f32 i) (R.of_int i)) =
  of_int_approx #f32 (I64.int_to_t i)

inline_for_extraction noextract
let rotary_value_refinement (p:table_position) (d:Shared.rotary_index)
  : Lemma (
      v_approximates (rotary_value 64 p 0 d) (table_cosine p d) /\
      v_approximates (rotary_value 64 p 1 d) (table_sine p d)) =
  let f = d % 64 in
  let base = of_int #f32 1000000L in
  let numerator = coordinate_f32 f in
  let denominator = coordinate_f32 64 in
  let exponent = div numerator denominator in
  let power = pow base exponent in
  let inverse = div (one #f32) power in
  let position = coordinate_f32 p in
  let phase = mul inverse position in
  let real_exponent = R.of_int f /. 64.0R in
  let real_power = Pow.pow Shared.rope_theta real_exponent in
  of_int_approx #f32 1000000L;
  coordinate_approx f;
  coordinate_approx 64;
  coordinate_approx p;
  div_approx numerator denominator (R.of_int f) 64.0R;
  F32.pow_approx base exponent Shared.rope_theta real_exponent;
  div_approx (one #f32) power 1.0R real_power;
  a_mul inverse position (1.0R /. real_power) (R.of_int p);
  frequency_refinement d;
  assert (v_approximates phase (table_angle p d));
  F32.cos_approx phase (table_angle p d);
  F32.sin_approx phase (table_angle p d);
  FC.fcast_approx #f32 #bf16 (cos phase) (Trig.cos (table_angle p d));
  FC.fcast_approx #f32 #bf16 (sin phase) (Trig.sin (table_angle p d))

inline_for_extraction noextract
let source_refines_physical
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (table:table_values)
  : Lemma
      (requires
        embedding_post 151936 1024 7 8 weights ts hidden /\
        rope_post 4096 64 table)
      (ensures physical_postcondition weights ts hidden table) =
  let hidden_at (p:physical_row) (h:Shared.hidden_index)
    : Lemma (
        S.index hidden (hidden_offset p h) ==
          S.index weights (embedding_offset (physical_token ts p) h)) =
    assert (padded_token ts p == physical_token ts p)
  in
  C.forall_intro_2 hidden_at;
  let rotary_at (p:table_position) (d:Shared.rotary_index)
    : Lemma (
        v_approximates (S.index table (table_offset 0 p d)) (table_cosine p d) /\
        v_approximates (S.index table (table_offset 1 p d)) (table_sine p d)) =
    assert (S.index table (table_offset 0 p d) == rotary_value 64 p 0 d);
    assert (S.index table (table_offset 1 p d) == rotary_value 64 p 1 d);
    rotary_value_refinement p d
  in
  C.forall_intro_2 rotary_at;
  let halves_at (p:table_position) (r:Shared.frequency_index)
    : Lemma (
        S.index table (table_offset 0 p r) ==
          S.index table (table_offset 0 p (r + Shared.half_dim)) /\
        S.index table (table_offset 1 p r) ==
          S.index table (table_offset 1 p (r + Shared.half_dim))) =
    Kuiper.Math.div_mod_of_mul_add 64 0 r;
    Kuiper.Math.div_mod_of_mul_add 64 1 r;
    assert (S.index table (table_offset 0 p r) == rotary_value 64 p 0 r);
    assert (S.index table (table_offset 1 p r) == rotary_value 64 p 1 r);
    assert (S.index table (table_offset 0 p (r + 64)) == rotary_value 64 p 0 (r + 64));
    assert (S.index table (table_offset 1 p (r + 64)) == rotary_value 64 p 1 (r + 64));
    assert (rotary_value 64 p 0 r == rotary_value 64 p 0 (r + 64));
    assert (rotary_value 64 p 1 r == rotary_value 64 p 1 (r + 64))
  in
  C.forall_intro_2 halves_at

inline_for_extraction noextract
let source_refines_semantic
  (weights:embedding_values) (ts:token_values)
  (hidden:hidden_values) (table:table_values)
  : Lemma
      (requires
        embedding_post 151936 1024 7 8 weights ts hidden /\
        rope_post 4096 64 table)
      (ensures semantic_postcondition weights ts hidden table) =
  source_refines_physical weights ts hidden table;
  establish_semantic_postcondition weights ts hidden table

inline_for_extraction noextract
let coordinate (i:I32.t{0 <= I32.v i})
  : r:f32{r == coordinate_f32 (I32.v i)} =
  let integer = IC.int32_to_int64 i in
  I64.v_inj integer (I64.int_to_t (I32.v i));
  of_int #f32 integer

inline_for_extraction noextract
let index64 (value:I64.t{0 <= I64.v value /\ SZ.fits (I64.v value)})
  : i:sz{SZ.v i == I64.v value} =
  assert_norm (pow2 32 == 0x100000000);
  SZ.uint32_to_sizet (IC.int64_to_uint32 value)

inline_for_extraction noextract
let index32 (value:I32.t{0 <= I32.v value})
  : i:sz{SZ.v i == I32.v value} =
  assert_norm (pow2 32 == 0x100000000);
  SZ.uint32_to_sizet (IC.int32_to_uint32 value)

inline_for_extraction noextract
fn read_token
  (prompt : larray I64.t Shared.prompt_length)
  (row : I32.t{0 <= I32.v row /\ I32.v row < 8})
  (#prompt_perm : perm)
  (#t : erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves gpu ** prompt |-> Frac prompt_perm (reveal t)
  returns token : I64.t
  ensures pure (0 <= I64.v token /\ I64.v token < 151936 /\
    I64.v token == padded_token t (I32.v row))
{
  if (I32.lt row 7l) {
    let i = index32 row;
    prompt.(i)
  } else {
    0L
  }
}

inline_for_extraction noextract
fn embed
  (weights : larray bf16 embedding_count)
  (prompt : larray I64.t Shared.prompt_length)
  (embedding : larray bf16 hidden_count)
  (#weights_perm #prompt_perm : perm)
  (#w : erased embedding_values)
  (#t : erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  (#before : erased hidden_values)
  preserves gpu **
    weights |-> Frac weights_perm (reveal w) **
    prompt |-> Frac prompt_perm (reveal t)
  requires embedding |-> before
  ensures exists* (after : hidden_values). embedding |-> after **
    pure (embedding_post 151936 1024 7 8 w t after)
{
  let mut row = 0l;
  while (I32.lt !row 8l)
    invariant exists* (r : I32.t{0 <= I32.v r /\ I32.v r <= 8})
      (values : hidden_values).
      row |-> r ** embedding |-> values **
      pure (embedding_ready 151936 1024 7 8 w t values (I32.v r) 0)
    decreases (8 - I32.v !row)
  {
    let r = !row;
    let token = read_token prompt r #prompt_perm #t;
    let mut column = 0l;
    while (I32.lt !column 1024l)
      invariant exists* (c : I32.t{0 <= I32.v c /\ I32.v c <= 1024})
        (values : hidden_values).
        column |-> c ** embedding |-> values **
        pure (embedding_ready 151936 1024 7 8 w t values (I32.v r) (I32.v c))
      decreases (1024 - I32.v !column)
    {
      let c = !column;
      let source = I64.add (I64.mul token 1024L) (IC.int32_to_int64 c);
      let destination = I32.add (I32.mul r 1024l) c;
      let source_index = index64 source;
      let destination_index = index32 destination;
      let value = weights.(source_index);
      with old. assert embedding |-> old;
      embedding.(destination_index) <- value;
      embedding_step 151936 1024 7 8 w t old (I32.v r) (I32.v c) value;
      column := I32.add !column 1l;
    };
    row := I32.add !row 1l;
  };
  with after. assert embedding |-> after;
  embedding_complete 151936 1024 7 8 w t after;
  assert pure (embedding_post 151936 1024 7 8 w t after);
}

inline_for_extraction noextract
fn write_rotary_pair
  (frequency : I32.t{0 <= I32.v frequency /\ I32.v frequency < 64})
  (position : I32.t{0 <= I32.v position /\ I32.v position < 4096})
  (cosine sine : bf16)
  (table : larray bf16 table_count)
  (#before : erased table_values)
  preserves gpu
  requires table |-> before **
    pure (rope_ready 4096 64 before (I32.v frequency) (I32.v position) /\
      cosine == rotary_value 64 (I32.v position) 0 (I32.v frequency) /\
      sine == rotary_value 64 (I32.v position) 1 (I32.v frequency))
  ensures exists* (after : table_values). table |-> after **
    pure (rope_ready 4096 64 after (I32.v frequency) (I32.v position + 1))
{
  let a = I32.add (I32.mul position 128l) frequency;
  let b = I32.add (I32.add 524288l (I32.mul position 128l)) frequency;
  let a0 = index32 a;
  let a1 = index32 (I32.add a 64l);
  let b0 = index32 b;
  let b1 = index32 (I32.add b 64l);
  table.(a0) <- cosine;
  pts_to_len table;
  table.(a1) <- cosine;
  pts_to_len table;
  table.(b0) <- sine;
  pts_to_len table;
  table.(b1) <- sine;
  pts_to_len table;
  with after. assert table |-> after;
  assert pure (after == S.upd (S.upd (S.upd (S.upd before
    (I32.v a) cosine) (I32.v a + 64) cosine)
    (I32.v b) sine) (I32.v b + 64) sine);
  rope_step 4096 64 before (I32.v frequency) (I32.v position) cosine sine;
  assert pure (rope_ready 4096 64 after (I32.v frequency) (I32.v position + 1));
}

inline_for_extraction noextract
fn make_rope
  (table : larray bf16 table_count)
  (#before : erased table_values)
  preserves gpu
  requires table |-> before
  ensures exists* (after : table_values). table |-> after **
    pure (rope_post 4096 64 after)
{
  let denominator = coordinate 64l;
  let mut frequency = 0l;
  while (I32.lt !frequency 64l)
    invariant exists* (f : I32.t{0 <= I32.v f /\ I32.v f <= 64})
      (values : table_values).
      frequency |-> f ** table |-> values **
      pure (rope_ready 4096 64 values (I32.v f) 0)
    decreases (64 - I32.v !frequency)
  {
    let f = !frequency;
    let inverse = div (one #f32)
      (pow (of_int #f32 1000000L) (div (coordinate f) denominator));
    assert pure (inverse == inverse_frequency 64 (I32.v f));
    let mut position = 0l;
    while (I32.lt !position 4096l)
      invariant exists* (p : I32.t{0 <= I32.v p /\ I32.v p <= 4096})
        (values : table_values).
        position |-> p ** table |-> values **
        pure (rope_ready 4096 64 values (I32.v f) (I32.v p))
      decreases (4096 - I32.v !position)
    {
      let p = !position;
      let phase = mul inverse (coordinate p);
      let cosine = FC.fcast #f32 #bf16 (cos phase);
      let sine = FC.fcast #f32 #bf16 (sin phase);
      Kuiper.Math.div_mod_of_mul_add 64 0 (I32.v f);
      assert pure (cosine == rotary_value 64 (I32.v p) 0 (I32.v f) /\
        sine == rotary_value 64 (I32.v p) 1 (I32.v f));
      write_rotary_pair f p cosine sine table;
      position := I32.add !position 1l;
    };
    with current. assert table |-> current;
    rope_next_frequency 4096 64 current (I32.v f);
    assert pure (rope_ready 4096 64 current (I32.v f + 1) 0);
    frequency := I32.add !frequency 1l;
  };
  with after. assert table |-> after;
  rope_complete 4096 64 after;
  assert pure (rope_post 4096 64 after);
}

inline_for_extraction noextract
fn family_kernel
  (weights : larray bf16 embedding_count)
  (prompt : larray I64.t Shared.prompt_length)
  (embedding : larray bf16 hidden_count)
  (table : larray bf16 table_count)
  (#weights_perm #prompt_perm : perm)
  (#w : erased embedding_values)
  (#t : erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  (#embedding_before : erased hidden_values)
  (#table_before : erased table_values)
  preserves gpu **
    weights |-> Frac weights_perm (reveal w) **
    prompt |-> Frac prompt_perm (reveal t)
  requires embedding |-> embedding_before ** table |-> table_before
  ensures exists*
    (embedding_after : hidden_values)
    (table_after : table_values).
    embedding |-> embedding_after ** table |-> table_after **
    pure (embedding_post 151936 1024 7 8 w t embedding_after /\
      rope_post 4096 64 table_after)
{
  embed weights prompt embedding;
  make_rope table;
}

inline_for_extraction noextract
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global_array arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (weights:larray bf16 embedding_count{is_global_array weights})
  (prompt:larray I64.t Shared.prompt_length{is_global_array prompt})
  (embedding:larray bf16 hidden_count{is_global_array embedding})
  (table:larray bf16 table_count{is_global_array table})
  (#weights_perm #prompt_perm:perm)
  (#w:erased embedding_values)
  (#t:erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves cpu **
    on gpu_loc (
      weights |-> Frac weights_perm (reveal w) **
      prompt |-> Frac prompt_perm (reveal t))
  requires exists* (embedding_before:hidden_values) (table_before:table_values).
    on gpu_loc (embedding |-> embedding_before ** table |-> table_before)
  ensures exists* (embedding_after:hidden_values) (table_after:table_values).
    on gpu_loc (embedding |-> embedding_after ** table |-> table_after) **
    pure (semantic_postcondition w t embedding_after table_after)
{
  with embedding_before table_before.
    assert on gpu_loc (embedding |-> embedding_before ** table |-> table_before);
  launch_kernel_1 (fun () ->
    family_kernel weights prompt embedding table
      #weights_perm #prompt_perm #w #t #embedding_before #table_before);
  with embedding_after table_after.
    assert on gpu_loc (embedding |-> embedding_after ** table |-> table_after);
  source_refines_semantic w t embedding_after table_after;
  assert pure (semantic_postcondition w t embedding_after table_after);
}
