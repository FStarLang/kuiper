module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Seq.Common
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Trade
open Pulse.Lib.Array

module S = FStar.Seq
module SZ = Kuiper.SizeT
module U64 = FStar.UInt64
module G = Kuiper.Spec.GEMM
module MMA = Kuiper.TensorCore.MMA
module FM = Kuiper.Float32.FastMath
module SM = Kuiper.Spec.Softmax

inline_for_extraction noextract
let flat_view (#et:Type0) (rows columns:nat)
  (values:lseq et (rows * columns)) : GTot (chest2 et rows columns) =
  mk2 (fun row column -> S.index values (row * columns + column))

inline_for_extraction noextract
let softmax_matrix (#n:nat) (#e:pos) (r:chest2 real n e) : chest2 real n e =
  mk2 (fun token expert -> acc1 (SM.softmax_real (chest2_row r token)) expert)

inline_for_extraction noextract
let matrix_post (n:nat) (e:pos) (s result:lseq f32 (n * e)) : prop =
  forall (r:chest2 real n e).
    flat_view n e s %~ r ==> flat_view n e result %~ softmax_matrix r

inline_for_extraction noextract
let matrix_prefix (n:nat) (e:pos) (s result:lseq f32 (n * e))
  (token:nat{token <= n}) (expert:nat{expert <= e}) : prop =
  forall (r:chest2 real n e) (t:natlt n) (j:natlt e).
    flat_view n e s %~ r /\
    (t < token \/ (t == token /\ j < expert)) ==>
    S.index result (t * e + j) %~ acc2 (softmax_matrix r) t j

inline_for_extraction noextract
let exp_sum (#e:pos) (r:lseq real e) (i:nat{i <= e}) : real =
  rsum (seq_map exp (S.slice r 0 i))

inline_for_extraction noextract
let denominator (#e:pos) (r:lseq real e) (i:nat{i <= e}) (m:f32) : real =
  exp_sum r i /. exp (to_real m)

inline_for_extraction noextract
let exp_sum_step (#e:pos) (r:lseq real e) (i:nat{i < e})
  : Lemma (exp_sum r (i + 1) == exp_sum r i +. exp (S.index r i))
  = assert (S.equal (S.slice r 0 (i + 1))
      (S.append (S.slice r 0 i) (S.create 1 (S.index r i))));
    assert (S.equal (seq_map exp (S.slice r 0 (i + 1)))
      (S.append (seq_map exp (S.slice r 0 i))
        (S.create 1 (exp (S.index r i)))));
    rsum_append (seq_map exp (S.slice r 0 i))
      (S.create 1 (exp (S.index r i)))

inline_for_extraction noextract
let field_step (s x old_m new_m:real)
  : Lemma (
      (s /. exp old_m) *. exp (old_m -. new_m) +. exp (x -. new_m)
      == (s +. exp x) /. exp new_m)
  = exp_sub old_m new_m;
    exp_sub x new_m

inline_for_extraction noextract
let denominator_step (#e:pos) (r:lseq real e) (i:nat{i < e}) (m m':f32)
  : Lemma (
      denominator r i m *. exp (to_real m -. to_real m') +.
        exp (S.index r i -. to_real m')
      == denominator r (i + 1) m')
  = exp_sum_step r i;
    field_step (exp_sum r i) (S.index r i) (to_real m) (to_real m')

inline_for_extraction noextract
let denominator_positive (#e:pos) (r:lseq real e) (m:f32)
  : Lemma (denominator r e m >. 0.0R)
  = assert (S.equal (S.slice r 0 e) r);
    let _ : real = rsum (seq_map exp r) in
    ()

inline_for_extraction noextract
let normalized_value (#e:pos) (r:lseq real e) (m:f32) (i:nat{i < e})
  : Lemma (
      exp (S.index r i -. to_real m) /. denominator r e m
      == S.index (SM.softmax_real_seq r) i)
  = denominator_positive r m;
    assert (S.equal (S.slice r 0 e) r);
    exp_sub (S.index r i) (to_real m)

inline_for_extraction noextract
let denominator_step_all (#e:pos) (i:nat{i < e}) (m m':f32)
  : Lemma (forall (r:lseq real e).
      denominator r i m *. exp (to_real m -. to_real m') +.
        exp (S.index r i -. to_real m')
      == denominator r (i + 1) m')
  = let proof (r:lseq real e)
      : Lemma (
          denominator r i m *. exp (to_real m -. to_real m') +.
            exp (S.index r i -. to_real m')
          == denominator r (i + 1) m')
      = denominator_step r i m m'
    in
    FStar.Classical.forall_intro proof

inline_for_extraction noextract
let normalized_value_all (#e:pos) (m:f32) (i:nat{i < e})
  : Lemma (forall (r:lseq real e).
      denominator r e m >. 0.0R /\
      exp (S.index r i -. to_real m) /. denominator r e m
        == S.index (SM.softmax_real_seq r) i)
  = let proof (r:lseq real e)
      : Lemma (
          denominator r e m >. 0.0R /\
          exp (S.index r i -. to_real m) /. denominator r e m
            == S.index (SM.softmax_real_seq r) i)
      = denominator_positive r m;
        normalized_value r m i
    in
    FStar.Classical.forall_intro proof

inline_for_extraction noextract
let input_row (n:nat) (e:pos) (s:lseq f32 (n * e)) (t:natlt n)
  : GTot (lseq f32 e) =
  S.init_ghost e (fun j -> S.index s (t * e + j))

inline_for_extraction noextract
fn normalizer
  (n:szp) (e:szp{SZ.fits (SZ.v n * SZ.v e)})
  (scores:larray f32 (SZ.v n * SZ.v e))
  (t:szlt (SZ.v n))
  (#f:perm)
  (#s:erased (lseq f32 (SZ.v n * SZ.v e)))
  preserves gpu ** scores |-> Frac f (reveal s)
  returns md:(f32 & f32)
  ensures pure (
    forall (r:lseq real (SZ.v e)).
      seq_approximates (input_row n e s t) r ==>
        snd md %~ denominator r e (fst md))
{
  let mut maximum:f32 = neg infinity;
  let mut d:f32 = zero;
  let mut i = 0sz;
  while (!i <^ e)
    invariant
      live maximum ** live d ** live i **
      pure (!i <= e /\
        (forall (r:lseq real (SZ.v e)).
          seq_approximates (input_row n e s t) r ==>
            !d %~ denominator r !i !maximum))
    decreases (e - !i)
  {
    let old_m = !maximum;
    let old_d = !d;
    let index = !i;
    let x = scores.(t *^ e +^ index);
    let new_m = fmax old_m x;
    to_real_ok old_m;
    to_real_ok new_m;
    let delta_m = FM.sub_rn old_m new_m;
    let adjustment = FM.exp delta_m;
    let delta_x = FM.sub_rn x new_m;
    let numerator = FM.exp delta_x;
    let new_d = FM.fma_rn old_d adjustment numerator;
    assert pure (
      forall (r:lseq real (SZ.v e)).
        seq_approximates (input_row n e s t) r ==> new_d %~
          (denominator r index old_m *.
            exp (to_real old_m -. to_real new_m) +.
            exp (S.index r index -. to_real new_m)));
    denominator_step_all #(SZ.v e) index old_m new_m;
    maximum := new_m;
    d := new_d;
    i := !i +^ 1sz;
  };
  (!maximum, !d)
}

inline_for_extraction noextract
let value_to_matrix (n:nat) (e:pos) (s:lseq f32 (n * e))
  (t:natlt n) (j:natlt e) (value:f32)
  : Lemma
      (requires forall (r:lseq real e).
        seq_approximates (input_row n e s t) r ==>
        value %~ S.index (SM.softmax_real_seq r) j)
      (ensures forall (r:chest2 real n e).
        flat_view n e s %~ r ==> value %~ acc2 (softmax_matrix r) t j)
  = let proof (r:chest2 real n e)
      : Lemma (flat_view n e s %~ r ==> value %~ acc2 (softmax_matrix r) t j)
      = SM.lem_softmax_real_to_seq (chest2_row r t);
        assert (flat_view n e s %~ r ==>
          seq_approximates (input_row n e s t) (chest1_to_seq (chest2_row r t)))
    in
    FStar.Classical.forall_intro proof

inline_for_extraction noextract
let matrix_prefix_full (n:nat) (e:pos) (s result:lseq f32 (n * e))
  : Lemma (requires matrix_prefix n e s result n 0)
          (ensures matrix_post n e s result)
  = let point (r:chest2 real n e) (index:abs (n @| e @| INil))
      : Lemma (flat_view n e s %~ r ==>
        acc (flat_view n e result) index %~ acc (softmax_matrix r) index)
      = let (t, (j, ())) = index in
        assert (flat_view n e s %~ r ==>
          S.index result (t * e + j) %~ acc2 (softmax_matrix r) t j);
        assert (acc (flat_view n e result) index == S.index result (t * e + j));
        assert (acc (softmax_matrix r) index == acc2 (softmax_matrix r) t j)
    in
    FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
fn matrix_kernel
  (n:szp) (e:szp{SZ.fits (SZ.v n * SZ.v e)})
  (scores output:larray f32 (SZ.v n * SZ.v e))
  (#f:perm)
  (#s:erased (lseq f32 (SZ.v n * SZ.v e)))
  ()
  preserves gpu ** scores |-> Frac f (reveal s)
  requires exists* (before:lseq f32 (SZ.v n * SZ.v e)). output |-> before
  ensures exists* (result:lseq f32 (SZ.v n * SZ.v e)).
    output |-> result ** pure (matrix_post n e s result)
{
  let mut token = 0sz;
  while (!token <^ n)
    invariant
      live token **
      (exists* (current:lseq f32 (SZ.v n * SZ.v e)).
        output |-> current **
        pure (!token <= n /\ matrix_prefix n e s current !token 0))
    decreases (n - !token)
  {
    let t = !token;
    let (maximum, d) = normalizer n e scores t;
    to_real_ok maximum;
    let mut expert = 0sz;
    while (!expert <^ e)
      invariant
        live expert **
        (exists* (current:lseq f32 (SZ.v n * SZ.v e)).
          output |-> current **
          pure (!expert <= e /\ matrix_prefix n e s current t !expert))
      decreases (e - !expert)
    {
      let j = !expert;
      let x = scores.(t *^ e +^ j);
      let delta = FM.sub_rn x maximum;
      let numerator = FM.exp delta;
      let value = FM.divide numerator d;
      normalized_value_all #(SZ.v e) maximum j;
      assert pure (
        forall (r:lseq real (SZ.v e)).
          seq_approximates (input_row n e s t) r ==>
          value %~ S.index (SM.softmax_real_seq r) j);
      value_to_matrix n e s t j value;
      output.(t *^ e +^ j) <- value;
      expert := !expert +^ 1sz;
    };
    token := !token +^ 1sz;
  };
  with result. assert output |-> result;
  matrix_prefix_full n e s result;
}

inline_for_extraction noextract
fn kernel_thread
  (n:szp) (e:szp{SZ.fits (SZ.v n * SZ.v e)})
  (scores output:larray f32 (SZ.v n * SZ.v e))
  (#f:perm)
  (#s #before:erased (lseq f32 (SZ.v n * SZ.v e)))
  ()
  norewrite
  requires gpu ** (scores |-> Frac f (reveal s) ** output |-> reveal before)
  ensures gpu **
    (scores |-> Frac f (reveal s) **
      (exists* (result:lseq f32 (SZ.v n * SZ.v e)).
        output |-> result ** pure (matrix_post n e s result)))
{
  matrix_kernel n e scores output ();
}

inline_for_extraction noextract
instance send_global_array_contents
  (#et:Type0) (storage:array et{is_global_array storage})
  (#f:perm) (values:seq et)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to storage #f values)
  = Kuiper.Array.Core.is_send_pts_to storage #f values

inline_for_extraction noextract
let softmax_kernel
  (n:szp) (e:szp{SZ.fits (SZ.v n * SZ.v e)})
  (scores:larray f32 (SZ.v n * SZ.v e){is_global_array scores})
  (output:larray f32 (SZ.v n * SZ.v e){is_global_array output})
  (#f:perm)
  (#s #before:erased (lseq f32 (SZ.v n * SZ.v e)))
  : kernel_desc_1_1
      (scores |-> Frac f (reveal s) ** output |-> reveal before)
      (scores |-> Frac f (reveal s) **
        (exists* (result:lseq f32 (SZ.v n * SZ.v e)).
          output |-> result ** pure (matrix_post n e s result)))
  = {
      f = kernel_thread n e scores output #f #s #before;
      full_pre_sendable = solve;
      full_post_sendable = solve;
    }

inline_for_extraction noextract
let zero_tile:chest2 f32 16 8 = const (16 @| 8 @| INil) zero

inline_for_extraction noextract
let real_zero_tile:chest2 real 16 8 = const (16 @| 8 @| INil) 0.0R

inline_for_extraction noextract
let rec mma_prefix
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (tile:natlt (m / 16)) (group:natlt (e / 8)) (steps:nat{steps <= h / 16})
  : GTot (chest2 f32 16 8) (decreases steps)
  = if steps = 0 then zero_tile
    else MMA.emma (mma_prefix a b tile group (steps - 1))
      (ematrix_subtile a 16 16 tile (steps - 1))
      (ematrix_subtile b 16 8 (steps - 1) group)

inline_for_extraction noextract
let mma_prefix_step
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (tile:natlt (m / 16)) (group:natlt (e / 8)) (step:natlt (h / 16))
  : Lemma (
      mma_prefix a b tile group (step + 1) ==
      MMA.emma (mma_prefix a b tile group step)
        (ematrix_subtile a 16 16 tile step)
        (ematrix_subtile b 16 8 step group))
  = ()

inline_for_extraction noextract
let real_prefix
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 real m h) (b:chest2 real h e)
  (tile:natlt (m / 16)) (group:natlt (e / 8)) (steps:nat{steps <= h / 16})
  : GTot (chest2 real 16 8)
  = G.__gmatmul_single real_zero_tile G.matmul G.matplus
      (ematrix_tiled a 16 16) (ematrix_tiled b 16 8) tile group steps

inline_for_extraction noextract
let rec mma_prefix_approx
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (ra:chest2 real m h) (rb:chest2 real h e)
  (tile:natlt (m / 16)) (group:natlt (e / 8)) (steps:nat{steps <= h / 16})
  : Lemma (requires a %~ ra /\ b %~ rb)
          (ensures mma_prefix a b tile group steps %~ real_prefix ra rb tile group steps)
          (decreases steps)
  = if steps = 0 then ()
    else (
      mma_prefix_approx a b ra rb tile group (steps - 1);
      MMA.emma_approx_lemma
        (mma_prefix a b tile group (steps - 1))
        (ematrix_subtile a 16 16 tile (steps - 1))
        (ematrix_subtile b 16 8 (steps - 1) group)
        (real_prefix ra rb tile group (steps - 1))
        (ematrix_subtile ra 16 16 tile (steps - 1))
        (ematrix_subtile rb 16 8 (steps - 1) group);
      G.__gmatmul_single_lemma real_zero_tile G.matmul G.matplus
        (ematrix_tiled ra 16 16) (ematrix_tiled rb 16 8) tile group steps;
      mma_prefix_step a b tile group (steps - 1)
    )

inline_for_extraction noextract
let mma_tile_approx
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (ra:chest2 real m h) (rb:chest2 real h e)
  (tile:natlt (m / 16)) (group:natlt (e / 8))
  : Lemma (requires a %~ ra /\ b %~ rb)
          (ensures mma_prefix a b tile group (h / 16) %~
            ematrix_subtile (G.matmul ra rb) 16 8 tile group)
  = mma_prefix_approx a b ra rb tile group (h / 16);
    G.matmul_tiles_lemma (fun _ -> ()) (fun _ _ _ -> ())
      16 8 16 real_zero_tile ra rb tile group;
    G.matmul_decompose_lemma ra rb 16 8 tile group;
    let result = ematrix_subtile (G.matmul ra rb) 16 8 tile group in
    assert (equal (G.matplus real_zero_tile result) result)

inline_for_extraction noextract
let outputs_to
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (group:natlt (e / 8)) (tiles:nat{tiles <= m / 16})
  : GTot (chest2 f32 m e)
  = mk2 (fun row expert ->
      if row / 16 < tiles && expert / 8 = group then
        acc2 (mma_prefix a b (row / 16) group (h / 16)) (row % 16) (expert % 8)
      else acc2 before row expert)

inline_for_extraction noextract
let outputs_to_zero
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (group:natlt (e / 8))
  : Lemma (outputs_to a b before group 0 == before)
  = assert (equal (outputs_to a b before group 0) before)

inline_for_extraction noextract
let outputs_to_step
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (group:natlt (e / 8)) (tile:natlt (m / 16))
  : Lemma (
      update_tile (outputs_to a b before group tile) 16 8 tile group
        (mma_prefix a b tile group (h / 16))
      == outputs_to a b before group (tile + 1))
  = assert (equal
      (update_tile (outputs_to a b before group tile) 16 8 tile group
        (mma_prefix a b tile group (h / 16)))
      (outputs_to a b before group (tile + 1)))

inline_for_extraction noextract
let columns_to
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (groups:nat{groups <= e / 8})
  : GTot (chest2 f32 m e)
  = mk2 (fun row expert ->
      if expert / 8 < groups then
        acc2 (mma_prefix a b (row / 16) (expert / 8) (h / 16))
          (row % 16) (expert % 8)
      else acc2 before row expert)

inline_for_extraction noextract
let columns_to_zero
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e) (before:chest2 f32 m e)
  : Lemma (columns_to a b before 0 == before)
  = assert (equal (columns_to a b before 0) before)

inline_for_extraction noextract
let columns_to_step
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (group:natlt (e / 8))
  : Lemma (
      outputs_to a b (columns_to a b before group) group (m / 16) ==
      columns_to a b before (group + 1))
  = assert (equal
      (outputs_to a b (columns_to a b before group) group (m / 16))
      (columns_to a b before (group + 1)))

inline_for_extraction noextract
let projection_approx
  (#m:pos{m % 16 == 0}) (#h:pos{h % 16 == 0}) (#e:pos{e % 8 == 0})
  (a:chest2 bf16 m h) (b:chest2 bf16 h e)
  (before:chest2 f32 m e) (ra:chest2 real m h) (rb:chest2 real h e)
  : Lemma (requires a %~ ra /\ b %~ rb)
          (ensures columns_to a b before (e / 8) %~ G.matmul ra rb)
  = let point (row:natlt m) (expert:natlt e)
      : Lemma (
          acc2 (columns_to a b before (e / 8)) row expert %~
          acc2 (G.matmul ra rb) row expert)
      = mma_tile_approx a b ra rb (row / 16) (expert / 8);
        assert (acc2 (mma_prefix a b (row / 16) (expert / 8) (h / 16))
          (row % 16) (expert % 8) %~
          acc2 (ematrix_subtile (G.matmul ra rb) 16 8 (row / 16) (expert / 8))
            (row % 16) (expert % 8))
    in
    FStar.Classical.forall_intro_2 point

inline_for_extraction noextract
fn multiply_step
  (m:szp{m % 16 == 0})
  (h:szp{h % 16 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e))
  (fr:MMA.fragment) (tile:szlt (m / 16)) (group:szlt (e / 8)) (step:szlt (h / 16))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e)
  preserves gpu ** a |-> Frac fa va ** b |-> Frac fb vb
  requires fr |-> mma_prefix va vb tile group step
  ensures fr |-> mma_prefix va vb tile group (step + 1)
{
  let atile = array2_extract_tile_ro' a 16 16 (SZ.v tile) (SZ.v step);
  let btile = array2_extract_tile_ro' b 16 8 (SZ.v step) (SZ.v group);
  MMA.mma_sync atile btile fr;
  elim_trade
    (atile |-> Frac fa (ematrix_subtile va 16 16 tile step))
    (a |-> Frac fa va);
  elim_trade
    (btile |-> Frac fb (ematrix_subtile vb 16 8 step group))
    (b |-> Frac fb vb);
  mma_prefix_step va vb tile group step;
}

inline_for_extraction noextract
fn multiply_tile
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (tile:szlt (m / 16)) (group:szlt (e / 8))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  preserves gpu ** a |-> Frac fa va ** b |-> Frac fb vb
  requires c |-> Frac (1.0R /. 32.0R) vc
  ensures c |-> Frac (1.0R /. 32.0R)
    (update_tile vc 16 8 tile group (mma_prefix va vb tile group (h / 16)))
{
  let fr = MMA.alloc_fragment ();
  MMA.fill fr zero;
  let mut hidden_tile = 0sz;
  while (!hidden_tile <^ h)
    invariant
      exists* (ht:sz{ht <= h /\ ht % 64 == 0}).
        hidden_tile |-> ht ** fr |-> mma_prefix va vb tile group (ht / 16)
    decreases (h - !hidden_tile)
  {
    let base = !hidden_tile;
    let mut hidden_step = 0sz;
    while (!hidden_step <^ 64sz)
      invariant
        exists* (hs:sz{hs <= 64 /\ hs % 16 == 0 /\ base + hs <= h}).
          hidden_step |-> hs **
          fr |-> mma_prefix va vb tile group ((base + hs) / 16)
      decreases (64 - !hidden_step)
    {
      let step = !hidden_step;
      let hidden = base +^ step;
      multiply_step m h e a b fr tile group (hidden /^ 16sz);
      hidden_step := !hidden_step +^ 16sz;
    };
    hidden_tile := !hidden_tile +^ 64sz;
  };
  array2_extract_tile c 16 8 tile group;
  let ctile = array2_subtile c 16 8 (SZ.v tile) (SZ.v group);
  assert (rewrites_to ctile (array2_subtile c 16 8 (SZ.v tile) (SZ.v group)));
  MMA.store fr ctile;
  Pulse.Lib.Forall.elim_forall (mma_prefix va vb tile group (h / 16));
  elim_trade
    (ctile |-> Frac (1.0R /. 32.0R) (mma_prefix va vb tile group (h / 16)))
    (c |-> Frac (1.0R /. 32.0R)
      (update_tile vc 16 8 tile group (mma_prefix va vb tile group (h / 16))));
  drop_ (fr |-> mma_prefix va vb tile group (h / 16));
}

inline_for_extraction noextract
fn multiply_all
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (group:szlt (e / 8))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  preserves gpu ** a |-> Frac fa va ** b |-> Frac fb vb
  requires c |-> Frac (1.0R /. 32.0R) vc
  ensures c |-> Frac (1.0R /. 32.0R) (outputs_to va vb vc group (m / 16))
{
  outputs_to_zero va vb vc group;
  let mut token_base = 0sz;
  while (!token_base <^ m)
    invariant
      exists* (tb:sz{tb <= m /\ tb % 16 == 0}).
        token_base |-> tb **
        c |-> Frac (1.0R /. 32.0R) (outputs_to va vb vc group (tb / 16))
    decreases (m - !token_base)
  {
    let base = !token_base;
    let tile = base /^ 16sz;
    multiply_tile m h e a b c tile group;
    outputs_to_step va vb vc group tile;
    token_base := !token_base +^ 16sz;
  };
}

(* All lanes visit the same groups and tiles and execute every MMA collectively. *)
inline_for_extraction noextract
fn multiply_groups
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  preserves gpu ** a |-> Frac fa va ** b |-> Frac fb vb
  requires c |-> Frac (1.0R /. 32.0R) vc
  ensures c |-> Frac (1.0R /. 32.0R) (columns_to va vb vc (e / 8))
{
  columns_to_zero va vb vc;
  let mut group = 0sz;
  while (!group <^ e /^ 8sz)
    invariant exists* (g:sz{g <= e / 8}).
      group |-> g ** c |-> Frac (1.0R /. 32.0R) (columns_to va vb vc g)
    decreases (e / 8 - !group)
  {
    let g = !group;
    multiply_all m h e a b c g;
    columns_to_step va vb vc g;
    group := !group +^ 1sz;
  };
}

inline_for_extraction noextract
ghost fn warp_setup
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  ()
  norewrite
  requires a |-> Frac fa va ** b |-> Frac fb vb ** c |-> vc
  ensures (forall+ (_:natlt 32).
    a |-> Frac (fa /. 32.0R) va **
    b |-> Frac (fb /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
{
  tensor_share_n a 32;
  tensor_share_n b 32;
  tensor_share_n c 32;
  forevery_zip3
    (fun (_:natlt 32) -> a |-> Frac (fa /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (fb /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
}

inline_for_extraction noextract
ghost fn warp_teardown
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  ()
  norewrite
  requires (forall+ (_:natlt 32).
    a |-> Frac (fa /. 32.0R) va **
    b |-> Frac (fb /. 32.0R) vb **
    c |-> Frac (1.0R /. 32.0R) vc) ** emp
  ensures a |-> Frac fa va ** b |-> Frac fb vb ** c |-> vc
{
  forevery_unzip3
    (fun (_:natlt 32) -> a |-> Frac (fa /. 32.0R) va)
    (fun (_:natlt 32) -> b |-> Frac (fb /. 32.0R) vb)
    (fun (_:natlt 32) -> c |-> Frac (1.0R /. 32.0R) vc);
  tensor_gather_n a 32;
  tensor_gather_n b 32;
  tensor_gather_n c 32;
}

inline_for_extraction noextract
fn warp_thread
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h)) (b:array2 bf16 (cm h e)) (c:array2 f32 (rm m e))
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  (tid:szlt 32)
  ()
  norewrite
  requires
    gpu **
    (a |-> Frac (fa /. 32.0R) va **
     b |-> Frac (fb /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) vc) ** thread_id 32 tid
  ensures
    gpu **
    (a |-> Frac (fa /. 32.0R) va **
     b |-> Frac (fb /. 32.0R) vb **
     c |-> Frac (1.0R /. 32.0R) (columns_to va vb vc (e / 8))) ** thread_id 32 tid
{
  multiply_groups m h e a b c;
}

inline_for_extraction noextract
let projection_kernel
  (m:szp{m % 16 == 0})
  (h:szp{h % 64 == 0 /\ SZ.fits (m * h)})
  (e:szp{e % 8 == 0 /\ SZ.fits (e * h) /\ SZ.fits (m * e)})
  (a:array2 bf16 (rm m h){is_global a})
  (b:array2 bf16 (cm h e){is_global b})
  (c:array2 f32 (rm m e){is_global c})
  (#fa #fb:perm)
  (#va:chest2 bf16 m h) (#vb:chest2 bf16 h e) (#vc:chest2 f32 m e)
  : kernel_desc_1_n
      (a |-> Frac fa va ** b |-> Frac fb vb ** c |-> vc)
      (a |-> Frac fa va ** b |-> Frac fb vb ** c |-> columns_to va vb vc (e / 8))
  = {
      nthr = 32sz;
      frame = emp;
      kpre = (fun _ ->
        a |-> Frac (fa /. 32.0R) va **
        b |-> Frac (fb /. 32.0R) vb **
        c |-> Frac (1.0R /. 32.0R) vc);
      kpost = (fun _ ->
        a |-> Frac (fa /. 32.0R) va **
        b |-> Frac (fb /. 32.0R) vb **
        c |-> Frac (1.0R /. 32.0R) (columns_to va vb vc (e / 8)));
      block_setup = warp_setup m h e a b c #fa #fb #va #vb #vc;
      block_teardown = warp_teardown m h e a b c #fa #fb #va #vb
        #(columns_to va vb vc (e / 8));
      f = warp_thread m h e a b c #fa #fb #va #vb #vc;
      full_pre_sendable = solve;
      full_post_sendable = solve;
      kpre_sendable = (fun _ -> solve);
      kpost_sendable = (fun _ -> solve);
    }

inline_for_extraction noextract
let row_layout (#et:Type0) (rows columns:nat) (values:lseq et (rows * columns))
  : Lemma (from_seq (rm rows columns) values == row_major_view rows columns values)
  = assert (equal (from_seq (rm rows columns) values) (row_major_view rows columns values))

inline_for_extraction noextract
let row_roundtrip (#et:Type0) (rows columns:nat) (values:chest2 et rows columns)
  : Lemma (row_major_view rows columns (to_seq (rm rows columns) values) == values)
  = assert (equal (row_major_view rows columns (to_seq (rm rows columns) values)) values)

inline_for_extraction noextract
let weight_layout (hidden experts:nat) (weights:lseq bf16 (experts * hidden))
  : Lemma (
      from_seq (cm hidden experts) weights ==
      mtranspose (row_major_view experts hidden weights))
  = let point (h:natlt hidden) (expert:natlt experts)
      : Lemma (
          acc2 (from_seq (cm hidden experts) weights) h expert ==
          acc2 (mtranspose (row_major_view experts hidden weights)) h expert)
      = assert ((cm hidden experts).imap.f (idx2 h expert) == expert * hidden + h)
    in
    FStar.Classical.forall_intro_2 point;
    assert (equal (from_seq (cm hidden experts) weights)
      (mtranspose (row_major_view experts hidden weights)))

#push-options "--z3rlimit 20"
inline_for_extraction noextract
let compose
  (tokens hidden:nat) (experts:nat{admissible_shape tokens hidden experts})
  (x:lseq bf16 (tokens * hidden)) (w:lseq bf16 (experts * hidden))
  (before:chest2 f32 tokens experts) (normalized:lseq f32 (tokens * experts))
  : Lemma
      (requires matrix_post tokens experts
        (to_seq (rm tokens experts)
          (columns_to
            (from_seq (rm tokens hidden) x)
            (from_seq (cm hidden experts) w)
            before (experts / 8)))
        normalized)
      (ensures projection_softmax_post tokens hidden experts x w normalized)
  = row_layout tokens hidden x;
    weight_layout hidden experts w;
    let projected = columns_to
      (from_seq (rm tokens hidden) x)
      (from_seq (cm hidden experts) w) before (experts / 8) in
    let scores = to_seq (rm tokens experts) projected in
    introduce forall (rx:chest2 real tokens hidden) (rw:chest2 real experts hidden).
      row_major_view tokens hidden x %~ rx /\
      row_major_view experts hidden w %~ rw ==>
      row_major_view tokens experts normalized %~ projection_softmax_real rx rw
    with introduce _ ==> _ with (
      projection_approx
        (from_seq (rm tokens hidden) x)
        (from_seq (cm hidden experts) w) before rx (mtranspose rw);
      row_roundtrip tokens experts projected;
      assert (flat_view tokens experts scores == row_major_view tokens experts scores);
      assert (flat_view tokens experts scores == projected);
      assert_norm (projection_real rx rw == G.matmul rx (mtranspose rw));
      assert (flat_view tokens experts scores %~ projection_real rx rw);
      assert (flat_view tokens experts normalized %~ softmax_matrix (projection_real rx rw));
      assert (flat_view tokens experts normalized == row_major_view tokens experts normalized);
      assert_norm (softmax_matrix (projection_real rx rw) == projection_softmax_real rx rw);
      ()
    )
#pop-options

inline_for_extraction noextract
ghost fn restore_array
  (#et:Type0) (#rank:nat) (#dimensions:shape rank)
  (layout:tlayout dimensions{is_full layout})
  (storage:larray et (tlayout_ulen layout))
  (#f:perm) (#values:lseq et (tlayout_ulen layout))
  requires from_array layout storage |-> Frac f (from_seq layout values)
  ensures storage |-> Frac f values
{
  tensor_concr (from_array layout storage);
  rewrite each (core (from_array layout storage)) as storage;
  to_from layout values;
  rewrite each (to_seq layout (from_seq layout values)) as values;
}

fn kplay_run
  (token_count:u64{0 < U64.v token_count /\ U64.v token_count <= 2147483647})
  (hidden_count:u64{0 < U64.v hidden_count /\ U64.v hidden_count <= 2147483647})
  (expert_count:u64{admissible_shape
    (U64.v token_count) (U64.v hidden_count) (U64.v expert_count)})
  (tokens:larray bf16 (U64.v token_count * U64.v hidden_count){is_global_array tokens})
  (weights:larray bf16 (U64.v expert_count * U64.v hidden_count){is_global_array weights})
  (normalized_rows:larray f32 (U64.v token_count * U64.v expert_count){
    is_global_array normalized_rows})
  (#token_perm #weight_perm:perm)
  (#x:erased (lseq bf16 (U64.v token_count * U64.v hidden_count)))
  (#w:erased (lseq bf16 (U64.v expert_count * U64.v hidden_count)))
  preserves cpu ** on gpu_loc (
    tokens |-> Frac token_perm (reveal x) ** weights |-> Frac weight_perm (reveal w))
  requires
    pure (input_domain (U64.v token_count) (U64.v hidden_count) (U64.v expert_count) x w) **
    (exists* (before:lseq f32 (U64.v token_count * U64.v expert_count)).
      on gpu_loc (normalized_rows |-> before))
  returns modeled:erased (chest2 real (U64.v token_count) (U64.v expert_count))
  ensures exists* (normalized:lseq f32 (U64.v token_count * U64.v expert_count)).
    on gpu_loc (normalized_rows |-> normalized) **
    pure (semantic_post (U64.v token_count) (U64.v hidden_count) (U64.v expert_count)
      x x w w normalized (reveal modeled))
{
  SZ.fits_ok (U64.v token_count);
  SZ.fits_ok (U64.v hidden_count);
  SZ.fits_ok (U64.v expert_count);
  let m = FStar.SizeT.uint64_to_sizet token_count;
  let h = FStar.SizeT.uint64_to_sizet hidden_count;
  let e = FStar.SizeT.uint64_to_sizet expert_count;
  let scores = gpu_array_alloc #f32 (m *^ e);
  with initial_scores. assert on gpu_loc (scores |-> initial_scores);
  let a = from_array (rm m h) tokens;
  let b = from_array (cm h e) weights;
  let c = from_array (rm m e) scores;
  assert (rewrites_to a (from_array (rm m h) tokens));
  assert (rewrites_to b (from_array (cm h e) weights));
  assert (rewrites_to c (from_array (rm m e) scores));
  map_loc gpu_loc
    #(tokens |-> Frac token_perm (reveal x) **
      weights |-> Frac weight_perm (reveal w) ** scores |-> initial_scores)
    #(a |-> Frac token_perm (from_seq (rm m h) x) **
      b |-> Frac weight_perm (from_seq (cm h e) w) **
      c |-> from_seq (rm m e) initial_scores)
    fn () {
      tensor_abs' (rm m h) tokens;
      tensor_abs' (cm h e) weights;
      tensor_abs' (rm m e) scores;
    };
  with before. assert on gpu_loc (c |-> before);
  launch_sync (projection_kernel m h e a b c);
  with projected. assert on gpu_loc (c |-> projected);
  map_loc gpu_loc
    #(a |-> Frac token_perm (from_seq (rm m h) x) **
      b |-> Frac weight_perm (from_seq (cm h e) w) ** c |-> projected)
    #(tokens |-> Frac token_perm (reveal x) **
      weights |-> Frac weight_perm (reveal w) ** scores |-> to_seq (rm m e) projected)
    fn () {
      restore_array (rm m h) tokens;
      restore_array (cm h e) weights;
      tensor_concr c;
      rewrite each (core c) as scores;
    };
  launch_sync (softmax_kernel m e scores normalized_rows);
  with normalized. assert on gpu_loc (normalized_rows |-> normalized);
  compose m h e x w before normalized;
  completed_post m h e x w normalized;
  gpu_array_free scores;
  let modeled = hide (projection_softmax_real
    (to_real_chest (row_major_view (U64.v token_count) (U64.v hidden_count) x))
    (to_real_chest (row_major_view (U64.v expert_count) (U64.v hidden_count) w)));
  modeled
}
