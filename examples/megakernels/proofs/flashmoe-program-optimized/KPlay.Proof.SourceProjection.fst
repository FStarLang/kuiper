module KPlay.Proof.SourceProjection
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm, l2_col_major as cm }
open Pulse.Lib.Array

module S = FStar.Seq
module C = Kuiper.Chest
module P = KPlay.Proof.OptimizedMMA
module Layout = KPlay.Proof.SourceLayouts
module Shared = KPlay.SharedSpec
module Projection = KPlay.Proof.Shared.ProjectionSoftmax
module B = Kuiper.Floating.Bounds
module PA = Pulse.Lib.Array

inline_for_extraction noextract
let projected (x:Layout.token_values) (w:Layout.weight_values) : GTot (C.chest2 f32 256 8) =
  C.mk2 (fun t e -> C.acc2 (Layout.mma_prefix x w (t / 16) 32) (t % 16) e)

inline_for_extraction noextract
let scores (x:Layout.token_values) (w:Layout.weight_values) : GTot Layout.score_values =
  to_seq (rm 256 8) (projected x w)

inline_for_extraction noextract
let layout_inputs (x:Layout.token_values) (w:Layout.weight_values)
  : Lemma (
      from_seq (rm 256 512) x == Layout.logical_tokens x /\
      from_seq (cm 512 8) w == Layout.logical_weights w)
  =
  assert (C.equal (from_seq (rm 256 512) x) (Layout.logical_tokens x));
  assert (C.equal (from_seq (cm 512 8) w) (Layout.logical_weights w))

inline_for_extraction noextract
let rec prefix_identity (x:Layout.token_values) (w:Layout.weight_values)
  (tile:natlt 16) (n:nat{n <= 32})
  : Lemma
      (ensures Layout.mma_prefix x w tile n ==
        P.mma_prefix (Layout.logical_tokens x) (Layout.logical_weights w) tile 0 n)
      (decreases n)
  =
  if n = 0 then assert (C.equal (Layout.mma_prefix x w tile n) P.zero_tile)
  else (
    prefix_identity x w tile (n - 1);
    assert (C.equal (Layout.operand_a x tile (n - 1))
      (ematrix_subtile (Layout.logical_tokens x) 16 16 tile (n - 1)));
    assert (C.equal (Layout.operand_b w (n - 1))
      (ematrix_subtile (Layout.logical_weights w) 16 8 (n - 1) 0))
  )

inline_for_extraction noextract
let completed_identity (x:Layout.token_values) (w:Layout.weight_values)
  (before:C.chest2 f32 256 8)
  : Lemma (
      P.outputs_to (Layout.logical_tokens x) (Layout.logical_weights w) before 0 16 ==
        projected x w /\
      P.columns_to (Layout.logical_tokens x) (Layout.logical_weights w) before 1 ==
        projected x w)
  =
  let cell (t:natlt 256) (e:natlt 8)
    : Lemma (
      C.acc2 (P.outputs_to (Layout.logical_tokens x) (Layout.logical_weights w)
        before 0 16) t e == C.acc2 (projected x w) t e /\
      C.acc2 (P.columns_to (Layout.logical_tokens x) (Layout.logical_weights w)
        before 1) t e == C.acc2 (projected x w) t e)
    = prefix_identity x w (t / 16) 32
  in
  FStar.Classical.forall_intro_2 cell;
  assert (C.equal (P.outputs_to (Layout.logical_tokens x) (Layout.logical_weights w)
    before 0 16) (projected x w));
  assert (C.equal (P.columns_to (Layout.logical_tokens x) (Layout.logical_weights w)
    before 1) (projected x w))

inline_for_extraction noextract
let score_certificate (x:Layout.token_values) (w:Layout.weight_values)
  (t:natlt 256) (e:natlt 8)
  : Lemma
      (requires Shared.input_domain (Layout.logical_tokens x) (Layout.logical_weights w))
      (ensures
        B.magnitude 65536 (S.index (scores x w) (t * 8 + e)) /\
        S.index (scores x w) (t * 8 + e) == C.acc2 (projected x w) t e /\
        v_approximates (S.index (scores x w) (t * 8 + e))
          (C.acc2 (Projection.projection true (Layout.logical_tokens x)
            (Layout.logical_weights w)) t e))
  =
  let before:C.chest2 f32 256 8 = C.mk2 (fun _ _ -> zero) in
  Layout.mma_prefix_bounds x w (t / 16) 32;
  assert (B.magnitude 65536
    (C.acc (Layout.mma_prefix x w (t / 16) 32) (t % 16, (e, ()))));
  assert (C.acc2 (projected x w) t e ==
    C.acc2 (Layout.mma_prefix x w (t / 16) 32) (t % 16) e);
  P.row_roundtrip 256 8 (projected x w);
  assert (S.index (scores x w) (t * 8 + e) == C.acc2 (projected x w) t e);
  completed_identity x w before;
  C.lemma_to_real_chest_approximates (Layout.logical_tokens x);
  C.lemma_to_real_chest_approximates (Layout.logical_weights w);
  P.projection_approx (Layout.logical_tokens x) (Layout.logical_weights w) before
    (C.to_real_chest (Layout.logical_tokens x)) (C.to_real_chest (Layout.logical_weights w));
  assert (v_approximates (C.acc2 (projected x w) t e)
    (C.acc2 (Projection.projection true (Layout.logical_tokens x)
      (Layout.logical_weights w)) t e))

inline_for_extraction noextract
let garray (a:Type0) (n:nat) = p:larray a n{is_global_array p}

inline_for_extraction noextract
instance send_global_contents
  (#a:Type0) (storage:PA.array a{is_global_array storage})
  (#f:perm) (values:S.seq a)
  : is_send_across gpu_of (PA.pts_to storage #f values) =
  Kuiper.Array.Core.is_send_pts_to storage #f values

inline_for_extraction noextract
let pre (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (output:garray f32 2048) (x:Layout.token_values) (w:Layout.weight_values)
  (fp fw:perm) : slprop =
  PA.pts_to tokens #fp x ** PA.pts_to weights #fw w **
    (exists* (old:lseq f32 2048). PA.pts_to output old)

inline_for_extraction noextract
let post (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (output:garray f32 2048) (x:Layout.token_values) (w:Layout.weight_values)
  (fp fw:perm) : slprop =
  PA.pts_to tokens #fp x ** PA.pts_to weights #fw w ** PA.pts_to output (scores x w)

inline_for_extraction noextract
let lane_pre
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
  b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
  (exists* (old:C.chest2 f32 256 8). c |-> Frac (1.0R /. 32.0R) old)

inline_for_extraction noextract
let lane_post
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) : slprop =
  a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
  b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
  c |-> Frac (1.0R /. 32.0R) (projected x w)

inline_for_extraction noextract
ghost
fn forget_old
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm)
  (old:C.chest2 f32 256 8) (_:natlt 32)
  requires
    a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
    b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
    c |-> Frac (1.0R /. 32.0R) old
  ensures lane_pre a b c x w fp fw
{
  fold lane_pre a b c x w fp fw;
}

inline_for_extraction noextract
ghost
fn setup_tensors
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm)
  (old:C.chest2 f32 256 8) ()
  norewrite
  requires a |-> Frac fp (Layout.logical_tokens x) **
    b |-> Frac fw (Layout.logical_weights w) ** c |-> old
  ensures (forall+ (_:natlt 32). lane_pre a b c x w fp fw) ** emp
{
  P.warp_setup 256sz 512sz 8sz a b c #fp #fw
    #(Layout.logical_tokens x) #(Layout.logical_weights w) #old ();
  Kuiper.ForEvery.forevery_map
    (fun (_:natlt 32) ->
      a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
      b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
      c |-> Frac (1.0R /. 32.0R) old)
    (fun (_:natlt 32) -> lane_pre a b c x w fp fw)
    (forget_old a b c x w fp fw old);
}

inline_for_extraction noextract
ghost
fn setup (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (output:garray f32 2048) (x:Layout.token_values) (w:Layout.weight_values)
  (fp fw:perm) ()
  norewrite
  requires pre tokens weights output x w fp fw
  ensures
    (forall+ (_:natlt 32).
      lane_pre (from_array (rm 256 512) tokens) (from_array (cm 512 8) weights)
        (from_array (rm 256 8) output) x w fp fw) ** emp
{
  unfold pre tokens weights output x w fp fw;
  with old. assert PA.pts_to output old;
  tensor_abs' (rm 256 512) tokens;
  tensor_abs' (cm 512 8) weights;
  tensor_abs' (rm 256 8) output;
  layout_inputs x w;
  let a = from_array (rm 256 512) tokens;
  let b = from_array (cm 512 8) weights;
  let c = from_array (rm 256 8) output;
  assert (rewrites_to a (from_array (rm 256 512) tokens));
  assert (rewrites_to b (from_array (cm 512 8) weights));
  assert (rewrites_to c (from_array (rm 256 8) output));
  rewrite each (from_seq (rm 256 512) x) as (Layout.logical_tokens x);
  rewrite each (from_seq (cm 512 8) w) as (Layout.logical_weights w);
  setup_tensors a b c x w fp fw (from_seq (rm 256 8) old) ();
}

inline_for_extraction noextract
fn thread
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (#x:erased Layout.token_values) (#w:erased Layout.weight_values) (fp fw:perm)
  (tid:szlt 32) ()
  norewrite
  requires gpu ** lane_pre a b c x w fp fw ** thread_id 32 tid
  ensures gpu ** lane_post a b c x w fp fw ** thread_id 32 tid
{
  unfold lane_pre a b c x w fp fw;
  with old. assert c |-> Frac (1.0R /. 32.0R) old;
  P.multiply_all 256sz 512sz 8sz a b c 0sz
    #(fp /. 32.0R) #(fw /. 32.0R) #(Layout.logical_tokens x) #(Layout.logical_weights w) #old;
  completed_identity x w old;
  fold lane_post a b c x w fp fw;
}

inline_for_extraction noextract
ghost
fn expose_lane
  (a:array2 bf16 (rm 256 512)) (b:array2 bf16 (cm 512 8))
  (c:array2 f32 (rm 256 8))
  (x:Layout.token_values) (w:Layout.weight_values) (fp fw:perm) (_:natlt 32)
  requires lane_post a b c x w fp fw
  ensures
    a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
    b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
    c |-> Frac (1.0R /. 32.0R) (projected x w)
{
  unfold lane_post a b c x w fp fw;
}

inline_for_extraction noextract
ghost
fn teardown (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (output:garray f32 2048) (x:Layout.token_values) (w:Layout.weight_values)
  (fp fw:perm) ()
  norewrite
  requires
    (forall+ (_:natlt 32).
      lane_post (from_array (rm 256 512) tokens) (from_array (cm 512 8) weights)
        (from_array (rm 256 8) output) x w fp fw) ** emp
  ensures post tokens weights output x w fp fw
{
  let a = from_array (rm 256 512) tokens;
  let b = from_array (cm 512 8) weights;
  let c = from_array (rm 256 8) output;
  assert (rewrites_to a (from_array (rm 256 512) tokens));
  assert (rewrites_to b (from_array (cm 512 8) weights));
  assert (rewrites_to c (from_array (rm 256 8) output));
  Kuiper.ForEvery.forevery_map
    (fun (_:natlt 32) -> lane_post a b c x w fp fw)
    (fun (_:natlt 32) ->
      a |-> Frac (fp /. 32.0R) (Layout.logical_tokens x) **
      b |-> Frac (fw /. 32.0R) (Layout.logical_weights w) **
      c |-> Frac (1.0R /. 32.0R) (projected x w))
    (expose_lane a b c x w fp fw);
  P.warp_teardown 256sz 512sz 8sz a b c #fp #fw
    #(Layout.logical_tokens x) #(Layout.logical_weights w) #(projected x w) ();
  layout_inputs x w;
  P.restore_array (rm 256 512) tokens #fp #x;
  P.restore_array (cm 512 8) weights #fw #w;
  tensor_concr c;
  rewrite each (core (from_array (rm 256 8) output)) as output;
  fold post tokens weights output x w fp fw;
}

inline_for_extraction noextract
let kernel (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (output:garray f32 2048) (#x:erased Layout.token_values) (#w:erased Layout.weight_values)
  (fp fw:perm)
  : kernel_desc_1_n
      (pre tokens weights output x w fp fw)
      (post tokens weights output x w fp fw) =
  {
    nthr = 32sz;
    frame = emp;
    kpre = (fun _ -> lane_pre (from_array (rm 256 512) tokens)
      (from_array (cm 512 8) weights) (from_array (rm 256 8) output) x w fp fw);
    kpost = (fun _ -> lane_post (from_array (rm 256 512) tokens)
      (from_array (cm 512 8) weights) (from_array (rm 256 8) output) x w fp fw);
    block_setup = setup tokens weights output x w fp fw;
    block_teardown = teardown tokens weights output x w fp fw;
    f = thread (from_array (rm 256 512) tokens) (from_array (cm 512 8) weights)
      (from_array (rm 256 8) output) #x #w fp fw;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = (fun _ -> solve);
    kpost_sendable = (fun _ -> solve)
  }
