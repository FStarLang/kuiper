module KPlay.Proof.OptimizedMMA

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
let row_major_view (#et:Type0) (rows columns:nat)
  (values:lseq et (rows * columns)) : GTot (chest2 et rows columns) =
  mk2 (fun row column -> S.index values (row * columns + column))

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

