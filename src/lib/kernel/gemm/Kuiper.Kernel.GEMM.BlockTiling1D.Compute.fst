module Kuiper.Kernel.GEMM.BlockTiling1D.Compute

#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Chest
open Kuiper.Tensor.Tiling
open Kuiper.Kernel.GEMMGPU.Type
open Kuiper.Kernel.GEMM.FlipFlopColumns
open Pulse.Lib.Trade
module T = Kuiper.Tensor
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT

#set-options "--z3rlimit 60"

let tile_quotient (n : nat) (tile : pos)
  : Lemma (n * tile / tile == n) [SMTPat (n * tile / tile)]
  = Kuiper.Math.div_mod_of_mul_add tile n 0

(* Updating one index leaves every other prefix predicate unchanged. *)
let prefix_frame (n : nat) (done : nat)
  (p : natlt n -> bool -> GTot slprop)
  : Lemma (forall (i : natlt n{i =!= done}).
      p i (i < done + 1) == p i (i < done))
  = ()

(* Tile accumulation is a contiguous extension of the full dot product. *)
let rec tiled_step
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (tile : valid_tile) (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #n : nat)
  (eA : chest2 ta (m * tile) (k * tile))
  (eB : chest2 tb (k * tile) (n * tile))
  (row : natlt m) (col : natlt n)
  (i j : natlt tile) (bk : natlt k) (d : nat{d <= tile})
  : Lemma
    (MS.__gmatmul_single
      (MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) (bk * tile))
      mul add
      (chest_map mapA (ematrix_subtile eA tile tile row bk))
      (chest_map mapB (ematrix_subtile eB tile tile bk col)) i j d
      == MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) (bk * tile + d))
    (decreases d)
  = if d = 0 then ()
    else begin
      tiled_step tile mapA mapB eA eB row col i j bk (d - 1);
      MS.__gmatmul_single_lemma
        (MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
          (row * tile + i) (col * tile + j) (bk * tile)) mul add
        (chest_map mapA (ematrix_subtile eA tile tile row bk))
        (chest_map mapB (ematrix_subtile eB tile tile bk col)) i j d;
      MS.__gmatmul_single_lemma zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) (bk * tile + d)
    end

let advance_columns
  (#ta #tb #tacc : Type0) {| scalar tacc |}
  (tile : valid_tile) (mapA : ta -> tacc) (mapB : tb -> tacc)
  (#m #k #n : nat)
  (eA : chest2 ta (m * tile) (k * tile))
  (eB : chest2 tb (k * tile) (n * tile))
  (row : natlt m) (col : natlt n) (j : natlt tile) (bk : natlt k)
  (s0 s1 : lseq tacc tile)
  : Lemma
    (requires
      (forall (i : natlt tile). s0 @! i ==
        MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
          (row * tile + i) (col * tile + j) (bk * tile)) /\
      (forall (i : natlt tile). s1 @! i ==
        MS.__gmatmul_single (s0 @! i) mul add
          (chest_map mapA (ematrix_subtile eA tile tile row bk))
          (chest_map mapB (ematrix_subtile eB tile tile bk col)) i j tile))
    (ensures forall (i : natlt tile). s1 @! i ==
      MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) ((bk + 1) * tile))
  = introduce forall (i : natlt tile). s1 @! i ==
      MS.__gmatmul_single zero mul add (chest_map mapA eA) (chest_map mapB eB)
        (row * tile + i) (col * tile + j) ((bk + 1) * tile)
    with tiled_step tile mapA mapB eA eB row col i j bk tile

(* Keep the quantified frame opaque; expose only the cell being written. *)
[@@ "opaque_to_smt"]
let column_cell
  (#et : Type0) (#tile : nat) (#l : layout2 tile tile)
  (m : array2 et l) (em : chest2 et tile tile) (tid : natlt tile)
  (initialized : bool) (ii : natlt tile)
  : slprop =
  exists* (value : et). tensor_pts_to_cell m (idx2 ii tid) value **
    pure (initialized ==> value == acc2 em ii tid)

inline_for_extraction noextract
fn bring_2cols
  (tile : valid_tile)
  (#ta #tb #tacc : Type0) {| scalar ta, scalar tb, scalar tacc |}
  (mapA : ta -> tacc)
  (mapB : tb -> tacc)
  (#m #n #k : erased nat)
  (#lA : layout2 (m * tile) (k * tile))
  (#lB : layout2 (k * tile) (n * tile))
  {| T.ctlayout lA, T.ctlayout lB |}
  (gA : array2 ta lA)
  (gB : array2 tb lB)
  (#l1 #l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (sa1 : array2 tacc l1) (sa2 : array2 tacc l2)
  (mm : szlt m)
  (kk : szlt k)
  (nn : szlt n)
  (tid : szlt tile)
  (#fA #fB : perm)
  (#eA : chest2 ta (m * tile) (k * tile))
  (#eB : chest2 tb (k * tile) (n * tile))
  preserves
    gpu **
    gA |-> Frac fA eA **
    gB |-> Frac fB eB
  requires
    own_1_col sa1 tid **
    own_1_col sa2 tid
  ensures
    own_1_col_at sa1 (chest_map mapA (ematrix_subtile eA tile tile mm kk)) tid **
    own_1_col_at sa2 (chest_map mapB (ematrix_subtile eB tile tile kk nn)) tid
{
  let vA = chest_map mapA (ematrix_subtile eA tile tile mm kk);
  let vB = chest_map mapB (ematrix_subtile eB tile tile kk nn);
  assert rewrites_to vA (chest_map mapA (ematrix_subtile eA tile tile mm kk));
  assert rewrites_to vB (chest_map mapB (ematrix_subtile eB tile tile kk nn));
  unfold own_1_col sa1 tid;
  unfold own_1_col sa2 tid;
  forevery_map
    (fun (ii : natlt tile) -> exists* (value : tacc). tensor_pts_to_cell sa1 (idx2 ii tid) value)
    (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < 0) ii)
    fn ii { fold column_cell sa1 vA tid (ii < 0) ii };
  forevery_map
    (fun (ii : natlt tile) -> exists* (value : tacc). tensor_pts_to_cell sa2 (idx2 ii tid) value)
    (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < 0) ii)
    fn ii { fold column_cell sa2 vB tid (ii < 0) ii };
  let mut i = 0sz;
  while (!i <^ tile)
    invariant live i ** pure (!i <= tile)
    invariant forall+ (ii : natlt tile). column_cell sa1 vA tid (ii < !i) ii
    invariant forall+ (ii : natlt tile). column_cell sa2 vB tid (ii < !i) ii
    decreases (tile - !i)
  {
    {
      let ci = !i;
      forevery_extract' #(natlt tile) ci
        (fun ii -> column_cell sa1 vA tid (ii < ci) ii);
      unfold column_cell sa1 vA tid (ci < ci) ci;
      let tileM = array2_extract_tile_ro' gA (SZ.v tile) (SZ.v tile) (SZ.v mm) (SZ.v kk);
      let v1 = tensor_read tileM ((ci <: szlt _), ((tid <: szlt _), ()));
      tensor_write_cell sa1 ((ci <: szlt _), ((tid <: szlt _), ())) (mapA v1);
      ambig_trade_elim ();
      fold column_cell sa1 vA tid true ci;
      Pulse.Lib.Forall.elim_forall (fun ii -> column_cell sa1 vA tid (ii < ci + 1) ii);
      rewrite column_cell sa1 vA tid true ci as column_cell sa1 vA tid (ci < ci + 1) ci;
      prefix_frame tile ci (fun ii initialized -> column_cell sa1 vA tid initialized ii);
      elim_trade _ _;
    };
    {
      let ci = !i;
      forevery_extract' #(natlt tile) ci
        (fun ii -> column_cell sa2 vB tid (ii < ci) ii);
      unfold column_cell sa2 vB tid (ci < ci) ci;
      let tileM = array2_extract_tile_ro' gB (SZ.v tile) (SZ.v tile) (SZ.v kk) (SZ.v nn);
      let v2 = tensor_read tileM ((ci <: szlt _), ((tid <: szlt _), ()));
      tensor_write_cell sa2 ((ci <: szlt _), ((tid <: szlt _), ())) (mapB v2);
      ambig_trade_elim ();
      fold column_cell sa2 vB tid true ci;
      Pulse.Lib.Forall.elim_forall (fun ii -> column_cell sa2 vB tid (ii < ci + 1) ii);
      rewrite column_cell sa2 vB tid true ci as column_cell sa2 vB tid (ci < ci + 1) ci;
      prefix_frame tile ci (fun ii initialized -> column_cell sa2 vB tid initialized ii);
      elim_trade _ _;
    };
    let prev = !i;
    let next = !i +^ 1sz;
    forevery_ext (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < prev + 1) ii)
      (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < next) ii);
    forevery_ext (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < prev + 1) ii)
      (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < next) ii);
    i := next;
  };
  let done = !i;
  forevery_ext (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < done) ii)
    (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < tile) ii);
  forevery_ext (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < done) ii)
    (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < tile) ii);
  forevery_map
    (fun (ii : natlt tile) -> column_cell sa1 vA tid (ii < tile) ii)
    (fun (ii : natlt tile) -> tensor_pts_to_cell sa1 (idx2 ii tid) (acc2 vA ii tid))
    fn ii { unfold column_cell sa1 vA tid (ii < tile) ii };
  fold own_1_col_at sa1 vA tid;
  forevery_map
    (fun (ii : natlt tile) -> column_cell sa2 vB tid (ii < tile) ii)
    (fun (ii : natlt tile) -> tensor_pts_to_cell sa2 (idx2 ii tid) (acc2 vB ii tid))
    fn ii { unfold column_cell sa2 vB tid (ii < tile) ii };
  fold own_1_col_at sa2 vB tid;
}

let column_dot_step
  (#et : Type0) {| scalar et |}
  (tile : nat) (acc0 : lseq et tile) (em1 em2 : chest2 et tile tile)
  (j sk : natlt tile)
  : Lemma (forall (ii : natlt tile).
      MS.__gmatmul_single (acc0 @! ii) mul add em1 em2 ii j (sk + 1)
      == add (MS.__gmatmul_single (acc0 @! ii) mul add em1 em2 ii j sk)
             (mul (acc2 em1 ii sk) (acc2 em2 sk j)))
  = introduce forall (ii : natlt tile).
      MS.__gmatmul_single (acc0 @! ii) mul add em1 em2 ii j (sk + 1)
      == add (MS.__gmatmul_single (acc0 @! ii) mul add em1 em2 ii j sk)
             (mul (acc2 em1 ii sk) (acc2 em2 sk j))
    with MS.__gmatmul_single_lemma (acc0 @! ii) mul add em1 em2 ii j (sk + 1)

inline_for_extraction noextract
fn subproduct_cols
  (#et : Type0) {| scalar et |}
  (tile : sz)
  (acc : array et)
  (#l1 #l2 : layout2 tile tile)
  {| T.ctlayout l1, T.ctlayout l2 |}
  (m1 : array2 et l1)
  (m2 : array2 et l2)
  (j : szlt tile)
  (#acc0 : erased (lseq et tile))
  (#em1 #em2 : chest2 et tile tile)
  (#f : perm)
  preserves
    gpu **
    m1 |-> Frac f em1 **
    m2 |-> Frac f em2
  requires
    acc |-> acc0
  ensures
    exists* (acc' : lseq et tile).
      acc |-> acc' ** pure (forall (i : natlt tile).
        acc' @! i == MS.__gmatmul_single (acc0 @! i) mul add em1 em2 i j tile)
{
  Pulse.Lib.Array.pts_to_len acc;
  let mut sk : szle tile = 0sz;
  while (!sk <^ tile)
    invariant live sk ** pure (!sk <= tile)
    invariant exists* (s : lseq et tile). acc |-> s **
      pure (forall (i : natlt tile).
        s @! i == MS.__gmatmul_single (acc0 @! i) mul add em1 em2 i j !sk)
    decreases (tile - !sk)
  {
    with s0. assert acc |-> s0;
    let mut i = 0sz;
    (* We can read em2 out of the inner loop, this is extremely
       important for performance. NVCC may realize this is invariant
       across iterations and hoist it out, but don't rely on it. *)
    let csk = !sk;
    let v2 = tensor_read m2 ((csk <: szlt _), ((j <: szlt _), ()));
    while (!i <^ tile)
      invariant live i ** pure (!i <= tile)
      invariant exists* (s : lseq et tile). acc |-> s **
        pure (forall (ii : natlt tile).
          s @! ii == (if ii < !i
            then add (s0 @! ii) (mul (acc2 em1 ii !sk) (acc2 em2 !sk j))
            else s0 @! ii))
      decreases (tile - !i)
    {
      let ci = !i;
      let csk2 = !sk;
      let v1 = tensor_read m1 ((ci <: szlt _), ((csk2 <: szlt _), ()));

      open Pulse.Lib.Array;
      pts_to_len acc;
      acc.(!i) <- acc.(!i) `add` (v1 `mul` v2);
      i := !i +^ 1sz;
    };
    column_dot_step tile acc0 em1 em2 j csk;
    sk := !sk +^ 1sz;
  };
  Pulse.Lib.Array.pts_to_len acc;
}
