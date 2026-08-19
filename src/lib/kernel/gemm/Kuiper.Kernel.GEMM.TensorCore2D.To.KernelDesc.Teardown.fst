module Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc.Teardown

#lang-pulse

open Kuiper
open Kuiper.Array.Vectorized { has_vec_cpy, chunk }
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor
open Kuiper.Array2.Strided
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major as rm }
open Kuiper.Kernel.GEMM.Tiled.Common.Vec
open Kuiper.TensorCore

module SZ = Kuiper.SizeT
module MS = Kuiper.Spec.GEMM

open Kuiper.Kernel.GEMM.TensorCore2D.KernelDesc

open Kuiper.Kernel.GEMM.TensorCore2D.To.KernelDesc

(* (i * n + j) / n == i and (i * n + j) % n == j, when j < n. Kept as a
   top-level pure lemma so the nonlinear division/modulo facts type-check in
   a minimal context: inside the large ambient proof state of gather_block
   and gather_output (with many size-refinement facts already in scope), Z3
   does not reliably close this goal when it is asserted inline. *)
let div_mod_of_mul_add (n : pos) (i : nat) (j : natlt n)
  : Lemma ((i * n + j) / n == i /\ (i * n + j) % n == j)
  = FStar.Math.Lemmas.lemma_div_plus j i n;
    FStar.Math.Lemmas.small_div j n;
    FStar.Math.Lemmas.lemma_mod_plus j i n;
    FStar.Math.Lemmas.small_mod j n

(* [teardown_block_output]/[teardown_warp_output]/[teardown_lane_output] name
   the real-valued expected tile at, respectively, the block/warp/lane
   granularity: the corresponding subtile of the full [MS.mmcomb comb_r rC rA
   rB] result.

   These are deliberately *named* abbreviations rather than the equivalent
   [ematrix_subtile (ematrix_subtile (MS.mmcomb comb_r rC rA rB) ...) ...]
   terms written out inline at every call site. gather_output threads this
   value (or a subtile of it) through half a dozen [forevery_*] combinator
   calls; spelling it out inline at each of those call sites makes every one
   of those calls compare two large, differently-nested-but-definitionally-
   equal copies of the [MS.mmcomb]/[ematrix_subtile] term, which sends F*'s
   core typechecker into a deep delta dive through the [chest]
   representation over and over. Naming the term once and reusing the exact
   same (folded) application everywhere keeps those comparisons a cheap,
   linear head/argument match instead. This is the same fix already applied
   to [Kuiper.Kernel.GEMM.TensorCore2D.To.Finish]'s [epilogue_warp_output];
   see that module for the original profiling writeup. Do not inline these
   definitions into a specification. *)
let teardown_block_output_at
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (br : natlt (m / bm))
  (bc : natlt (n / bn))
  : chest2 real bm bn
= ematrix_subtile (MS.mmcomb comb_r rC rA rB) bm bn br bc

let teardown_block_output
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  : chest2 real bm bn
= teardown_block_output_at comb_r bm bn bk tm tn tk wm wn rA rB rC
    (bid / (n / bn)) (bid % (n / bn))

(* One-step unfolding bridge from [teardown_block_output] to
   [teardown_block_output_at], analogous to [teardown_lane_is_warp_output]
   above and needed for the same reason at the [gather_output]/[teardown_to]
   block-level [forevery_factor'] call. *)
let teardown_block_is_block_at
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  : Lemma (teardown_block_output comb_r bm bn bk tm tn tk wm wn
             nblk rA rB rC bid
           == teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
             rA rB rC (bid / (n / bn)) (bid % (n / bn)))
  = ()

let teardown_warp_output
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  (wid : natlt (bm / (wm * tm) * (bn / (wn * tn))))
  : chest2 real (wm * tm) (wn * tn)
= ematrix_subtile
    (teardown_block_output comb_r bm bn bk tm tn tk wm wn nblk rA rB rC bid)
    (wm * tm) (wn * tn)
    (wid / (bn / (wn * tn)))
    (wid % (bn / (wn * tn)))

let teardown_lane_output
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  (tid : natlt nthr)
  : chest2 real (wm * tm) (wn * tn)
= teardown_warp_output comb_r bm bn bk tm tn tk wm wn nblk rA rB rC bid
    (tid / warp_size)

(* [teardown_lane_output] is defined as exactly [teardown_warp_output] applied
   to [tid / warp_size]; this one-step unfolding lemma lets call sites bridge
   between the two named forms with a single cheap [()] proof (a definitional
   unfold at the F* typechecker level) instead of asking the ambient SMT query
   to rediscover that fact on its own (which, since neither name carries the
   [unfold] qualifier, it does not do automatically). *)
let teardown_lane_is_warp_output
  (comb_r : binop real)
  (#m #n #k : szp)
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  (bid : natlt nblk)
  (tid : natlt nthr)
  : Lemma (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
             nblk nthr rA rB rC bid tid
           == teardown_warp_output comb_r bm bn bk tm tn tk wm wn
             nblk rA rB rC bid (tid / warp_size))
  = ()

#push-options "--ifuel 1 --initial_fuel 0 --max_fuel 1 --z3rlimit 15"

let in_lane_covers_all
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  : Lemma (exists lane. in_lane rows cols lane ij)
= let lane : natlt warp_size =
    (ij._1 * cols + ij._2) % warp_size in
  assert (in_lane rows cols lane ij);
  ()

let in_lane_no_overlap
  (rows cols : nat)
  (ij : natlt rows & natlt cols)
  (lane1 lane2 : natlt warp_size)
  : Lemma
      (requires in_lane rows cols lane1 ij /\ in_lane rows cols lane2 ij)
      (ensures lane1 == lane2)
= ()

let lane_coincide
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  : prop
= forall (i : natlt rows) (j : natlt cols).
    in_lane rows cols lane (i, j) ==> acc2 em1 i j == acc2 em2 i j

ghost
fn own_lane_cells_rw
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (lane : natlt warp_size)
  (em1 em2 : chest2 et rows cols)
  (#_ : squash (lane_coincide lane em1 em2))
  requires own_lane_cells m em1 lane
  ensures own_lane_cells m em2 lane
{
  unfold own_lane_cells m em1 lane;
  forevery_map
    #(ij : (natlt rows & natlt cols){in_lane rows cols lane ij})
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2))
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2))
    fn ij {
      rewrite
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em1 ij._1 ij._2)
      as
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em2 ij._1 ij._2);
    };
  fold own_lane_cells m em2 lane;
}

ghost
fn join_array2_from_lane_cells
  (#et : Type0) {| scalar et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (#em : chest2 et rows cols)
  requires forall+ (lane : natlt warp_size). own_lane_cells m em lane
  ensures m |-> em
{
  forevery_map
    (fun lane -> own_lane_cells m em lane)
    (fun lane ->
      forall+ (ij : (natlt rows & natlt cols){in_lane rows cols lane ij}).
        tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2))
    fn lane { unfold own_lane_cells m em lane };
  forevery_join_or_n
    (fun (lane : natlt warp_size) ij -> in_lane rows cols lane ij)
    (fun ij -> tensor_pts_to_cell m (idx2 ij._1 ij._2) (acc2 em ij._1 ij._2));
  Classical.forall_intro (in_lane_covers_all rows cols);
  Classical.forall_intro_3
    (fun ij lane1 -> Classical.move_requires
      (in_lane_no_overlap rows cols ij lane1));
  forevery_refine_ext #_
    #(fun (ij : natlt rows & natlt cols) ->
      exists lane. in_lane rows cols lane ij)
    (fun _ -> True) _;
  forevery_unflatten' _;
  tensor_iraise2 m;
}

ghost
fn join_lane_cells_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (m : array2 et l)
  (r : chest2 real rows cols)
  requires
    forall+ (lane : natlt warp_size).
      exists* (em : chest2 et rows cols).
        own_lane_cells m em lane ** pure (em %~ r)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists #(natlt warp_size)
    (fun lane em -> own_lane_cells m em lane ** pure (em %~ r));
  let em' : chest2 et rows cols =
    mk2 (fun i j ->
      let lane : natlt warp_size = (i * cols + j) % warp_size in
      acc2 (ff lane) i j);
  forevery_unzip
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> pure (ff lane %~ r));
  forevery_elim_pure (fun lane -> ff lane %~ r);
  assert pure (em' %~ r);
  forevery_map
    (fun lane -> own_lane_cells m (ff lane) lane)
    (fun lane -> own_lane_cells m em' lane)
    fn lane {
      assert pure (lane_coincide lane (ff lane) em');
      own_lane_cells_rw m lane (ff lane) em';
    };
  join_array2_from_lane_cells m;
}

ghost
fn array2_untile_approximates
  (#et : Type0) {| scalar et, real_like et |}
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (m : array2 et l)
  (trows : pos{trows /? rows})
  (tcols : pos{tcols /? cols})
  {| enumerable (natlt (rows / trows)),
     enumerable (natlt (cols / tcols)) |}
  (r : chest2 real rows cols)
  (#_ : squash (SZ.fits l.ulen))
  (#_ : squash (SZ.fits (rows / trows)))
  (#_ : squash (SZ.fits (cols / tcols)))
  requires
    forall+ (tr : natlt (rows / trows))
             (tc : natlt (cols / tcols)).
      exists* (em : chest2 et trows tcols).
        array2_subtile m trows tcols tr tc |-> em **
        pure (em %~ ematrix_subtile r trows tcols tr tc)
  ensures
    exists* (em : chest2 et rows cols).
      m |-> em ** pure (em %~ r)
{
  let ff = forevery_exists_2
    #(natlt (rows / trows)) #_ #(natlt (cols / tcols)) #_
    (fun tr tc (em : chest2 et trows tcols) ->
      array2_subtile m trows tcols tr tc |-> em **
      pure (em %~ ematrix_subtile r trows tcols tr tc));
  forevery_extract_pure_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      ff tr tc %~ ematrix_subtile r trows tcols tr tc)
    fn tr tc { () };
  assert pure (forall (tr : natlt (rows / trows))
                      (tc : natlt (cols / tcols)).
    ff tr tc %~ ematrix_subtile r trows tcols tr tc);
  forevery_map_2
    #(natlt (rows / trows)) #(natlt (cols / tcols))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc **
      pure (ff tr tc %~ ematrix_subtile r trows tcols tr tc))
    (fun tr tc ->
      array2_subtile m trows tcols tr tc |-> ff tr tc)
    fn tr tc { () };
  array2_untile' m trows tcols ff;
  assert pure (ematrix_from_tiles trows tcols ff %~ r);
}

#push-options ""

ghost
fn gather_warp
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (wid : natlt (nthr / warp_size))
  (rWarp : chest2 real (wm * tm) (wn * tn))
  requires
    forall+ (lane : natlt warp_size).
      output_lane_approximates
        gD bm bn tm tn wm wn bid (wid * warp_size + lane)
        rWarp
  ensures
    exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
      warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
        (wm * tm) (wn * tn) wid |-> eWarp **
      pure (eWarp %~ rWarp)
{
  assert pure (wid < bm / (wm * tm) * (bn / (wn * tn)));
  assert pure (wid / (bn / (wn * tn)) < bm / (wm * tm));
  assert pure (wid % (bn / (wn * tn)) < bn / (wn * tn));
  Math.Lemmas.lemma_div_exact (wm * tm) tm;
  Math.Lemmas.lemma_div_exact (wn * tn) tn;
  assert pure ((wm * tm) / tm == wm);
  assert pure ((wn * tn) / tn == wn);
  forevery_map
    (fun (lane : natlt warp_size) ->
      output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp)
    (fun (lane : natlt warp_size) ->
      forall+ (mi : natlt wm) (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn lane {
      assert pure ((wid * warp_size + lane) / warp_size == wid);
      assert pure ((wid * warp_size + lane) % warp_size == lane);
      unfold output_lane_approximates gD bm bn tm tn wm wn bid
        (wid * warp_size + lane) rWarp;
      rewrite each ((wid * warp_size + lane) / warp_size) as wid;
      rewrite each ((wid * warp_size + lane) % warp_size) as lane;
    };
  forevery_commute
    (fun (lane : natlt warp_size) (mi : natlt wm) ->
      forall+ (nj : natlt wn).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  forevery_mid_flip
    (fun (mi : natlt wm) (lane : natlt warp_size) (nj : natlt wn) ->
      exists* (eFrag : chest2 et_cd tm tn).
        own_lane_cells
          (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
          eFrag lane **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  forevery_map_2 #(natlt wm) #(natlt wn)
    (fun mi nj ->
      forall+ (lane : natlt warp_size).
        exists* (eFrag : chest2 et_cd tm tn).
          own_lane_cells
            (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
            eFrag lane **
          pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      join_lane_cells_approximates
        (output_fragment gD bm bn tm tn wm wn bid wid mi nj)
        (ematrix_subtile rWarp tm tn mi nj);
    };
  let dWarp =
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
  forevery_map_2 #(natlt wm) #(natlt wn)
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        output_fragment gD bm bn tm tn wm wn bid wid mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    (fun mi nj ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj))
    fn mi nj {
      rewrite each output_fragment gD bm bn tm tn wm wn bid wid mi nj
        as array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj;
    };
  forevery_rw_size2 wm ((wm * tm) / tm) wn ((wn * tn) / tn)
    #(fun (mi : natlt wm) (nj : natlt wn) ->
      exists* (eFrag : chest2 et_cd tm tn).
        array2_subtile dWarp (SZ.v tm) (SZ.v tn) mi nj |-> eFrag **
        pure (eFrag %~ ematrix_subtile rWarp tm tn mi nj));
  array2_untile_approximates dWarp (SZ.v tm) (SZ.v tn) rWarp;
  rewrite each dWarp as
    warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
      (wm * tm) (wn * tn) wid;
}

#pop-options

ghost
fn gather_block
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (#m #n : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (SZ.fits ((rm m n).ulen)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (bid : natlt nblk)
  (rBlock : chest2 real bm bn)
  requires
    forall+ (wid : natlt (nthr / warp_size)).
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn))))
  ensures
    exists* (eBlock : chest2 et_cd bm bn).
      block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
      pure (eBlock %~ rBlock)
{
  forevery_map
    #(natlt (nthr / warp_size))
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn) wid |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))))
    (fun (wid : natlt (nthr / warp_size)) ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
            wid % (bn / (wn * tn))) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn)
            (wid / (bn / (wn * tn)))
            (wid % (bn / (wn * tn)))))
    fn wid {
      FStar.Math.Lemmas.euclidean_division_definition wid (bn / (wn * tn));
      rewrite each wid as
        ((wid / (bn / (wn * tn))) * (bn / (wn * tn)) +
          wid % (bn / (wn * tn)));
    };
  forevery_factor'
    (nthr / warp_size)
    (bm / (wm * tm))
    (bn / (wn * tn))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
          (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc));
  let dBlock = block_tile gD (SZ.v bm) (SZ.v bn) bid;
  rewrite each block_tile gD (SZ.v bm) (SZ.v bn) bid as dBlock;
  forevery_map_2
    #(natlt (bm / (wm * tm)))
    #(natlt (bn / (wn * tn)))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc) |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc))
    (fun wr wc ->
      exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
        array2_subtile dBlock (wm * tm) (wn * tn) wr wc |-> eWarp **
        pure (eWarp %~
          ematrix_subtile rBlock (wm * tm) (wn * tn) wr wc))
    fn wr wc {
      div_mod_of_mul_add (bn / (wn * tn)) wr wc;
      rewrite each
        warp_tile dBlock (wm * tm) (wn * tn)
          (wr * (bn / (wn * tn)) + wc)
      as array2_subtile dBlock (wm * tm) (wn * tn) wr wc;
    };
  array2_untile_approximates dBlock (wm * tm) (wn * tn) rBlock;
  rewrite each dBlock as block_tile gD (SZ.v bm) (SZ.v bn) bid;
}

ghost
fn gather_output
  (#et_cd : Type0) {| scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      output_lane_approximates gD bm bn tm tn wm wn bid tid
        (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid)) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB)
{
  forevery_map
    (fun (bid : natlt nblk) ->
      forall+ (tid : natlt nthr).
        output_lane_approximates gD bm bn tm tn wm wn bid tid
          (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
            nblk nthr rA rB rC bid tid))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          teardown_block_output comb_r bm bn bk tm tn tk wm wn
            nblk rA rB rC bid))
    fn bid {
      forevery_map
        #(natlt nthr)
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid tid
            (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
              nblk nthr rA rB rC bid tid))
        (fun (tid : natlt nthr) ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            ((tid / warp_size) * warp_size + tid % warp_size)
            (teardown_warp_output comb_r bm bn bk tm tn tk wm wn
              nblk rA rB rC bid (tid / warp_size)))
        fn tid {
          (* Bridge from the lane-indexed named form to the warp-indexed
             named form with one cheap definitional-unfold lemma, instead of
             letting the ambient SMT query try (and fail within budget) to
             discover [teardown_lane_output ... tid == teardown_warp_output
             ... (tid / warp_size)] on its own. *)
          teardown_lane_is_warp_output comb_r bm bn bk tm tn tk wm wn
            nblk nthr rA rB rC bid tid;
          FStar.Math.Lemmas.euclidean_division_definition tid warp_size;
          rewrite each
            output_lane_approximates gD bm bn tm tn wm wn bid tid
              (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
                nblk nthr rA rB rC bid tid)
          as
            output_lane_approximates gD bm bn tm tn wm wn bid
              ((tid / warp_size) * warp_size + tid % warp_size)
              (teardown_warp_output comb_r bm bn bk tm tn tk wm wn
                nblk rA rB rC bid (tid / warp_size));
        };
      forevery_factor' nthr (nthr / warp_size) warp_size
        (fun wid lane ->
          output_lane_approximates gD bm bn tm tn wm wn bid
            (wid * warp_size + lane)
            (teardown_warp_output comb_r bm bn bk tm tn tk wm wn
              nblk rA rB rC bid wid));
      forevery_map
        (fun (wid : natlt (nthr / warp_size)) ->
          forall+ (lane : natlt warp_size).
            output_lane_approximates gD bm bn tm tn wm wn bid
              (wid * warp_size + lane)
              (teardown_warp_output comb_r bm bn bk tm tn tk wm wn
                nblk rA rB rC bid wid))
        (fun (wid : natlt (nthr / warp_size)) ->
          exists* (eWarp : chest2 et_cd (wm * tm) (wn * tn)).
            warp_tile (block_tile gD (SZ.v bm) (SZ.v bn) bid)
              (wm * tm) (wn * tn) wid |-> eWarp **
            pure (eWarp %~
              teardown_warp_output comb_r bm bn bk tm tn tk wm wn
                nblk rA rB rC bid wid))
        fn wid {
          gather_warp gD bm bn bk tm tn tk wm wn
            nblk nthr bid wid
            (teardown_warp_output comb_r bm bn bk tm tn tk wm wn
              nblk rA rB rC bid wid);
        };
      gather_block gD bm bn bk tm tn tk wm wn
        nblk nthr bid
        (teardown_block_output comb_r bm bn bk tm tn tk wm wn
          nblk rA rB rC bid);
    };
  forevery_map
    #(natlt nblk)
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn) bid |-> eBlock **
        pure (eBlock %~
          teardown_block_output comb_r bm bn bk tm tn tk wm wn
            nblk rA rB rC bid))
    (fun (bid : natlt nblk) ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          ((bid / (n / bn)) * (n / bn) + bid % (n / bn)) |-> eBlock **
        pure (eBlock %~
          teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
            rA rB rC (bid / (n / bn)) (bid % (n / bn))))
    fn bid {
      (* Bridge from the block-indexed named form to the (br,bc)-indexed
         named form with one cheap definitional-unfold lemma, mirroring
         [teardown_lane_is_warp_output] above, before the ambient index is
         also rearranged so it lines up with [forevery_factor']'s shape. *)
      teardown_block_is_block_at comb_r bm bn bk tm tn tk wm wn
        nblk rA rB rC bid;
      rewrite each
        teardown_block_output comb_r bm bn bk tm tn tk wm wn nblk rA rB rC bid
      as
        teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
          rA rB rC (bid / (n / bn)) (bid % (n / bn));
      FStar.Math.Lemmas.euclidean_division_definition bid (n / bn);
      (* This rewrite reports "No rewrites performed" (a harmless warning,
         not an error): Pulse's rewrite tactic doesn't find a *syntactic*
         [bid] occurrence to replace here (the name-swap rewrite above
         already normalized the [pure] conjunct), but retaining the call
         still matters -- removing it changes how Pulse discharges the
         final index-equality obligation on the [block_tile gD ... bid]
         hypothesis against this branch's [(bid/(n/bn))*(n/bn)+bid%(n/bn)]
         postcondition index, and doing so causes verification to fail. *)
      rewrite each bid as
        ((bid / (n / bn)) * (n / bn) + bid % (n / bn));
    };
  forevery_factor' nblk (m / bm) (n / bn)
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
            rA rB rC br bc));
  forevery_map_2
    #(natlt (m / bm)) #(natlt (n / bn))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        block_tile gD (SZ.v bm) (SZ.v bn)
          (br * (n / bn) + bc) |-> eBlock **
        pure (eBlock %~
          teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
            rA rB rC br bc))
    (fun br bc ->
      exists* (eBlock : chest2 et_cd bm bn).
        array2_subtile gD (SZ.v bm) (SZ.v bn) br bc |-> eBlock **
        pure (eBlock %~
          teardown_block_output_at comb_r bm bn bk tm tn tk wm wn
            rA rB rC br bc))
    fn br bc {
      div_mod_of_mul_add (n / bn) br bc;
      rewrite each block_tile gD (SZ.v bm) (SZ.v bn)
        (br * (n / bn) + bc)
      as array2_subtile gD (SZ.v bm) (SZ.v bn) br bc;
    };
  array2_untile_approximates gD (SZ.v bm) (SZ.v bn)
    (MS.mmcomb comb_r rC rA rB);
}

#pop-options
#push-options ""

let teardown_inputs_pre
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n)) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= (forall+ (bid : natlt nblk) (tid : natlt nthr).
    kpost1_to comb_r gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn fA fB fC rA rB rC
      nblk nthr bid tid) **
  pure (SZ.fits ((rm m n).ulen))

let teardown_inputs_post
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA) (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB) (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n)) (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  : slprop
= gA |-> Frac fA eA **
  gB |-> Frac fB eB **
  gC |-> Frac fC eC **
  (forall+ (bid : natlt nblk) (tid : natlt nthr).
    output_lane_approximates gD bm bn tm tn wm wn bid tid
      (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
        nblk nthr rA rB rC bid tid)) **
  pure (SZ.fits ((rm m n).ulen))

ghost
fn gather_kernel_outputs
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (bm bn bk tm tn tk wm wn : szp {
    constraints bm bn bk tm tn tk wm wn })
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (nblk : szp { SZ.v nblk == m / bm * (n / bn) })
  (nthr : szp {
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size })
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    teardown_inputs_pre comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
  ensures
    teardown_inputs_post comb_r
      gA eA gB eB gC eC gD
      bm bn bk tm tn tk wm wn nblk nthr
      fA fB fC rA rB rC
{
  unfold teardown_inputs_pre comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  forevery_map_2
    #(natlt nblk)
    #(natlt nthr)
    (fun bid tid ->
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid)
    (fun bid tid ->
      gA |-> Frac (fA /. (nblk * nthr)) eA **
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid))
    fn bid tid {
      unfold kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid;
      (* [kpost1_to] unfolds to the spelled-out nested [ematrix_subtile]
         term (it has to stay [unfold] for its many other call sites); fold
         it back into the named abbreviation here, once, so every
         subsequent step below deals with the small, opaque
         [teardown_lane_output] application instead. *)
      rewrite each
        (ematrix_subtile
          (ematrix_subtile (MS.mmcomb comb_r rC rA rB)
            bm bn (bid / (n / bn)) (bid % (n / bn)))
          (wm * tm) (wn * tn)
          ((tid / warp_size) / (bn / (wn * tn)))
          ((tid / warp_size) % (bn / (wn * tn))))
      as
        teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid;
    };
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA)
    (fun bid tid ->
      gB |-> Frac (fB /. (nblk * nthr)) eB **
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB)
    (fun bid tid ->
      gC |-> Frac (fC /. (nblk * nthr)) eC **
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid));
  forevery_unzip_2
    #(natlt nblk)
    #(natlt nthr)
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC)
    (fun bid tid ->
      output_lane_approximates
        gD bm bn tm tn wm wn bid tid
        (teardown_lane_output comb_r bm bn bk tm tn tk wm wn
          nblk nthr rA rB rC bid tid));

  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gA |-> Frac (fA /. (nblk * nthr)) eA);
  tensor_gather_n gA (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gB |-> Frac (fB /. (nblk * nthr)) eB);
  tensor_gather_n gB (nblk * nthr);
  forevery_unfactor' (nblk * nthr) nblk nthr
    (fun _ _ -> gC |-> Frac (fC /. (nblk * nthr)) eC);
  tensor_gather_n gC (nblk * nthr);
  fold teardown_inputs_post comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
}

#pop-options

ghost
fn teardown_to
  (#et_ab #et_cd : Type0)
  {| scalar et_ab, scalar et_cd, real_like et_cd |}
  (comb_r : binop real)
  (#m #n #k : szp)
  (#lA : layout2 m k)
  (#lB : layout2 k n)
  (gA : array2 et_ab lA)
  (eA : chest2 et_ab m k)
  (gB : array2 et_ab lB)
  (eB : chest2 et_ab k n)
  (gC : array2 et_cd (rm m n))
  (eC : chest2 et_cd m n)
  (gD : array2 et_cd (rm m n))
  (#_ : squash (SZ.fits (m * n)))
  (bm bn bk tm tn tk wm wn : szp{
    constraints bm bn bk tm tn tk wm wn})
  (#_ : squash (bm /?+ m /\ bn /?+ n))
  (#_ : squash (wm * tm /?+ bm /\ wn * tn /?+ bn))
  (#_ : squash (tm /?+ (wm * tm) /\ tn /?+ (wn * tn)))
  (nblk : szp{SZ.v nblk == m / bm * (n / bn)})
  (nthr : szp{
    SZ.v nthr == bm / (wm * tm) * (bn / (wn * tn)) * warp_size})
  (fA fB fC : perm)
  (rA : chest2 real m k)
  (rB : chest2 real k n)
  (rC : chest2 real m n)
  ()
  norewrite
  requires
    (forall+ (bid : natlt nblk) (tid : natlt nthr).
      kpost1_to comb_r gA eA gB eB gC eC gD
        bm bn bk tm tn tk wm wn fA fB fC rA rB rC
        nblk nthr bid tid) **
    pure (SZ.fits ((rm m n).ulen))
  ensures
    gA |-> Frac fA eA **
    gB |-> Frac fB eB **
    gC |-> Frac fC eC **
    (exists* (eD : chest2 et_cd m n).
      gD |-> eD ** pure (eD %~ MS.mmcomb comb_r rC rA rB))
{
  fold teardown_inputs_pre comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  gather_kernel_outputs comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC ();
  unfold teardown_inputs_post comb_r
    gA eA gB eB gC eC gD
    bm bn bk tm tn tk wm wn nblk nthr
    fA fB fC rA rB rC;
  gather_output comb_r gD
    bm bn bk tm tn tk wm wn nblk nthr rA rB rC ();
}
