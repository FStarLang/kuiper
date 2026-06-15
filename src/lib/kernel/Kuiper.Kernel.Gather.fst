module Kuiper.Kernel.Gather

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module SZF = FStar.SizeT
module Cast = FStar.Int.Cast
module U64 = FStar.UInt64
module U32 = FStar.UInt32
module Array1 = Kuiper.Array1
open Kuiper.Array1
open Kuiper.Seq.Common
open Kuiper.Tensor { ctlayout }

ghost
fn explode_setup_gather2
  (#et : Type0)
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 })
  (#ls : Array1.layout lensrc)
  (#li : Array1.layout lenout)
  (#lo : Array1.layout lenout)
  (src : Array1.t et ls)
  (idx : Array1.t u64 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq et lenout))
  (#fs #fi : perm)
  ()
  norewrite
  requires
    (src |-> Frac fs ss) ** (idx |-> Frac fi si) ** (out |-> so)
  ensures
    (forall+ (k : natlt lenout).
      src |-> Frac (fs /. lenout) ss **
      idx |-> Frac (fi /. lenout) si **
      Cell out k |-> (so @! k)) **
    pure (SZ.fits (Array1.layout_size lo))
{
  Array1.share_n src lenout;
  Array1.share_n idx lenout;
  Array1.pts_to_ref out;
  Array1.explode out;
  forevery_zip
    (fun (_ : natlt lenout) -> idx |-> Frac (fi /. lenout) si)
    (fun (k : natlt lenout) -> Cell out k |-> (so @! k));
  forevery_zip
    (fun (_ : natlt lenout) -> src |-> Frac (fs /. lenout) ss)
    (fun (k : natlt lenout) ->
      idx |-> Frac (fi /. lenout) si ** Cell out k |-> (so @! k));
  ()
}

ghost
fn explode_teardown_gather2
  (#et : Type0)
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 })
  (#ls : Array1.layout lensrc)
  (#li : Array1.layout lenout)
  (#lo : Array1.layout lenout)
  (src : Array1.t et ls)
  (idx : Array1.t u64 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#fs #fi : perm)
  (_ : squash (idx_ok (SZ.v lensrc / SZ.v cols) si))
  ()
  norewrite
  requires
    (forall+ (k : natlt lenout).
      src |-> Frac (fs /. lenout) ss **
      idx |-> Frac (fi /. lenout) si **
      Cell out k |-> (access_gather cols lensrc lenout ss si k)) **
    pure (SZ.fits (Array1.layout_size lo))
  ensures
    (src |-> Frac fs ss) **
    (idx |-> Frac fi si) **
    (out |-> (lseq_gather_2d_dim0 cols lensrc lenout ss si <: lseq et lenout))
{
  forevery_unzip
    (fun (_ : natlt lenout) -> src |-> Frac (fs /. lenout) ss)
    _;
  Array1.gather_n src lenout;
  forevery_unzip
    (fun (_ : natlt lenout) -> idx |-> Frac (fi /. lenout) si)
    _;
  Array1.gather_n idx lenout;
  forevery_map
    (fun (k : natlt lenout) ->
      Cell out k |-> (access_gather cols lensrc lenout ss si k))
    (fun (k : natlt lenout) ->
      Cell out k |-> ((lseq_gather_2d_dim0 cols lensrc lenout ss si) @! k))
    fn x { () };
  Array1.implode out;
  ()
}

inline_for_extraction noextract
fn kf_gather2
  (#et : Type0) {| sized et |}
  (cols : szp)
  (#lensrc : erased nat { lensrc % SZ.v cols == 0 /\ SZ.fits lensrc })
  (#lenout : erased nat { lenout % SZ.v cols == 0 })
  (#ls : Array1.layout lensrc) {| ctlayout ls |}
  (#li : Array1.layout lenout) {| ctlayout li |}
  (#lo : Array1.layout lenout) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u64 li)
  (out : Array1.t et lo)
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq et lenout))
  (#fs #fi : perm)
  (#_ : squash (idx_ok (lensrc / SZ.v cols) si))
  (k : szlt lenout)
  ()
  requires
    gpu **
    src |-> Frac fs ss **
    idx |-> Frac fi si **
    Cell out (k <: natlt lenout) |-> (so @! k)
  ensures
    gpu **
    src |-> Frac fs ss **
    idx |-> Frac fi si **
    Cell out (k <: natlt lenout) |->
      (access_gather cols lensrc lenout ss si k)
{
  let ix : u64 = Array1.read idx k;
  FStar.Math.Lemmas.lemma_mult_le_right (SZ.v cols) (U64.v ix) (lensrc / SZ.v cols - 1);
  let row_u32 : U32.t = Cast.uint64_to_uint32 ix;
  let row_sz : SZ.t = SZ.uint32_to_sizet row_u32;
  let j_sz : SZ.t = k %^ cols;
  let src_off : SZ.t = (row_sz *^ cols) +^ j_sz;
  let x = Array1.read src src_off;
  Array1.write_cell out k x;
}

inline_for_extraction noextract
let kgather2
  (#et : Type0) {| sized et |}
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 /\ lenout <= max_blocks * max_threads })
  (#ls : Array1.layout lensrc) {| ctlayout ls |}
  (#li : Array1.layout lenout) {| ctlayout li |}
  (#lo : Array1.layout lenout) {| ctlayout lo |}
  (src : Array1.t et ls)
  (idx : Array1.t u64 li)
  (out : Array1.t et lo)
  (#_ : squash (Array1.is_global src))
  (#_ : squash (Array1.is_global idx))
  (#_ : squash (Array1.is_global out))
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq et lenout))
  (#fs #fi : perm)
  (#_ : squash (idx_ok (SZ.v lensrc / SZ.v cols) si))
  : kernel_desc
      (requires (src |-> Frac fs ss) ** (idx |-> Frac fi si) ** (out |-> so))
      (ensures  (src |-> Frac fs ss) ** (idx |-> Frac fi si) **
                (out |-> (lseq_gather_2d_dim0 cols lensrc lenout ss si <: lseq et lenout)))
= {
    nthr = lenout;
    f = kf_gather2 cols src idx out #ss #si #so;

    frame = pure (SZ.fits (Array1.layout_size lo));
    teardown = explode_teardown_gather2 cols lensrc lenout src idx out ();
    setup = explode_setup_gather2 cols lensrc lenout src idx out;
    kpre = (fun (k : natlt lenout) ->
      src |-> Frac (fs /. lenout) ss **
      idx |-> Frac (fi /. lenout) si **
      Cell out k |-> (so @! k));
    kpost = (fun (k : natlt lenout) ->
      src |-> Frac (fs /. lenout) ss **
      idx |-> Frac (fi /. lenout) si **
      Cell out k |-> (access_gather cols lensrc lenout ss si k));
    kpost_sendable = solve;
    kpre_sendable  = solve;
  } <: kernel_desc_n _ _

inline_for_extraction noextract
fn gather_2d_dim0_gpu
  (#et : Type0) {| sized et |}
  (cols : szp)
  (lensrc : szp { SZ.v lensrc % SZ.v cols == 0 })
  (lenout : szp { SZ.v lenout % SZ.v cols == 0 /\ lenout <= max_blocks * max_threads })
  (#ls : Array1.layout lensrc) {| ctlayout ls |}
  (#li : Array1.layout lenout) {| ctlayout li |}
  (#lo : Array1.layout lenout) {| ctlayout lo |}
  (src : Array1.t et ls { Array1.is_global src })
  (idx : Array1.t u64 li { Array1.is_global idx })
  (out : Array1.t et lo { Array1.is_global out })
  (#ss : erased (lseq et lensrc))
  (#si : erased (lseq u64 lenout))
  (#so : erased (lseq et lenout))
  (#fs #fi : perm)
  (_ : squash (idx_ok (SZ.v lensrc / SZ.v cols) si))
  norewrite
  preserves cpu
  preserves on gpu_loc (src |-> Frac fs ss) ** on gpu_loc (idx |-> Frac fi si)
  requires
    on gpu_loc (out |-> so)
  ensures
    on gpu_loc (out |-> (lseq_gather_2d_dim0 cols lensrc lenout ss si <: lseq et lenout))
{
  launch_sync (kgather2 cols lensrc lenout src idx out);
}
