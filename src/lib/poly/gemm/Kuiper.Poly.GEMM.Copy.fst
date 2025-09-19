module Kuiper.Poly.GEMM.Copy

#lang-pulse

module SZ = FStar.SizeT

inline_for_extraction noextract
fn cp_matrix
  (#et : Type0) {| scalar et |}
  (rows cols: sz)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#f : perm)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (nthr : sz)
  (tid : szlt nthr)
  preserves
    gpu **
    pure (SZ.fits (rows * cols + nthr - 1)) **
    src |-> Frac f esrc **
    live_tile_stride_cells dst nthr tid
{
  let mlen = rows *^ cols;

  let mut i : sz = tid;
  while (SZ.(!i <^ mlen))
    invariant
      exists* (vi : sz).
        pure (vi >= tid) **
        pure (vi % nthr == tid) **
        i |-> vi **
        live_tile_stride_cells dst nthr tid **
        pure (vi < mlen + nthr)
  {
    let v = gpu_matrix_read src (!i /^ cols) (!i %^ cols);

    let ite : erased (natlt (div_ceil (rows*cols) nthr)) = (!i - tid) / nthr;

    unfold live_tile_stride_cells dst nthr tid;
    forevery_extract (reveal ite) _;

    rewrite each
      (((tid + ite * nthr) / cols < rows) &&
       ((tid + ite * nthr) % cols < cols))
    as true;

    assert (rewrites_to (tid + ite * nthr) !i);

    // Manual unfold and fold necessary.
    //  If unfold was automatic the vi in live_cell would not be rewritten above
    //  because it is under an exists*.
    unfold live_cell dst (!i / cols) (!i % cols);
    gpu_matrix_write_cell dst (!i /^ cols) (!i %^ cols) v;
    fold live_cell dst (!i / cols) (!i % cols);

    rewrite each SZ.v !i as (tid + ite * nthr);

    Pulse.Lib.Trade.elim_trade _ _;
    fold live_tile_stride_cells dst nthr tid;

    Math.Lemmas.modulo_addition_lemma !i nthr 1;
    i := !i +^ nthr;
  };

  ()
}

inline_for_extraction noextract
fn cp_matrix_one_cell_per_thread
  (#et : Type0) {| scalar et |}
  (#rows #cols : szp)
  (#lsrc #ldst : mlayout rows cols)
  {| clayout lsrc, clayout ldst |}
  (src : gpu_matrix et lsrc)
  (#esrc : ematrix et rows cols)
  (dst : gpu_matrix et ldst)
  (#f : perm)
  (#nthr : erased nat{nthr == rows * cols})
  (tid : szlt nthr)
  preserves
    gpu **
    src |-> Frac f esrc
  requires
    live_cell dst (tid/cols) (tid%cols)
  ensures
    gpu_matrix_pts_to_cell dst (tid/cols) (tid%cols) (macc esrc (tid/cols) (tid%cols))
{
  unfold live_cell;
  let v = gpu_matrix_read src (tid /^ cols) (tid %^ cols);
  gpu_matrix_write_cell dst (tid /^ cols) (tid %^ cols) v;
}
