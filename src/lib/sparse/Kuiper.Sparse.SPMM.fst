module Kuiper.Sparse.SPMM

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.SPMM.LoadSparse
open Kuiper.Sparse.SPMM.LoadDense
open Kuiper.Sparse.SPMM.StoreDense
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier
open Kuiper.EMatrix
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Tensor.Layout { ctlayout }
open Kuiper.Tensor.Layout.Alg
open Kuiper.Array2.Strided
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute


#push-options "--split_queries always"

unfold
let block_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  thread_live_tile_vec
    gC
    (brow p bid |~> row_perm)
    (tcol p bid tid)
    (p.blockItemsX / p.blockWidth)
    p.blockWidth
  // forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
  //   when__
  //     (bcol p bid + k * p.blockWidth + tid < p.cols)
  //     (fun _ -> matrix_live_cell
  //       gC (brow p bid |~> row_perm) (bcol p bid + k * p.blockWidth + tid))


unfold
let block_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  thread_pts_to_tile_vec
    gC
    (brow p bid |~> row_perm)
    (tcol p bid tid)
    (MS.matmul eA eB)
    (p.blockItemsX / p.blockWidth)
    p.blockWidth

unfold
let kpre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // TODO no se si puedo hacer eso
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let sh_e : larray et p.blockItemsK = fst sh in
  let sh_i : larray sz p.blockItemsK = fst (snd sh) in
  block_pre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  thread_live_chunks sh_e p.blockWidth tid **
  thread_live_chunks sh_i p.blockWidth tid **
  pure (
    aligned 16 sh_e /\
    aligned 16 sh_i
  )

unfold
let kpost
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (eB : chest2 et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  let sh_e : larray et p.blockItemsK = fst sh in
  let sh_i : larray sz p.blockItemsK = fst (snd sh) in
  block_post
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  live sh_e #(1.0R /. p.blockWidth) **
  live sh_i #(1.0R /. p.blockWidth)

// Lemas para prueba de setup
// TODO mover de acá
ghost
fn forevery_ext_3
  (#a #b #c : Type0)
  (f g : a -> b -> c -> slprop)
  requires
    pure (forall x y z. f x y z == g x y z)
  requires
    forall+ (x:a) (y:b) (z:c). f x y z
  ensures
    forall+ (x:a) (y:b) (z:c). g x y z
{
  forevery_map_2
    (fun x y -> forall+ z. f x y z)
    (fun x y -> forall+ z. g x y z)
    fn x y {
      forevery_ext (fun z -> f x y z) (fun z -> g x y z)
    };
}

ghost
fn forevery_assoc_2
  (#a:Type0)
  (#b:Type0)
  (p1 p2 p3 : a -> b -> slprop)
  requires
    forall+ (x:a) (y:b). (p1 x y ** p2 x y) ** p3 x y
  ensures
    forall+ (x:a) (y:b). p1 x y ** p2 x y ** p3 x y
{
  forevery_map_2
    (fun x y -> (p1 x y ** p2 x y) ** p3 x y)
    (fun x y -> p1 x y ** p2 x y ** p3 x y)
    fn x y {};
}

let lem_div1 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) / d == n)
= Math.Lemmas.lemma_div_plus r n d

let lem_div2 (n : nat) (d : pos) (r : natlt d)
: Lemma (requires true) (ensures (n * d + r) % d == r)
= Math.Lemmas.lemma_mod_plus r n d
//

ghost
fn setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  ()
  norewrite
  requires
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    row_indices |-> Frac fri (ordering row_perm) **
    gB |-> Frac fB eB **
    live gC
  ensures
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_pre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        bid tid
    ) **
      emp
{
  with eC. assert gC |-> eC;
  tensor_explode2 gC;
  forevery_unflatten' _;

  forevery_map
    (fun r ->
      forall+ (c : natlt p.cols).
        tensor_pts_to_cell gC (idx2 (r) (c)) (acc2 eC r c)
    )
    (fun r ->
      forall+
        (b : natlt (divup p.cols p.blockItemsX)) (tid : natlt p.blockWidth)
        (k : natlt(p.blockItemsX /^ p.blockWidth)).
          when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
            (fun _ ->
              matrix_live_cell gC r
              (b * p.blockItemsX + k * p.blockWidth + tid)
            )
    )
    fn (r : natlt p.rows) {
      forevery_factor_ p.cols p.blockItemsX _;

      forevery_map
        #(natlt (divup p.cols p.blockItemsX))
        (fun b ->
          forall+ (ix : natlt p.blockItemsX { b * p.blockItemsX + ix < p.cols }).
            tensor_pts_to_cell gC (idx2 (r) (b * p.blockItemsX + ix))
              (acc2 eC r (b * p.blockItemsX + ix))
        )
        (fun b ->
          forall+
            (tid : natlt p.blockWidth)
            (k : natlt(p.blockItemsX /^ p.blockWidth)).
            when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
              (fun _ ->
                matrix_live_cell gC r
                (b * p.blockItemsX + k * p.blockWidth + tid)
              )
        )
        fn b {
          forevery_map
            #(ix : natlt p.blockItemsX { b * p.blockItemsX + ix  < p.cols })
            (fun ix ->
              tensor_pts_to_cell gC (idx2 (r) (b * p.blockItemsX + ix))
                (acc2 eC r (b * p.blockItemsX + ix))
            )
            (fun ix ->
              matrix_live_cell gC r (b * p.blockItemsX + ix)
            )
            fn ix {
              fold matrix_live_cell gC r (b * p.blockItemsX + ix);
            };
          forevery_unrefine_pred' #(natlt p.blockItemsX)
            (fun ix -> b * p.blockItemsX + ix  < p.cols)
            (fun ix _ -> matrix_live_cell gC r (b * p.blockItemsX + ix));

          prod_divides p.blockWidth (chunk et) p.blockItemsX;
          forevery_rw_size p.blockItemsX
            ((p.blockItemsX /^ p.blockWidth) * p.blockWidth);
          forevery_factor
            ((p.blockItemsX /^ p.blockWidth) * p.blockWidth)
            (p.blockItemsX /^ p.blockWidth) p.blockWidth
            (fun ix ->
              when__ (b * p.blockItemsX + ix < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + ix))
            );

          forevery_commute
            #(natlt (p.blockItemsX /^ p.blockWidth)) #(natlt p.blockWidth)
            (fun k tid ->
              when__ (b * p.blockItemsX + (k * p.blockWidth + tid) < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + (k * p.blockWidth + tid)))
            );
          forevery_ext_2
            #(natlt p.blockWidth)
            #(natlt (p.blockItemsX /^ p.blockWidth))
            (fun tid k ->
              when__ (b * p.blockItemsX + (k * p.blockWidth + tid) < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + (k * p.blockWidth + tid)))
            )
            (fun tid k ->
              when__ (b * p.blockItemsX + k * p.blockWidth + tid < p.cols)
                (fun _ -> matrix_live_cell gC r (b * p.blockItemsX + k * p.blockWidth + tid))
            );
        };
    };
  forevery_iso (Kuiper.Bijection.bij_sym row_perm) _;

  forevery_unfactor' (nblocks p) _ _ _;
  forevery_ext_3
    #(natlt (nblocks p))
    #(natlt p.blockWidth)
    #(natlt (p.blockItemsX /^ p.blockWidth))
    _
    (fun bid tid k ->
      when__ (bcol p bid + k * p.blockWidth + tid < p.cols)
        (fun _ ->
          matrix_live_cell gC
            (brow p bid |~> row_perm)
            (bcol p bid + k * p.blockWidth + tid)
        )
    );

  Kuiper.Array.Extra.array_share row_indices (allthreads p);
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  tensor_share_n gB (allthreads p) #fB;
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  forevery_zip3_2
    (fun _ _ ->
      row_indices |-> Frac (fri /. (allthreads p)) (ordering row_perm)
    )
    (fun _ _ -> gB |-> Frac (fB /. (allthreads p)) eB)
    _;

  smatrix_share_n' gA #fA elems col_ind row_off eA (allthreads p);
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  forevery_zip_2
    (fun _ _ ->
      smatrix_pts_to' gA #(fA /. (allthreads p)) elems col_ind row_off eA
    )
    _;

  // TODO: scalar->vectorized ownership conversion for gC tile
  // (fold `forall+ (z). matrix_live_cell` into `thread_live_tile_vec`);
  // needs a new bundling lemma. Admitted for now.
  admit();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_pre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        bid tid
    )
  ensures
    (forall+ (tid : natlt p.blockWidth).
      kpre
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        sh
        bid tid
    ) **
      emp
{
  unfold_c_shmems sh (`%shmems_desc);

  let sh_e : larray et p.blockItemsK = fst sh;
  assert rewrites_to sh_e (fst sh);
  let sh_i : larray sz p.blockItemsK = fst (snd sh);
  assert rewrites_to sh_i (fst (snd sh));

  thread_share_chunks sh_e p.blockWidth;
  thread_share_chunks sh_i p.blockWidth;

  // TODO de donde sacamos esto?
  assume pure ( aligned 16 sh_e /\ aligned 16 sh_i);

  forevery_intro_pure #(natlt p.blockWidth)
    (fun _ ->
        aligned 16 sh_e /\
        aligned 16 sh_i
    );
  forevery_zip3 #(natlt p.blockWidth)
    (fun tid -> thread_live_chunks sh_e p.blockWidth tid)
    (fun tid -> thread_live_chunks sh_i p.blockWidth tid)
    (fun tid ->
      pure (
        aligned 16 sh_e /\
        aligned 16 sh_i
      )
    );

  forevery_zip #(natlt p.blockWidth)
    _
    (fun tid ->
      thread_live_chunks sh_e p.blockWidth tid **
      thread_live_chunks sh_i p.blockWidth tid **
      pure (
        aligned 16 sh_e /\
        aligned 16 sh_i
      )
    );

}

ghost
fn block_teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p ))
  (bid : natlt (nblocks p))
  ()
  norewrite
  requires
    (forall+ (tid : natlt p.blockWidth).
      kpost
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        sh bid tid
    ) **
    emp
  ensures
    live_c_shmems sh **
    (forall+ (tid : natlt p.blockWidth).
      block_post
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        bid tid
    )
{
  let sh_e : larray et p.blockItemsK = fst sh;
  assert rewrites_to sh_e (fst sh);
  let sh_i : larray sz p.blockItemsK = fst (snd sh);
  assert rewrites_to sh_i (fst (snd sh));

  forevery_unzip3 _ _ _;
  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      pts_to sh_e #(1.0R /. p.blockWidth) x
    );
  with elems_tile.
    assert pts_to sh_e #(1.0R /. p.blockWidth) elems_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to sh_e #(1.0R /. p.blockWidth) elems_tile)
    (fun tid -> exists* x.
      pts_to sh_e #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to sh_e #(1.0R /. p.blockWidth) elems_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq sh_e;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to sh_e #(1.0R /. p.blockWidth) elems_tile
    );

  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      pts_to sh_i #(1.0R /. p.blockWidth) x
    );
  with col_ind_tile.
    assert pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile)
    (fun tid -> exists* x.
      pts_to sh_i #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq sh_i;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to sh_i #(1.0R /. p.blockWidth) col_ind_tile
    );

  Kuiper.Array.Extra.array_gather sh_e       p.blockWidth;
  Kuiper.Array.Extra.array_gather sh_i p.blockWidth;

  fold_c_shmems sh (`%shmems_desc);

  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  (gC : array2 et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  ()
  norewrite
  requires
    (forall+ (bid : natlt (nblocks p)) (tid : natlt p.blockWidth).
      block_post
        p row_perm
        gA row_indices gB gC
        elems col_ind row_off
        eA eB
        fA fri fB
        bid tid
    ) **
    emp
  ensures
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    row_indices |-> Frac fri (ordering row_perm) **
    gB |-> Frac fB eB **
    gC |-> MS.matmul eA eB
{
  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ ->
      smatrix_pts_to' gA #(fA /. allthreads p) elems col_ind row_off eA
    );
  smatrix_gather_n' gA #fA elems col_ind row_off eA (allthreads p);

  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ ->
      row_indices |-> Frac (fri /. allthreads p) (ordering row_perm)
    );
  Kuiper.Array.Extra.array_gather row_indices (allthreads p);
  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ -> gB |-> Frac (fB /. allthreads p) eB);
  tensor_gather_n gB (allthreads p) #fB;


  // TODO: vectorized->scalar ownership conversion for gC tile
  // (unfold `forall+ (x)(y). thread_pts_to_tile_vec` into the scalar
  // `forall+ (x)(tid)(k). tensor_pts_to_cell` form the reshape below
  // consumes); needs the inverse of setup's bundling lemma. Admitted for now.
  admit();

  forevery_map #(natlt (nblocks p))
    (fun bid ->
      forall+
        (tid: natlt p.blockWidth)
        (k: natlt (p.blockItemsX /^ p.blockWidth)).
        when__ (bcol p bid + k * v p.blockWidth + tid < v p.cols)
          (fun _ ->
            tensor_pts_to_cell gC
              (idx2 (brow p bid |~> row_perm) (bcol p bid + k * v p.blockWidth + tid))
              (MS.matmul_single eA
                  eB
                  (brow p bid |~> row_perm)
                  (bcol p bid + k * v p.blockWidth + tid)))
    )
    (fun bid ->
      forall+ (ix : natlt p.blockItemsX).
        when__
          (bcol p bid + ix < p.cols)
          (fun _ ->
            tensor_pts_to_cell gC
              (idx2 (brow p bid |~> row_perm) (bcol p bid + ix))
              (MS.matmul_single eA eB
                (brow p bid |~> row_perm)
                (bcol p bid + ix)
              )
          )
    )
    fn bid {
      forevery_ext_2
        #(natlt p.blockWidth)
        #(natlt (p.blockItemsX /^ p.blockWidth))
        (fun tid k ->
          when__ (bcol p bid + k * v p.blockWidth + tid < v p.cols)
            (fun _ ->
              tensor_pts_to_cell gC
                (idx2 (brow p bid |~> row_perm) (bcol p bid + k * v p.blockWidth + tid))
                (MS.matmul_single eA
                    eB
                    (brow p bid |~> row_perm)
                    (bcol p bid + k * v p.blockWidth + tid)))
        )
        (fun tid k ->
          when__ (bcol p bid + (k * v p.blockWidth + tid) < v p.cols)
            (fun _ ->
              tensor_pts_to_cell gC
                (idx2 (brow p bid |~> row_perm) (bcol p bid + (k * v p.blockWidth + tid)))
                (MS.matmul_single eA
                    eB
                    (brow p bid |~> row_perm)
                    (bcol p bid + (k * v p.blockWidth + tid)))
            )
        );
      forevery_commute _;
      prod_divides p.blockWidth (chunk et) p.blockItemsX;
      forevery_unfactor
        ((p.blockItemsX /^ p.blockWidth) * p.blockWidth)
        (p.blockItemsX /^ p.blockWidth) p.blockWidth
        (fun ix ->
          when__
            (bcol p bid + ix < p.cols)
            (fun _ ->
              tensor_pts_to_cell gC
                (idx2 (brow p bid |~> row_perm) (bcol p bid + ix))
                (MS.matmul_single eA eB
                  (brow p bid |~> row_perm)
                  (bcol p bid + ix)
                )
            )
        );
      forevery_rw_size
        ((p.blockItemsX /^ p.blockWidth) * p.blockWidth) p.blockItemsX;
    };
  forevery_factor (nblocks p) p.rows (divup p.cols p.blockItemsX) _;
  forevery_map #(natlt p.rows)
    (fun r ->
      forall+
        (b: natlt (divup p.cols p.blockItemsX))
        (ix : natlt p.blockItemsX).
        when__
          (bcol p (r * divup p.cols p.blockItemsX + b) + ix < p.cols)
          (fun _ ->
            tensor_pts_to_cell gC
              (idx2 (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm) (bcol p (r * divup p.cols p.blockItemsX + b) + ix))
              (MS.matmul_single eA eB
                (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm)
                (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
              )
          )
    )
    (fun r ->
      forall+ (c : natlt p.cols).
        tensor_pts_to_cell gC (idx2 (r |~> row_perm) (c))
          (MS.matmul_single eA eB (r |~> row_perm) c)
    )
    fn r {
      forevery_map_2 #(natlt (divup p.cols p.blockItemsX)) #(natlt p.blockItemsX)
        (fun b ix ->
          when__
            (bcol p (r * divup p.cols p.blockItemsX + b) + ix < p.cols)
            (fun _ ->
              tensor_pts_to_cell gC
                (idx2 (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm) (bcol p (r * divup p.cols p.blockItemsX + b) + ix))
                (MS.matmul_single eA eB
                  (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm)
                  (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
                )
            )
        )
        (fun b ix ->
          when__
            (b * p.blockItemsX + ix < p.cols)
            (fun _ ->
              tensor_pts_to_cell gC
                (idx2 (r |~> row_perm) (b * p.blockItemsX + ix))
                (MS.matmul_single eA eB (r |~> row_perm) (b * p.blockItemsX + ix))
            )
        )
        fn b ix {
          lem_div1 r (divup p.cols p.blockItemsX) b;
          rewrite each (brow p (r * divup p.cols p.blockItemsX + b))
            as r;
          lem_div2 r (divup p.cols p.blockItemsX) b;
          rewrite each (bcol p (r * divup p.cols p.blockItemsX + b))
            as (b * p.blockItemsX);
        };
      forevery_map #(natlt (divup p.cols p.blockItemsX))
        (fun b ->
          forall+ (ix : natlt p.blockItemsX).
            when__
              (b * p.blockItemsX + ix < p.cols)
              (fun _ ->
                tensor_pts_to_cell gC
                  (idx2 (r |~> row_perm) (b * p.blockItemsX + ix))
                  (MS.matmul_single eA eB
                    (r |~> row_perm) (b * p.blockItemsX + ix)
                  )
              )
        )
        (fun b ->
          forall+ (ix : natlt p.blockItemsX {b * p.blockItemsX + ix < p.cols}).
            tensor_pts_to_cell gC
              (idx2 (r |~> row_perm) (b * p.blockItemsX + ix))
              (MS.matmul_single
                eA eB (r |~> row_perm) (b * p.blockItemsX + ix)
              )
        )
        fn b {
          forevery_refine_pred' #(natlt p.blockItemsX)
            (fun ix -> b * p.blockItemsX + ix < p.cols) _;
        };

      forevery_unfactor_
        p.cols
        (p.blockItemsX)
        (fun c ->
          tensor_pts_to_cell gC (idx2 (r |~> row_perm) (c))
            (MS.matmul_single eA eB (r |~> row_perm) c)
        );
    };

  forevery_iso row_perm (fun r ->
    forall+ (c : natlt p.cols).
      tensor_pts_to_cell gC (idx2 (r |~> row_perm) (c))
        (MS.matmul_single eA eB (r |~> row_perm) c)
  );
  forevery_ext_2
    (fun r c ->
      tensor_pts_to_cell gC
        (idx2 (row_perm.gg r |~> row_perm) (c))
        (MS.matmul_single eA eB (row_perm.gg r |~> row_perm) c)
    )
    (fun r c -> tensor_pts_to_cell gC (idx2 (r) (c)) (acc2 (MS.matmul eA eB) r c));
  forevery_flatten _;
  forevery_ext
    (fun (rc : natlt p.rows & natlt p.cols) -> tensor_pts_to_cell gC (idx2 (rc._1) (rc._2)) (acc2 (MS.matmul eA eB) rc._1 rc._2))
    (fun (rc : natlt p.rows & natlt p.cols) -> tensor_pts_to_cell gC (idx2 (rc._1) (rc._2)) (acc2 (MS.matmul eA eB) rc._1 rc._2));
  tensor_implode2 gC;

  ();
}


inline_for_extraction noextract
fn sparse_load_main
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  // matriz sparse gA
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#eA : chest2 et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p (reveal col_ind) (reveal row_off)))
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{SZ.v ri == round2 (max (chunk et) (chunk sz)) (reveal row_off @! (brow p bid |~> row_perm))})
  (re : sz{re == reveal row_off @! (brow p bid |~> row_perm) + 1})
  (idx : sz { idx > 0 })
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK + p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA (reveal elems) col_ind row_off eA **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s)
  ensures
    B.barrier_state ((idx + 1) * 2) **
    elems_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice elems
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK)) **
    col_ind_tile |-> Frac (1.0R /. p.blockWidth)
      (Seq.slice col_ind
        (ri + idx * p.blockItemsK) (ri + idx * p.blockItemsK + p.blockItemsK))
{
  let off : (o:sz{SZ.v o == SZ.v ri + SZ.v idx * SZ.v p.blockItemsK})
    = ri +^ idx *^ p.blockItemsK;

  barrier_in_fold_main_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid;

  barrier_out_unfold_main_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  let ri_ = hide (row_off @! (brow p bid |~> row_perm));

  offset_aligned_lemma_et p gA.elems (SZ.v ri_) (SZ.v idx);
  assert pure (aligned' 16 gA.elems off);
  load_array_vec elems_tile gA.elems off p.blockWidth tid;

  offset_aligned_lemma_sz p gA.col_ind (SZ.v ri_) (SZ.v idx);
  assert pure (aligned' 16 gA.col_ind off);
  load_array_vec col_ind_tile gA.col_ind off p.blockWidth tid;

  barrier_in_fold_main_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2 + 1) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid;

  barrier_out_unfold_main_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri re idx tid;

  ();

}

inline_for_extraction noextract
fn sparse_load_residue
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  // matriz sparse gA
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#eA : chest2 et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (idx residue : sz { residue_pred p.blockItemsK ri ri' re idx residue })
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (idx * 2) tid
  ensures
    B.barrier_state ((idx + 1) * 2) **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice col_ind (re - residue) re) **
    slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK
{
  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2) tid;

  barrier_out_unfold_residue_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  load2_array
    elems_tile col_ind_tile
    residue
    gA.elems gA.col_ind (re -^ residue)
    p.blockWidth tid;

  barrier_in_fold_residue_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout (idx * 2 + 1) tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid (idx * 2 + 1) tid;

  barrier_out_unfold_residue_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid idx residue;
}

noextract
let barrier_count
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : nat)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (bid : natlt (nblocks p))
: Ghost nat
  (requires
    valid_smatrix p.rows p.shared (cast_pos col_ind) (cast_pos row_off)
  )
  (ensures fun _ -> true)
=
  let ri = row_off @! (brow p bid |~> row_perm) in
  let re = row_off @! (brow p bid |~> row_perm) + 1 in
  let ri' = round2 (max (chunk et) (chunk sz)) ri in
  ((re - ri') / p.blockItemsK + 1) * 2

// Defs de algignment
// TODO mover a otro lado
noextract inline_for_extraction
let align_sz (size : szp) (x : sz) : sz =
  (x /^ size) *^ size

noextract inline_for_extraction
let align_offset
  (et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (off : sz)
: Pure sz
  (requires true)
  (ensures fun r -> SZ.v r == round2 (max (chunk et) (chunk sz)) off)
  (* chunk sz y chunk et son potencias de dos, así que alinear
     a la mayor es alinear a ambas. *)
= if chunk sz <^ chunk et
    then align_sz (chunk et) off
    else align_sz (chunk sz) off

let offset_aligned_lemma_et'
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray et n { aligned 16 x })
  (i : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk et /? i';
  ()

let offset_aligned_lemma_sz'
  (et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : larray sz n { aligned 16 x })
  (i : nat)
: Lemma
  (requires true)
  (ensures aligned' 16 x
    (round2 (max (chunk et) (chunk sz)) i)
  )
=
  let i' = round2 (max (chunk et) (chunk sz)) i in
  round2_chunk_lemma et sz i;
  assert chunk sz /? i';
  ()
//

open Kuiper.Seq.Common { (@+) }
open Kuiper.Sparse.SPMM.Mask

// Defs de secuencias
// TODO hacen falta? mover a otro lado o quitar
let lslice
  (#a : Type0)
  (#n : nat)
  (s : lseq a n)
  (i k : nat { i + k <= n })
: lseq a k
= Seq.slice s i (i + k)

let lslice'
  (#a : Type0)
  (#n : nat)
  (s : lseq a n)
  (i j : natle n { i <= j })
: lseq a (j - i)
= Seq.slice s i j

let seq_mask
  (#et : Type0) {| scalar et |}
  (k : nat)
  (#n : nat)
  (s : lseq et n)
: lseq et (k + n)
= Seq.create k zero @+ s
//


let lem0
  (a : nat) (c d : pos)
: Lemma (requires (c * d) /? a) (ensures c /? (a / d))
=
  let open FStar.Math.Lemmas in
  calc (==) {
    a;
    == { cancel_mul_div a (c * d) }
    a / (c * d) * (c * d);
    == { paren_mul_right (a / (c * d)) c d; paren_mul_right (a / (c * d)) c d }
    (a / (c * d) * c) * d;
  };
  calc (==) {
    a / d;
    == {}
    ((a / (c * d) * c) * d) / d;
    == { cancel_mul_div (a / (c * d) * c) d }
    a / (c * d) * c;
  };
  intro_divides c (a / (c * d)) (a / d)

// TODO mover y renombrar
let lem
  (a : nat) (c d : pos)
: Lemma (requires (c * d) /? a) (ensures c /? (a / d) /\ d /? (a / c))
= lem0 a c d; lem0 a d c

let threadItemsX
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et |}
  (p : parameters et)
: Ghost nat (requires true) (ensures fun k -> chunk et /? k)
=
  lem p.blockItemsX p.blockWidth (chunk et);
  p.blockItemsX / p.blockWidth

// TODO refactorizar? Dividir en dos partes?
#push-options "--z3rlimit 45"
inline_for_extraction noextract
fn kf_head
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#lB : layout2 p.shared p.cols) {| ctlB : ctlayout lB, srmB : strided_row_major lB |}
  (gB : array2 et lB)
  (#tx : sz { SZ.v tx == threadItemsX p })
  (#ldt : layout2 p.blockItemsK tx) {| ctldt : ctlayout ldt, srmdt : strided_row_major ldt |}
  (dense_tile : array2 et ldt)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : chest2 et p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  // TODO creo tid : natlt p.blockWidth esta bien
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (#_ : squash (ri' + p.blockItemsK <= re))
  norewrite
  preserves gpu
  preserves smatrix_pts_to' gA #fA elems col_ind row_off eA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srmB)
  preserves live dense_tile
  requires  pure (aligned 16 (core dense_tile))
  requires  pure (aligned_strided_row_major (chunk et) srmdt)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires  thread_live_chunks elems_tile p.blockWidth tid
  requires  thread_live_chunks col_ind_tile p.blockWidth tid
  requires  out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero
  requires  B.barrier_state 0
  ensures   exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures   exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures   B.barrier_state 2
  ensures   exists* vout.
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_vmprod_prop #_ #_ #_ #hvc
        #(p.blockItemsK)
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) d.zero)
        (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
        (lslice (cast_pos col_ind) ri' p.blockItemsK)
        eB n_idx p.blockWidth vout
    )
{
  offset_aligned_lemma_et' p gA.elems ri;
  assert pure (aligned' 16 gA.elems ri');
  load_array_vec elems_tile gA.elems ri' p.blockWidth tid;

  offset_aligned_lemma_sz' et p gA.col_ind ri;
  assert pure (aligned' 16 gA.col_ind ri');
  load_array_vec col_ind_tile gA.col_ind ri' p.blockWidth tid;

  barrier_in_fold_mask_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin 0 tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout 0 tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 0 tid;

  barrier_out_unfold_mask_pre p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  mask_array elems_tile (ri -^ ri') zero p.blockWidth tid;

  barrier_in_fold_mask_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;

  rewrite barrier_in p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid
  as (barrier_contract p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid).rin 1 tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm elems col_ind
    row_off elems_tile col_ind_tile bid).rout 1 tid
  as barrier_out p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid 1 tid;

  barrier_out_unfold_mask_post p row_perm elems col_ind row_off
    elems_tile col_ind_tile bid ri ri' re tid;
  
  // TODO mejores nombres
  let elems'   : erased (lseq et (re - ri')) = Seq.create (ri - ri') zero @+ Seq.slice elems ri re;
  let col_ind' : erased (lseq nat (re - ri')) = Seq.slice (cast_pos col_ind) ri' re;

  // TODO usar modulo qualified? diria que si
  load_dense_matrix
    dense_tile
    gB
    n_idx p.blockWidth col_ind_tile;

  with edt. assert dense_tile |-> edt;

  assert pure (
    Seq.equal
      (Seq.slice col_ind' 0 p.blockItemsK)
      (cast_pos #p.blockItemsK (Seq.slice col_ind ri' (ri' + p.blockItemsK)))
  );

  assert pure (
    chest2_tile_prop
      eB
      (Seq.slice col_ind' 0 p.blockItemsK)
      n_idx p.blockWidth edt
  );


  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  // TODO
  assume pure (chunk et /? (p.blockItemsK / p.blockWidth));

  Compute.tile_vmprod_prop_lemma0
    #_ #_ #_ #_
    #(p.blockItemsX / p.blockWidth)
    out0
    eB n_idx p.blockWidth;


  Seq.lemma_empty (Seq.slice elems' 0 0);
  Seq.lemma_empty (Seq.slice col_ind' 0 0);

  assert pure (
    Compute.tile_vmprod_prop
      out0
      (Seq.slice elems' 0 0 <: lseq et 0)
      (Seq.slice col_ind' 0 0)
      eB
      n_idx p.blockWidth
      out0
  );

  assert pure (
    Seq.equal
      (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
      (Seq.slice elems' 0 p.blockItemsK)
  );

  Compute.tile_vmprod
    out
    (Seq.create (p.blockItemsX / p.blockWidth) d.zero <: lseq et (p.blockItemsX / p.blockWidth))
    elems_tile #(1.0R /. p.blockWidth)
    #(re - ri')
    elems' col_ind'
    0
    #ldt #ctldt
    dense_tile #edt #1.0R
    eB
    n_idx p.blockWidth #();

  ();
}

inline_for_extraction noextract
fn kf_main
  (#et : Type0) {| d : scalar et, sized et, hvc : has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#lB : layout2 p.shared p.cols) {| ctlB : ctlayout lB, srmB : strided_row_major lB |}
  (gB : array2 et lB)
  (#tx : sz { SZ.v tx == threadItemsX p })
  (#ldt : layout2 p.blockItemsK tx) {| ctldt : ctlayout ldt, srmdt : strided_row_major ldt |}
  (dense_tile : array2 et ldt)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : chest2 et p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  // TODO creo tid : natlt p.blockWidth esta bien
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (nnz idx : ref sz) // realmente no necesito una referencia a idx
  norewrite
  preserves gpu
  preserves smatrix_pts_to' gA #fA elems col_ind row_off eA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srmB)
  preserves live dense_tile
  requires  pure (aligned 16 (core dense_tile))
  requires  pure (aligned_strided_row_major (chunk et) srmdt)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires  thread_live_chunks elems_tile p.blockWidth tid
  requires  thread_live_chunks col_ind_tile p.blockWidth tid
  requires  B.barrier_state 0
  requires  out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero
  requires  live idx
  requires  nnz |-> (re -^ ri')
  // MAYBE usar round2 p.blockItemsK (re - ri)
  ensures exists* (v_idx :sz) (residue : szle (re - ri)) vout.
    idx |-> v_idx **
    nnz |-> (residue <: sz) **
    pure (residue_pred p.blockItemsK ri ri' re v_idx residue) **
    B.barrier_state (v_idx * 2) **
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (v_idx * 2) tid **
    out |-> (vout <: seq et) **
    pure (
      Compute.tile_vmprod_prop #_ #_ #_ #_
        #(re - residue - ri)
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) d.zero)
        (Seq.slice elems ri (re - residue))
        (lslice' (cast_pos col_ind) ri (re - residue))
        eB n_idx p.blockWidth vout
    )

{
  let out0 : erased (lseq et (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  if (!nnz >=^ p.blockItemsK)
  {
    let row_elems_ : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
    let row_elems : erased (lseq et (re - ri')) = seq_mask (ri - ri') #(re - ri) row_elems_;

    let row_ind : erased (lseq nat (re - ri')) = hide (Seq.slice (cast_pos col_ind) ri' re);

    kf_head
      p row_perm
      gA gB dense_tile
      eA
      out
      elems_tile col_ind_tile
      bid
      ri ri' re
      tid n_idx;

    idx := 1sz;
    nnz := !nnz -^ p.blockItemsK;

    assert pure (
      Seq.equal
        (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
        (Seq.slice row_elems 0 (!idx * p.blockItemsK))
    );
    assert pure (
      Seq.equal
        (lslice (cast_pos col_ind) ri' p.blockItemsK)
        (Seq.slice row_ind 0 (!idx * p.blockItemsK))
    );

    with v_out. assert out |-> v_out;
    assert pure (
      Compute.tile_vmprod_prop #_ #_ #_ #_
        #(!idx * p.blockItemsK) #(threadItemsX p)
        out0
        (Seq.slice row_elems 0 (!idx * p.blockItemsK))
        (Seq.slice row_ind 0 (!idx * p.blockItemsK))
        eB n_idx p.blockWidth v_out
    );

    while (!nnz >=^ p.blockItemsK)
      invariant
        (exists* v_out.
          out |-> v_out **
          live idx **
          live nnz **
          B.barrier_state (!idx * 2) **
          (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
          (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
          pure (
            !idx > 0 /\ !idx <= (re - ri') / p.blockItemsK /\
            SZ.v !nnz == re - ri' - !idx * p.blockItemsK /\
            len v_out == threadItemsX p /\
            Compute.tile_vmprod_prop #_ #_ #_ #_
              #(!idx * p.blockItemsK) #(threadItemsX p)
              out0
              (lslice row_elems 0 (!idx * p.blockItemsK))
              (Seq.slice row_ind 0 (!idx * p.blockItemsK))
              eB n_idx p.blockWidth v_out
          )
        )
      decreases SZ.v !nnz
    {
      sparse_load_main p row_perm gA #_ #_ #_ #_ #eA
        elems_tile col_ind_tile bid ri' re !idx tid;

      Pulse.Lib.Array.pts_to_len out;
      // with (v_out : lseq _ (p.blockItemsX / p.blockWidth)).
      with v_out. assert out |-> v_out;
      assert pure (len v_out == p.blockItemsX / p.blockWidth);

      load_dense_matrix
        dense_tile
        gB
        n_idx p.blockWidth col_ind_tile;

      with edt. assert dense_tile |-> edt;

      assume pure (
        Seq.equal
          (Seq.slice
            row_ind
            (!idx * p.blockItemsK) (!idx * p.blockItemsK + p.blockItemsK))
          (cast_pos #p.blockItemsK (
            Seq.slice col_ind
              (ri' + !idx * p.blockItemsK)
              (ri' + !idx * p.blockItemsK + p.blockItemsK)
          ))
      );

      assert pure (
        chest2_tile_prop
          eB
          (Seq.slice
            row_ind
            (!idx * p.blockItemsK) (!idx * p.blockItemsK + p.blockItemsK))
          n_idx p.blockWidth edt
      );
      assume pure (
        Seq.equal
        (Seq.slice elems
          (ri' + !idx * p.blockItemsK)
          (ri' + !idx * p.blockItemsK + p.blockItemsK))
        (Seq.slice row_elems
          (!idx * p.blockItemsK)
          (!idx * p.blockItemsK + p.blockItemsK))
      );
      assume pure (chunk et /? p.cols);
      assert pure (in_bounds 0 p.shared row_ind);
      assert pure (!idx * p.blockItemsK + p.blockItemsK <= re - ri');
      
      Compute.tile_vmprod
        out
        out0
        elems_tile
        row_elems row_ind
        (!idx * p.blockItemsK)
        #ldt #ctldt
        dense_tile
        eB
        n_idx p.blockWidth #();

      idx := !idx +^ 1sz;
      nnz := !nnz -^ p.blockItemsK;

      // TODO
      admit();
    };


    with v_out. assert out |-> v_out;

    assert pure (
      Seq.equal
        (lslice row_elems 0 (!idx * p.blockItemsK))
        (Seq.append
          (Seq.create (ri - ri') zero)
          (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri))
        )
    );

    Compute.tile_mask_lemma
      #_ #_ #_ #_ #_ #_
      (!idx * p.blockItemsK)
      (ri - ri')
      (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri))
      (Seq.slice row_ind 0 (!idx * p.blockItemsK))
      eB n_idx p.blockWidth
      out0 v_out;

    assert pure (
      Seq.equal
        (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri))
        (Seq.slice elems ri (re - !nnz))
    );
    assert pure (
      Seq.equal
        (Seq.slice
          (lslice row_ind 0 (!idx * p.blockItemsK))
          (ri - ri') (!idx * p.blockItemsK)
        )
        (lslice' (cast_pos col_ind) ri (re - !nnz))
    );

    barrier_in_fold_residue_pre p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid ri' re tid;

    assert pure (SZ.v !idx == (re - ri') / p.blockItemsK);
    rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid ((re - ri') / p.blockItemsK * 2) tid
    as barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (!idx * 2) tid;

    ();
  }
  else {
    idx := 0sz;
    nnz := re -^ ri;

    Compute.tile_vmprod_prop_lemma0
      out0
      eB n_idx p.blockWidth;

    assert pure (
      Seq.equal
        Seq.empty
        (Seq.slice elems ri (re - !nnz))
    );
    assert pure (
      Seq.equal
        Seq.empty
        (lslice' (cast_pos col_ind) ri (re - !nnz))
    );
    
    barrier_in_fold_residue0_pre p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid ri' re tid;

    rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid 0 tid
    as barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (!idx * 2) tid;
    ();
  };
}

inline_for_extraction noextract
fn kf_residue
  (#et : Type0) {| d : scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srm : strided_row_major lB |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (gB : array2 et lB)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : chest2 et p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : larray et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : larray sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == tcol p bid tid })
  (idx residue : sz {residue_pred p.blockItemsK ri ri' re idx residue})
  norewrite
  preserves gpu
  preserves smatrix_pts_to' gA #fA elems col_ind row_off eA
  requires  pure (aligned 16 gA.elems)
  requires  pure (aligned 16 gA.col_ind)
  preserves gB |-> Frac (fB /. allthreads p) eB
  requires  pure (aligned 16 (core gB))
  requires  pure (aligned_strided_row_major (chunk et) srm)
  preserves
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    )
  preserves thread_id p.blockWidth tid
  requires B.barrier_state (idx * 2)
  requires
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (idx * 2) tid
  requires exists* vout.
    out |-> vout **
    pure (
      Compute.tile_vmprod_prop #_ #_ #_ #_
        #(re - residue - ri)
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) d.zero)
        (Seq.slice elems ri (re - residue))
        (lslice' (cast_pos col_ind) ri (re - residue))
        eB n_idx p.blockWidth vout
    )
  ensures B.barrier_state ((idx + 1) * 2)
  ensures exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s
  ensures exists* vout.
    out |-> vout **
    pure (
      Compute.tile_vmprod_prop #_ #_ #_ #_
        #(re - ri)
        #(threadItemsX p)
        (Seq.create (p.blockItemsX / p.blockWidth) d.zero)
        (Seq.slice elems ri re)
        (lslice' (cast_pos col_ind) ri re)
        eB n_idx p.blockWidth vout
    )
{
  let out0 : erased (lseq et (threadItemsX p)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;
  let row_elems : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
  let row_ind : erased (lseq sz (re - ri)) = hide (Seq.slice col_ind ri re);

  assert pure (
    Seq.equal
      (Seq.slice elems ri (re - residue))
      (lslice' row_elems 0 ((re - ri) - residue))
  );
  assert pure (
    Seq.equal
      (lslice' (cast_pos col_ind) ri (re - residue))
      (lslice' (cast_pos row_ind) 0 ((re - ri) - residue))
  );

  with vout. assert out |-> vout;
  assert pure (
    Compute.tile_vmprod_prop
      out0
      (lslice' row_elems 0 ((re - ri) - residue))
      (lslice' (cast_pos row_ind) 0 ((re - ri) - residue))
      eB n_idx p.blockWidth vout
  );

  sparse_load_residue p row_perm gA #_ #row_off #elems #col_ind #eA #fA
    elems_tile col_ind_tile bid ri ri' re tid idx residue;
  
  assert pure (
    Seq.equal
      (Seq.slice elems (re - residue) re)
      (Seq.slice row_elems ((re - ri) - residue) (re - ri))
  );
  assert pure (
    Seq.equal
      (Seq.slice col_ind (re - residue) re)
      (Seq.slice row_ind ((re - ri) - residue) (re - ri))
  );

  Compute.tile_load_vmprod
    out out0
    elems_tile col_ind_tile
    row_elems row_ind
    gB n_idx p.blockWidth
    ((re - ri) - residue) (re - ri) residue;
  
  assert pure (
    Seq.equal
      (Seq.slice row_elems 0 (re - ri))
      (Seq.slice elems ri re)
  );
  assert pure (
    Seq.equal
      (Seq.slice (cast_pos row_ind) 0 (re - ri))
      (lslice' (cast_pos col_ind) ri re)
  );

  unfold slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  slice_concat elems_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  slice_concat col_ind_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;

  add_full_slice elems_tile 0 p.blockItemsK p.blockItemsK;
  slice_to_array elems_tile;
  add_full_slice col_ind_tile 0 p.blockItemsK p.blockItemsK;
  slice_to_array col_ind_tile;
  ();
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 p.rows p.cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // layout para fragmento denso
  (dtsize : sz { SZ.v dtsize = p.blockItemsK * threadItemsX p })
  (ldt : layout2 p.blockItemsK (p.blockItemsX /^ p.blockWidth) { is_full ldt }) {| ctlayout ldt, srmdt : strided_row_major ldt |}
  (#_ : squash (aligned_strided_row_major (chunk et) srmdt))
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB)
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (gC : array2 et lC)
  // TODO esto tiene que estar acá? podria estar en block_pre?
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#eA : chest2 et p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  (sh : c_shmems (shmems_desc p))
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  ()
  norewrite
  requires
    gpu **
    kpre
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      sh
      bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        (fst sh) (fst (snd sh)) bid
    ) **
    B.barrier_state 0
  ensures
    gpu **
    kpost
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      sh
      bid tid **
    thread_id p.blockWidth tid **
    block_id (nblocks p) bid **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        (fst sh) (fst (snd sh)) bid
    ) **
    B.barrier_state (barrier_count p row_perm col_ind row_off bid)
{
  let m_idx = Pulse.Lib.Array.(row_indices.(brow_ p bid));
  assert rewrites_to m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
  // let n_idx = bcol_ p bid;
  let n_idx = tcol_ p bid tid;
  assert rewrites_to n_idx (SZ.uint_to_t (tcol p bid tid));

  let (elems_tile0, (col_ind_tile0, _)) = sh;

  // pts_to_len elems_tile0;
  // pts_to_len col_ind_tile0;

  (* This incantation here improves the generated code by actually defining
  these variables at this point. *)
  let elems_tile   = elems_tile0;     assert rewrites_to elems_tile   elems_tile0;
  let col_ind_tile = col_ind_tile0;   assert rewrites_to col_ind_tile col_ind_tile0;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  let ri = Pulse.Lib.Array.(gA.row_off.(m_idx));
  let re = Pulse.Lib.Array.(gA.row_off.(m_idx +^ 1sz));

  let ri' = align_offset et ri;

  let mut nnz : sz = re -^ ri';
  let mut idx = 0sz;

  (* GM: Nota: no podemos tener una expresión de división como la
  longitud del array, porque sería un VLA. Por eso agregué un argumento
  al kernel (blockChunks) que tiene un refinamiento que asegura que
  es igual (p.blockItemsK / p.blockWidth). *)
  let mut out = [| zero #et #_; blockChunks |];
  assume pure (aligned 16 out);

  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  //------------------main-----------------------------------------

  let mut dtcore = [| zero #et #_; dtsize |];
  assume pure (aligned 16 dtcore);

  let dense_tile : array2 et ldt = from_array ldt dtcore;
  assert rewrites_to dense_tile (from_array ldt dtcore);
  tensor_abs' ldt dtcore;


  assert pure (threadItemsX p > 0);
  kf_main
    p row_perm
    gA gB dense_tile
    eA
    out
    elems_tile col_ind_tile
    bid
    ri ri' re
    tid n_idx
    nnz idx;

    tensor_concr dense_tile;
    rewrite each (core (from_array ldt dtcore)) as dtcore;


  //------------------residue-----------------------------------------

  kf_residue
    p row_perm
    gA gB eA
    out
    elems_tile col_ind_tile
    bid ri ri' re tid n_idx !idx !nnz;

  //------------------dense store-----------------------------------------

  with v_out. assert out |-> v_out;
  Pulse.Lib.Array.pts_to_len out;

  unsparse_row_lemma
    p.rows p.shared
    elems (cast_pos col_ind) (cast_pos row_off) m_idx;

  assert pure (
    Seq.equal
      (cast_pos #(re - ri) (Seq.slice col_ind ri re))
      (Seq.slice (cast_pos col_ind) ri re)
  );

  Compute.vmprod_is_tile
    eA m_idx
    #(re - ri)
    (Seq.slice elems ri re) (Seq.slice (cast_pos col_ind) ri re)
    eB n_idx p.blockWidth v_out;

  assert pure ((brow p (SizeT.v bid) |~> row_perm) == SZ.v m_idx);

  // no esta funcionando bien el rewrites_to m_idx acá; lo hacemos a mano
  rewrite each (brow p bid |~> row_perm)
  as (SizeT.v (Seq.Base.index (ordering row_perm) (SizeT.v (brow_ p bid))));

  gpu_matrix_store_tile_vec
    gC m_idx n_idx (Kuiper.Spec.GEMM.matmul eA eB)
    (p.blockItemsX /^ p.blockWidth) out p.blockWidth;

  rewrite each (SizeT.v (Seq.Base.index (ordering row_perm) (SizeT.v (brow_ p bid))))
  as (brow p bid |~> row_perm);


  //------------------exit-----------------------------------------

  slice_to_array row_indices;

  Pulse.Lib.Array.PtsTo.pts_to_len elems_tile;
  assert is_full_slice elems_tile p.blockItemsK;
  slice_to_array elems_tile;

  Pulse.Lib.Array.PtsTo.pts_to_len col_ind_tile;
  assert is_full_slice col_ind_tile p.blockItemsK;
  slice_to_array col_ind_tile;

  ()
}
#pop-options


inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lB : layout2 p.shared p.cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 p.rows p.cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (dtsize : sz { SZ.v dtsize = p.blockItemsK * threadItemsX p })
  (ldt : layout2 p.blockItemsK (p.blockItemsX /^ p.blockWidth) { is_full ldt }) {| ctlayout ldt, srmdt : strided_row_major ldt |}
  (#_ : squash (aligned_strided_row_major (chunk et) srmdt))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : larray sz p.rows)
  (gB : array2 et lB {is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (gC : array2 et lC {is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (p.rows + 1)))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (#_ : squash ((chunk et * p.blockWidth) /? p.blockItemsK))
  (#_ : squash ((chunk sz * p.blockWidth) /? p.blockItemsK))
  : kernel_desc
    (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      live gC
    )
    (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      gC |-> MS.matmul eA eB
    )
= {
  nblk = nblocks p;
  // nthr = nthreads p;
  nthr = p.blockWidth;

  barrier_contract = (fun bid ptrs ->
    barrier_contract p row_perm elems col_ind row_off
      (fst ptrs) (fst (snd ptrs)) bid);
  barrier_count = (fun bid -> barrier_count p row_perm col_ind row_off bid);
  barrier_ok = (fun bid ptrs -> magic());

  shmems_desc = shmems_desc p;

  frame = emp;

  block_pre  = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_pre
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      bid tid
  );
  block_post = (fun bid -> forall+ (tid : natlt p.blockWidth).
    block_post
      p row_perm
      gA row_indices gB gC
      elems col_ind row_off
      eA eB
      fA fri fB
      bid tid
  );
  setup = setup
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #_ #_ #fA;
  teardown = teardown
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #_ #_ #fA;

  block_frame    = (fun _ar _bid -> emp);
  block_setup = block_setup
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #eA #eB
    #fA #fri #fB;
  block_teardown = block_teardown
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    #eA #eB
    #fA #fri #fB;

  kpre  = kpre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB;
  kpost = kpost
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB;

  f =
    kf
      p row_perm blockChunks
      gA
      dtsize ldt // estos parametros estan en cualquier lado
      row_indices gB gC;

  block_pre_sendable=magic();
  block_post_sendable=magic();
  kpre_sendable=magic();
  kpost_sendable=magic();
}

inline_for_extraction noextract
fn spmm_on
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (dtsize : sz { SZ.v dtsize = blockItemsK * (blockItemsX / blockWidth) })
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (rows + 1)))
  (#eA : chest2 et rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  //(#_ : size_req rows shared cols)
  (s : stream_t)
  (#e : epoch_t)
  norewrite
  preserves cpu ** stream_live s ** epoch_live s e
  requires
    pure (blockItemsX /? cols) **
    on gpu_loc (smatrix_pts_to' gA #fA elems col_ind row_off eA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB) **
    on gpu_loc (live gC) **
    pure (rows * (cols `divup` blockItemsX) <= max_blocks) **
    pure (blockWidth <= max_threads)
  ensures
    pledge0 (epoch_done s e) (on gpu_loc (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      gC |-> MS.matmul eA eB
    ))
{
  dguard (rows <^ 10000sz);
  dguard (shared <^ 10000sz);
  dguard (cols <^ 10000sz);
  dguard (blockItemsK <^ 10000sz);
  dguard (blockItemsX <^ 10000sz);
  // ^ FIXME: propagate preconditions instead of dynamically aborting
  assert pure (rows * (cols `divup` blockItemsX) <= max_blocks);
  assert pure (size_req #et ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth }));
  assert pure ((chunk et * blockWidth) /? blockItemsK);
  assert pure ((chunk sz * blockWidth) /? blockItemsK);
  
  assume pure (chunk et /? blockChunks);
  lemma_aligned_strided_row_major_l2_row_major
    #(SZ.v blockItemsK) #(SZ.v blockChunks) (chunk et);

  launch (
    kdesc #et #_
      ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth })
      row_perm blockChunks #lB #_ #_ #lC dtsize (l2_row_major _ _) #(c_l2_row_major (SZ.v blockItemsK) blockChunks) #(strided_row_major_l2_row_major #(SZ.v blockItemsK) #(SZ.v blockChunks))
      gA row_indices gB gC elems col_ind row_off eA
      #eB #fA #fri #fB
  ) s;
}

(* Same structure as Kuiper.Kernel.launch_kernel_full_sync, one level up. *)
inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (rows shared cols : szp { chunk et /? cols })
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    (k * chunk et) /? blockItemsX
  }))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (dtsize : sz { SZ.v dtsize = blockItemsK * (blockItemsX / blockWidth) })
  (#lB : layout2 shared cols) {| ctlayout lB, srmB : strided_row_major lB |}
  (#lC : layout2 rows cols)   {| ctlayout lC, srmC : strided_row_major lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#_ : squash (aligned 16 (core gB)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmB))
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  (#_ : squash (aligned 16 (core gC)))
  (#_ : squash (aligned_strided_row_major (chunk et) srmC))
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (rows + 1)))
  (#eA : chest2 et rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : chest2 et shared cols)
  (#eC : chest2 et rows cols)
  //(#_ : size_req rows shared cols)
  norewrite
  preserves
    cpu **
    //on gpu_loc (gA |-> Frac fA eA) **
    on gpu_loc (smatrix_pts_to' gA #fA elems col_ind row_off eA) **
    on gpu_loc (row_indices |-> Frac fri (ordering row_perm)) **
    on gpu_loc (gB |-> Frac fB eB)
  requires
    pure (blockItemsX /? cols) **
    on gpu_loc (live gC) **
    pure (rows * (cols `divup` blockItemsX) <= max_blocks) **
    pure (blockWidth <= max_threads)
  ensures on gpu_loc (gC |-> MS.matmul eA eB)
{
  let s = fresh_stream ();
  get_epoch s ();
  spmm_on
    rows shared cols
    blockItemsK blockItemsX blockWidth
    blockChunks dtsize
    gA row_indices fri gB gC
    elems col_ind row_off
    #eA row_perm #eB #eC
    s;
  sync_stream s;
  redeem_pledge emp_inames (epoch_done s _) (on gpu_loc (
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      row_indices |-> Frac fri (ordering row_perm) **
      gB |-> Frac fB eB **
      gC |-> MS.matmul eA eB
  ));
  drop_ (epoch_done s _);
  drop_ (epoch_live s _);
  destroy_stream s;
}
