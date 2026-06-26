module Kuiper.Sparse.SPMM

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute
module Array2 = Kuiper.Array2
open Kuiper.Sparse
open Kuiper.Sparse.Load
open Kuiper.EMatrix
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier
open Kuiper.Tensor.Layout { ctlayout }

unfold
let block_pre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> matrix_live_cell
        gC (brow p bid |~> row_perm) (bcol p bid + k * p.blockWidth + tid))


unfold
let block_post
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  smatrix_pts_to' gA #(fA /. allthreads p)
    elems col_ind row_off eA **
  row_indices |-> Frac (fri /. allthreads p) (ordering row_perm) **
  gB |-> Frac (fB /. allthreads p) eB **
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + k * p.blockWidth + tid)
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + k * p.blockWidth + tid)))

unfold
let kpre
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // TODO no se si puedo hacer eso
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  block_pre
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  thread_live_chunks (fst sh) p.blockWidth tid **
  thread_live_chunks (fst (snd sh)) p.blockWidth tid **
  pure (
    aligned 16 (fst sh) /\
    aligned 16 (fst (snd sh))
  )

unfold
let kpost
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (eB : ematrix et p.shared p.cols)
  (fA fri fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  block_post
    p row_perm
    gA row_indices gB gC
    elems col_ind row_off
    eA eB
    fA fri fB
    bid tid **
  live (fst sh) #(1.0R /. p.blockWidth) **
  live (fst (snd sh)) #(1.0R /. p.blockWidth)

let divup_factor (n : nat) (d : pos) =
  (i : natlt (divup n d) & (j : natlt d {i * d + j < n }))

let bij_divup_factor (n : nat) (d : pos)
: Kuiper.Bijection.bijection (natlt n) (divup_factor n d)
=
{
  ff = (fun (i : natlt n) -> (|i / d, i % d|) <: divup_factor n d);
  gg = (fun (|j, k|) -> j * d + k);

  ff_gg = (fun _ -> ());
  gg_ff = (fun _ -> ());
}

ghost
fn forevery_factor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i:natlt n). p i
  ensures forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
{
  forevery_iso (bij_divup_factor n d) p;
  forevery_ext #(divup_factor n d)
    (fun q -> p ((bij_divup_factor n d).gg q))
    (fun q -> p (q._1 * d + q._2));
  forevery_unflatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
}

ghost
fn forevery_unfactor_
  (n : nat)
  (d : pos)
  (p : natlt n -> slprop)
  requires forall+ (i1:natlt (divup n d)) (i2:natlt d {i1 * d + i2 < n}).
    p (i1 * d + i2)
  ensures forall+ (i:natlt n). p i
{
  forevery_flatten_dep
    #(natlt (divup n d)) #(fun i1 -> (i2 : natlt d {i1 * d + i2 < n}))
    (fun i1 i2 -> p (i1 * d + i2));
  forevery_iso (Kuiper.Bijection.bij_sym (bij_divup_factor n d )) _;
  forevery_ext _ (fun i -> p i);
}


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

ghost
fn forevery_refine_pred'
  (#a:Type0)
  (f: a -> prop)
  (p: (x:a) -> squash (f x) -> slprop)
  requires
    forall+ (x:a). when__ (f x) (p x)
  ensures
    forall+ (x:a { f x }). p x ()
{
  forevery_refine_split (fun x -> when__ (f x) (p x)) f;
  drop_ (forall+ (x:a { ~(f x) }). when__ (f x) (p x));
  forevery_ext (fun (x:a { f x }) -> when__ (f x) (p x)) (fun x -> p x ());
}


ghost
fn setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
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
  Array2.explode gC;
  forevery_unflatten' _;

  forevery_map
    (fun r ->
      forall+ (c : natlt p.cols).
        Array2.pts_to_cell gC (r, c) (macc eC r c)
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
            Array2.pts_to_cell gC (r, b * p.blockItemsX + ix)
              (macc eC r (b * p.blockItemsX + ix))
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
              Array2.pts_to_cell gC (r, b * p.blockItemsX + ix)
                (macc eC r (b * p.blockItemsX + ix))
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

          forevery_factor p.blockItemsX
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

  gpu_slice_share row_indices 0 p.rows (allthreads p);
  forevery_factor (allthreads p) (nblocks p) p.blockWidth _;

  Array2.share_n gB (allthreads p) #fB;
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

  ();
}

// #push-options "--print_implicits"
ghost
fn block_setup
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
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

  thread_share_chunks (fst sh) p.blockWidth;
  thread_share_chunks (fst (snd sh)) p.blockWidth;

  // TODO de donde sacamos esto?
  assume pure ( aligned 16 (fst sh) /\ aligned 16 (fst (snd sh)));

  forevery_intro_pure #(natlt p.blockWidth)
    (fun _ ->
        aligned 16 (fst sh) /\
        aligned 16 (fst (snd sh))
    );
  forevery_zip3 #(natlt p.blockWidth)
    (fun tid -> thread_live_chunks (fst sh) p.blockWidth tid)
    (fun tid -> thread_live_chunks (fst (snd sh)) p.blockWidth tid)
    (fun tid ->
      pure (
        aligned 16 (fst sh) /\
        aligned 16 (fst (snd sh))
      )
    );

  forevery_zip #(natlt p.blockWidth)
    _
    (fun tid ->
      thread_live_chunks (fst sh) p.blockWidth tid **
      thread_live_chunks (fst (snd sh)) p.blockWidth tid **
      pure (
        aligned 16 (fst sh) /\
        aligned 16 (fst (snd sh))
      )
    );

}

ghost
fn block_teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
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
  forevery_unzip3 _ _ _;
  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    );
  with elems_tile.
    assert gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      elems_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      elems_tile)
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    )
    (fun tid ->
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        elems_tile
    )
    fn tid {
      gpu_slice_pts_to_eq (fst sh) 0 p.blockItemsK (1.0R /. p.blockWidth)
        #_ #elems_tile;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      gpu_pts_to_slice (fst sh) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        elems_tile
    );

  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    );
  with col_ind_tile.
    assert gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      col_ind_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
      col_ind_tile)
    (fun tid -> exists* x.
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK x
    )
    (fun tid ->
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        col_ind_tile
    )
    fn tid {
      gpu_slice_pts_to_eq (fst (snd sh)) 0 p.blockItemsK (1.0R /. p.blockWidth)
        #_ #col_ind_tile;
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      gpu_pts_to_slice (fst (snd sh)) #(1.0R /. p.blockWidth) 0 p.blockItemsK
        col_ind_tile
    );

  gpu_slice_gather (fst sh) 0 p.blockItemsK p.blockWidth;
  gpu_slice_gather (fst (snd sh)) 0 p.blockItemsK p.blockWidth;

  fold_c_shmems sh (`%shmems_desc);

  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB)
  (gC : Array2.t et lC)
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (#eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
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
  gpu_slice_gather row_indices 0 p.rows (allthreads p);
  forevery_unzip_2 _ _;
  forevery_unfactor' (allthreads p) _ _
    (fun _ _ -> gB |-> Frac (fB /. allthreads p) eB);
  Array2.gather_n gB (allthreads p) #fB;


  forevery_map #(natlt (nblocks p))
    (fun bid ->
      forall+
        (tid: natlt p.blockWidth)
        (k: natlt (p.blockItemsX /^ p.blockWidth)).
        when__ (bcol p bid + k * v p.blockWidth + tid < v p.cols)
          (fun _ ->
            Array2.pts_to_cell gC
              (brow p bid |~> row_perm,
               bcol p bid + k * v p.blockWidth + tid)
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
            Array2.pts_to_cell gC
              (brow p bid |~> row_perm,
               bcol p bid + ix)
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
              Array2.pts_to_cell gC
                (brow p bid |~> row_perm,
                 bcol p bid + k * v p.blockWidth + tid)
                (MS.matmul_single eA
                    eB
                    (brow p bid |~> row_perm)
                    (bcol p bid + k * v p.blockWidth + tid)))
        )
        (fun tid k ->
          when__ (bcol p bid + (k * v p.blockWidth + tid) < v p.cols)
            (fun _ ->
              Array2.pts_to_cell gC
                (brow p bid |~> row_perm,
                 bcol p bid + (k * v p.blockWidth + tid))
                (MS.matmul_single eA
                    eB
                    (brow p bid |~> row_perm)
                    (bcol p bid + (k * v p.blockWidth + tid)))
            )
        );
      forevery_commute _;
      forevery_unfactor p.blockItemsX (p.blockItemsX /^ p.blockWidth) _
        (fun ix ->
          when__
            (bcol p bid + ix < p.cols)
            (fun _ ->
              Array2.pts_to_cell gC
                (brow p bid |~> row_perm,
                 bcol p bid + ix)
                (MS.matmul_single eA eB
                  (brow p bid |~> row_perm)
                  (bcol p bid + ix)
                )
            )
        );
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
            Array2.pts_to_cell gC
              (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm,
               bcol p (r * divup p.cols p.blockItemsX + b) + ix)
              (MS.matmul_single eA eB
                (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm)
                (bcol p (r * divup p.cols p.blockItemsX + b) + ix)
              )
          )
    )
    (fun r ->
      forall+ (c : natlt p.cols).
        Array2.pts_to_cell gC (r |~> row_perm, c)
          (MS.matmul_single eA eB (r |~> row_perm) c)
    )
    fn r {
      forevery_map_2 #(natlt (divup p.cols p.blockItemsX)) #(natlt p.blockItemsX)
        (fun b ix ->
          when__
            (bcol p (r * divup p.cols p.blockItemsX + b) + ix < p.cols)
            (fun _ ->
              Array2.pts_to_cell gC
                (brow p (r * divup p.cols p.blockItemsX + b) |~> row_perm,
                 bcol p (r * divup p.cols p.blockItemsX + b) + ix)
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
              Array2.pts_to_cell gC
                (r |~> row_perm, b * p.blockItemsX + ix)
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
                Array2.pts_to_cell gC
                  (r |~> row_perm, b * p.blockItemsX + ix)
                  (MS.matmul_single eA eB
                    (r |~> row_perm) (b * p.blockItemsX + ix)
                  )
              )
        )
        (fun b ->
          forall+ (ix : natlt p.blockItemsX {b * p.blockItemsX + ix < p.cols}).
            Array2.pts_to_cell gC
              (r |~> row_perm, b * p.blockItemsX + ix)
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
          Array2.pts_to_cell gC (r |~> row_perm, c)
            (MS.matmul_single eA eB (r |~> row_perm) c)
        );
    };

  forevery_iso row_perm (fun r ->
    forall+ (c : natlt p.cols).
      Array2.pts_to_cell gC (r |~> row_perm, c)
        (MS.matmul_single eA eB (r |~> row_perm) c)
  );
  forevery_ext_2
    (fun r c ->
      Array2.pts_to_cell gC
        (row_perm.gg r |~> row_perm, c)
        (MS.matmul_single eA eB (row_perm.gg r |~> row_perm) c)
    )
    (fun r c -> Array2.pts_to_cell gC (r, c) (macc (MS.matmul eA eB) r c));
  forevery_flatten _;
  forevery_ext
    (fun (rc : natlt p.rows & natlt p.cols) -> Array2.pts_to_cell gC (rc._1, rc._2) (macc (MS.matmul eA eB) rc._1 rc._2))
    (fun (rc : Array2.ait p.rows p.cols) -> Array2.pts_to_cell gC rc (macc (MS.matmul eA eB) rc._1 rc._2));
  Array2.implode gC;

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
  (#eA : ematrix et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p (reveal col_ind) (reveal row_off)))
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
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
  let off : sz = ri +^ idx *^ p.blockItemsK;

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
  (#eA : ematrix et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
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
    gpu_pts_to_slice elems_tile #(1.0R /. p.blockWidth) 0 residue
      (Seq.slice elems (re - residue) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK **
    gpu_pts_to_slice col_ind_tile #(1.0R /. p.blockWidth) 0 residue
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

  load2_array_to
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

ghost
fn when__intro_true (b:bool{b == true}) (p : slprop)
  requires p
  ensures when__ b (fun _ -> p)
{
  rewrite p as when__ b (fun _ -> p);
}

ghost
fn when__intro_false (b : bool{b == false}) (p : slprop)
  ensures when__ b (fun _ -> p)
{
  rewrite emp as when__ b (fun _ -> p);
}

ghost
fn when__elim_true (b:bool{b == true}) (p : slprop)
  requires when__ b (fun _ -> p)
  ensures p
{
  rewrite when__ b (fun _ -> p) as p;
}

ghost
fn when__elim_false (b:bool{b == false}) (p : slprop)
  requires when__ b (fun _ -> p)
  ensures emp
{
  rewrite when__ b (fun _ -> p) as emp;
}

inline_for_extraction noextract
fn store_out
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lC |}
  (gC : Array2.t et lC)
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  (#v_out : erased (seq et){length out == len v_out})
  (bid : szlt (nblocks p))
  (tid : szlt p.blockWidth)
  (m_idx : szlt p.rows{SZ.v m_idx == (brow p bid |~> row_perm)})
  (n_idx : szlt p.cols{SZ.v n_idx == bcol p bid})
  (x : szlt (p.blockItemsX /^ p.blockWidth))
  requires
    when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      matrix_live_cell gC
        (brow p bid |~> row_perm)
        (bcol p bid + x * p.blockWidth + tid)
    )
    ** out |-> v_out
  ensures
    when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
        Array2.pts_to_cell gC
          (brow p bid |~> row_perm,
           bcol p bid + x * p.blockWidth + tid)
          (v_out @! x)
    )
    ** out |-> v_out
{
  block_lemma_off p.blockItemsX p.blockWidth x tid;

  let out_off = n_idx +^ x *^ p.blockWidth +^ tid;
  assert rewrites_to out_off (n_idx +^ x *^ p.blockWidth +^ tid);

  if (out_off <^ p.cols) {
    when__elim_true _ _;
    unfold matrix_live_cell;

    open Pulse.Lib.Array;
    let c = out.(x);
    assert pure (n_idx +^ x *^ p.blockWidth +^ tid <^ p.cols);

    assert rewrites_to #sz m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
    assert rewrites_to #sz n_idx (SZ.uint_to_t (bcol p bid));

    Array2.write_cell gC ((m_idx <: sz), n_idx +^ x *^ p.blockWidth +^ tid) c;

    assert Array2.pts_to_cell gC
      (brow p bid |~> row_perm, bcol p bid + x * p.blockWidth + tid)
      (v_out @! x);
    when__intro_true (bcol p bid + x * p.blockWidth + tid < p.cols)
      (Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + x * p.blockWidth + tid)
        (v_out @! x)
      );
  }
  else {
    rewrite when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      matrix_live_cell gC
        (brow p bid |~> row_perm)
        (bcol p bid + x * p.blockWidth + tid)
    ) as when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      Array2.pts_to_cell gC
        (brow p bid |~> row_perm, bcol p bid + x * p.blockWidth + tid)
        (v_out @! x)
    );
  };

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
  (x : gpu_array et n { aligned 16 x })
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
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (p : parameters et)
  (#n : nat)
  (x : gpu_array et n { aligned 16 x })
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

open Kuiper.Seq.Common { (@+) }

inline_for_extraction noextract
fn kf_head
  (#et : Type0) {| d : scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lb : Array2.layout p.shared p.cols)
  {| ctlayout lb |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (gB : Array2.t et lb)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == bcol p bid })
  (#_ : squash (ri' + p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac (fB /. allthreads p) eB **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid **
    out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero **
    B.barrier_state 0
  ensures
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
    B.barrier_state 2 **
    out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX #p.blockItemsK
        (Seq.create (ri - ri') zero @+ Seq.slice elems ri (ri' + p.blockItemsK))
        // (Seq.slice (cast_pos col_ind) ri (ri + p.blockItemsK))
        (cast_pos (Seq.slice col_ind ri' (ri' + p.blockItemsK)))
        eB
        (Seq.create (p.blockItemsX / p.blockWidth) zero)
        tid (bcol p bid)
{
  offset_aligned_lemma_et' p gA.elems ri;
  assert pure (aligned' 16 gA.elems ri');
  load_array_vec elems_tile gA.elems ri' p.blockWidth tid;

  offset_aligned_lemma_sz' p gA.elems ri;
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

  mask_array_to elems_tile (ri -^ ri') zero p.blockWidth tid;

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

  Compute.compute
    p.blockWidth p.blockItemsK p.blockItemsX
    elems_tile col_ind_tile p.blockItemsK gB out tid n_idx;
}

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

ghost
fn rewrite_seq_slice
  (#et : Type0)
  (#n : nat)
  (a : gpu_array et n)
  (#f : perm)
  (#m : nat)
  (s : lseq et m)
  (i j : natle m { i <= j })
  (k1 k2 : natle (j - i) { k1 <= k2 })
  (s' : lseq et (j - i))
  requires
    a |-> Frac f (Seq.slice s (i + k1) (i + k2)) **
    pure (
      s' == (Seq.slice s i j)
    )
  ensures
    a |-> Frac f (Seq.slice s' k1 k2)
{
  let t = Seq.slice s (i + k1) (i + k2);
  let t' = Seq.slice s' k1 k2;

  assert pure (t `Seq.equal` t');
}

ghost
fn rewrite_seq_mask_slice
  (#et : Type0) {| scalar et |}
  (#n : nat)
  (a : gpu_array et n)
  (#f : perm)
  (#m : nat)
  (s : lseq et m)
  (i j : natle m { i <= j })
  (i' : natle i)
  (k1 k2 : natle (j - i') { k1 <= k2 })
  (s' : lseq et (j - i'))
  requires
    a |-> Frac f (Seq.slice s (i' + k1) (i' + k2)) **
    pure (
      i - i' <= k1 /\
      s' == seq_mask (i - i') #(j - i) (Seq.slice s i j)
    )
  ensures
    a |-> Frac f (Seq.slice s' k1 k2)
{
  let t = Seq.slice s (i' + k1) (i' + k2);
  let t' = Seq.slice s' k1 k2;

  assert pure (t `Seq.equal` t');
}

ghost
fn rewrite_compute_step
  (#et : Type0) {| scalar et |}
  (#shared #cols : nat)
  (bw bx : pos{bw /? bx})
  (#nnz : nat)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz{ in_bounds 0 shared (cast_pos col_ind) })
  (eB : ematrix et shared cols)
  (#l : nat)
  (out : larray et l)
  (v_out0 : seq et { len v_out0 == bx / bw})
  (v_out : seq et { len v_out == bx / bw})
  (off : natlt bw)
  (n : natlt cols)
  (from to : natle nnz{from <= to})
  requires
    out |->
      Compute.compute_result
        bw bx #(to - from)
        (Seq.slice elems from to)
        (cast_pos (Seq.slice col_ind from to))
        eB v_out off n **
    pure (
      v_out ==
      Compute.compute_result
        bw bx #from
        (Seq.slice elems 0 from)
        (cast_pos (Seq.slice col_ind 0 from))
        eB v_out0 off n
    )
  ensures
    out |->
    Compute.compute_result
      bw bx #to
      (Seq.slice elems 0 to)
      (cast_pos (Seq.slice col_ind 0 to))
      eB v_out0 off n
{
  assert pure (
    Seq.equal
      (cast_pos #from (Seq.slice col_ind 0 from))
      (Seq.slice (cast_pos col_ind) 0 from)
  );
  assert pure (
    Seq.equal
      (Seq.slice (cast_pos col_ind) from to)
      (cast_pos #(to - from) (Seq.slice col_ind from to))
  );
  assert pure (
    Seq.equal
      (Seq.slice (cast_pos col_ind) 0 to)
      (cast_pos #to (Seq.slice col_ind 0 to))
  );
  Compute.compute_step
    bw bx
    elems
    (cast_pos col_ind)
    eB v_out0 off n
    from
    to;
  assert pure (
    v_out `Seq.equal`
    Compute.compute_result
      bw bx #from
      (Seq.slice elems 0 from)
      ((Seq.slice (cast_pos col_ind) 0 from))
      eB v_out0 off n
  );
}

#push-options "--z3rlimit 25"
inline_for_extraction noextract
fn kf_main
  (#et : Type0) {| d : scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lb : Array2.layout p.shared p.cols)
  {| ctlayout lb |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (gB : Array2.t et lb)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == bcol p bid })
  (nnz idx : ref sz) // realmente no necesito una referencia a idx
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac (fB /. allthreads p) eB **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    thread_live_chunks elems_tile p.blockWidth tid **
    thread_live_chunks col_ind_tile p.blockWidth tid **
    B.barrier_state 0 **
    out |-> Seq.create (p.blockItemsX / p.blockWidth) d.zero **
    live idx **
    nnz |-> (re -^ ri')
  ensures
    // MAYBE usar round2 p.blockItemsK (re - ri)
    (exists* (v_idx :sz) (residue : szle (re - ri)).
      idx |-> v_idx **
      nnz |-> (residue <: sz) **
      pure (residue_pred p.blockItemsK ri ri' re v_idx residue) **
      B.barrier_state (v_idx * 2) **
      barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (v_idx * 2) tid **
      out |->
        Compute.compute_result
          p.blockWidth p.blockItemsX
          #(re - residue - ri)
          (Seq.slice elems ri (re - residue))
          (cast_pos (lslice' col_ind ri (re - residue)))
          eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
          tid n_idx
    )
{
  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  if (!nnz >=^ p.blockItemsK)
  {
    let row_elems_ : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
    let row_elems : erased (lseq et (re - ri')) = seq_mask (ri - ri') #(re - ri) row_elems_;

    let row_ind : erased (lseq sz (re - ri')) = hide (Seq.slice col_ind ri' re);

    kf_head
      p row_perm
      gA gB eA
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
        (cast_pos #(p.blockItemsK) (lslice' col_ind ri' (ri' + p.blockItemsK)))
        (cast_pos #(p.blockItemsK) (lslice' row_ind 0 (!idx * p.blockItemsK)))
    );

    assert out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX #(!idx * p.blockItemsK)
        (Seq.slice row_elems 0 (!idx * p.blockItemsK))
        // (Seq.slice row_pos 0 (!idx * p.blockItemsK))
        (cast_pos #(p.blockItemsK) (Seq.slice row_ind 0 (!idx * p.blockItemsK)))
        eB out0 tid n_idx;

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
            v_out ==
            Compute.compute_result
              p.blockWidth p.blockItemsX #(!idx * p.blockItemsK)
              (lslice row_elems 0 (!idx * p.blockItemsK))
              (cast_pos (lslice row_ind 0 (!idx * p.blockItemsK)))
              eB out0 tid n_idx
          )
        )
    {
      sparse_load_main p row_perm gA #_ #_ #_ #_ #eA
        elems_tile col_ind_tile bid ri' re !idx tid;

      // assert pure (ri_ - ri <= p.blockItemsK);
      rewrite_seq_mask_slice
        elems_tile #(1.0R /. p.blockWidth)
        elems
        ri re ri'
        (!idx * p.blockItemsK) (!idx * p.blockItemsK + p.blockItemsK)
        row_elems;
      rewrite_seq_slice
        col_ind_tile #(1.0R /. p.blockWidth)
        col_ind
        ri' re
        (!idx * p.blockItemsK) (!idx * p.blockItemsK + p.blockItemsK)
        row_ind;

      Pulse.Lib.Array.pts_to_len out;
      // with (v_out : lseq _ (p.blockItemsX / p.blockWidth)).
      with v_out. assert out |-> v_out;
      assert pure (len v_out == p.blockItemsX / p.blockWidth);

      Compute.compute
        p.blockWidth p.blockItemsK p.blockItemsX
        elems_tile col_ind_tile p.blockItemsK gB out tid n_idx;

      rewrite_compute_step
        p.blockWidth p.blockItemsX
        row_elems row_ind
        eB
        out
        out0
        v_out
        tid n_idx
        (!idx * p.blockItemsK)
        ((!idx + 1) * p.blockItemsK);

      idx := !idx +^ 1sz;
      nnz := !nnz -^ p.blockItemsK;
    };

    with v_out. assert out |-> v_out;

    assert pure (
      Seq.equal
        (lslice row_elems 0 (!idx * p.blockItemsK))
        (seq_mask (ri - ri') (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri)))
    );

    Compute.compute_mask_lemma
      p.blockWidth p.blockItemsX
      (ri - ri')
      (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri))
      (cast_pos (lslice row_ind 0 (!idx * p.blockItemsK)))
      eB out0 tid n_idx;

    assert pure (
      Seq.equal
        (lslice row_elems_ 0 (ri' + !idx * p.blockItemsK - ri))
        (Seq.slice elems ri (re - !nnz))
    );
    assert pure (
      Seq.equal
        (Seq.slice
          (cast_pos (lslice row_ind 0 (!idx * p.blockItemsK)))
          (ri - ri') (!idx * p.blockItemsK)
        )
        (cast_pos (lslice' col_ind ri (re - !nnz)))
    );

    barrier_in_fold_residue_pre p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid ri' re tid;

    assert pure (SZ.v !idx == (re - ri') / p.blockItemsK);
    rewrite barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid ((re - ri') / v p.blockItemsK * 2) tid
    as barrier_in p row_perm elems col_ind row_off elems_tile col_ind_tile
      bid (!idx * 2) tid;

    ();
  }
  else {
    idx := 0sz;
    nnz := re -^ ri;
    assert pure (
      Seq.equal
        out0
        (Compute.compute_result
          p.blockWidth p.blockItemsX
          #(re - !nnz - ri)
          (Seq.slice elems ri (re - !nnz))
          (cast_pos #(re - !nnz - ri) (Seq.slice col_ind ri (re - !nnz)))
          eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
          tid n_idx)
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
  (#lb : Array2.layout p.shared p.cols)
  {| ctlayout lb |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (gB : Array2.t et lb)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  // salida
  (out : larray et (p.blockItemsX /^ p.blockWidth))
  // shmem
  (elems_tile : gpu_array et p.blockItemsK { aligned 16 elems_tile })
  (col_ind_tile : gpu_array sz p.blockItemsK { aligned 16 col_ind_tile })
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (ri' : sz{SZ.v ri' == round2 (max (chunk et) (chunk sz)) ri})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (tid : szlt p.blockWidth)
  (n_idx : sz {SZ.v n_idx == bcol p bid })
  (idx residue : sz {residue_pred p.blockItemsK ri ri' re idx residue})
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    gB |-> Frac (fB /. allthreads p) eB **
    B.barrier_tok (
      barrier_contract p row_perm elems col_ind row_off
        elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    barrier_in p row_perm elems col_ind row_off
      elems_tile col_ind_tile bid (idx * 2) tid **
    out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX
        (Seq.slice elems ri (re - residue))
          (cast_pos (lslice' col_ind ri (re - residue)))
        eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
        tid n_idx
  ensures
    B.barrier_state ((idx + 1) * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
    out |->
      Compute.compute_result
        p.blockWidth p.blockItemsX
        (Seq.slice elems ri re)
        (cast_pos (lslice' col_ind ri re))
        eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
        tid n_idx
{
  let row_elems : erased (lseq et (re - ri)) = hide (Seq.slice elems ri re);
  let row_ind : erased (lseq sz (re - ri)) = hide (Seq.slice col_ind ri re);

  assert pure (
    Seq.equal
      (Seq.slice elems ri (re - residue))
      (lslice' row_elems 0 ((re - ri) - residue))
  );
  assert pure (
    Seq.equal
      (cast_pos (lslice' col_ind ri (re - residue)))
      (lslice' (cast_pos row_ind) 0 ((re - ri) - residue))
  );

  assert out |->
    Compute.compute_result
      p.blockWidth p.blockItemsX
      (lslice' row_elems 0 ((re - ri) - residue))
      (lslice' (cast_pos row_ind) 0 ((re - ri) - residue))
      eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
      tid n_idx;

  sparse_load_residue p row_perm gA #_ #row_off #elems #col_ind #eA #fA
    elems_tile col_ind_tile bid ri ri' re tid idx residue;

  Compute.compute
    p.blockWidth p.blockItemsK p.blockItemsX
    elems_tile col_ind_tile residue gB out tid n_idx;

  assert pure (
    Seq.equal
      (Seq.slice elems (re - residue) re)
      (lslice' row_elems ((re - ri) - residue) (re - ri))
  );
  assert pure (
    Seq.equal
      (cast_pos (lslice' col_ind (re - residue) re))
      (lslice' (cast_pos row_ind) ((re - ri) - residue) (re - ri))
  );
  Compute.compute_step
    p.blockWidth p.blockItemsX
    #(re - ri)
    row_elems (cast_pos row_ind)
    eB
    (Seq.create (p.blockItemsX / p.blockWidth) zero)
    tid n_idx ((re - ri) - residue) (re - ri);

  unfold slice_live elems_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  gpu_slice_concat elems_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth) residue p.blockItemsK;
  gpu_slice_concat col_ind_tile #(1.0R /. p.blockWidth)
    0 residue p.blockItemsK;
}

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et { size_req p })
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lb : Array2.layout p.shared p.cols)
  (#lc : Array2.layout p.rows p.cols)
  {| ctlayout lb, ctlayout lc |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lb)
  (gC : Array2.t et lc)
  // matriz sparse ga
  (#elems : erased (lseq et gA.nnz))
  (#col_ind : erased (lseq sz gA.nnz))
  (#row_off : erased (lseq sz (p.rows + 1)))
  (#eA : ematrix et p.rows p.shared)
  // matriz densa gb
  (#eB : ematrix et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
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
  let m_idx = gpu_array_read row_indices (brow_ p bid);
  assert rewrites_to m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
  let n_idx = bcol_ p bid;

  let (elems_tile0, (col_ind_tile0, _)) = sh;

  (* This incantation here improves the generated code by actually defining
  these variables at this point. *)
  let elems_tile   = elems_tile0;     assert rewrites_to elems_tile   elems_tile0;
  let col_ind_tile = col_ind_tile0;   assert rewrites_to col_ind_tile col_ind_tile0;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  let ri = gpu_array_read gA.row_off m_idx;
  let re = gpu_array_read gA.row_off (m_idx +^ 1sz);

  let ri' = align_offset et ri;

  let mut nnz : sz = re -^ ri';
  let mut idx = 0sz;

  (* GM: Nota: no podemos tener una expresión de división como la
  longitud del array, porque sería un VLA. Por eso agregué un argumento
  al kernel (blockChunks) que tiene un refinamiento que asegura que
  es igual (p.blockItemsK / p.blockWidth). *)
  let mut out = [| zero #et #_; blockChunks |];
  let out0 : erased (lseq et (p.blockItemsX / p.blockWidth)) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  //------------------main-----------------------------------------

  kf_main
    p row_perm
    gA gB eA
    out
    elems_tile col_ind_tile
    bid
    ri ri' re
    tid n_idx
    nnz idx;

  //------------------residue-----------------------------------------

  kf_residue
    p row_perm
    gA gB eA
    out
    elems_tile col_ind_tile
    bid ri ri' re tid n_idx !idx !nnz;

  //------------------------------------------------------------------

  with v_out. assert out |-> v_out;
  Pulse.Lib.Array.pts_to_len out;

  foreach (p.blockItemsX /^ p.blockWidth)
    (fun x -> when__ (bcol p bid + x * p.blockWidth + tid < p.cols)
      (fun _ ->
        matrix_live_cell gC
          (brow p bid |~> row_perm)
          (bcol p bid + x * p.blockWidth + tid)
      )
    )
    (fun x -> when__ (bcol p bid + x * p.blockWidth + tid < p.cols)
        (fun _ ->
          Array2.pts_to_cell gC
            (brow p bid |~> row_perm,
             bcol p bid + x * p.blockWidth + tid)
            (v_out @! x)
        )
    )
    (store_out p row_perm gC out bid tid m_idx n_idx);

  unsparse_row_lemma
    p.rows p.shared
    elems (cast_pos col_ind) (cast_pos row_off) m_idx;

  forevery_refine_pred' #(natlt (p.blockItemsX /^ p.blockWidth))
    (fun x -> bcol p bid + x * p.blockWidth + tid < p.cols)
    (fun x _ ->
      Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + x * p.blockWidth + tid)
        (v_out @! x));

  assert pure (
    Seq.equal
      (cast_pos #(re - ri) (Seq.slice col_ind ri re))
      (Seq.slice (cast_pos col_ind) ri re)
  );
  forevery_map
    #(x : natlt (p.blockItemsX /^ p.blockWidth){
      bcol p bid + x * p.blockWidth + tid < p.cols
    })
    (fun x ->
      Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + x * p.blockWidth + tid)
        (v_out @! x)
    )
    (fun x ->
      Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + x * p.blockWidth + tid)
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + x * p.blockWidth + tid)
        )
    )
    fn x {
      Compute.compute_lemma
        p.blockWidth p.blockItemsX
        #(re - ri)
        (Seq.slice elems ri re)
        (Seq.slice (cast_pos col_ind) ri re)
        eA eB out0
        tid n_idx m_idx x;
      rewrite each (
        Compute.compute_result
          p.blockWidth p.blockItemsX
          (Seq.Base.slice elems (v ri) (v re))
          (cast_pos (lslice' col_ind (v ri) (v re)))
          eB (Seq.create (p.blockItemsX / p.blockWidth) zero)
          tid n_idx @! x
      ) as (
        MS.matmul_single eA eB m_idx (n_idx + tid + x * p.blockWidth)
      );
      ();
    };

  forevery_unrefine_pred' #(natlt (p.blockItemsX /^ p.blockWidth))
    (fun x -> bcol p bid + x * p.blockWidth + tid < p.cols)
    (fun x _ ->
      Array2.pts_to_cell gC
        (brow p bid |~> row_perm,
         bcol p bid + x * p.blockWidth + tid)
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + x * p.blockWidth + tid)
        )
    );

  assert pure (SZ.v !idx = (re - ri') / p.blockItemsK);

  ()
}
 #pop-options


inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (p : parameters et {size_req p})
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lB : Array2.layout p.shared p.cols)
  (#lC : Array2.layout p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (row_indices : gpu_array sz p.rows)
  (gB : Array2.t et lB {Array2.is_global gB})
  (gC : Array2.t et lC {Array2.is_global gC})
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (p.rows + 1)))
  (eA : ematrix et p.rows p.shared)
  // matrices densas
  (#eB : ematrix et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
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

  f = kf p row_perm blockChunks gA row_indices gB gC;

  block_pre_sendable=magic();
  block_post_sendable=magic();
  kpre_sendable=magic();
  kpost_sendable=magic();
}

inline_for_extraction noextract
fn spmm
  (#et : Type0) {| scalar et, sized et, has_vec_cpy et |}
  (rows shared cols : szp)
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {
    (k * chunk et) /? blockItemsK /\
    (k * chunk sz) /? blockItemsK /\
    k /? blockItemsX
  }))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (#lB : Array2.layout shared cols)
  (#lC : Array2.layout rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#_ : squash (aligned 16 gA.elems /\ aligned 16 gA.col_ind))
  (#fA : perm)
  (row_indices : gpu_array sz rows)
  (fri : perm)
  (gB : Array2.t et lB{Array2.is_global gB})
  (#fB : perm)
  (gC : Array2.t et lC{Array2.is_global gC})
  // matriz sparse gA
  (elems : erased (lseq et gA.nnz))
  (col_ind : erased (lseq sz gA.nnz))
  (row_off : erased (lseq sz (rows + 1)))
  (#eA : ematrix et rows shared)
  // permutacion de filas
  (row_perm : permutation (natlt rows))
  // matrices densas
  (#eB : ematrix et shared cols)
  (#eC : ematrix et rows cols)
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
  dguard (rows <^ 10000sz);
  dguard (shared <^ 10000sz);
  dguard (cols <^ 10000sz);
  dguard (blockItemsK <^ 10000sz);
  dguard (blockItemsX <^ 10000sz);
  // ^ FIXME: propagate preconditions instead of dynamically aborting
  assert pure (rows * (cols `divup` blockItemsX) <= max_blocks);
  assert pure (size_req #et ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth }));
  launch_sync (
    kdesc #et #_
      ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth })
      row_perm blockChunks #lB #lC
      gA row_indices gB gC elems col_ind row_off eA
      #eB #fA #fri #fB
  );
}
