module Kuiper.Sparse.SPMM

#lang-pulse

open Kuiper
module MS = Kuiper.Spec.GEMM
module SZ = Kuiper.SizeT
module B = Kuiper.Barrier
module Compute = Kuiper.Sparse.SPMM.Compute
open Kuiper.Sparse
open Kuiper.EMatrix
open Kuiper.Math { even, odd, even_2x, odd_2x1 }
open Kuiper.Bijection { ( |~> ) }
open Kuiper.Kernel.GEMMGPU.Type { size_req_t }
open Kuiper.Sparse.SPMM.Defs
open Kuiper.Sparse.SPMM.Barrier
open Kuiper.Tensor
open Kuiper.Seq.Common { op_At_Bang }

let matrix_live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : layout2 rows cols)
  (gm : array2 et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. tensor_pts_to_cell gm (idx2 (i) (j)) v

unfold
let block_pre
  (#et : Type0) {| scalar et |}
  (p : parameters{size_req p})
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
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> matrix_live_cell
        gC (brow p bid |~> row_perm) (bcol p bid + k * p.blockWidth + tid))


unfold
let block_post
  (#et : Type0) {| scalar et |}
  (p : parameters{ size_req p })
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
  forall+ (k : natlt(p.blockItemsX /^ p.blockWidth)).
    when__
      (bcol p bid + k * p.blockWidth + tid < p.cols)
      (fun _ -> tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + k * p.blockWidth + tid))
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + k * p.blockWidth + tid)))

let barrier_contract
  (#et : Type0)
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#nnz : sz)
  (elems : lseq et nnz)
  (col_ind : lseq sz nnz)
  (row_off : lseq sz (p.rows + 1))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (#_ : squash (well_formed p col_ind row_off))
  (bid : natlt (nblocks p))
  : B.contract p.blockWidth =
  {
    rin  = barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
    rout = barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid;
  }

unfold
let kpre
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
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
  (sh : c_shmems (shmems_desc et p))
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
  (exists* (s : seq et). fst sh |-> Frac (1.0R /. p.blockWidth) s) **
  (exists* (s : seq sz). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s)

unfold
let kpost
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
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
  (sh : c_shmems (shmems_desc et p))
  (bid : natlt (nblocks p))
  (tid : natlt p.blockWidth)
  : slprop
  =
  //let (elems_tile, (col_ind_tile, _)) = sh in
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
  (#et : Type0) {| scalar et |}
  (p : parameters{size_req p})
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

  ();
}

ghost
fn block_setup
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
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
  (sh : c_shmems (shmems_desc et p ))
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
  with (x : seq _). assert fst sh |-> x;
  with (c : seq _). assert fst (snd sh) |-> c;

  Kuiper.Array.Extra.array_share (fst sh) p.blockWidth;
  forevery_map #(natlt p.blockWidth)
    (fun _ -> fst sh |-> Frac (1.0R /. p.blockWidth) x)
    (fun _ -> (exists* (s : seq _). fst sh |-> Frac (1.0R /. p.blockWidth) s))
    fn _ {};

  Kuiper.Array.Extra.array_share (fst (snd sh)) p.blockWidth;
  forevery_map #(natlt p.blockWidth)
    (fun _ -> fst (snd sh) |-> Frac (1.0R /. p.blockWidth) c)
    (fun _ -> (exists* (s : seq _). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s))
    fn _ {};

  forevery_zip3 #(natlt p.blockWidth)
    _
    (fun _ -> (exists* (s : seq _). fst sh |-> Frac (1.0R /. p.blockWidth) s))
    (fun _ -> (exists* (s : seq _). fst (snd sh) |-> Frac (1.0R /. p.blockWidth) s));

}

ghost
fn block_teardown
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
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
  (sh : c_shmems (shmems_desc et p ))
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
      pts_to (fst sh) #(1.0R /. p.blockWidth) x
    );
  with elems_tile.
    assert pts_to (fst sh) #(1.0R /. p.blockWidth) elems_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to (fst sh) #(1.0R /. p.blockWidth) elems_tile)
    (fun tid -> exists* x.
      pts_to (fst sh) #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to (fst sh) #(1.0R /. p.blockWidth) elems_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq (fst sh);
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to (fst sh) #(1.0R /. p.blockWidth) elems_tile
    );

  forevery_natlt_pop p.blockWidth
    (fun tid -> exists* x.
      pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) x
    );
  with col_ind_tile.
    assert pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) col_ind_tile;
  forevery_map_extra #(natlt (p.blockWidth - 1))
    (pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) col_ind_tile)
    (fun tid -> exists* x.
      pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) x
    )
    (fun tid ->
      pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) col_ind_tile
    )
    fn tid {
      Pulse.Lib.Array.pts_to_injective_eq (fst (snd sh));
    };
  forevery_natlt_push p.blockWidth
    (fun tid ->
      pts_to (fst (snd sh)) #(1.0R /. p.blockWidth) col_ind_tile
    );

  Kuiper.Array.Extra.array_gather (fst sh)       p.blockWidth;
  Kuiper.Array.Extra.array_gather (fst (snd sh)) p.blockWidth;

  fold_c_shmems sh (`%shmems_desc);

  ();
}

ghost
fn teardown
  (#et : Type0) {| scalar et |}
  (p : parameters{ size_req p })
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
      forevery_unfactor p.blockItemsX (p.blockItemsX /^ p.blockWidth) _
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


open Kuiper.Bijection

let natlt_refined_bij (m n : nat)
: bijection (a : natlt m {a < n}) (natlt (min m n))
= {
  ff = (fun (a : natlt m {a < n}) -> let a' : natlt (min m n) = a in a');
  gg = (fun (b : natlt (min m n)) -> b);
}

let natlt_is_between (n : nat) : Lemma (natlt n == between 0 n)
  =
  FStar.RefinementExtensionality.refext
    nat
    (fun (x:nat) -> x < n)
    (fun (x:nat) -> 0 <= x /\ x < n);
  assert (x:nat{x < n} == x:nat{0 <= x /\ x < n});
  assert (natlt n == x:nat{x < n});
  assert_norm (between 0 n == x:nat{0 <= x /\ x < n});
  ()


inline_for_extraction noextract
fn foreach
  (n : sz)
  (p q : natlt n -> slprop)
  (#frame : slprop)
  (f : (i : szlt n) -> stt unit (p i ** frame) (fun _ -> q i ** frame))
  preserves
    frame
  requires
    (forall+ (k : natlt n). p k)
  ensures
    (forall+ (k : natlt n). q k)
{
  natlt_is_between n;
  assert pure (natlt n == between 0sz n);
  forevery_rw_type (natlt n) (between 0sz n) p;
  Kuiper.For.for_loop' 0sz n
    p q
    frame
    fn x { f x };
  forevery_rw_type (between 0sz n) (natlt n) q;
}

inline_for_extraction noextract
fn sparse_load_one
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : chest2 et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (ri re : sz{ri < re /\ re <= gA.nnz})
  (idx : sz)
  (tid : szlt (p.blockWidth))
  (k : szlt (p.blockItemsK /^ p.blockWidth))
  (#_ : squash (ri + idx * p.blockItemsK + k * p.blockWidth + tid < re))
  requires
    barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k **
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA
  ensures
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k **
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA
{
  let tile_off = k *^ p.blockWidth +^ tid;
  assert rewrites_to tile_off (k *^ p.blockWidth +^ tid);

  let off = ri +^ idx *^ p.blockItemsK;
  assert rewrites_to off (ri +^ idx *^ p.blockItemsK);

  unfold barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k;
  unfold array_live_cell elems_tile;
  unfold array_live_cell col_ind_tile;

  let x = slice_read gA.elems (off +^ tile_off);
  slice_write elems_tile tile_off x;
  with s. assert pts_to_slice elems_tile tile_off (tile_off + 1) s;
  assert pure (Seq.equal s seq![elems @! off +^ tile_off]);
  assert pts_to_cell elems_tile tile_off
      (elems @! off +^ tile_off);

  let c = slice_read gA.col_ind (off +^ tile_off);
  slice_write col_ind_tile tile_off c;
  with s. assert pts_to_slice col_ind_tile tile_off (tile_off + 1) s;
  assert pure (Seq.equal s seq![col_ind @! off +^ tile_off]);
  assert pts_to_cell col_ind_tile tile_off
    (col_ind @! off +^ tile_off);

  slice_to_array gA.elems;
  slice_to_array gA.col_ind;
  fold barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k;
}

#push-options "--z3rlimit 20"
inline_for_extraction noextract
fn sparse_load
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : chest2 et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : sz)
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK + p.blockItemsK <= re))
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
  let off = ri +^ idx *^ p.blockItemsK;

  barrier_p_fold_even p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  rewrite barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid
       as (barrier_contract p row_perm elems col_ind row_off
            elems_tile col_ind_tile bid).rin (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2) tid
       as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid;

  barrier_q_unfold_even p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  foreach (p.blockItemsK /^ p.blockWidth)
    (fun ki -> barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid ki)
    (fun ki -> barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid ki)
    (fun k ->
      sparse_load_one p gA #row_off #elems #col_ind #eA elems_tile col_ind_tile
        ri re idx tid k
    );

  barrier_p_fold_odd p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  rewrite barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid
       as (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2 + 1) tid
       as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid;

  barrier_q_unfold_odd p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;


  let elems_slice   : erased (seq et) = Seq.slice elems   off (off + p.blockItemsK);
  let col_ind_slice : erased (seq sz) = Seq.slice col_ind off (off + p.blockItemsK);

  ();
}

let between_coerce_down
  (#i #j #j' : nat{i < j' /\ j' <= j})
  (k : between i j{k < j'})
: GTot (between i j')
= k

unfold
let between_coerce_up
  (#i #j #i' : nat{i <= i' /\ i' < j})
  (k : between i j{i' <= k})
: GTot (between i' j)
= k

let between_restrict_shift_down (i j j' : nat { i < j' /\ j' <= j }) (p: between i j' -> slprop) =
  forevery_refine_ext' #nat #(fun k -> i <= k /\ k < j /\ k < j')
    (fun k -> i <= k /\ k < j') (fun k -> p k)

let between_restrict_shift_up (i j i' : nat { i <= i' /\ i' < j }) (p: between i' j -> slprop) =
  forevery_refine_ext' #nat #(fun k -> i <= k /\ k < j /\ i' <= k)
    (fun k -> i' <= k /\ k < j) (fun k -> p k)

ghost
fn forevery_between_restrict_down
  (i j j' : nat{i < j' /\ j' <= j})
  (p : between i j' -> slprop)
  requires forall+ (k : between i j {k < j'}). p (between_coerce_down k)
  ensures  forall+ (k : between i j'). p k
{
  between_restrict_shift_down i j j' p;
}

ghost
fn forevery_between_restrict_up
  (i j i' : nat{i <= i' /\ i' < j})
  (p : between i' j -> slprop)
  requires forall+ (k : between i j {i' <= k}). p (between_coerce_up k)
  ensures  forall+ (k : between i' j). p k
{
  between_restrict_shift_up i j i' p;
}

ghost
fn rec gpu_forall_cell_to_slice_
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (i j : nat {i < j})
  (#v : erased (seq a))
  (#_ : squash (Seq.length v == j - i))
  requires
    (forall+ (k : between i j).
      pts_to_cell arr #f k (v @! k - i))
  ensures pts_to_slice arr #f i j v
  decreases j
{
  let j' = j - 1;
  if (j' = i) {
    forevery_singleton_elim' #(between i j) _ j';
    assert pure (Seq.equal seq![v @! 0] v);
    rewrite pts_to_slice arr #f j' (j' + 1) seq![v @! 0]
      as pts_to_slice arr #f i j v;
    ()
  } else {
    forevery_remove #(between i j)
      (fun k -> pts_to_cell arr #f k (v @! k - i))
      j';
    forevery_refine_ext #(between i j)
      (fun k -> k < j')
      (fun k -> pts_to_cell arr #f k (v @! k - i));
    forevery_between_restrict_down i j j'
      (fun k -> pts_to_cell arr #f k (v @! k - i));
    forevery_ext #(between i j')
      (fun k -> pts_to_cell arr #f k (v @! k - i))
      (fun k -> pts_to_cell arr #f k (Seq.slice v 0 (j' - i) @! k - i));
    gpu_forall_cell_to_slice_ arr i j';
    slice_concat arr #f i _ _;
    assert pure (Seq.equal (Seq.append (Seq.slice v 0 (j' - i)) seq![v @! j' - i]) v);
    rewrite pts_to_slice arr #f i (j' + 1) (Seq.append (Seq.slice v 0 (j' - i)) seq![v @! j' - i])
      as pts_to_slice arr #f i j v;
  }
}

ghost
fn gpu_forall_cell_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (i n #m : nat {i <= n /\ n <= m})
  (#v : erased (seq a))
  (#_ : squash (Seq.length v == n - i))
  requires
    (forall+ (k : between i n).
      pts_to_cell arr #f k (v @! k - i))
  preserves
    slice_live arr #f n m
  ensures pts_to_slice arr #f i n v
{
  if (i < n)
  {
    gpu_forall_cell_to_slice_ arr #f i n
  }
  else {
    forevery_elim_empty _;
    unfold slice_live;
    assert pure (Seq.empty `Seq.equal` v);

    with s. assert pts_to_slice arr #f n m s;
    assert pure (Seq.append v s `Seq.equal` s);
    assert pts_to_slice arr #f n m (Seq.append v s);
    slice_split arr #f #v #s n n m;


    fold slice_live arr #f n m;
    rewrite pts_to_slice arr #f n n v
    as pts_to_slice arr #f i n v;
  };

}

unfold
let coerce_fun (#a : Type0) (#b #c : Type{a == b}) (p : a -> c) (x : b) : c = p x

ghost
fn forevery_rw_type_ref
  (a:Type0)
  (b:Type{a == b})
  (p : a -> prop)
  (f : a -> slprop)
  requires
    forall+ (x:a{p x}). f x
  ensures
    forall+ (x:b{p x}). coerce_fun #a #b f x
{
  forevery_rw_type (x : a{p x}) (x : b{p x}) f;
}

let between_to_natlt (#m #n : nat{m <= n}) (a : between m n) : GTot (natlt (n - m)) = a - m
let natlt_to_between (#m #n : nat{m <= n}) (a : natlt (n - m)) : GTot (between m n) = a + m

let bij_between_natlt (m n : nat{m <= n})
: bijection (between m n) (natlt (n - m))
= {
  ff = between_to_natlt;
  gg = natlt_to_between;
}

instance enumerable_between (m n:nat{m <= n}) : enumerable (between m n) = {
  _cardinal = n - m;
  bij = bij_between_natlt m n;
}

ghost
fn gpu_forall_live_cell_to_slice
  (#a:Type u#0)
  (#sz:nat)
  (arr : larray a sz)
  (#f : perm)
  (i j : nat {i < j})
  requires forall+ (k : between i j).
    exists* x. pts_to_cell arr #f k x
  ensures exists* v.
    pts_to_slice arr #f i j v
{
  let y = forevery_exists #(between i j) (pts_to_cell arr #f);
  let v = Seq.init_ghost (j - i) (fun k -> y (k + i));
  forevery_ext #(between i j)
    (fun k -> pts_to_cell arr #f k (y k))
    (fun k -> pts_to_cell arr #f k (v @! k - i));
  gpu_forall_cell_to_slice_ arr i j;
}

inline_for_extraction noextract
fn sparse_load_residue
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  // matriz sparse gA
  (#row_off : lseq sz (p.rows + 1))
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#eA : chest2 et p.rows p.shared)
  (#fA : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (elems_tile : larray et p.blockItemsK)
  (col_ind_tile : larray sz p.blockItemsK)
  (bid : szlt (nblocks p))
  (ri : sz{ri == row_off @! (brow p bid |~> row_perm)})
  (re : sz{re == row_off @! (brow p bid |~> row_perm) + 1})
  (idx : sz)
  (tid : szlt p.blockWidth)
  (#_ : squash(ri + idx * p.blockItemsK <= re))
  norewrite
  preserves
    gpu **
    smatrix_pts_to' gA #fA elems col_ind row_off eA **
    B.barrier_tok (
      barrier_contract p row_perm
        elems col_ind row_off elems_tile col_ind_tile bid
    ) **
    thread_id p.blockWidth tid
  requires
    B.barrier_state (idx * 2) **
    (exists* (s : seq et). elems_tile |-> Frac (1.0R /. p.blockWidth) s) **
    (exists* (s : seq sz). col_ind_tile |-> Frac (1.0R /. p.blockWidth) s) **
    pure (
      re - (ri + idx * p.blockItemsK) < p.blockItemsK
    )
  ensures
    B.barrier_state ((idx + 1) * 2) **
    pts_to_slice elems_tile #(1.0R /. p.blockWidth)
      0 (re - (ri + idx * p.blockItemsK))
      (Seq.slice elems (ri + idx * p.blockItemsK) re) **
    pts_to_slice col_ind_tile #(1.0R /. p.blockWidth)
      0 (re - (ri + idx * p.blockItemsK))
      (Seq.slice col_ind (ri + idx * p.blockItemsK) re) **
    slice_live elems_tile #(1.0R /. p.blockWidth)
      (re - (ri + idx * p.blockItemsK)) p.blockItemsK **
    slice_live col_ind_tile #(1.0R /. p.blockWidth)
      (re - (ri + idx * p.blockItemsK)) p.blockItemsK **
    is_full_slice elems_tile p.blockItemsK **
    is_full_slice col_ind_tile p.blockItemsK
{

  let off = ri +^ idx *^ p.blockItemsK;

  barrier_p_fold_even p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  rewrite barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid
       as (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2) tid
       as barrier_q p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2) tid;

  barrier_q_unfold_even p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  let tresidue : sz = (re -^ off +^ (p.blockWidth -^ 1sz) -^ tid) /^ p.blockWidth;

  forevery_refine_split #(natlt (p.blockItemsK /^ p.blockWidth))
    (barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid)
    (fun k ->
      k < SZ.v tresidue);

  forevery_natlt_restrict #tresidue
    (p.blockItemsK /^ p.blockWidth)
    (fun (k : natlt tresidue) ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid (natlt_coerce k));

  foreach tresidue
    (fun (ki : natlt tresidue) ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid (natlt_coerce ki))
    (fun (ki : natlt tresidue) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid (natlt_coerce ki))
    #(
      gpu **
      smatrix_pts_to' gA #fA elems col_ind row_off eA **
      B.barrier_tok (
        barrier_contract p row_perm
          elems col_ind row_off elems_tile col_ind_tile bid
      ) **
      thread_id p.blockWidth tid
    )
    fn (k : szlt tresidue)
    {
      let k1 : sz = k;
      assert rewrites_to k1 k;
      sparse_load_one p gA #row_off #elems #col_ind #eA elems_tile col_ind_tile
        ri re idx tid k1;
    };

  forevery_natlt_extend (p.blockItemsK /^ p.blockWidth)
    (fun (k : natlt tresidue) ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid (natlt_coerce k)
    );

  forevery_ext #(k : natlt (p.blockItemsK /^ p.blockWidth){k < tresidue})
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid
        (natlt_coerce (natlt_coerce #tresidue k))
    )
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    );

  forevery_map #(k : natlt (p.blockItemsK /^ p.blockWidth) {~(k < tresidue)})
    (fun k ->
      barrier_q_even p gA.nnz elems_tile col_ind_tile ri re idx tid k
    )
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    )
    fn k {
      unfold barrier_q_even;
      unfold array_live_cell elems_tile;
      unfold array_live_cell col_ind_tile;
      fold barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k;
    };

  forevery_refine_join
    (fun k ->
      barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
    )
    (fun k -> k < tresidue) (fun k -> ~(k < tresidue));

  forevery_unrefine (fun k ->
    barrier_p_odd p elems col_ind elems_tile col_ind_tile ri re idx tid k
  );

  barrier_p_fold_odd p row_perm elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  rewrite barrier_p p row_perm elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid
       as (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rin
            (idx * 2 + 1) tid;

  B.barrier_wait ();

  rewrite (barrier_contract p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid).rout
            (idx * 2 + 1) tid
       as barrier_q p row_perm
            elems col_ind row_off elems_tile col_ind_tile bid
            (idx * 2 + 1) tid;

  barrier_q_unfold_odd_residue p row_perm
    elems col_ind row_off elems_tile col_ind_tile
    bid ri re idx tid;

  forevery_refine_split
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile ri re idx)
    (fun (k : natlt p.blockItemsK) -> k < re - off);

  let elems_slice   : erased (seq et) = Seq.slice elems   off re;
  let col_ind_slice : erased (seq sz) = Seq.slice col_ind off re;

  // el residuo
  forevery_map #(k : natlt p.blockItemsK {k < re - off})
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile
      ri re idx)
    (fun k ->
      pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k)
    )
    fn k { unfold barrier_q_odd };

  forevery_natlt_restrict #(re - off) p.blockItemsK
    (fun k ->
      pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k)
    );

  natlt_is_between (re - off);
  forevery_rw_type (natlt (re - off)) (between 0 (re - off)) _;

  forevery_ext #(between 0 (re - off))
    (fun k ->
      pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k) **
      pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k))
    (fun k ->
      pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k - 0) **
      pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k - 0));

  forevery_unzip #(between 0 (re - off))
    (fun k -> pts_to_cell elems_tile #(1.0R /. p.blockWidth) k
        (elems_slice @! k - 0))
    (fun k -> pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k
        (col_ind_slice @! k - 0));


  // el resto
  forevery_map #(k : natlt p.blockItemsK {~(k < re - off)})
    (barrier_q_odd p elems col_ind elems_tile col_ind_tile
      ri re idx)
    (fun k ->
      (exists* x. pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    )
    fn k { unfold barrier_q_odd };


  natlt_is_between p.blockItemsK;
  forevery_rw_type_ref
    (natlt p.blockItemsK)
    (between 0 p.blockItemsK)
    (fun (k : natlt p.blockItemsK) -> ~(k < re - off))
    (fun (k : natlt p.blockItemsK) ->
      (exists* x. pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    );


  forevery_refine_ext #(between 0 p.blockItemsK)
    (fun k -> re - off <= k)
    (fun k ->
      (exists* x. pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c)
    );

  forevery_between_restrict_up 0 p.blockItemsK (re - off)
    (fun k ->
      (exists* x. pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x) **
      (exists* c. pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c));

  forevery_unzip #(between (re - off) p.blockItemsK)
    (fun k ->
      (exists* x. pts_to_cell elems_tile   #(1.0R /. p.blockWidth) k x))
    (fun k ->
      (exists* c. pts_to_cell col_ind_tile #(1.0R /. p.blockWidth) k c));

  gpu_forall_live_cell_to_slice elems_tile   (re - off) p.blockItemsK;
  gpu_forall_live_cell_to_slice col_ind_tile (re - off) p.blockItemsK;

  fold slice_live elems_tile   #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;
  fold slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;

  gpu_forall_cell_to_slice elems_tile 0 (re - off);
  gpu_forall_cell_to_slice col_ind_tile 0 (re - off);

  (* [elems_tile]/[col_ind_tile] : larray _ p.blockItemsK are full; materialize
     [is_full_slice] from the residue slice, which grounds frame inference. *)
  unfold slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;
  add_full_slice elems_tile (re - off) p.blockItemsK p.blockItemsK;
  fold slice_live elems_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;
  add_full_slice col_ind_tile (re - off) p.blockItemsK p.blockItemsK;
  fold slice_live col_ind_tile #(1.0R /. p.blockWidth) (re - off) p.blockItemsK;

  rewrite each off as (ri +^ idx *^ p.blockItemsK);

  ();
}
#pop-options

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
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lC |}
  (gC : array2 et lC)
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
        tensor_pts_to_cell gC
          (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
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

    tensor_write_cell gC ((m_idx <: szlt _), ((n_idx +^ x *^ p.blockWidth +^ tid <: szlt _), ())) c;

    assert tensor_pts_to_cell gC
      (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
      (v_out @! x);
    when__intro_true (bcol p bid + x * p.blockWidth + tid < p.cols)
      (tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (v_out @! x)
      );
  }
  else {
    rewrite when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      matrix_live_cell gC
        (brow p bid |~> row_perm)
        (bcol p bid + x * p.blockWidth + tid)
    ) as when__ (bcol p bid + x * p.blockWidth + tid < p.cols) (fun _ ->
      tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (v_out @! x)
    );
  };

}

noextract
let barrier_count
  (p : parameters { size_req p })
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
  ((re - ri) / p.blockItemsK + 1) * 2

#push-options "--z3rlimit 15"

inline_for_extraction noextract
fn kf
  (#et : Type0) {| scalar et |}
  (p : parameters { size_req p })
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lb : layout2 p.shared p.cols)
  (#lc : layout2 p.rows p.cols)
  {| ctlayout lb, ctlayout lc |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared))
  (row_indices : larray sz p.rows)
  (gB : array2 et lb)
  (gC : array2 et lc)
  // matriz sparse ga
  (#elems : lseq et gA.nnz)
  (#col_ind : lseq sz gA.nnz)
  (#row_off : lseq sz (p.rows + 1))
  (#eA : chest2 et p.rows p.shared)
  // matriz densa gb
  (#eB : chest2 et p.shared p.cols)
  (#fA #fri #fB : perm)
  (#_ : squash (well_formed p col_ind row_off))
  (sh : c_shmems (shmems_desc et p))
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
  let m_idx = slice_read row_indices (brow_ p bid);
  assert rewrites_to m_idx (SZ.uint_to_t (brow p bid |~> row_perm));
  let n_idx = bcol_ p bid;

  let (elems_tile0, (col_ind_tile0, _)) = sh;

  pts_to_len elems_tile0;
  pts_to_len col_ind_tile0;

  (* This incantation here improves the generated code by actually defining
  these variables at this point. *)
  let elems_tile   = elems_tile0;     assert rewrites_to elems_tile   elems_tile0;
  let col_ind_tile = col_ind_tile0;   assert rewrites_to col_ind_tile col_ind_tile0;

  assert rewrites_to elems_tile (fst sh);
  assert rewrites_to col_ind_tile (fst (snd sh));

  Pulse.Lib.Array.pts_to_len elems_tile;
  Pulse.Lib.Array.pts_to_len col_ind_tile;

  let ri = slice_read gA.row_off m_idx;
  let re = slice_read gA.row_off (m_idx +^ 1sz);

  let row_elems : lseq et (re - ri) = hide (Seq.slice elems ri re);
  let row_pos : lseq nat (re - ri) = hide (Seq.slice (cast_pos col_ind) ri re);

  assert pure (valid_pos #(re - ri) p.shared row_pos);

  (* GM: Nota: no podemos tener una expresión de división como la
  longitud del array, porque sería un VLA. Por eso agregué un argumento
  al kernel (blockChunks) que tiene un refinamiento que asegura que
  es igual (p.blockItemsK / p.blockWidth). *)
  let mut out = [| zero #et #_; blockChunks |];
  let out0 : lseq et (p.blockItemsX / p.blockWidth) =
    Seq.create (p.blockItemsX / p.blockWidth) zero;

  let mut nnz : sz = re -^ ri;
  let mut idx = 0sz;

  assert pure (SZ.fits (re - ri));
  assert pure (SZ.fits ((re - ri) / p.blockItemsK));

  assert pure (ri == row_off @! (brow p bid |~> row_perm));
  assert pure (re == row_off @! (brow p bid |~> row_perm) + 1);

  assert pure (
    Seq.equal
      out0
      (Compute.compute_result
        p.blockWidth p.blockItemsX #0
        (Seq.slice row_elems 0 0)
        (Seq.slice row_pos 0 0)
        eB out0 tid n_idx)
  );

  assert is_full_slice elems_tile   p.blockItemsK;
  assert is_full_slice col_ind_tile p.blockItemsK;

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
          !idx <= (re - ri) / p.blockItemsK /\
          SZ.v !nnz == re - ri - SZ.v !idx * p.blockItemsK /\
          v_out ==
          Compute.compute_result
            p.blockWidth p.blockItemsX #(!idx * p.blockItemsK)
            (Seq.slice row_elems 0 (!idx * p.blockItemsK))
            (Seq.slice row_pos 0 (!idx * p.blockItemsK))
            eB out0 tid n_idx
        )
      )
    decreases SZ.v !nnz
  {
    assert pure (ri + (!idx + 1) * p.blockItemsK <= gA.nnz);
    assert pure (!idx < (re - ri) / p.blockItemsK);

    slice_to_array gA.row_off;
    slice_to_array elems_tile;
    slice_to_array col_ind_tile;
    sparse_load p row_perm gA #row_off #elems #col_ind #eA
      elems_tile col_ind_tile bid ri re !idx tid #();

    Seq.Properties.slice_slice elems ri re
      (!idx * p.blockItemsK) ((!idx + 1) * p.blockItemsK);

    assert pure (
      Seq.equal
        (Seq.slice elems
          (ri + !idx * p.blockItemsK)
          (ri + (!idx + 1) * p.blockItemsK))
        (Seq.slice row_elems
          (!idx * p.blockItemsK)
          ((!idx + 1) * p.blockItemsK))
    );

    Seq.Properties.slice_slice (cast_pos col_ind) ri re
      (!idx * p.blockItemsK) ((!idx + 1) * p.blockItemsK);

    assert pure (
      Seq.equal
        (cast_pos #p.blockItemsK (
          Seq.slice col_ind
            (ri + !idx * p.blockItemsK)
            (ri + (!idx + 1) * p.blockItemsK)
        ))
        (Seq.slice row_pos (!idx * p.blockItemsK)
          ((!idx + 1) * p.blockItemsK))
    );

    Pulse.Lib.Array.pts_to_len out;

    Compute.compute
      p.blockWidth p.blockItemsK p.blockItemsX
      elems_tile col_ind_tile p.blockItemsK gB out tid n_idx;

    Compute.compute_step
      p.blockWidth p.blockItemsX
      row_elems row_pos eB out0 tid n_idx
      (!idx * p.blockItemsK) ((!idx + 1) * p.blockItemsK);

    idx := !idx +^ 1sz;
    nnz := !nnz -^ p.blockItemsK;

    slice_to_array gA.row_off;
    slice_to_array (fst sh);
    slice_to_array (fst (snd sh));
    ()
  };

  Pulse.Lib.Array.pts_to_len elems_tile;
  Pulse.Lib.Array.pts_to_len col_ind_tile;
  assert is_full_slice elems_tile   p.blockItemsK;
  assert is_full_slice col_ind_tile p.blockItemsK;

  //------------------residue-----------------------------------------
  assert pure (ri + !idx * p.blockItemsK <= re);
  assert pure (re - (ri + !idx * p.blockItemsK) < p.blockItemsK);


  // let residue : sz = re -^ (ri +^ !idx *^ p.blockItemsK);
  (* ^ This is provably equal to the current value of nnz, using that
  to avoid the extra computation. *)

  slice_to_array gA.row_off;

  sparse_load_residue p row_perm gA #row_off #elems #col_ind #eA
    elems_tile col_ind_tile bid ri re !idx tid;

  assert pure (
    Seq.equal
      (Seq.slice elems (ri + !idx * p.blockItemsK) re)
      (Seq.slice row_elems
        (!idx * p.blockItemsK)
        (!idx * p.blockItemsK + !nnz)
      )
  );
  assert pure (
    Seq.equal
      (cast_pos #!nnz (Seq.slice col_ind (ri + !idx * p.blockItemsK) re))
      (Seq.slice row_pos
        (!idx * p.blockItemsK)
        (!idx * p.blockItemsK + !nnz)
      )
  );

  Compute.compute
    p.blockWidth p.blockItemsK p.blockItemsX
    elems_tile col_ind_tile !nnz gB out tid n_idx;
  Compute.compute_step
    p.blockWidth p.blockItemsX
    row_elems row_pos eB out0 tid n_idx
    (!idx * p.blockItemsK) (!idx * p.blockItemsK + !nnz );

  assert out |->
    Compute.compute_result
      p.blockWidth p.blockItemsX
      row_elems row_pos eB out0 tid n_idx;


  unfold slice_live elems_tile #(1.0R /. p.blockWidth)
    (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;
  slice_concat elems_tile #(1.0R /. p.blockWidth)
    0 (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;

  unfold slice_live col_ind_tile #(1.0R /. p.blockWidth)
    (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;
  slice_concat col_ind_tile #(1.0R /. p.blockWidth)
    0 (re - (ri + !idx * p.blockItemsK)) p.blockItemsK;

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
          tensor_pts_to_cell gC
            (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
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
      tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (v_out @! x));

  forevery_map
    #(x : natlt (p.blockItemsX /^ p.blockWidth){
      bcol p bid + x * p.blockWidth + tid < p.cols
    })
    (fun x ->
      tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (v_out @! x)
    )
    (fun x ->
      tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + x * p.blockWidth + tid)
        )
    )
    fn x {
      assert tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (Compute.compute_result
          p.blockWidth p.blockItemsX
          row_elems row_pos
          eB out0
          tid n_idx @! x
        );
      Compute.compute_lemma
        p.blockWidth p.blockItemsX
        row_elems row_pos
        eA eB out0
        tid n_idx m_idx x;
      rewrite each (
        Compute.compute_result
          p.blockWidth p.blockItemsX
          row_elems row_pos
          eB out0
          tid n_idx @! x
      ) as (
        MS.matmul_single eA eB m_idx (n_idx + tid + x * p.blockWidth)
      );
      ();
    };

  forevery_unrefine_pred' #(natlt (p.blockItemsX /^ p.blockWidth))
    (fun x -> bcol p bid + x * p.blockWidth + tid < p.cols)
    (fun x _ ->
      tensor_pts_to_cell gC
        (idx2 (brow p bid |~> row_perm) (bcol p bid + x * p.blockWidth + tid))
        (MS.matmul_single eA eB
          (brow p bid |~> row_perm)
          (bcol p bid + x * p.blockWidth + tid)
        )
    );

  assert pure (SZ.v !nnz == re - ri - SZ.v !idx * p.blockItemsK);
  assert pure (SZ.v !nnz < p.blockItemsK);
  assert pure (SZ.v !idx <= (re - ri) / p.blockItemsK);
  FStar.Math.Lemmas.small_div (SZ.v !nnz) p.blockItemsK;
  FStar.Math.Lemmas.lemma_div_mod_plus (SZ.v !nnz) (SZ.v !idx) p.blockItemsK;
  assert pure (SZ.v !idx == (re - ri) / p.blockItemsK);

  slice_to_array row_indices;

  assert is_full_slice (fst sh) p.blockItemsK;
  assert is_full_slice (fst (snd sh)) p.blockItemsK;
  slice_to_array (fst sh);
  slice_to_array (fst (snd sh));

  ()
}
#pop-options

inline_for_extraction noextract
let kdesc
  (#et : Type0) {| scalar et |}
  (p : parameters{size_req p})
  (row_perm : permutation (natlt p.rows))
  (blockChunks : sz{SZ.v blockChunks == p.blockItemsX / p.blockWidth}) // Ver nota abajo
  (#lB : layout2 p.shared p.cols)
  (#lC : layout2 p.rows p.cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v p.rows) (SZ.v p.shared){is_global_smatrix gA})
  (row_indices : larray sz p.rows)
  (gB : array2 et lB {is_global gB})
  (gC : array2 et lC {is_global gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (p.rows + 1))
  (eA : chest2 et p.rows p.shared)
  // matrices densas
  (#eB : chest2 et p.shared p.cols)
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
  nthr = nthreads p;

  barrier_contract = (fun bid ptrs ->
    barrier_contract p row_perm elems col_ind row_off
      (fst ptrs) (fst (snd ptrs)) bid);
  barrier_count = (fun bid -> barrier_count p row_perm col_ind row_off bid);
  barrier_ok = (fun bid ptrs -> magic());

  shmems_desc = shmems_desc et p;

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
  (#et : Type0) {| scalar et |}
  (rows shared cols : szp)
  (blockItemsK : szp)
  (blockItemsX : szp)
  (blockWidth : (k : szp {k /? blockItemsK /\ k /? blockItemsX}))
  (blockChunks : sz{SZ.v blockChunks == blockItemsX / blockWidth}) // Ver nota abajo
  (#lB : layout2 shared cols)
  (#lC : layout2 rows cols)
  {| ctlayout lB, ctlayout lC |}
  (gA : smatrix et (SZ.v rows) (SZ.v shared){is_global_smatrix gA})
  (#fA : perm)
  (row_indices : larray sz rows)
  (fri : perm)
  (gB : array2 et lB{is_global gB})
  (#fB : perm)
  (gC : array2 et lC{is_global gC})
  // matriz sparse gA
  (elems : lseq et gA.nnz)
  (col_ind : lseq sz gA.nnz)
  (row_off : lseq sz (rows + 1))
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
  dguard (rows <^ 10000sz);
  dguard (shared <^ 10000sz);
  dguard (cols <^ 10000sz);
  dguard (blockItemsK <^ 10000sz);
  dguard (blockItemsX <^ 10000sz);
  // ^ FIXME: propagate preconditions instead of dynamically aborting
  assert pure (rows * (cols `divup` blockItemsX) <= max_blocks);
  assert pure (size_req ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth }));
  launch_sync (
    kdesc #et #_
      ({ rows; shared; cols; blockItemsK; blockItemsX; blockWidth })
      row_perm blockChunks #lB #lC
      gA row_indices gB gC elems col_ind row_off eA
      #eB #fA #fri #fB
  );
}
