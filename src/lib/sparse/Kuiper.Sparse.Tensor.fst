module Kuiper.Sparse.Tensor

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Array2.Strided { strided_row_major, aligned_strided_row_major }
open FStar.Tactics.Typeclasses { no_method }
open Kuiper.Array.Vectorized
open Kuiper.Seq.Common { (@!) }
open Kuiper.Tensor.Layout.Slice
open Kuiper.Array2.Strided { cell_of_pos }

module Math = Kuiper.Sparse.Math
module SZ = FStar.SizeT
module Chest = Kuiper.Chest

#push-options "--z3rlimit 20"

let aligned_cell_strided_row_major
  (#et:Type0) {| sized et, has_vec_cpy et |}
  (#rows #cols : pos { chunk et /? cols })
  (#l : layout2 rows cols) {| strided : strided_row_major l |}
  (gm : array2 et l)
  (i : natlt rows)
  (j : natlt cols { chunk et /? j })
: Lemma
  (requires
    aligned 16 (core gm) /\
    aligned_strided_row_major (chunk et) strided
  )
  (ensures aligned' 16 (core gm) (cell_of_pos l i j))
=
  strided.pf i j;
  Math.lineal_divides (chunk et) strided.offset strided.stride i;
  lemma_divides_sum (chunk et) (strided.offset + strided.stride * i) j;
  ()
#pop-options

instance cont_layout_l1_forward (#n : erased nat)
  : cont_layout (l1_forward n) = {
    offset  = 0sz;
    pf = ez
  }

instance cont_layout_strided_row_major
  (#rows #cols : erased nat)
  (l : layout2 rows cols { 0 < cols }) {| srm : strided_row_major l |}
  (#_: squash (fits (tlayout_ulen l)))
  (i : natlt rows)
  {| conc_i : concrete_sz i |}
  : cont_layout #cols (tlayout_slice l 0 i) = {
    offset  = (srm.pf i 0; srm.offset +^ srm.stride *^ (concr' conc_i));
    pf = srm.pf i;
  }

let aligned_cont_strided_row_major
  (#rows #cols : erased nat { 0 < cols })
  (l : layout2 rows cols) {| srm : strided_row_major l |}
  (#_: squash (fits (tlayout_ulen l)))
  (k : pos)
  (i : szlt rows)
: Lemma
  (requires aligned_strided_row_major k srm)
  (ensures aligned_cont_layout k (cont_layout_strided_row_major l i))
=
  srm.pf i 0;
  Math.lineal_divides k srm.offset srm.stride i

open Kuiper.Bijection

let prod_unit_bij (a : Type) : (a & unit =~ a) =
{
  ff = (fun (x, ()) -> x);
  gg = (fun x -> (x, ()));

  ff_gg = ez;
  gg_ff = ez;
}

ghost
fn forevery_abs1_iso
  (#n : nat)
  (p : abs (n @| INil) -> slprop)
  requires forall+ (i : abs (n @| INil)). p i
  ensures  forall+ (i : natlt n). p (idx1 i)
{
  forevery_iso (abs_bring_forward_bij 0 (n @| INil)) _;
  rewrite each ((n @| INil) @! 0) as n;
  rewrite each abs (modulo_i 0 (n @| INil)) as unit;
  forevery_iso (prod_unit_bij (natlt n)) _;
  forevery_ext _ (fun i -> p (idx1 i));
}

ghost
fn forevery_abs1_iso_back
  (#n : nat)
  (p : abs (n @| INil) -> slprop)
  requires forall+ (i : natlt n). p (idx1 i)
  ensures  forall+ (i : abs (n @| INil)). p i
{
  forevery_iso (bij_sym (prod_unit_bij (natlt n))) _;
  rewrite each n as ((n @| INil) @! 0);
  rewrite each unit as abs (modulo_i 0 (n @| INil));
  forevery_iso (bij_sym (abs_bring_forward_bij 0 (n @| INil))) _;
  forevery_ext _ p;
}

ghost
fn lower_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (#f : perm)
  (#s : chest1 et sz)
  requires a |-> Frac f s
  ensures  pts_to_slice (core a) #f cl.offset (cl. offset + sz) (chest1_to_seq s)
{
  tensor_explode a;
  let s' = chest1_to_seq s;

  forevery_abs1_iso _;

  forevery_map #(natlt sz)
    (fun i -> Cell a (idx1 i) |-> Frac f (acc s (idx1 i)))
    (fun i -> Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s' i))
    fn i {
      cl.pf i;
      tensor_pts_to_cell_eq a (idx1 i) f (acc s (idx1 i));
      rewrite  Cell a (idx1 i) |-> Frac f (acc s (idx1 i))
      as pts_to_cell (core a) #f (cl.offset + i) (Seq.index s' i);
    };
  cl.pf (sz - 1);
  forevery_rw_size sz ((cl.offset + sz) - cl.offset);
  cells_to_nonempty_slice (core a) cl.offset (cl.offset + sz);
}

ghost
fn raise_cont
  (#et : Type u#0)
  (#sz : pos)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (#f : perm)
  (#s : lseq et sz)
  requires pure (fits (tlayout_ulen l))
  requires pts_to_slice (core a) #f cl.offset (cl. offset + sz) s
  ensures  a |-> Frac f (seq_to_chest1 s)
{
  let s' = seq_to_chest1 s;
  cl.pf (sz - 1);
  slice_to_cells (core a) cl.offset (cl.offset + sz);

  forevery_rw_size ((cl.offset + sz) - cl.offset) sz;
  forevery_map #(natlt sz)
    (fun i ->
      Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s i))
    (fun i -> Cell a (idx1 i) |-> Frac f (acc s' (idx1 i)))
    fn i {
      cl.pf i;
      tensor_pts_to_cell_eq a (idx1 i) f (Seq.index s i);
      rewrite Cell (core a <: array et) (cl.offset + i <: nat) |-> Frac f (Seq.index s i)
      as  Cell a (idx1 i) |-> Frac f (acc s' (idx1 i));
    };
  forevery_abs1_iso_back #sz
    (fun i -> tensor_pts_to_cell a #f i (acc s' i));
  tensor_implode_with_exists a;
}

let aligned_cont_offset
  (#et : Type u#0) {| sized et, has_vec_cpy et |}
  (#sz : erased nat)
  (#l : layout1 sz) {| cl : cont_layout l |}
  (a : array1 et l)
  (off : erased nat { chunk et /? off })
: Lemma
  (requires
    aligned 16 (core a) /\
    aligned_cont_layout (chunk et) cl
  )
  (ensures aligned' 16 (core a) (cl.offset + off))
= ()
open Pulse.Lib.Trade { (@==>) }

let chest2_upd_row
  (#et : Type0)
  (#rows #cols : erased nat)
  (em : chest2 et rows cols)
  (i : natlt rows)
  (new_row : chest1 et cols)
  : chest2 et rows cols
  = mk2 fun i' j ->
      if i' = i
      then acc new_row (idx1 j)
      else acc2 em i' j

ghost
fn tensor_extract_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    tensor_row a i |-> Frac f (chest2_row s i) **
    (forall* (s' : chest1 et cols).
      tensor_row a i |-> Frac f s' @==>
      a |-> Frac f (chest2_upd_row s i s'))
{
  tensor_extract_slice a 0 i;
  rewrite each sliceof a 0 i as tensor_row a i;
  rewrite each chest (modulo_i 0 (ICons rows (ICons cols INil))) et as chest1 et cols;
  rewrite each modulo_i 0 (rows @| (cols @| INil)) as (cols @| INil);

  Pulse.Lib.Forall.intro_forall
    #_
    #(fun (s' : chest1 et cols) ->
      tensor_row a i |-> Frac f s' @==>
      a |-> Frac f (chest2_upd_row s i s'))
    (forall* (s' : chest1 et cols).
      tensor_row a i |-> Frac f s' @==>
      a |-> Frac f (chest_update_slice 0 i s s'))
    fn s'{
      Pulse.Lib.Forall.elim_forall s';
      rewrite each chest_update_slice 0 i s s' as chest2_upd_row s i s';
    };
}

ghost
fn tensor_extract_row_ro
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    a |-> Frac f s
  ensures
    factored
      (tensor_row a i |-> Frac f (chest2_row s i))
      (a |-> Frac f s)
{
  tensor_extract_row a i;
  Pulse.Lib.Forall.elim_forall (chest_slice 0 i s <: chest1 et cols);
  assert pure (
    equal
      (chest2_upd_row s i (chest2_row s i))
      s
  );
}

ghost
fn tensor_restore_row
  (#et : Type0)
  (#rows #cols : nat)
  (#l : layout2 rows cols)
  (a : array2 et l)
  (i : natlt rows)
  (#f : perm)
  (#s : chest2 et rows cols)
  requires
    factored
      (tensor_row a i |-> Frac f (chest_slice 0 i s))
      (a |-> Frac f s)
  ensures
    a |-> Frac f s
{
  Pulse.Lib.Trade.elim_trade _ _;
}