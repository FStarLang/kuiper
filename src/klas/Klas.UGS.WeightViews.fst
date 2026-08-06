module Klas.UGS.WeightViews
#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.View
open Kuiper.VArray
open Kuiper.Injection

module SZ = Kuiper.SizeT
module Math = FStar.Math.Lemmas

let packed_cols_bound (n : nat) (c : natlt n)
  : Lemma (requires n % pair_group_n == 0)
          (ensures gate_col_of c < 2 * n /\ up_col_of c < 2 * n)
  = Math.lemma_div_mod n pair_group_n;
    Math.lemma_div_le c (n - 1) pair_group_n

let gate_col_injective (x y : nat)
  : Lemma (requires gate_col_of x == gate_col_of y)
          (ensures x == y)
  = Math.lemma_div_mod x pair_group_n;
    Math.lemma_div_mod y pair_group_n

let up_col_injective (x y : nat)
  : Lemma (requires up_col_of x == up_col_of y)
          (ensures x == y)
  = gate_col_injective x y

let gate_up_disjoint (x y : nat)
  : Lemma (ensures gate_col_of x =!= up_col_of y)
  = Math.lemma_div_mod x pair_group_n;
    Math.lemma_div_mod y pair_group_n

let packed_row_eq
  (n rx ry px py : nat)
  : Lemma (requires
      n > 0 /\ px < 2 * n /\ py < 2 * n /\
      rx * (2 * n) + px == ry * (2 * n) + py)
    (ensures rx == ry)
  = Math.lemma_div_plus px rx (2 * n);
    Math.lemma_div_plus py ry (2 * n)

let packed_disjoint
  (n rx ry px py : nat)
  : Lemma (requires
      n > 0 /\ px < 2 * n /\ py < 2 * n /\ px =!= py)
    (ensures rx * (2 * n) + px =!= ry * (2 * n) + py)
  = if t2b (rx * (2 * n) + px == ry * (2 * n) + py)
    then packed_row_eq n rx ry px py
    else ()

let gate_layout (k n : nat { n % pair_group_n == 0 }) : layout2 k n = {
  ulen = k * (2 * n);
  imap = {
    f = (fun (rc : abs (k @| n @| INil)) ->
      match rc with
      | (r, (c, ())) ->
        packed_cols_bound n c;
        (r * (2 * n) + gate_col_of c <: natlt (k * (2 * n))));
    is_inj = (fun x y ->
      match x, y with
      | (rx, (cx, ())), (ry, (cy, ())) ->
        packed_cols_bound n cx;
        packed_cols_bound n cy;
        packed_row_eq n rx ry (gate_col_of cx) (gate_col_of cy);
        gate_col_injective cx cy)
  };
}

let up_layout (k n : nat { n % pair_group_n == 0 }) : layout2 k n = {
  ulen = k * (2 * n);
  imap = {
    f = (fun (rc : abs (k @| n @| INil)) ->
      match rc with
      | (r, (c, ())) ->
        packed_cols_bound n c;
        (r * (2 * n) + up_col_of c <: natlt (k * (2 * n))));
    is_inj = (fun x y ->
      match x, y with
      | (rx, (cx, ())), (ry, (cy, ())) ->
        packed_cols_bound n cx;
        packed_cols_bound n cy;
        packed_row_eq n rx ry (up_col_of cx) (up_col_of cy);
        up_col_injective cx cy)
  };
}

let gate_layout_index
  (#k #n : nat { n % pair_group_n == 0 })
  (r : natlt k) (c : natlt n)
  : Lemma (
      (gate_layout k n).imap.f (idx2 r c) ==
      r * (2 * n) + gate_col_of c)
  = ()

let up_layout_index
  (#k #n : nat { n % pair_group_n == 0 })
  (r : natlt k) (c : natlt n)
  : Lemma (
      (up_layout k n).imap.f (idx2 r c) ==
      r * (2 * n) + up_col_of c)
  = ()

inline_for_extraction noextract
let concrete_gate_index
  (#k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  (rc : conc (k @| n @| INil))
  : out:SZ.t {
      SZ.v out ==
      (gate_layout k n).imap.f (up rc) }
  = match rc with
    | (r, (c, ())) ->
      packed_cols_bound n (SZ.v c);
      let group = c /^ 64sz in
      let lane = c %^ 64sz in
      let gate_col : szlt (2 * n) = (128sz *^ group) +^ lane in
      (r *^ (2sz *^ n)) +^ gate_col

inline_for_extraction noextract
let concrete_up_index
  (#k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  (rc : conc (k @| n @| INil))
  : out:SZ.t {
      SZ.v out ==
      (up_layout k n).imap.f (up rc) }
  = match rc with
    | (r, (c, ())) ->
      packed_cols_bound n (SZ.v c);
      let group = c /^ 64sz in
      let lane = c %^ 64sz in
      let gate_col = (128sz *^ group) +^ lane in
      let up_col : szlt (2 * n) = gate_col +^ 64sz in
      (r *^ (2sz *^ n)) +^ up_col

inline_for_extraction noextract
instance c_gate_layout
  (k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  : ctlayout (gate_layout k n) = {
    ulen_fits = ();
    all_fit = ();
    cimap = concrete_gate_index n;
  }

inline_for_extraction noextract
instance c_up_layout
  (k : erased nat { SZ.fits k })
  (n : sz { n % pair_group_n == 0 /\
            SZ.fits (2 * n) /\
            SZ.fits (k * (2 * n)) })
  : ctlayout (up_layout k n) = {
    ulen_fits = ();
    all_fit = ();
    cimap = concrete_up_index n;
  }

let gate_view (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n)
  = tensor_aview half (gate_layout k n)

let up_view (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n)
  = tensor_aview half (up_layout k n)

let gate_up_no_overlap (k n : nat { n % pair_group_n == 0 })
  : Lemma (
      no_overlap
        (gate_view k n).iview.step.imap.f
        (up_view k n).iview.step.imap.f)
  = introduce forall
      (x : (gate_view k n).iview.ait)
      (y : (up_view k n).iview.ait).
      (gate_view k n).iview.step.imap.f x =!=
      (up_view k n).iview.step.imap.f y
    with (
      match x, y with
      | (rx, (cx, ())), (ry, (cy, ())) ->
        packed_cols_bound n cx;
        packed_cols_bound n cy;
        gate_up_disjoint cx cy;
        packed_disjoint n rx ry (gate_col_of cx) (up_col_of cy)
    )

let packed_aview (k n : nat { n % pair_group_n == 0 })
  : aview half (chest2 half k n & chest2 half k n)
  = gate_up_no_overlap k n;
    sum_aview (gate_view k n) (up_view k n) #()

inline_for_extraction noextract
fn split
  (#k #n : nat { n % pair_group_n == 0 })
  (a : varray (packed_aview k n))
  (#f : perm)
  (#vGate : erased (chest2 half k n))
  (#vUp : erased (chest2 half k n))
  requires
    a |-> Frac f (reveal vGate, reveal vUp)
  returns
    p : array2 half (gate_layout k n) & array2 half (up_layout k n)
  ensures
    fst p |-> Frac f (reveal vGate) **
    snd p |-> Frac f (reveal vUp) **
    pure (Kuiper.Tensor.core (fst p) == Kuiper.VArray.core a) **
    pure (Kuiper.Tensor.core (snd p) == Kuiper.VArray.core a)
{
  gate_up_no_overlap k n;
  let gate_v, up_v = varray_split2
    (gate_view k n)
    (up_view k n)
    #()
    a
    #f
    #(reveal vGate, reveal vUp);

  varray_iconcr gate_v;
  let gate_t =
    Kuiper.Tensor.from_array
      (gate_layout k n)
      (Kuiper.VArray.core gate_v);
  Kuiper.Tensor.lem_from_array_core
    #half
    #2
    #(k @| n @| INil)
    #(gate_layout k n)
    (Kuiper.VArray.core gate_v);
  forevery_rw_type _ (abs (k @| n @| INil)) _;
  forevery_ext
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.VArray.core gate_v)
        #f
        ((gate_layout k n).imap.f i)
        ((gate_view k n).ctn.acc (reveal vGate) i))
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.Tensor.core gate_t)
        #f
        ((gate_layout k n).imap.f i)
        (acc (reveal vGate) i));
  Kuiper.Tensor.tensor_iraise gate_t #f #(reveal vGate);

  varray_iconcr up_v;
  let up_t =
    Kuiper.Tensor.from_array
      (up_layout k n)
      (Kuiper.VArray.core up_v);
  Kuiper.Tensor.lem_from_array_core
    #half
    #2
    #(k @| n @| INil)
    #(up_layout k n)
    (Kuiper.VArray.core up_v);
  forevery_rw_type _ (abs (k @| n @| INil)) _;
  forevery_ext
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.VArray.core up_v)
        #f
        ((up_layout k n).imap.f i)
        ((up_view k n).ctn.acc (reveal vUp) i))
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.Tensor.core up_t)
        #f
        ((up_layout k n).imap.f i)
        (acc (reveal vUp) i));
  Kuiper.Tensor.tensor_iraise up_t #f #(reveal vUp);

  (gate_t, up_t)
}

inline_for_extraction noextract
fn join
  (#k #n : nat { n % pair_group_n == 0 })
  (gate : array2 half (gate_layout k n))
  (up : array2 half (up_layout k n))
  (#f : perm)
  (#vGate : erased (chest2 half k n))
  (#vUp : erased (chest2 half k n))
  requires
    pure (Kuiper.Tensor.core gate == Kuiper.Tensor.core up) **
    gate |-> Frac f (reveal vGate) **
    up |-> Frac f (reveal vUp)
  returns
    a : varray (packed_aview k n)
  ensures
    a |-> Frac f (reveal vGate, reveal vUp) **
    pure (Kuiper.VArray.core a == Kuiper.Tensor.core gate)
{
  gate_up_no_overlap k n;
  Kuiper.Tensor.tensor_ilower gate;
  let gate_v =
    Kuiper.VArray.from_array
      (gate_view k n)
      (Kuiper.Tensor.core gate);
  Kuiper.VArray.lem_core_from_array
    #half
    #(chest2 half k n)
    #(gate_view k n)
    (Kuiper.Tensor.core gate);
  forevery_ext
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.Tensor.core gate)
        #f
        ((gate_layout k n).imap.f i)
        (acc (reveal vGate) i))
    (fun (i : (gate_view k n).iview.ait) ->
      pts_to_cell
        (Kuiper.VArray.core gate_v)
        #f
        ((gate_view k n).iview.step.imap.f i)
        ((gate_view k n).ctn.acc (reveal vGate) i));
  forevery_rw_type _ (gate_view k n).iview.ait _;
  varray_iabs gate_v #f #(reveal vGate);

  Kuiper.Tensor.tensor_ilower up;
  let up_v =
    Kuiper.VArray.from_array
      (up_view k n)
      (Kuiper.Tensor.core up);
  Kuiper.VArray.lem_core_from_array
    #half
    #(chest2 half k n)
    #(up_view k n)
    (Kuiper.Tensor.core up);
  forevery_ext
    (fun (i : abs (k @| n @| INil)) ->
      pts_to_cell
        (Kuiper.Tensor.core up)
        #f
        ((up_layout k n).imap.f i)
        (acc (reveal vUp) i))
    (fun (i : (up_view k n).iview.ait) ->
      pts_to_cell
        (Kuiper.VArray.core up_v)
        #f
        ((up_view k n).iview.step.imap.f i)
        ((up_view k n).ctn.acc (reveal vUp) i));
  forevery_rw_type _ (up_view k n).iview.ait _;
  varray_iabs up_v #f #(reveal vUp);

  varray_join2
    #half
    #(chest2 half k n)
    #(chest2 half k n)
    #(gate_view k n)
    #(up_view k n)
    #()
    gate_v
    up_v
    #f
    #(reveal vGate)
    #(reveal vUp)
}
