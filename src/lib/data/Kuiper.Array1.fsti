module Kuiper.Array1

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Injection
open Kuiper.Index
open FStar.Tactics.Typeclasses { no_method }
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Pulse.Lib.Trade
module B = Kuiper.Array
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let desc (len : nat) : idesc 1 =
  len @| INil

// Even if this is trivial, it seems to help in some contexts.
let sizeof_desc (rows : nat)
  : Lemma (sizeof (desc rows) == rows)
          [SMTPat (sizeof (desc rows))]
  = ()

let ait (len : nat) = natlt len

let adapt_idx (#len : nat) (idx : abs (desc len)) : ait len =
  match idx with
  | (i, ()) -> i

let adapt_idx_back (#len : nat) (idx : ait len) : abs (desc len) =
  (idx, ())

let raw_cit = sz

let cit_fits (len : nat) (idx : raw_cit) : prop =
  idx < len

[@@erasable]
type layout (len : nat) = tlayout (desc len)

type full_layout (len : nat) = l : layout len { is_full l }

let layout_size (#len : nat) (l : layout len) : GTot nat = l.ulen

(* From an underlying sequence to the viewed one. *)
let from_seq (#et:Type) (#len : nat) (l : full_layout len) (s : lseq et len)
  : GTot (lseq et len)
  = Seq.init_ghost len (fun i -> Seq.index s (l.imap.f (adapt_idx_back i)))

let to_seq (#et:Type) (#len : nat) (l : full_layout len) (s : lseq et len)
  : GTot (lseq et len)
  = Seq.init_ghost len (fun i ->
      let x = Kuiper.Injection.inverse_f l.imap i in
      Seq.index s x._1)

val to_from (#et:Type) (#len : nat) (l : full_layout len) (s : lseq et len)
  : Lemma (to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]

inline_for_extraction noextract
val t (et : Type0) (#len : nat) (l : layout len) : Type0

unfold let array1 = t

val is_global (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  : prop

inline_for_extraction noextract
val from_array
  (#et : Type0) (#len : erased nat)
  (l : layout len)
  (a : larray et (layout_size l))
  : t et l

inline_for_extraction noextract
val core
  (#et : Type0) (#len : erased nat) (#l : layout len)
  (a : t et l)
  : larray et (layout_size l)

val lem_core_from_array
  (#et : Type) (#len : erased nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]

val lem_from_array_core
  (#et : Type) (#len : erased nat)
  (l : layout len)
  (p : larray et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]

val lem_is_global_iff_core
  (#et : Type0) (#len : nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]

val pts_to
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop

instance
val is_send_across_global
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l { is_global a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (pts_to a #f s)

unfold
instance has_pts_to_inst (et : Type) (len : erased nat) (l : _)
  : has_pts_to (t et l) (lseq et len)
  = { pts_to }

inline_for_extraction noextract
fn alloc0
  (#et:Type) {| sized et |}
  (len : szp)
  (l : layout len { is_full l })
  preserves
    cpu
  returns
    p : t et l
  ensures
    exists* em. on gpu_loc (p |-> em)
  ensures
    pure (is_global p) **
    pure (is_full_array (core p))

inline_for_extraction noextract
fn free
  (#et:Type)
  (#len : erased nat)
  (#l : layout len { is_full l })
  (p : t et l)
  (#em : erased (lseq et len))
  preserves
    cpu
  requires
    on gpu_loc (p |-> em) **
    pure (is_full_array (core p))
  ensures
    emp

ghost
fn pts_to_ref
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm) (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))

ghost
fn lower
  (#et : Type) (#len : nat)
  (#l : full_layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    a |-> Frac f s
  ensures
    core a |-> Frac f (to_seq l s)

ghost
fn raise
  (#et : Type) (#len : nat)
  (l : full_layout len)
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : lseq et len)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s

ghost
fn raise'
  (#et : Type) (#len : nat)
  (l : full_layout len)
  (p : larray et (layout_size l))
  (#f : perm)
  (#s : lseq et len)
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)

ghost
fn share_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s

ghost
fn gather_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s

inline_for_extraction noextract
fn read
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (#f : perm)
  (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  returns
    v : et
  ensures
    pure (v == Seq.index s i)

inline_for_extraction noextract
fn write
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (v : et)
  (#s : erased (lseq et len))
  requires
    a |-> s
  ensures
    a |-> (Seq.upd s i v <: lseq et len)

val pts_to_cell
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] i : ait len)
  (v : et)
  : slprop

[@@pulse_unfold; FStar.Tactics.Typeclasses.noinst]
instance cell_pts_to (#et : Type) (#len : nat) (#l : layout len)
  : has_pts_to (cell (t et l) (ait len)) et
= {
  pts_to = (fun (Cell ar i) #f v -> pts_to_cell ar #f i v);
}

val pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l) (i : ait len) (f : perm) (v : et)
  : Lemma (Cell a i |-> Frac f v
           ==
           B.pts_to_cell (core a) #f (l.imap.f (adapt_idx_back i)) v)

instance
val is_send_across_global_cell
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l { is_global a })
  (#f : perm) (i : ait len) (v : et)
  : is_send_across gpu_of (pts_to_cell a #f i v)

ghost
fn explode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires a |-> Frac f s
  ensures
    forall+ (i : ait len).
      Cell a i |-> Frac f (Seq.index s i)

ghost
fn implode
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm)
  (#s : lseq et len)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (i : ait len).
      Cell a i |-> Frac f (Seq.index s i)
  ensures
    a |-> Frac f s

ghost
fn extract_cell
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (i : natlt len)
  (#f : perm)
  (#s : lseq et len)
  requires
    a |-> Frac f s ** 
    pure (SZ.fits (layout_size l))
  ensures
    Cell a i |-> Frac f (Seq.index s i) ** 
    (forall* (si': et).   
      Cell a i |-> Frac f si' @==> a |-> Frac f (Seq.upd s i si' <: (lseq et len)))

// Just an elim_trade wrapper
ghost
fn restore_cell
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  (i : natlt len)
  (#f : perm)
  (#si': et)
  (#s : lseq et len)
  requires
    Cell a i |-> Frac f si' **
    (forall* (si': et).   
      Cell a i |-> Frac f si' @==> a |-> Frac f (Seq.upd s i si' <: (lseq et len)))
  ensures
    a |-> Frac f (Seq.upd s i si' <: (lseq et len))

inline_for_extraction noextract
fn read_cell
  (#et : Type0) (#len : erased nat)
  (#l : layout len ) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (#f : perm)
  (#s : erased et)
  preserves
    Cell a (i <: natlt len) |-> Frac f s
  returns
    v : et
  ensures
    pure (v == s)

inline_for_extraction noextract
fn write_cell
  (#et : Type0) (#len : erased nat)
  (#l : layout len ) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (v : et)
  (#s : erased et)
  requires
    Cell a (i <: natlt len) |-> s
  ensures
    Cell a (i <: natlt len) |-> v

(* Syntax, in lieu of a typeclass *)
inline_for_extraction noextract
unfold let op_Array_Access
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (#f : perm)
  (#s : erased (lseq et len))
  = read #et #len #l a i #f #s

inline_for_extraction noextract
unfold let op_Array_Assignment
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| ctlayout l |}
  (a : t et l)
  (i : raw_cit{cit_fits len i})
  (v : et)
  (#s : erased (lseq et len))
  = write #et #len #l a i v #s

inline_for_extraction noextract
fn memcpy_host_to_device
  (#et:Type) {| sized et |} (#len : erased nat) (#l : full_layout len)
  (dst : t et l) (src : vec et) (n : sz)
  (#vsrc : erased (lseq et len))
  (#vdst : erased (lseq et len))
  preserves
    cpu
  preserves
    src |-> vsrc
  requires
    pure (SZ.v n == len)
  requires
    on gpu_loc (dst |-> vdst)
  ensures
    on gpu_loc (dst |-> from_seq l vsrc)

inline_for_extraction noextract
fn memcpy_device_to_host'
  (#et : Type u#0) {| sized et |}
  (#dst_sz : erased nat)
  (dst : vec et)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src : t et (Kuiper.Tensor.Layout.Alg.l1_forward src_sz))
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz
  })
  (#f : perm)
  (#v : erased (lseq et src_sz))
  (#gv : erased (lseq et dst_sz))
  preserves
    cpu **
    on gpu_loc (src |-> Frac f v)
  requires
    dst |-> gv
  ensures
    exists* (s' : lseq et dst_sz).
      dst |-> s' **
      pure (s' == Kuiper.Seq.Common.seq_blit gv dst_off v src_off cnt)

inline_for_extraction noextract
fn memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (#sz : erased nat)
  (#l : full_layout sz)
  (dst : t a l)
  (src : t a l)
  (cnt : SZ.t)
  (#f : perm)
  (#vsrc #vdst : erased (lseq a sz))
  preserves
    cpu **
    on gpu_loc (src |-> Frac f vsrc)
  requires
    on gpu_loc (dst |-> vdst) **
    pure (SZ.v cnt == sz)
  ensures
    on gpu_loc (dst |-> vsrc)

(* Random helper. From the CPU, read one element from a flat array1. *)
inline_for_extraction noextract
fn arr_read_1
  (#et : Type0) {| sized et |}
  (#len : erased nat)
  (a : t et (l1_forward len))
  (i : szlt len)
  (#f : perm)
  (#va : erased (lseq et len))
  preserves
    cpu
  preserves
    on gpu_loc (a |-> Frac f va)
  returns
    x : et
  ensures
    pure (x == Seq.index va i)

ghost
fn array1_collect_approx
  (#et : Type0) {| floating et, real_like et |}
  (#len : nat)
  (#l : layout len)
  (a : array1 et l)
  (ra : lseq real len)
  requires
    pure (SZ.fits (layout_size l))
  requires
    forall+ (i:natlt len).
      exists* (v: et). Cell a i |-> v ** pure (v %~ (ra `Seq.index` i))
  ensures
    exists* (va: lseq et len). (a |-> va) ** pure (va %~ ra)

val ref_of_array_cell
  (#et : Type0)
  (#len : nat)
  (#l : layout len)
  (a : array1 et l)
  (i : natlt len)
  : GTot (ref et)

inline_for_extraction noextract
fn get_ref_of_array_cell
  (#et : Type0)
  (#len : erased nat)
  (#l : layout len) {| c : ctlayout l |}
  (a : array1 et l)
  (i : szlt len)
  returns
    r : ref et
  ensures
    pure (r == ref_of_array_cell a i)

ghost
fn array1_cell_to_ref
  (#et : Type0)
  (#len : nat)
  (#l : layout len)
  (a : array1 et l)
  (i : natlt len)
  (#f : perm)
  (#v : erased et)
  requires
    Cell a i |-> Frac f v
  ensures
    ref_of_array_cell a i |-> Frac f v

ghost
fn array1_cell_from_ref
  (#et : Type0)
  (#len : nat)
  (#l : layout len)
  (a : array1 et l)
  (i : natlt len)
  (#f : perm)
  (#v : erased et)
  requires
    ref_of_array_cell a i |-> Frac f v
  ensures
    Cell a i |-> Frac f v
