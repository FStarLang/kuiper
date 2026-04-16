module Kuiper.Array1
#lang-pulse

open Kuiper
open Kuiper.Chest
open Kuiper.Bijection
module T = Kuiper.Tensor
module SZ = Kuiper.SizeT
module Tac = FStar.Tactics.V2

let abs_bij (#len : nat) : (abs (desc len) =~ ait len) =
  {
    ff = (fun (i, ()) -> i);
    gg = (fun i -> (i, ()));
    ff_gg = ez;
    gg_ff = ez;
  }

let tr_val (#et : Type) (#len : nat) (s : lseq et len) : chest (desc len) et =
  Chest.mk (desc len) (fun (i, ()) -> s @! i)

let backtr_val (#et : Type) (#len : nat) (c : chest (desc len) et) : GTot (lseq et len) =
  Seq.init_ghost len (fun i -> Chest.acc c (i, ()))

inline_for_extraction noextract
let adapt_cit_back (len : erased nat) (idx : raw_cit{cit_fits len idx}) : conc (desc len) =
  (idx, ())

let to_from (#et:Type) (#len : nat) (l : full_layout len) (s : lseq et len)
  : Lemma (to_seq l (from_seq l s) == s)
          [SMTPat (to_seq l (from_seq l s))]
  = let aux (i : natlt len) : Lemma (to_seq l (from_seq l s) @! i == Seq.index s i) =
      ()
    in
    // ^ Why is this needed?
    Classical.forall_intro aux;
    assert (Seq.equal (to_seq l (from_seq l s)) s);
    ()

let to_seq_rel (#et:Type) (#len : nat)
  (l : full_layout len) (s : lseq et len)
  : Lemma (to_seq l s == T.to_seq l (tr_val s))
  = let aux (i : natlt len)
      : Lemma (to_seq l s @! i == T.to_seq l (tr_val s) @! i) = ()
    in
    Classical.forall_intro aux;
    assert (Seq.equal (to_seq l s) (T.to_seq l (tr_val s)))

let from_seq_l1_fwd (#et : Type) (#len : nat) (s : lseq et len)
  : Lemma (from_seq (Kuiper.Tensor.Layout.Alg.l1_forward len) s == s)
  = let l = Kuiper.Tensor.Layout.Alg.l1_forward len in
    let aux (i : natlt len)
      : Lemma (from_seq l s @! i == s @! i) = ()
    in
    Classical.forall_intro aux;
    assert (Seq.equal (from_seq l s) s)

let to_seq_l1_fwd (#et : Type) (#len : nat) (s : lseq et len)
  : Lemma (to_seq (Kuiper.Tensor.Layout.Alg.l1_forward len) s == s)
  = let l = Kuiper.Tensor.Layout.Alg.l1_forward len in
    from_seq_l1_fwd #et #len s;
    assert (to_seq l (from_seq l s) == s)

let t (et : Type0) (#len : nat) (l : layout len) : Type0 =
  T.tensor et l

let is_global (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l)
  : prop =
  T.is_global a

let from_array
  (#et : Type0) (#len : erased nat)
  (l : layout len)
  (a : gpu_array et (layout_size l))
  : t et l
  = T.from_array _ a

let core
  (#et : Type0) (#len : erased nat) (#l : layout len)
  (a : t et l)
  : gpu_array et (layout_size l)
  = T.core a

let lem_core_from_array
  (#et : Type) (#len : erased nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures from_array l (core a) == a)
          [SMTPat (core a)]
   = ()

let lem_from_array_core
  (#et : Type) (#len : erased nat)
  (l : layout len)
  (p : gpu_array et (layout_size l))
  : Lemma (ensures core (from_array l p) == p)
          [SMTPat (from_array l p)]
  = ()

let lem_is_global_iff_core
  (#et : Type0) (#len : nat)
  (#l : layout len)
  (a : t et l)
  : Lemma (ensures is_global a <==> is_global_array (core a))
          [SMTPat (is_global a)]
  = ()

let pts_to
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  (s : lseq et len)
  : slprop
  = T.tensor_pts_to a #f (tr_val s)

instance is_send_across_global
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l { is_global a })
  (#f : perm) (s : lseq et len)
  : is_send_across gpu_of (pts_to a #f s)
  = solve

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
    pure (is_global p)
{
  let t = T.alloc0 #et len l;
  with em. assert on gpu_loc (T.tensor_pts_to t em);
  assert pure (Chest.equal em (tr_val (backtr_val em)));
  rewrite on gpu_loc (T.tensor_pts_to t em)
       as on gpu_loc (pts_to t (backtr_val em));
  t
}

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
    on gpu_loc (p |-> em)
  ensures
    emp
{
  rewrite on gpu_loc (pts_to p em)
       as on gpu_loc (T.tensor_pts_to p (tr_val em));
  T.free p;
}

ghost
fn pts_to_ref
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l)
  (#f : perm) (#s : erased (lseq et len))
  preserves
    a |-> Frac f s
  ensures
    pure (SZ.fits (layout_size l))
{
  unfold pts_to a #f s;
  T.tensor_pts_to_ref a;
  fold pts_to a #f s;
}

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
{
  unfold pts_to a #f s;
  T.tensor_concr a;
  to_seq_rel l s;
  rewrite T.core a |-> Frac f (T.to_seq l (tr_val s))
       as core a |-> Frac f (to_seq l s);
}

ghost
fn raise
  (#et : Type) (#len : nat)
  (l : full_layout len)
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : lseq et len)
  requires
    p |-> Frac f (to_seq l s)
  ensures
    from_array l p |-> Frac f s
{
  to_seq_rel l s;
  rewrite
    p |-> Frac f (to_seq l s)
  as
    p |-> Frac f (T.to_seq l (tr_val s));
  T.tensor_abs l p;
  fold pts_to (from_array l p) #f s;
}

ghost
fn raise'
  (#et : Type) (#len : nat)
  (l : full_layout len)
  (p : gpu_array et (layout_size l))
  (#f : perm)
  (#s : lseq et len)
  requires
    p |-> Frac f s
  ensures
    from_array l p |-> Frac f (from_seq l s)
{
  rewrite each s as to_seq l (from_seq l s);
  raise l p;
}

ghost
fn share_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    a |-> Frac f s
  ensures
    forall+ (_:natlt k). a |-> Frac (f /. k) s
{
  unfold pts_to a #f s;
  T.tensor_share_n a k;
  forevery_map
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) (tr_val s))
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    fn i { fold pts_to a #(f /. k) s };
}

ghost
fn gather_n
  (#et : Type0) (#len : nat) (#l : layout len)
  (a : t et l) (k : pos)
  (#f : perm) (#s : lseq et len)
  requires
    forall+ (_:natlt k). a |-> Frac (f /. k) s
  ensures
    a |-> Frac f s
{
  forevery_map
    (fun (i:natlt k) -> pts_to a #(f /. k) s)
    (fun (i:natlt k) -> T.tensor_pts_to a #(f /. k) (tr_val s))
    fn i { unfold pts_to a #(f /. k) s };
  T.tensor_gather_n a k;
  fold pts_to a #f s;
}

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
{
  unfold pts_to a #f s;
  let v = T.tensor_read a (adapt_cit_back len i);
  fold pts_to a #f s;
  v
}

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
{
  unfold pts_to a s;
  T.tensor_write a (adapt_cit_back len i) v;
  with cs'. assert T.tensor_pts_to a cs';
  assert pure (Chest.equal cs' (tr_val (Seq.upd s i v)));
  fold pts_to a (Seq.upd s i v);
  ()
}

let pts_to_cell
  (#et : Type) (#len : nat) (#l : layout len)
  ([@@@mkey] a : t et l)
  (#[Tac.exact (`1.0R)] f : perm)
  ([@@@mkey] i : ait len)
  (v : et)
  : slprop
  = T.tensor_pts_to_cell a #f (i, ()) v

let pts_to_cell_eq
  (#et : Type) (#len : nat) (#l : layout len)
  (a : t et l) (i : ait len) (f : perm) (v : et)
  : Lemma (pts_to_cell a #f i v
           ==
           gpu_pts_to_cell (core a) #f (l.imap.f (adapt_idx_back i)) v)
  = T.tensor_pts_to_cell_eq a (adapt_idx_back i) f v

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
{
  unfold pts_to a #f s;
  T.tensor_explode a;
  forevery_iso abs_bij _;
  forevery_ext _ (fun (i : ait len) -> Cell a i |-> Frac f (Seq.index s i));
  ()
}

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
{
  forevery_iso (bij_sym abs_bij) _;
  forevery_ext _ (fun (i : abs (desc len)) -> Cell a i |-> Frac f (acc (tr_val s) i));
  T.tensor_implode a;
  fold pts_to a #f s;
}

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
{
  unfold pts_to_cell a #f i s;
  let v = T.tensor_read_cell a (adapt_cit_back len i);
  fold pts_to_cell a #f i s;
  v
}

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
{
  unfold pts_to_cell a i s;
  T.tensor_write_cell a (adapt_cit_back len i) v;
  fold pts_to_cell a i v;
}

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
{
  map_loc gpu_loc
    #(dst |-> vdst)
    #(core dst |-> to_seq l vdst)
    fn _ { lower dst; };

  gpu_memcpy_host_to_device (core dst) src n;

  map_loc gpu_loc
    #(core dst |-> vsrc)
    #(dst |-> from_seq l vsrc)
    fn _ {
      raise' l (core dst);
      rewrite (from_array l (core dst) |-> from_seq l vsrc)
           as (dst |-> from_seq l vsrc);
    };
  ();
}

inline_for_extraction noextract
fn memcpy_device_to_host'
  (#a:Type u#0)
  {| sized a |}
  (#dst_sz : erased nat)
  (dst_arr : vec a)
  (dst_off : SZ.t)
  (#src_sz : erased nat)
  (src : t a (Kuiper.Tensor.Layout.Alg.l1_forward src_sz))
  (src_off : SZ.t)
  (cnt : SZ.t {
    dst_off + cnt <= dst_sz /\
          src_off + cnt <= src_sz
  })
  (#f : perm)
  (#v : erased (lseq a src_sz))
  (#gv : erased (lseq a dst_sz))
  preserves
    cpu **
    on gpu_loc (src |-> Frac f v)
  requires
    dst_arr |-> gv
  ensures
    exists* (s' : lseq a dst_sz).
      dst_arr |-> s' **
      pure (s' == Kuiper.Seq.Common.seq_blit gv dst_off v src_off cnt)
{
  let l = Kuiper.Tensor.Layout.Alg.l1_forward src_sz;
  map_loc gpu_loc
    #(src |-> Frac f v)
    #(core src |-> Frac f v)
    fn _ {
      lower src;
      to_seq_l1_fwd #a #src_sz v;
      rewrite (core src |-> Frac f (to_seq l v))
           as (core src |-> Frac f v);
    };

  (* Bulk copy at the gpu_array level *)
  gpu_memcpy_device_to_host' #_ #_ #dst_sz dst_arr dst_off #_ (core src) src_off cnt;

  map_loc gpu_loc
    #(core src |-> Frac f v)
    #(src |-> Frac f v)
    fn _ {
      raise' l (core src);
      from_seq_l1_fwd #a #src_sz v;
      rewrite (from_array l (core src) |-> Frac f (from_seq l v))
           as (src |-> Frac f v);
    };
  ()
}

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
{
  map_loc gpu_loc
    #(dst |-> vdst)
    #(core dst |-> to_seq l vdst)
    fn _ { lower dst; };
  map_loc gpu_loc
    #(src |-> Frac f vsrc)
    #(core src |-> Frac f (to_seq l vsrc))
    fn _ { lower src; };

  gpu_memcpy_device_to_device (core dst) (core src) cnt;

  map_loc gpu_loc
    #(core src |-> Frac f (to_seq l vsrc))
    #(src |-> Frac f vsrc)
    fn _ {
      raise l (core src);
      rewrite (from_array l (core src) |-> Frac f vsrc)
           as (src |-> Frac f vsrc);
    };
  map_loc gpu_loc
    #(core dst |-> to_seq l vsrc)
    #(dst |-> vsrc)
    fn _ {
      raise l (core dst);
      rewrite (from_array l (core dst) |-> vsrc)
           as (dst |-> vsrc);
    };
  ();
}
