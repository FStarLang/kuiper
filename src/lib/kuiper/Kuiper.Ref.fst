module Kuiper.Ref

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Sized

module A = Pulse.Lib.Array
module R = Pulse.Lib.Reference
module RA = Pulse.Lib.Reference.Array
module C = Kuiper.Array.Core

include Kuiper.Ref.Base

// Keep the dereference marker through specialization so accesses print as *r.
[@@FStar.Custard.no_specialize]
let deref_index : FStar.SizeT.t = FStar.SizeT.uint32_to_sizet _zero_for_deref

ghost
fn _unfold_loc (#a:Type0) (r:ref a) (#f:perm) (#v:a)
  requires on gpu_loc (pts_to r #f v)
  ensures on gpu_loc (A.pts_to (to_array r) #f seq![v])
{
  map_loc gpu_loc #(pts_to r #f v)
    #(A.pts_to (to_array r) #f seq![v])
    fn _ { unfold pts_to; };
}

ghost
fn _fold_loc (#a:Type0) (r:ref a)
  (#f:perm) (#v:Seq.seq a{Seq.length v > 0})
  requires on gpu_loc (A.pts_to (to_array r) #f v)
  ensures on gpu_loc (pts_to r #f (Seq.index v 0))
{
  map_loc gpu_loc #(A.pts_to (to_array r) #f v)
    #(pts_to r #f (Seq.index v 0))
    fn _ {
      A.pts_to_len (to_array r);
      assert pure (Seq.equal v seq![Seq.index v 0]);
      fold (pts_to r #f (Seq.index v 0));
    };
}

inline_for_extraction noextract
fn alloc0 (#a:Type0) {| sized a |} ()
  preserves cpu
  returns r:ref a
  ensures exists* (v:a). on gpu_loc (r |-> v)
  ensures pure (is_global_ref r /\ is_full_ref r)
{
  let arr = C.gpu_array_alloc #a 1sz;
  let r = of_array arr;
  with s. assert on gpu_loc (A.pts_to arr s);
  rewrite on gpu_loc (A.pts_to arr s)
       as on gpu_loc (A.pts_to (to_array r) s);
  _fold_loc r #1.0R #s;
  r
}

inline_for_extraction noextract
fn free (#a:Type0) (r:ref a)
  preserves cpu
  requires pure (is_full_ref r)
  requires on gpu_loc (r |-> ('v <: a))
  ensures emp
{
  _unfold_loc r;
  C.gpu_array_free (to_array r)
}

inline_for_extraction noextract
fn read (#a:Type0) (r:ref a) (#f:perm) (#v:erased a)
  preserves r |-> Frac f v
  returns x:a
  ensures pure (x == reveal v)
{
  unfold pts_to;
  let x = A.op_Dot_Lparen_Rparen (to_array r) deref_index;
  fold (pts_to r #f (reveal v));
  x
}

inline_for_extraction noextract
fn write (#a:Type0) (r:ref a) (v:a)
  requires r |-> ('old <: a)
  ensures r |-> v
{
  unfold pts_to;
  A.op_Dot_Lparen_Rparen_Less_Minus (to_array r) deref_index v;
  with s. assert A.pts_to (to_array r) s;
  assert pure (Seq.equal s seq![v]);
  fold (pts_to r v);
}

inline_for_extraction noextract
fn memcpy_host_to_device (#a:Type0) {| sized a |}
  (dst:ref a) (src:R.ref a)
  preserves cpu
  preserves src |-> Frac 'f ('v <: a)
  requires on gpu_loc (dst |-> ('old <: a))
  ensures on gpu_loc (dst |-> 'v)
{
  _unfold_loc dst;
  let src_arr = RA.to_array src;
  C.memcpy_host_to_device (to_array dst) src_arr 1sz;
  RA.return_to_array src;
  _fold_loc dst;
}

inline_for_extraction noextract
fn memcpy_device_to_host (#a:Type0) {| sized a |}
  (dst:R.ref a) (src:ref a)
  preserves cpu
  preserves on gpu_loc (src |-> Frac 'f ('v <: a))
  requires dst |-> ('old <: a)
  ensures dst |-> 'v
{
  _unfold_loc src;
  let dst_arr = RA.to_array dst;
  C.memcpy_device_to_host dst_arr (to_array src) 1sz;
  RA.return_to_array dst;
  _fold_loc src;
}

inline_for_extraction noextract
fn memcpy_device_to_device (#a:Type0) {| sized a |}
  (dst src:ref a)
  preserves cpu
  preserves on gpu_loc (src |-> Frac 'f ('v <: a))
  requires on gpu_loc (dst |-> ('old <: a))
  ensures on gpu_loc (dst |-> 'v)
{
  _unfold_loc dst;
  _unfold_loc src;
  C.memcpy_device_to_device (to_array dst) (to_array src) 1sz;
  _fold_loc dst;
  _fold_loc src;
}
