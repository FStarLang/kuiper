module Kuiper.Ref

// Pulse represents references as one-element arrays. Keep this representation
// dependency here; clients use the abstract reference type and its permissions.
friend Pulse.Lib.Reference
friend Kuiper.Array.Core

#lang-pulse

open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Sized
open Kuiper.Array

module R = Pulse.Lib.Reference
module A = Pulse.Lib.Array

ghost
fn _unfold_loc
  (l:loc_id)
  (#a:Type0)
  (r:ref a)
  (#f:perm)
  (#v:a)
  requires on l (r |-> Frac f v)
  ensures on l (A.pts_to r #f seq![v])
  ensures pure (A.length r == 1)
{
  map_loc l
    #(r |-> Frac f v)
    #(A.pts_to r #f seq![v] ** pure (A.length r == 1))
    fn _ {
      unfold R.pts_to;
      A.from_mask r;
      with s. assert A.pts_to r #f s;
      assert pure (Seq.equal s seq![v]);
      A.pts_to_len r;
    };
}

ghost
fn _fold_loc
  (l:loc_id)
  (#a:Type0)
  (r:ref a{A.length r == 1})
  (#f:perm)
  (#v:Seq.seq a{Seq.length v > 0})
  requires on l (A.pts_to r #f v)
  ensures on l (r |-> Frac f (Seq.index v 0))
{
  map_loc l
    #(A.pts_to r #f v)
    #(r |-> Frac f (Seq.index v 0))
    fn _ {
      A.pts_to_len r;
      A.to_mask r;
      fold R.pts_to r #f (Seq.index v 0);
    };
}

inline_for_extraction noextract
fn alloc0
  (#a:Type u#0)
  {| sized a |}
  ()
  preserves cpu
  requires emp
  returns x : ref a
  ensures exists* (v:a). on gpu_loc (x |-> v)
  ensures pure (R.visibility_of_ref x == gpu_of /\ R.is_full_ref x)
{
  let x = gpu_array_alloc #a 1sz;
  _fold_loc gpu_loc x;
  x
}

inline_for_extraction noextract
fn free
  (#a:Type u#0)
  (r : ref a)
  preserves cpu
  requires pure (R.is_full_ref r)
  requires on gpu_loc (r |-> ('v <: a))
  ensures emp
{
  _unfold_loc gpu_loc r;
  gpu_array_free r
}

noextract
fn memcpy_host_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_gr : ref a)
  (src_r  : ref a)
  preserves cpu
  preserves src_r |-> Frac 'f ('v <: a)
  requires on gpu_loc (dst_gr |-> ('gv <: a))
  ensures on gpu_loc (dst_gr |-> 'v)
{
  admit(); // Primitive cudaMemcpy, extracted by ExtractKuiper.
}

noextract
fn memcpy_device_to_host
  (#a:Type u#0)
  {| sized a |}
  (dst_r  : ref a)
  (src_gr : ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f ('gv <: a))
  requires dst_r |-> ('v <: a)
  ensures dst_r |-> 'gv
{
  admit(); // Primitive cudaMemcpy, extracted by ExtractKuiper.
}

inline_for_extraction noextract
fn memcpy_device_to_device
  (#a:Type u#0)
  {| sized a |}
  (dst_r src_gr : ref a)
  preserves cpu
  preserves on gpu_loc (src_gr |-> Frac 'f ('gv <: a))
  requires on gpu_loc (dst_r |-> ('v <: a))
  ensures on gpu_loc (dst_r |-> 'gv)
{
  _unfold_loc gpu_loc dst_r;
  _unfold_loc gpu_loc src_gr;
  Kuiper.Array.Core.memcpy_device_to_device #a #_ #1 dst_r src_gr 1sz;
  _fold_loc gpu_loc dst_r;
  _fold_loc gpu_loc src_gr;
}
