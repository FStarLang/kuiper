module Kuiper.IArray.VectorizedAccess
#lang-pulse

open Kuiper
open Kuiper.IArray
open Kuiper.IView
open Kuiper.VectorType

open FStar.Seq.Base

module SZ = FStar.SizeT
module V = Pulse.Lib.Vec

open Kuiper.IArray.Vectorized
open Kuiper.Injection
open Kuiper.Bijection

inline_for_extraction noextract
let base_iview (len : nat) : aiview = {
  len;
  sch = {
    ait      = natlt len;
    ait_enum = solve;
  };
  step = {
    imap     = inj_id;
  };
}

// instance base_ciview (#len : nat) (clen : sz{SZ.v clen == len}) : ciview (base_iview len) = {
//   clen;
//   sch = {
//     cit     = szlt clen;
//     bij     = natural;
//   };
//   step = {
//     cimap   = inj_id;
//     compat  = ez;
//   };
// }

noextract
unfold
let global_id #nblk #nthr (bid : natlt nblk) (tid : natlt nthr) : natlt (nblk * nthr) = bid * nthr + tid

noextract
unfold
let kpre
  (size:sz)
  (a : iarray float (base_iview size))
  (s:seq float{ len s ==  SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  iarray_pts_to_4cells a (it_of_nat (base_iview size) (global_id bid tid * 4))
    (index s (global_id bid tid * 4),
     index s (global_id bid tid * 4 + 1),
     index s (global_id bid tid * 4 + 2),
     index s (global_id bid tid * 4 + 3))

noextract
let scale_seq (#len : nat) (s : seq float{length s == len}) (v : float)
  = Seq.seq_of_list (List.mapT (fun x -> v `mul` x) (Seq.seq_to_list s))

noextract
unfold
let kpost
  (v : float)
  (size:sz)
  (a : iarray float (base_iview size))
  (s:seq float{ len s == SZ.v size })
  (nblk : natle max_blocks)
  (nthr : nat{(nthr*4 * nblk == SZ.v size)})
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop =
  iarray_pts_to_4cells a (it_of_nat (base_iview size) (global_id bid tid * 4))
    (index s (global_id bid tid * 4)      `mul` v,
    (index s (global_id bid tid * 4 + 1)) `mul` v,
    (index s (global_id bid tid * 4 + 2)) `mul` v,
    (index s (global_id bid tid * 4 + 3)) `mul` v)

inline_for_extraction noextract
// [@@CPrologue "__device__"] // no KrmlPrivate, example
fn kf
  (size:sz)
  (#s:erased (seq float) { len s == SZ.v size })
  (nblk : erased (natle max_blocks))
  (a : iarray float (base_iview size))
  (v : float)
  (nthr : sz{nthr*4 * nblk == SZ.v size})
  (bid : szlt nblk)
  (tid : szlt nthr)
  ()
requires
  gpu **
  kpre size a s nblk nthr bid tid
ensures
  gpu **
  kpost v size a s nblk nthr bid tid
{
  let global_idx = ((bid *^ nthr +^ tid) *^ 4sz);
  rewrite each ((SZ.v bid * SZ.v nthr + SZ.v tid) * 4) as SZ.v global_idx;
  let fv = iarray_vec4_read_cells a global_idx;

  let vec = make_float4 (getx fv `mul` v) (gety fv `mul` v) (getz fv `mul` v) (getw fv `mul` v);
  iarray_vec4_write_cells a global_idx vec;
  ()
}

// #push-options "--debug SMTFail --split_queries always"
#push-options "--print_implicits"
fn hf (v : V.vec float)
  preserves
    exists* s. (v |-> s) ** pure (Seq.length s == 4)
  preserves cpu
{
  open Pulse.Lib.Vec;
  let a = gpu_array_alloc #float 4sz;

  gpu_memcpy_host_to_device a v 4sz;

  // with s. assert v |-> s;
  // assert (pure (Seq.equal s (slice s 0 4)));
  // assert a |-> slice s 0 4;

  assert pure (forall (x : natlt 4). in_image (base_iview 4).step.imap.f ((it_to_nat (base_iview 4) (it_of_nat (base_iview 4) 0)) + x));
  let ia' = iarray_begin a;
  let ia = iarray_reindex (natural #(natlt 4) #(natlt 4)) ia';
  with s. assert (iarray_pts_to ia (oo (g_seq_acc s) (natural #(natlt 4) #(natlt 4)).gg));
  iarray_ext ia #1.0R (oo (g_seq_acc s) (natural #(natlt 4) #(natlt 4)).gg) (index s);
  let two = Kuiper.Float32.one `add` one;

  // assert pure (reindex_view (raw_view #(hide #nat 4)) #(natlt 4) #(Kuiper.Enumerable.enumerable_natlt 4)
  //     (natural #(natlt 4) #(natlt 4) #(nb_self (natlt 4))) == base_iview 4);
  iarray_explode ia;
  rewrite each (raw_view #(hide #nat 4)).sch.ait as natlt 4;
  forevery_extract #_ #(reindex_view raw_view natural).sch.ait_enum 0 (fun i -> iarray_pts_to_cell ia i (index s i));
  admit();

  assert iarray_pts_to_4cells ia' #1.0R (it_of_nat (base_iview 4) 0) (index s 0, index s 1, index s 2, index s 3);
  launch_kernel_1 (fun () -> kf 4sz 1 ia' two 1sz 0sz 0sz ());

  gpu_memcpy_device_to_host v a 4sz;

  gpu_array_free a;
  ()
}
