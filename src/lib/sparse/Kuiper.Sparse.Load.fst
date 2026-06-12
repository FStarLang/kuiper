module Kuiper.Sparse.Load

#lang-pulse

open Kuiper
open Kuiper.Sparse
open Kuiper.Sparse.Common
open Kuiper.Array.Vectorized

fn gpu_array_vec_cpy
  (#a : Type u#0) {| sized a, has_vec_cpy a |}
  (#dsz : erased nat)
  (d : gpu_array a dsz) (doff : sz)
  (#_ : squash (aligned' 16 d doff))
  (#ssz : erased nat)
  (s : gpu_array a ssz) (soff : sz)
  (#_ : squash (aligned' 16 s soff))
  (#i #j : erased nat)
  (#f : perm)
  (#v : erased (seq a))
  (#_ : squash (i <= soff /\ soff <= j - chunk a))
  (#_ : squash (len v == j - i))
  preserves gpu
  preserves gpu_pts_to_slice #a #ssz s #f i j v
  requires gpu_live_vec d doff
  requires pure (aligned' 16 s soff)
  ensures gpu_pts_to_vec' d doff v (soff - i)
{
  unfold gpu_live_vec d;
  with u_. assert gpu_pts_to_vec d doff u_;
  gpu_pts_to_slice_ref d doff (doff + chunk a);
  gpu_pts_to_slice_ref s i j;
  gpu_array_vec_cpy_dd d doff s soff;
  with u. assert gpu_pts_to_vec d doff u;
  assert pure (u `Seq.equal` Seq.slice v (soff - i) (soff - i + chunk a));
}

inline_for_extraction noextract
fn load_array_vec
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : sz)
  (x : gpu_array et n)
  (#_ : squash (aligned 16 x))
  (#m : sz)
  (y : gpu_array et m)
  (#f : perm)
  (#s : erased (lseq et m))
  (i : szle m)
  (#_ : squash (aligned' 16 y i))
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (nthr * chunk et /? n))
  (#_ : squash (i + n <= m))
  preserves gpu ** y |-> Frac f s
  requires thread_live_chunks x nthr tid
  ensures thread_pts_to_chunks x s i nthr tid
{
  unfold thread_live_chunks x nthr tid;

  forevery_rw_size (n / (nthr * (chunk et))) (n /^ nthr /^ chunk et);

  foreach (n /^ nthr /^ chunk et)
  (fun k -> gpu_live_vec x ((k * nthr + tid) * chunk et))
  (fun k ->
    gpu_pts_to_vec' x ((k * nthr + tid) * chunk et)
      s (i + (k * nthr + tid) * chunk et)
  )
  #(gpu ** y |-> Frac f s)
  fn k {
    gpu_array_vec_cpy
      x ((k *^ nthr +^ tid) *^ chunk et)
      y (i +^ ((k *^ nthr +^ tid) *^ chunk et));
  };

  forevery_rw_size (n /^ nthr /^ chunk et) (n / (nthr * (chunk et)));

  fold thread_pts_to_chunks x s i nthr tid;
}

inline_for_extraction noextract
fn gpu_load_cell
  (#et : Type0)
  (#m #n : sz)
  (x : gpu_array et m)
  (i : szlt m)
  (y : gpu_array et n)
  (#f : perm)
  (#s : erased (lseq et n))
  (j : szlt n)
  preserves gpu ** y |-> Frac f s
  requires array_live_cell x i
  ensures  gpu_pts_to_cell x i (s @! j)
{
  unfold array_live_cell x;
  gpu_array_write x i (gpu_array_read y j);
  with t. assert gpu_pts_to_slice x i (i + 1) t;
  assert pure (Seq.equal t seq![s @! j]);
}


// TODO podria generalizar y tomar from y to, aunque cuando
// repartimos por threads es medio raro
// Quizas cambiar la def de thread_pts_to

inline_for_extraction noextract
fn load2_array_to
  (#et1 #et2 : Type0)
  (#dsz : sz)
  (dst1 : gpu_array et1 dsz)
  (dst2 : gpu_array et2 dsz)
  (to : szle dsz)
  (#ssz : sz)
  (src1 : gpu_array et1 ssz)
  (src2 : gpu_array et2 ssz)
  (#f : perm)
  (#s1 : erased (lseq et1 ssz))
  (#s2 : erased (lseq et2 ssz))
  (i : szle ssz)
  (nthr : sz)
  (tid : szlt nthr)
  (#_ : squash (i + to <= ssz))
  preserves gpu ** src1 |-> Frac f s1 ** src2 |-> Frac f s2
  requires
    thread_slice_live dst1 0 to nthr tid **
    thread_slice_live dst2 0 to nthr tid
  ensures
    thread_slice_pts_to dst1 0 to s1 i nthr tid **
    thread_slice_pts_to dst2 0 to s2 i nthr tid
{
  unfold thread_slice_live dst1 0 to nthr tid;
  unfold thread_slice_live dst2 0 to nthr tid;

  forevery_zip #(natlt ((to - 0 - tid) `divup` nthr))
    (fun k -> array_live_cell dst1 (0 + k * nthr + tid))
    _;
  assume pure (fits (to + nthr));
  forevery_rw_size
    ((to - 0 - tid) `divup` nthr)
    ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr);

  foreach ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr)
    (fun k ->
      array_live_cell dst1 (0 + k * nthr + tid) **
      array_live_cell dst2 (0 + k * nthr + tid)
    )
    (fun k -> 
      gpu_pts_to_cell dst1 (k * nthr + tid) (s1 @! i + k * nthr + tid) **
      gpu_pts_to_cell dst2 (k * nthr + tid) (s2 @! i + k * nthr + tid)
    )
    #(gpu ** src1 |-> Frac f s1 ** src2 |-> Frac f s2)
    fn k {
      rewrite each (0 + k * nthr + tid) as (k * nthr + tid);
      gpu_load_cell dst1 (k *^ nthr +^ tid) src1 (i +^ k *^ nthr +^ tid);
      gpu_load_cell dst2 (k *^ nthr +^ tid) src2 (i +^ k *^ nthr +^ tid);
    };

  forevery_rw_size
    ((to +^ (nthr -^ 1sz) -^ tid) /^ nthr)
    ((to - 0 - tid) `divup` nthr);
  forevery_unzip _ _;

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun x ->
      gpu_pts_to_cell dst1 (x * nthr + tid) (s1 @! i + x * nthr + tid)
    )
    (fun x ->
      gpu_pts_to_cell dst1 (0 + x * nthr + tid) (s1 @! i + x * nthr + tid)
    );
  fold thread_slice_pts_to dst1 0 to s1 i nthr tid;

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun x ->
      gpu_pts_to_cell dst2 (x * nthr + tid) (s2 @! i + x * nthr + tid)
    )
    (fun x ->
      gpu_pts_to_cell dst2 (0 + x * nthr + tid) (s2 @! i + x * nthr + tid)
    );
  fold thread_slice_pts_to dst2 0 to s2 i nthr tid;
}

inline_for_extraction noextract
fn gpu_mask_cell
  (#et : Type0)
  (#n : sz)
  (x : gpu_array et n)
  (i : szlt n)
  (z : et)
  preserves gpu
  requires array_live_cell x i
  ensures  gpu_pts_to_cell x i z
{
  unfold array_live_cell x;
  gpu_array_write x i z;
  with s. assert gpu_pts_to_slice x i (i + 1) s;
  assert pure (Seq.equal s seq![z]);
}

// TODO podria generalizar y tomar from y to, aunque cuando
// repartimos por threads es medio raro
// Quizas cambiar la def de thread_pts_to

inline_for_extraction noextract
fn mask_array_to
  (#et : Type0)
  (#n : sz)
  (x : gpu_array et n)
  (to : szle n)
  (z : et)
  (nthr : sz)
  (tid : szlt nthr)
  preserves gpu
  requires thread_slice_live x 0 to nthr tid
  ensures thread_slice_pts_to_value x 0 to z nthr tid
{
  unfold thread_slice_live x 0 to nthr tid;

  assume pure (fits (to + nthr));
  let to_ : sz = (to +^ (nthr -^ 1sz) -^ tid) /^ nthr;
  forevery_rw_size ((to - 0 - tid) `divup` nthr) to_;

  foreach to_
    (fun k ->
      array_live_cell x (0 + k * nthr + tid)
    )
    (fun k ->
      gpu_pts_to_cell x (k * nthr + tid) z
    )
    #gpu
    fn k {
      rewrite each (0 + k * nthr + tid) as (k * nthr + tid);
      gpu_mask_cell x (k *^ nthr +^ tid) z;
    };

  forevery_rw_size to_ ((to - 0 - tid) `divup` nthr);

  forevery_ext #(natlt ((to - 0 - tid) `divup` nthr))
    (fun k -> gpu_pts_to_cell x (k * nthr + tid) z)
    (fun k -> gpu_pts_to_cell x (0 + k * nthr + tid) z);
  fold thread_slice_pts_to_value x 0 to z nthr tid;
}
