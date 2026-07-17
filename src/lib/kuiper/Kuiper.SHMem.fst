module Kuiper.SHMem

#lang-pulse

open Pulse.Lib.Core
open Pulse.Lib.Pervasives
open Kuiper.Base
open Kuiper.Array
open FStar.Tactics.V2
open Kuiper.ForEvery
open Kuiper.Common
module T = FStar.Tactics

//don't mark this an instance, to avoid clashing with other instances
//for visibility_of, gpu_of
let is_send_across_block_array
  (#et:Type0)
  (a : array et { is_block_array a })
  (#f:perm) (#s:_)
: is_send_across block_of (pts_to a #f s)
= let i : is_send_across (visibility_of a) (pts_to a #f s)
   = Tactics.Typeclasses.solve_debug #_ #_ in
  i

instance is_send_across_live_c_shmem #d (c:c_shmem d) #f (_:squash (c_shmem_inv c))
: is_send_across block_of (live_c_shmem #d c #f)
= match d with
  | SHArray ty len ->
    let c : array ty = c in
    let ff (v:_) : is_send_across block_of (pts_to c #f v) =
      is_send_across_block_array c #_ #_
    in
    let ff : is_send_across block_of (exists* v. pts_to c #f v) =
      is_send_across_exists _ #ff
    in
    let ff : is_send_across block_of (live_c_shmem #(SHArray ty len) c #f) =
      ff
    in
    ff

ghost
fn unfold_live_c_shmems_nil (c : c_shmems []) (#[T.exact (`1.0R)]f:_)
  requires live_c_shmems c #f
  ensures emp
{
  rewrite live_c_shmems c #f as emp;
}

ghost
fn fold_live_c_shmems_nil (c : c_shmems []) (#[T.exact (`1.0R)]f:_)
  ensures live_c_shmems c #f
{
  rewrite emp as live_c_shmems c #f;
}

ghost
fn unfold_live_c_shmems_cons #d #ds (c : c_shmems (d::ds)) (#[T.exact (`1.0R)]f:_)
  requires live_c_shmems c #f
  ensures live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f
{
  rewrite (live_c_shmems c #f)
      as  (live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f)
}

ghost
fn fold_live_c_shmems_cons #d #ds (c : c_shmems (d::ds)) (#[T.exact (`1.0R)]f:_)
requires live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f
ensures live_c_shmems c #f
{
  rewrite (live_c_shmem #d (fst c) #f ** live_c_shmems #ds (snd c) #f)
  as (live_c_shmems c #f);
}

let rec is_send_across_live_c_shmems_ #ds (c:c_shmems ds) #f (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c #f)
= match ds with
  | [] -> solve #(is_send_across block_of emp)
  | d::ds ->
    let c : c_shmem d & c_shmems ds = c in (* coerce *)
    let s = is_send_across_live_c_shmems_ #ds (snd c) #f () in
    let f : is_send_across block_of (live_c_shmem #d (fst c) #f) = solve in
    is_send_across_star _ _ #f #s

instance is_send_across_live_c_shmems #ds (c:c_shmems ds) #f (pf:squash (c_shmems_inv c))
: is_send_across block_of (live_c_shmems #ds c #f)
= is_send_across_live_c_shmems_ #ds c #f pf

ghost
fn unfold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
  requires live_c_shmems c #f
  ensures Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)
{
  reduce_with_steps (live_c_shmems c #f) [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]];
}


ghost
fn fold_c_shmems (#ds:_) (c:c_shmems ds) (#f:_) (desc:_)
  requires Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f)
  ensures live_c_shmems c #f
{
  norm_spec [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f);
  rewrite (Pervasives.norm [zeta; iota; delta_only [`%live_c_shmems; `%live_c_shmem; desc]] (live_c_shmems c #f))
  as (live_c_shmems c #f);
}

ghost
fn unfold_live_c_shmem #d (c : c_shmem d) #f
  requires live_c_shmem c #f
  ensures
    exists* (s : Seq.seq (d_ty d)).
      pts_to (c <: larray (d_ty d) (d_len d)) #f s
{
  rewrite each d as (SHArray (d_ty d) #(d_ty_sized d) (d_len d));
  reduce_with_steps (live_c_shmem #((SHArray (d_ty d) #(d_ty_sized d) (d_len d))) c #f)
                    [delta_only [`%live_c_shmem]; iota]
}

ghost
fn fold_live_c_shmem #d (c:c_shmem d) #f
  requires
    // Idem
    exists* (s:Seq.seq (d_ty d)).
      pts_to (c <: larray (d_ty d) (d_len d)) #f s
  ensures live_c_shmem c #f
{
  rewrite each d as (SHArray (d_ty d) #(d_ty_sized d) (d_len d));
  fold (live_c_shmem #((SHArray (d_ty d) #(d_ty_sized d) (d_len d))) c #f);
  rewrite each ((SHArray (d_ty d) #(d_ty_sized d) (d_len d))) as d;
}

ghost
fn gpu_live_c_shmem_share_underspec
    (#d:_) (c:c_shmem d) (#f:_) (#k:nat { k > 0 })
requires
  live_c_shmem c #f
ensures
  forall+ (_ : natlt k). live_c_shmem c #(f /. Real.of_int k)
{
  let c' : larray (d_ty d) (d_len d) = c; assert rewrites_to c' c;
  unfold_live_c_shmem c #f;
  with s.  assert A.pts_to c' #f s;
  A.pts_to_len c';
  array_to_slice c';
  assert pts_to_slice c' #f 0 _ s; // FIXME: using (d_len d) for _ fails!?!?
  with ll. assert pts_to_slice c' #f 0 ll s;
  slice_share c' 0 _ k #f;
  (* [array_to_slice] gave [pure (Seq.length s == length c')] and [ll == Seq.length s],
     so the fullness fact is pure/duplicable: thread it through every iteration. *)
  assert pure (Pulse.Lib.Array.length c' == ll);
  drop_ (is_full_slice c' ll);
  forevery_map_extra
    (pure (Pulse.Lib.Array.length c' == ll))
    (fun (_ : natlt k) -> pts_to_slice c' #(f /. Real.of_int k) 0 ll s)
    (fun (_ : natlt k) -> live_c_shmem c #(f /. Real.of_int k))
    fn _ {
      slice_to_array_full c';
      fold_live_c_shmem #d c #(f /. Real.of_int k);
      ()
    };
  ()
}

ghost
fn rec gpu_live_c_shmems_share_underspec
  (#ds:_) (c:c_shmems ds) (#f:_) (#k:nat { k > 0 })
  requires
    live_c_shmems c #f
  ensures
    forall+ (_ : natlt k). live_c_shmems c #(f /. Real.of_int k)
  decreases ds
{
  match ds {
    [] -> {
      unfold_live_c_shmems_nil c #f;
      forevery_emp_intro (natlt k);
      forevery_map
        (fun _ -> emp)
        (fun _ -> live_c_shmems c #(f /. Real.of_int k))
      (fun _ -> fold_live_c_shmems_nil c #(f/. Real.of_int k))

    }
    d::ds' ->  {
      unfold_live_c_shmems_cons c #f;
      gpu_live_c_shmem_share_underspec (fst c) #f #k;
      gpu_live_c_shmems_share_underspec (snd c) #f #k;
      forevery_zip
        (fun _ -> live_c_shmem (fst c) #(f /. Real.of_int k))
        _;
      forevery_map
        (fun _ ->
          live_c_shmem (fst c) #(f /. Real.of_int k) **
          live_c_shmems (snd c) #(f /. Real.of_int k))
        (fun _ ->
           live_c_shmems c #(f /. Real.of_int k))
      (fun _ ->
        fold_live_c_shmems_cons #d #ds' c  #(f /. Real.of_int k));
    }
  }
}

ghost
fn gpu_live_c_shmem_gather_underspec
  (#d:_) (c:c_shmem d) (#f:perm) (#k:nat { k > 0 })
  requires
    forall+ (_ : natlt k). live_c_shmem #d c #(f /. Real.of_int k)
  ensures
    live_c_shmem c #f
{
  let c' : larray (d_ty d) (d_len d) = c; assert rewrites_to c' c;
  forevery_map #(natlt k)
    (fun (_ : natlt k) -> live_c_shmem #d c #(f /. Real.of_int k))
    (fun (_ : natlt k) -> exists* v. pts_to c' #(f /. Real.of_int k) v)
    fn _ { unfold_live_c_shmem #d c #(f /. Real.of_int k); };
  array_gather_underspec c' k;
  fold_live_c_shmem c #f;
}

ghost
fn rec gpu_live_c_shmems_gather_underspec
  (#ds:_) (c:c_shmems ds) (#f:perm) (#k:nat { k > 0 })
  requires
    forall+ (_ : natlt k). live_c_shmems c #(f /. Real.of_int k)
  ensures
    live_c_shmems c #f
  decreases ds
{
  match ds {
    [] -> {
      drop_ (forall+ (_ : natlt k). live_c_shmems #[] c #(f /. Real.of_int k));
      fold_live_c_shmems_nil (c <: c_shmems []) #f;
      rewrite each (Nil #shmem_desc) as ds;
    }

    d::ds' -> {
      forevery_map #(natlt k)
        (fun _ -> live_c_shmems #(d::ds') c #(f /. Real.of_int k))
        (fun _ -> live_c_shmem #d (fst (c<: c_shmem d & c_shmems ds')) #(f /. Real.of_int k) **
                  live_c_shmems #ds' (snd (c<: c_shmem d & c_shmems ds')) #(f /. Real.of_int k))
        fn x {
          unfold_live_c_shmems_cons #d #ds' c #(f /. Real.of_int k);
        };
      forevery_unzip _ _;
      gpu_live_c_shmems_gather_underspec #ds' (snd c) #f #k;
      gpu_live_c_shmem_gather_underspec #d (fst c) #f #k;
      fold_live_c_shmems_cons #d #ds' c #f;
      rewrite each (d::ds') as ds;
    }
  }
}
