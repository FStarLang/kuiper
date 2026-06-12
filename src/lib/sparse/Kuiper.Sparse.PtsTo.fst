module Kuiper.Sparse.PtsTo

#lang-pulse

open Kuiper
open FStar.Tactics.V2 { exact }
open Kuiper.Array.Vectorized
open Kuiper.Sparse.Common


(* Live *)

let slice_live
  (#et : Type0)
  (#l : nat)
  (a : gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i j : nat)
  : slprop
  = exists* s. gpu_pts_to_slice a #f i j s

let array_live_cell
  (#et : Type0)
  (#l : nat)
  (a :gpu_array et l)
  (#[FStar.Tactics.exact (`1.0R)] f : perm)
  (i : natlt l)
  : slprop
  = exists* v. gpu_pts_to_cell a #f i v

module Array2 = Kuiper.Array2

let matrix_live_cell
  (#et : Type0)
  (#rows #cols : nat)
  (#lm : Array2.layout rows cols)
  (gm : Array2.t et lm)
  (i : natlt rows)
  (j : natlt cols)
  : slprop
  = exists* v. Array2.pts_to_cell gm (i, j) v


(* Vector *)

unfold
let gpu_pts_to_vec
  (#a:Type u#0) {| sized a, has_vec_cpy a |}
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (v : seq a)
: slprop
= 
  gpu_pts_to_slice x #f i (i + chunk a) v

unfold
let gpu_pts_to_vec'
  (#a:Type) {| sized a, has_vec_cpy a |}
  (#sz:nat)
  ([@@@mkey] x:gpu_array a sz)
  (#[exact (`1.0R)] f : perm)
  ([@@@mkey] i : nat)
  (v : seq a)
  (k : natle (len v - chunk a))
: slprop
= gpu_pts_to_vec x #f i (Seq.slice v k (k + chunk a))

let gpu_live_vec
  (#a:Type) {| sized a, has_vec_cpy a |}
  (#l : nat)
  (x :gpu_array a l)
  (#[exact (`1.0R)] f : perm)
  (i : nat)
: slprop
= exists* v. gpu_pts_to_vec x #f i v



(* Thread sharing *)

let thread_slice_pts_to
  (#et : Type0)
  (#n : nat)
  (a : gpu_array et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (s : lseq et m)
  (k : natle (m - (j - i)))
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (x : natlt ((j - i - tid) `divup` nthr)).
    gpu_pts_to_cell a (i + x * nthr + tid) (s @! k + x * nthr + tid)

let thread_slice_pts_to_value
  (#et : Type0)
  (#n : nat)
  (a : gpu_array et n)
  (i j : natle n { i <= j })
  (v : et)
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (x : natlt ((j - i - tid) `divup` nthr)).
    gpu_pts_to_cell a (i + x * nthr + tid) v

let thread_slice_live
  (#et : Type0)
  (#n : nat)
  (a : gpu_array et n)
  (i j : natle n {i <= j})
  (nthr : nat) (tid : natlt nthr)
: slprop
=
  forall+ (k : natlt ((j - i - tid) `divup` nthr)).
    array_live_cell a (i + k * nthr + tid)


(* Vector thread sharing *)

let thread_pts_to_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  ([@@@mkey] x : gpu_array et n)
  (#m : nat)
  (s : lseq et m)
  (i : nat)
  (nthr : nat)
  (tid : natlt nthr)
: Pure slprop
  (requires (nthr * chunk et) /? n /\ i + n <= m)
  (ensures fun _ -> true)
=
  forall+ (k : natlt (n / (nthr * chunk et))).
    gpu_pts_to_vec' x ((k * nthr + tid) * chunk et)
      s (i + (k * nthr + tid) * chunk et)

let thread_live_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  ([@@@mkey] x : gpu_array et n)
  (nthr : nat)
  (tid : natlt nthr)
: Pure slprop
  (requires (nthr * chunk et) /? n)
  (ensures fun _ -> true)
=
  forall+ (k : natlt (n / (nthr * chunk et))).
    gpu_live_vec x ((k * nthr + tid) * chunk et)



(* Helpers *)

open Kuiper.Bijection

// TODO por que esto no está definido?
ghost
fn gpu_array_slice_1'
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i j : natle sz)
  (#v : erased (seq a){ Seq.length v == j - i })
  requires gpu_pts_to_slice arr #f i j v
  ensures  forall+ (k: natlt (j - i)). gpu_pts_to_cell arr #f (i + k) (v @! k)
{
  admit()
}

ghost
fn gpu_array_unslice_1'
  (#a:Type u#0)
  (#sz:nat)
  (arr : gpu_array a sz)
  (#f : perm)
  (i j : natle sz)
  (#v : erased (seq a) { Seq.length v == j - i })
  requires forall+ (k: natlt (j - i)). gpu_pts_to_cell arr #f (i + k) (v @! k)
  ensures gpu_pts_to_slice arr #f i j v
{
  admit();
}

let share_thread_ff
  (n : nat) (nthr : pos) (k : natlt n)
: (tid : natlt nthr & natlt ((n + (nthr - 1) - tid) / nthr))
= (| k % nthr, k / nthr |)

let share_thread_gg
  (n : nat) (nthr : pos)
  (tid : natlt nthr) (k : natlt ((n - tid) `divup` nthr))
: (natlt n)
= (k * nthr + tid)

let share_thread_bij (n : nat) (nthr : pos)
: bijection
  (natlt n)
  (tid : natlt nthr & natlt ((n - tid) `divup` nthr))
=
{
  ff = (fun k -> share_thread_ff n nthr k);
  gg = (fun (| tid, k |) -> share_thread_gg n nthr tid k);

  ff_gg = ez;
  gg_ff = ez;
}

ghost
fn thread_slice_share
  (#et : Type0)
  (#n : nat)
  (x : gpu_array et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (nthr : pos)
  requires slice_live x i j
  ensures forall+ (tid : natlt nthr). thread_slice_live x i j nthr tid
{
  unfold slice_live;
  with s. assert gpu_pts_to_slice x i j s;
  gpu_pts_to_slice_ref x i j;

  gpu_array_slice_1' x i j;
  forevery_iso (share_thread_bij (j - i) nthr) _;
  forevery_ext
    (fun r -> 
      gpu_pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s @! (share_thread_bij (j - i) nthr).gg r)
    )
    (fun (| tid, k |) ->
      gpu_pts_to_cell x (i + k * nthr + tid ) (s @! k * nthr + tid)
    );
  forevery_unflatten_dep' _;

  forevery_map #(natlt nthr)
    (fun tid -> forall+ (k : natlt (((j -i) - tid) `divup` nthr)).
      gpu_pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid)
    ) 
    (fun tid -> thread_slice_live x i j nthr tid)
    fn tid {
      forevery_map #(natlt (((j -i) - tid) `divup` nthr))
        (fun k -> gpu_pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid))
        (fun k -> array_live_cell x (i + k * nthr + tid))
        fn k {
          fold array_live_cell x (i + k * nthr + tid);
        };
      fold thread_slice_live x i j nthr tid;
    };
}

ghost
fn thread_slice_gather
  (#et : Type0)
  (#n : nat)
  (x : gpu_array et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (s : lseq et m)
  (k : natle (m - (j - i)))
  (nthr : pos)
  requires forall+ (tid : natlt nthr). thread_slice_pts_to x i j s k nthr tid
  ensures gpu_pts_to_slice x i j (Seq.slice s k (k + (j - i)))
{
  let s' = Seq.slice s k (k + (j - i));

  forevery_map #(natlt nthr)
    (fun tid -> thread_slice_pts_to x i j s k nthr tid)
    (fun tid -> forall+ (h : natlt (((j -i) - tid) `divup` nthr)).
      gpu_pts_to_cell x (i + h * nthr + tid) (s' @! h * nthr + tid)
    ) 
    fn tid {
      unfold thread_slice_pts_to x i j s k nthr tid;
      forevery_ext #(natlt (((j -i) - tid) `divup` nthr))
        (fun h -> gpu_pts_to_cell x (i + h * nthr + tid) (s @! k + h * nthr + tid))
        (fun h -> gpu_pts_to_cell x (i + h * nthr + tid) (s' @! h * nthr + tid));
    };

  forevery_flatten_dep _;

  forevery_ext #(tid : natlt nthr & natlt (((j - i) - tid) `divup` nthr))
    (fun r ->
      gpu_pts_to_cell x (i + r._2 * nthr + r._1) (s' @! r._2 * nthr + r._1)
    )
    (fun r -> 
      gpu_pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s' @! (share_thread_bij (j - i) nthr).gg r)
    );

  forevery_iso_back (share_thread_bij (j - i) nthr)
    (fun r -> gpu_pts_to_cell x (i + r) (s' @! r));

  gpu_array_unslice_1' x i j;
}

ghost
fn thread_slice_gather_value
  (#et : Type0)
  (#n : nat)
  (x : gpu_array et n)
  (i j : natle n { i <= j })
  (v : et)
  (nthr : pos)
  requires forall+ (tid : natlt nthr). thread_slice_pts_to_value x i j v nthr tid
  ensures gpu_pts_to_slice x i j (Seq.create (j - i) v)
{
  let s = Seq.create (j - i) v;
  forevery_map #(natlt nthr)
    (fun tid -> thread_slice_pts_to_value x i j v nthr tid)
    (fun tid -> forall+ (k : natlt (((j -i) - tid) `divup` nthr)).
      gpu_pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid)
    ) 
    fn tid {
      unfold thread_slice_pts_to_value x i j v nthr tid;
      forevery_ext #(natlt (((j -i) - tid) `divup` nthr))
        (fun h -> gpu_pts_to_cell x (i + h * nthr + tid) v)
        (fun h -> gpu_pts_to_cell x (i + h * nthr + tid) (s @! h * nthr + tid));
    };
  forevery_flatten_dep _;
  forevery_ext #(tid : natlt nthr & natlt (((j - i) - tid) `divup` nthr))
    (fun r ->
      gpu_pts_to_cell x (i + r._2 * nthr + r._1) (s @! r._2 * nthr + r._1)
    )
    (fun r -> 
      gpu_pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s @! (share_thread_bij (j - i) nthr).gg r)
    );
  forevery_iso_back (share_thread_bij (j - i) nthr)
    (fun r -> gpu_pts_to_cell x (i + r) (s @! r));

  gpu_array_unslice_1' x i j;
}

ghost
fn thread_share_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (x : gpu_array et n)
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires live x
  ensures forall+ (tid : natlt nthr). thread_live_chunks x nthr tid
{
  with s. assert gpu_pts_to_slice x 0 n s;
  gpu_pts_to_slice_ref x 0 n;

  let ch = chunk et;

  gpu_array_slice_1 x;

  forevery_factor n (n / (nthr * ch)) (nthr * ch) _;

  forevery_map #(natlt (n / (nthr * ch)))
    (fun k ->
      forall+ (h : natlt (nthr * ch)).
        gpu_pts_to_cell x (k * (nthr * ch) + h) (s @! k * (nthr * ch) + h)
    )
    (fun k ->
      forall+ (tid : natlt nthr). gpu_live_vec x ((k * nthr + tid) * chunk et)
    )
    fn k {
      forevery_factor (nthr * ch) nthr ch _;
      forevery_map #(natlt nthr)
        (fun tid ->
          forall+ (i : natlt ch).
            gpu_pts_to_cell x
              (k * (nthr * ch) + (tid * ch + i))
              (s @! k * (nthr * ch) + (tid * ch + i))
        )
        (fun tid ->
          gpu_live_vec x ((k * nthr + tid) * chunk et)
        )
        fn tid {
          let v : lseq et ch = Seq.slice s 
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          forevery_ext #(natlt ch)
            _
            (fun i -> gpu_pts_to_cell x ((k * nthr + tid) * ch + i) (v @! i));
          forevery_rw_size ch
            (((k * nthr + tid) * ch) + ch - ((k * nthr + tid) * ch));

          gpu_array_unslice_1' x
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          rewrite each ch as (chunk et);
          fold gpu_live_vec x ((k * nthr + tid) * chunk et);
        };
    };
  rewrite each ch as (chunk et);

  forevery_commute _;

  forevery_map #(natlt nthr)
    (fun tid ->
      forall+ (k: natlt (n / (nthr * (chunk et)))).
        gpu_live_vec x ((k * nthr + tid) * (chunk et))
    )
    (fun tid -> thread_live_chunks x nthr tid)
    fn tid { fold thread_live_chunks x nthr tid };
}

ghost
fn thread_gather_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (x : gpu_array et n)
  (#m : nat)
  (s : lseq et m)
  (i : natle (m - n))
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires forall+ (tid : natlt nthr). thread_pts_to_chunks x s i nthr tid
  ensures x |-> Seq.slice s i (i + n)
{
  let ch = v (chunk et);
  forevery_map
    (fun tid -> thread_pts_to_chunks x s i nthr tid)
    (fun tid -> 
      forall+ (j : natlt ch) (k : natlt (n / (nthr * ch))).
        gpu_pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
          (s @! i + (k * (nthr * ch) + (tid * ch + j)))
    )
    fn tid {
      unfold thread_pts_to_chunks x s i nthr tid;
      forevery_map #(natlt (n / (nthr * (chunk et))))
        (fun k ->
          gpu_pts_to_vec' x ((k * nthr + tid) * chunk et)
            s (i + (k * nthr + tid) * chunk et)
        )
        (fun k ->
          forall+ (j : natlt ch).
            gpu_pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
              (s @! i + (k * (nthr * ch) + (tid * ch + j)))
        )
        fn k {
          rewrite each (v (chunk et)) as ch;

          gpu_array_slice_1' x
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          forevery_rw_size _ ch;

          forevery_ext #(natlt ch)
            _
            (fun j ->
              gpu_pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
                (s @! i + (k * (nthr * ch) + (tid * ch + j)))
            );
          ();
        };
      rewrite each (v (chunk et)) as ch;
      forevery_commute _;
    };

  forevery_unfactor (nthr * ch) _ _
    (fun h ->
      forall+ (k : natlt (n / (nthr * ch))).
        gpu_pts_to_cell x
          (k * (nthr * ch) + h)
          (s @! i + (k * (nthr * ch) + h))
    );
  forevery_commute _;
  forevery_unfactor n (n / (nthr * ch)) (nthr * ch)
    (fun k -> gpu_pts_to_cell x k (s @! i + k));
  forevery_ext _
    (fun (k : natlt n) ->
      gpu_pts_to_cell x k (Seq.slice s i (i + n) @! k)
    );
  gpu_array_unslice_1 x;
}