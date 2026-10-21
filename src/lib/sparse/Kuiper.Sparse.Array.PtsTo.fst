module Kuiper.Sparse.Array.PtsTo

#lang-pulse

open Kuiper
open Kuiper.Sparse.Common
open Kuiper.Sparse.Math { divup }
open FStar.Tactics.V2 { exact }
open Kuiper.Array.Vectorized

let chunk_cell_offset (k nthr tid ch i : nat)
  : Lemma (k * (nthr * ch) + (tid * ch + i) ==
      (k * nthr + tid) * ch + i /\
      i + (k * nthr + tid) * ch == (k * nthr + tid) * ch + i)
  = ()

(* Helpers *)

open Kuiper.Bijection

let share_thread_ff
  (n : nat) (nthr : pos) (k : natlt n)
: (tid : natlt nthr & natlt ((n + (nthr - 1) - tid) / nthr))
= (| k % nthr, k / nthr |)

let share_thread_gg
  (n : nat) (nthr : pos)
  (tid : natlt nthr) (k : natlt ((n - tid) `divup` nthr))
: (natlt n)
= (k * nthr + tid)

(* Euclidean round trip k -> (k%nthr, k/nthr) -> k, proved without NL search *)
let share_thread_gg_ff_lem (n : nat) (nthr : pos) (k : natlt n)
: Lemma (let (| tid, q |) = share_thread_ff n nthr k in
         share_thread_gg n nthr tid q == k)
= FStar.Math.Lemmas.euclidean_division_definition k nthr

(* Round trip (tid, k) -> k*nthr+tid -> (tid, k) *)
let share_thread_ff_gg_lem (n : nat) (nthr : pos)
  (tid : natlt nthr) (k : natlt ((n - tid) `divup` nthr))
: Lemma (share_thread_ff n nthr (share_thread_gg n nthr tid k) == (| tid, k |))
= FStar.Math.Lemmas.lemma_mod_plus tid k nthr;
  FStar.Math.Lemmas.lemma_div_plus tid k nthr;
  FStar.Math.Lemmas.small_mod tid nthr;
  FStar.Math.Lemmas.small_div tid nthr

let share_thread_bij (n : nat) (nthr : pos)
: bijection
  (natlt n)
  (tid : natlt nthr & natlt ((n - tid) `divup` nthr))
=
{
  ff = (fun k -> share_thread_ff n nthr k);
  gg = (fun (| tid, k |) -> share_thread_gg n nthr tid k);

  ff_gg = (fun (| tid, k |) -> share_thread_ff_gg_lem n nthr tid k);
  gg_ff = (fun k -> share_thread_gg_ff_lem n nthr k);
}

ghost
fn thread_slice_share
  (#et : Type0)
  (#n : nat)
  (x : larray et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (nthr : pos)
  requires slice_live x i j
  ensures
    (forall+ (tid : natlt nthr). thread_slice_live x i j nthr tid) **
    array_exists x
{
  unfold slice_live;
  with s. assert pts_to_slice x i j s;
  pts_to_slice_ref x i j;

  slice_to_cells x i j;
  forevery_iso (share_thread_bij (j - i) nthr) _;
  forevery_ext
    (fun r ->
      pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s @! (share_thread_bij (j - i) nthr).gg r)
    )
    (fun (| tid, k |) ->
      pts_to_cell x (i + k * nthr + tid ) (s @! k * nthr + tid)
    );
  forevery_unflatten_dep' _;

  forevery_map #(natlt nthr)
    (fun tid -> forall+ (k : natlt (((j -i) - tid) `divup` nthr)).
      pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid)
    )
    (fun tid -> thread_slice_live x i j nthr tid)
    fn tid {
      forevery_map #(natlt (((j -i) - tid) `divup` nthr))
        (fun k -> pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid))
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
  (x : larray et n)
  (i j : natle n { i <= j })
  (#m : nat)
  (s : lseq et m)
  (k : natle (m - (j - i)))
  (nthr : pos)
  requires array_exists x
  requires
    (forall+ (tid : natlt nthr). thread_slice_pts_to x i j s k nthr tid)
  ensures pts_to_slice x i j (Seq.slice s k (k + (j - i)))
{
  let s' = Seq.slice s k (k + (j - i));

  forevery_map #(natlt nthr)
    (fun tid -> thread_slice_pts_to x i j s k nthr tid)
    (fun tid -> forall+ (h : natlt (((j -i) - tid) `divup` nthr)).
      pts_to_cell x (i + h * nthr + tid) (s' @! h * nthr + tid)
    )
    fn tid {
      unfold thread_slice_pts_to x i j s k nthr tid;
      forevery_ext #(natlt (((j -i) - tid) `divup` nthr))
        (fun h -> pts_to_cell x (i + h * nthr + tid) (s @! k + h * nthr + tid))
        (fun h -> pts_to_cell x (i + h * nthr + tid) (s' @! h * nthr + tid));
    };

  forevery_flatten_dep _;

  forevery_ext #(tid : natlt nthr & natlt (((j - i) - tid) `divup` nthr))
    (fun r ->
      pts_to_cell x (i + r._2 * nthr + r._1) (s' @! r._2 * nthr + r._1)
    )
    (fun r ->
      pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s' @! (share_thread_bij (j - i) nthr).gg r)
    );

  forevery_iso_back (share_thread_bij (j - i) nthr)
    (fun r -> pts_to_cell x (i + r) (s' @! r));

  cells_to_slice x i j;
}

ghost
fn thread_slice_gather_value
  (#et : Type0)
  (#n : nat)
  (x : larray et n)
  (i j : natle n { i <= j })
  (v : et)
  (nthr : pos)
  requires array_exists x
  requires
    (forall+ (tid : natlt nthr). thread_slice_pts_to_value x i j v nthr tid)
  ensures pts_to_slice x i j (Seq.create (j - i) v)
{
  let s = Seq.create (j - i) v;
  forevery_map #(natlt nthr)
    (fun tid -> thread_slice_pts_to_value x i j v nthr tid)
    (fun tid -> forall+ (k : natlt (((j -i) - tid) `divup` nthr)).
      pts_to_cell x (i + k * nthr + tid) (s @! k * nthr + tid)
    )
    fn tid {
      unfold thread_slice_pts_to_value x i j v nthr tid;
      forevery_ext #(natlt (((j -i) - tid) `divup` nthr))
        (fun h -> pts_to_cell x (i + h * nthr + tid) v)
        (fun h -> pts_to_cell x (i + h * nthr + tid) (s @! h * nthr + tid));
    };
  forevery_flatten_dep _;
  forevery_ext #(tid : natlt nthr & natlt (((j - i) - tid) `divup` nthr))
    (fun r ->
      pts_to_cell x (i + r._2 * nthr + r._1) (s @! r._2 * nthr + r._1)
    )
    (fun r ->
      pts_to_cell x
        (i + (share_thread_bij (j - i) nthr).gg r)
        (s @! (share_thread_bij (j - i) nthr).gg r)
    );
  forevery_iso_back (share_thread_bij (j - i) nthr)
    (fun r -> pts_to_cell x (i + r) (s @! r));

  cells_to_slice x i j;
}

ghost
fn thread_share_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : nat)
  (x : larray et n)
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires live x
  ensures
    (forall+ (tid : natlt nthr). thread_live_chunks x nthr tid) **
    array_exists x
{
  array_to_slice x;
  with s. assert pts_to_slice x 0 n s;
  pts_to_slice_ref x 0 n;

  let ch = chunk et;

  array_slice_1 x;

  forevery_factor n (n / (nthr * ch)) (nthr * ch) _;

  forevery_map #(natlt (n / (nthr * ch)))
    (fun k ->
      forall+ (h : natlt (nthr * ch)).
        pts_to_cell x (k * (nthr * ch) + h) (s @! k * (nthr * ch) + h)
    )
    (fun k ->
      forall+ (tid : natlt nthr). live_vec x ((k * nthr + tid) * chunk et)
    )
    fn k {
      chunk_k_bound n (nthr * ch) k;
      chunk_tid_cells_bound nthr ch;
      forevery_factor (nthr * ch) nthr ch _;
      forevery_map #(natlt nthr)
        (fun tid ->
          forall+ (i : natlt ch).
            pts_to_cell x
              (k * (nthr * ch) + (tid * ch + i))
              (s @! k * (nthr * ch) + (tid * ch + i))
        )
        (fun tid ->
          live_vec x ((k * nthr + tid) * chunk et)
        )
        fn tid {
          chunk_slice_bound n nthr ch k tid;
          let v : lseq et ch = Seq.slice s
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          forevery_map #(natlt ch)
            (fun i ->
              pts_to_cell x
                (k * (nthr * ch) + (tid * ch + i))
                (s @! k * (nthr * ch) + (tid * ch + i)))
            (fun i ->
              pts_to_cell x ((k * nthr + tid) * ch + i) (v @! i))
            fn i {
              chunk_cell_offset k nthr tid ch i;
              Seq.lemma_index_slice s
                ((k * nthr + tid) * ch)
                ((k * nthr + tid) * ch + ch)
                i;
              rewrite each
                (k * (nthr * ch) + (tid * ch + i))
              as ((k * nthr + tid) * ch + i);
              rewrite each
                (s @! ((k * nthr + tid) * ch + i))
              as (v @! i);
            };
          forevery_rw_size ch
            (((k * nthr + tid) * ch) + ch - ((k * nthr + tid) * ch));

          cells_to_nonempty_slice x
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          rewrite each ch as (chunk et);
          fold live_vec x ((k * nthr + tid) * chunk et);
        };
    };
  rewrite each ch as (chunk et);

  forevery_commute _;

  forevery_map #(natlt nthr)
    (fun tid ->
      forall+ (k: natlt (n / (nthr * (chunk et)))).
        live_vec x ((k * nthr + tid) * (chunk et))
    )
    (fun tid -> thread_live_chunks x nthr tid)
    fn tid { fold thread_live_chunks x nthr tid };
}

let chunk_vec_bound (n m i nthr ch k tid : nat)
  : Lemma
    (requires nthr > 0 /\ ch > 0 /\ nthr * ch > 0 /\ (nthr * ch) /? n /\ i + n <= m /\
              k < n / (nthr * ch) /\ tid < nthr)
    (ensures i + (k * nthr + tid) * ch <= m - ch)
  = let q = n / (nthr * ch) in
    FStar.Math.Lemmas.lemma_div_exact n (nthr * ch);
    FStar.Math.Lemmas.lemma_mult_le_right nthr (k + 1) q;
    FStar.Math.Lemmas.lemma_mult_le_right ch (k * nthr + tid + 1) (q * nthr);
    FStar.Math.Lemmas.distributivity_add_left (k * nthr + tid) 1 ch;
    FStar.Math.Lemmas.paren_mul_right q nthr ch

let chunk_vec_bound_all (n m i nthr ch : nat)
  : Lemma
    (requires nthr > 0 /\ ch > 0 /\ nthr * ch > 0 /\ (nthr * ch) /? n /\ i + n <= m)
    (ensures
      (forall (k tid : nat). {:pattern ((k * nthr + tid) * ch)}
        k < n / (nthr * ch) /\ tid < nthr ==> i + (k * nthr + tid) * ch <= m - ch) /\
      (forall (k tid j : nat). {:pattern (k * (nthr * ch) + (tid * ch + j))}
        k < n / (nthr * ch) /\ tid < nthr /\ j < ch ==>
          i + (k * (nthr * ch) + (tid * ch + j)) < m))
  = introduce forall (k tid : nat).
      k < n / (nthr * ch) /\ tid < nthr ==> i + (k * nthr + tid) * ch <= m - ch
    with introduce _ ==> _
    with chunk_vec_bound n m i nthr ch k tid;
    introduce forall (k tid j : nat).
      k < n / (nthr * ch) /\ tid < nthr /\ j < ch ==>
        i + (k * (nthr * ch) + (tid * ch + j)) < m
    with introduce _ ==> _
    with (chunk_vec_bound n m i nthr ch k tid; chunk_cell_offset k nthr tid ch j)

#push-options "--z3rlimit_factor 2"
ghost
fn thread_gather_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : pos)
  (x : larray et n)
  (#m : nat)
  (s : lseq et m)
  (i : natle (m - n))
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires
    (forall+ (tid : natlt nthr). thread_pts_to_chunks x s i nthr tid)
  ensures x |-> Seq.slice s i (i + n)
{
  let ch = v (chunk et);
  chunk_vec_bound_all n m i nthr (v (chunk et));
  forevery_map
    (fun tid -> thread_pts_to_chunks x s i nthr tid)
    (fun tid ->
      forall+ (j : natlt ch) (k : natlt (n / (nthr * ch))).
        pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
          (s @! i + (k * (nthr * ch) + (tid * ch + j)))
    )
    fn tid {
      unfold thread_pts_to_chunks x s i nthr tid;
      forevery_map #(natlt (n / (nthr * (chunk et))))
        (fun k ->
          pts_to_vec' x ((k * nthr + tid) * chunk et)
            s (i + (k * nthr + tid) * chunk et)
        )
        (fun k ->
          forall+ (j : natlt ch).
            pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
              (s @! i + (k * (nthr * ch) + (tid * ch + j)))
        )
        fn k {
          rewrite each (v (chunk et)) as ch;

          slice_to_cells x
            ((k * nthr + tid) * ch)
            ((k * nthr + tid) * ch + ch);

          forevery_rw_size _ ch;
          forevery_map #(natlt ch)
            (fun j ->
              pts_to_cell x ((k * nthr + tid) * ch + j)
                (Seq.index
                  (Seq.slice s
                    (i + (k * nthr + tid) * ch)
                    (i + (k * nthr + tid) * ch + ch))
                  j)
            )
            (fun j ->
              pts_to_cell x (k * (nthr * ch) + (tid * ch + j))
                (s @! i + (k * (nthr * ch) + (tid * ch + j)))
            )
            fn j {
              chunk_cell_offset k nthr tid ch j;
              Seq.lemma_index_slice s
                (i + (k * nthr + tid) * ch)
                (i + (k * nthr + tid) * ch + ch)
                j;
              rewrite each
                ((k * nthr + tid) * ch + j)
              as (k * (nthr * ch) + (tid * ch + j));
              rewrite each
                (Seq.index
                  (Seq.slice s
                    (i + (k * nthr + tid) * ch)
                    (i + (k * nthr + tid) * ch + ch))
                  j)
              as (s @! i + (k * (nthr * ch) + (tid * ch + j)));
            };
        };
      rewrite each (v (chunk et)) as ch;
      forevery_commute _;
    };

  forevery_unfactor (nthr * ch) _ _
    (fun h ->
      forall+ (k : natlt (n / (nthr * ch))).
        pts_to_cell x
          (k * (nthr * ch) + h)
          (s @! i + (k * (nthr * ch) + h))
    );
  forevery_commute _;
  forevery_unfactor n (n / (nthr * ch)) (nthr * ch)
    (fun k -> pts_to_cell x k (s @! i + k));
  forevery_ext _
    (fun (k : natlt n) ->
      pts_to_cell x k (Seq.slice s i (i + n) @! k)
    );
  let _ = nonempty_intro (0 <: natlt n);
  array_unslice_1 x;
}
#pop-options


(* Join contiguous vector slices without imposing values on unwritten cells. *)
ghost
fn rec gather_live_vecs
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : pos)
  (x : larray et n)
  (count : pos { count * chunk et <= n })
  requires forall+ (k : natlt count). live_vec x (k * chunk et)
  ensures slice_live x 0 (count * chunk et)
  decreases count
{
  if (count = 1) {
    forevery_singleton_elim' (fun (k : natlt count) -> live_vec x (k * chunk et)) 0;
    rewrite each (0 * chunk et) as 0;
    unfold live_vec x 0;
    fold slice_live x 0 (count * chunk et);
  } else {
    forevery_natlt_pop count (fun k -> live_vec x (k * chunk et));
    gather_live_vecs x (count - 1);
    unfold slice_live x 0 ((count - 1) * chunk et);
    unfold live_vec x ((count - 1) * chunk et);
    slice_concat x 0 ((count - 1) * chunk et) (count * chunk et);
    fold slice_live x 0 (count * chunk et);
  }
}

ghost
fn thread_gather_live_chunks
  (#et : Type0) {| sized et, has_vec_cpy et |}
  (#n : pos)
  (x : larray et n)
  (nthr : pos)
  (#_: squash ((nthr * chunk et) /? n))
  requires forall+ (tid : natlt nthr). thread_live_chunks x nthr tid
  ensures live x
{
  let ch = v (chunk et);
  let count = n / (nthr * ch);
  assert pure (count * (nthr * ch) == n);
  assert pure (count > 0);
  forevery_map
    (fun tid -> thread_live_chunks x nthr tid)
    (fun tid -> forall+ (k : natlt count). live_vec x ((k * nthr + tid) * ch))
    fn tid {
      unfold thread_live_chunks x nthr tid;
      rewrite each (v (chunk et)) as ch;
      forevery_rw_size (n / (nthr * ch)) count;
    };
  forevery_commute _;
  forevery_unfactor (count * nthr) count nthr (fun k -> live_vec x (k * ch));
  rewrite each ch as (v (chunk et));
  gather_live_vecs x (count * nthr);
  rewrite each ((count * nthr) * chunk et) as n;
  unfold slice_live x 0 n;
  slice_to_array_full x;
  fold live x;
}
