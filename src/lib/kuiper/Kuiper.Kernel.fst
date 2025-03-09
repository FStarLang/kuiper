module Kuiper.Kernel
#lang-pulse

open Pulse.Lib.Core
open FStar.Ghost
open Kuiper.SizeT
open Kuiper.IntAliases
open Kuiper.Array
open Kuiper.Base
open Kuiper.Barrier.RPM
open FStar.Mul
module SZ = FStar.SizeT
open Kuiper.Kernel.Base
open Kuiper.Scalars {} // instances

inline_for_extraction noextract
fn launch_kernel_n_m_shmem
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (a : Type u#0) {| Kuiper.Sized.sized a |}
  (smem_sz : SZ.t)
  (#shared_pre #shared_post : gpu_array a smem_sz -> natlt nblk -> natlt nthr -> slprop)
  (setup : (ar: gpu_array a smem_sz) -> (bid: natlt nblk) ->
    stt_ghost unit emp_inames
      (block_setup nthr ** (exists* v. gpu_pts_to_array #a #smem_sz ar #1.0R v))
      (fun _ -> block_setup nthr ** (forall+ (i : natlt nthr). shared_pre ar bid i)))
  (k :
    (ar: erased (gpu_array a smem_sz)) ->
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu **
                thread_id etid **
                shmem_tok ar **
                shared_pre ar (bidx_x etid) (tidx_x etid) **
                pre (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu **
                thread_id etid **
                // shmem_tok ar **
                shared_post ar (bidx_x etid) (tidx_x etid) **
                post (bidx_x etid) (tidx_x etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)
{
  let e = get_epoch ();
  let e' = launch_kernel_n_m_shmem_async nblk nthr #pre #post a #_ smem_sz #shared_pre #shared_post setup k #e;
  sync ();
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

let barrier_shared_pre
  (nblk : nat { nblk <= max_blocks })
  (nthr : nat { nthr <= max_threads })
  (p : rpm_t nthr)
  (ar : gpu_array u32 0)
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
  = mbarrier_tok nthr p 0 tid

let barrier_shared_post
  (nblk : nat { nblk <= max_blocks })
  (nthr : nat { nthr <= max_threads })
  (p : rpm_t nthr)
  (ar : gpu_array u32 0)
  (bid : natlt nblk)
  (tid : natlt nthr)
  : slprop
  = exists* it. mbarrier_tok nthr p it tid

ghost
fn barrier_setup
  (nblk : pos { nblk <= max_blocks })
  (nthr : pos { nthr <= max_threads })
  (p : rpm_t nthr)
  (ar : gpu_array u32 0)
  (bid : natlt nblk)
  requires
    block_setup nthr ** (exists* v. gpu_pts_to_array #u32 #0 ar #1.0R v)
  ensures
    block_setup nthr **
    (forall+ (i : natlt nthr). barrier_shared_pre nblk nthr p ar bid i)
{
  (* TODO *)
  admit ();
}

inline_for_extraction noextract
fn kernel_no_shmem
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu **
                thread_id etid **
                mbarrier_tok nthr p 0 (tidx_x etid) **
                pre (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu **
                thread_id etid **
                (exists* it. mbarrier_tok nthr p it (tidx_x etid)) **
                post (bidx_x etid) (tidx_x etid))
  )
  (ar : erased (gpu_array u32 0))
  (etid : tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr })
  requires
    gpu **
    thread_id etid **
    shmem_tok ar **
    barrier_shared_pre nblk nthr p ar (bidx_x etid) (tidx_x etid) **
    pre (bidx_x etid) (tidx_x etid)
  ensures
    gpu **
    thread_id etid **
    barrier_shared_post nblk nthr p ar (bidx_x etid) (tidx_x etid) **
    post (bidx_x etid) (tidx_x etid)
{
  drop_ (shmem_tok ar);
  unfold barrier_shared_pre;
  k etid;
  fold barrier_shared_post nblk nthr p ar (bidx_x etid) (tidx_x etid);
}

inline_for_extraction noextract
fn launch_kernel_n_m_barrier_async
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu **
                thread_id etid **
                mbarrier_tok nthr p 0 (tidx_x etid) **
                pre (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu **
                thread_id etid **
                (exists* it. mbarrier_tok nthr p it (tidx_x etid)) **
                post (bidx_x etid) (tidx_x etid))
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e')
      (forall+ (b : natlt nblk) (t : natlt nthr). post b t) **
    pure (e' >= e)
{
  let e' =
  launch_kernel_n_m_shmem_async
    nblk nthr
    #pre #post
    u32 #_
    0sz
    #(barrier_shared_pre nblk nthr p)
    #(barrier_shared_post nblk nthr p)
    (barrier_setup nblk nthr p)
    (kernel_no_shmem nblk nthr #pre #post #p k)
    ;
  e'
}

inline_for_extraction noextract
fn launch_kernel_n_m_barrier
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu **
                thread_id etid **
                mbarrier_tok nthr p 0 (tidx_x etid) **
                pre (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu **
                thread_id etid **
                (exists* it. mbarrier_tok nthr p it (tidx_x etid)) **
                post (bidx_x etid) (tidx_x etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)
{
  get_epoch ();
  let e' = launch_kernel_n_m_barrier_async nblk nthr k;
  sync ();
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

let norpm (n:nat) : rpm_t n = fun _ _ _ -> emp

inline_for_extraction noextract
fn no_barrier
  (#nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k0 :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu ** thread_id etid ** pre  (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu ** thread_id etid ** post (bidx_x etid) (tidx_x etid))
  )
  (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok nthr (norpm nthr) 0 (tidx_x etid) **
    pre (bidx_x etid) (tidx_x etid)
  ensures
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok nthr (norpm nthr) it (tidx_x etid)) **
    post (bidx_x etid) (tidx_x etid)
{
  k0 etid;
}

inline_for_extraction noextract
fn launch_kernel_n_m_async
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu ** thread_id etid ** pre  (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu ** thread_id etid ** post (bidx_x etid) (tidx_x etid))
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e')
      (forall+ (b : natlt nblk) (t : natlt nthr). post b t) **
    pure (e' >= e)
{
  launch_kernel_n_m_barrier_async nblk nthr #pre #post (no_barrier k);
}

(* f<<<nblk, nthr>>>(...); *)
inline_for_extraction noextract
fn launch_kernel_n_m
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k :
    (etid: tid_t { gdim_x etid == nblk /\ bdim_x etid == nthr }) ->
    stt unit
      (         gpu ** thread_id etid ** pre  (bidx_x etid) (tidx_x etid))
      (fun _ -> gpu ** thread_id etid ** post (bidx_x etid) (tidx_x etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). pre b t)
  ensures
    cpu **
    (forall+ (b : natlt nblk) (t : natlt nthr). post b t)
{
  launch_kernel_n_m_barrier nblk nthr #pre #post (no_barrier k);
}

inline_for_extraction noextract
fn kernel_n_as_n_m
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (tid:nat{ 0 <= tid /\ tid < SZ.v nblk } -> slprop))
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (bidx_x etid))
             (fun _ -> gpu ** thread_id etid ** post (bidx_x etid))
  )
  (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz })
  requires
    gpu **
    thread_id etid **
    mbarrier_tok 1 (norpm 1) 0 (tidx_x etid) **
    pre (bidx_x etid)
  ensures
    gpu **
    thread_id etid **
    (exists* it. mbarrier_tok 1 (norpm 1) it (tidx_x etid)) **
    post (bidx_x etid)
{
  k etid;
}

inline_for_extraction noextract
fn launch_kernel_n_async
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : natlt nblk -> slprop)
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (bidx_x etid))
             (fun _ -> gpu ** thread_id etid ** post (bidx_x etid))
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    (forall+ (b : natlt nblk). pre b)
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e')
      (forall+ (b : natlt nblk). post b) **
    pure (e' >= e)
{
  forevery_factor nblk nblk 1 _;
  forevery_ext2
    #(natlt nblk) #_
    #(natlt 1) #_
    (fun (b:natlt nblk) (t:natlt 1) -> pre (b*1 + t))
    (fun (b:natlt nblk) (t:natlt 1) -> pre b);
  let e' = launch_kernel_n_m_barrier_async nblk 1sz #(fun b t -> pre b) #(fun b t -> post b) (kernel_n_as_n_m nblk #pre #post k);
  ghost
  fn aux ()
    requires 
      forall+ (b:natlt nblk) (t:natlt 1). post b
    ensures
      forall+ (b:natlt nblk). post b
  {
    forevery_ext2
      #(natlt nblk) #_
      #(natlt 1) #_
      (fun (b:natlt nblk) (t:natlt 1) -> post b)
      (fun (b:natlt nblk) (t:natlt 1) -> post (b*1 + t));
    forevery_unfactor nblk nblk 1 _;
  };
  rewrite_pledge _ _ aux;
  e'
}

inline_for_extraction noextract
fn launch_kernel_n
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : natlt nblk -> slprop)
  (k :
    (etid:tid_t { gdim_x etid == nblk /\ bdim_x etid == 1sz }) ->
    stt unit (gpu ** thread_id etid ** pre (thread_index etid))
             (fun _ -> gpu ** thread_id etid ** post (thread_index etid))
  )
  requires
    cpu **
    (forall+ (b : natlt nblk). pre b)
  ensures
    cpu **
    (forall+ (b : natlt nblk). post b)
{
  get_epoch ();
  let e' = launch_kernel_n_async nblk #pre #post k;
  sync ();
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

(* f<<<1, 1>>>(...); *)
// Private
inline_for_extraction noextract
fn kernel_1_as_n
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (etid:tid_t { gdim_x etid == 1sz /\ bdim_x etid == 1sz })
  requires gpu ** thread_id etid ** pre
  ensures  gpu ** thread_id etid ** post
{
  k ()
}

inline_for_extraction noextract
fn launch_kernel_1_async
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (#e : epoch_t)
  requires
    cpu **
    epoch_live e **
    pre
  returns
    e' : epoch_t
  ensures
    cpu **
    epoch_live e' **
    pledge0 (epoch_done e') post **
    pure (e' >= e)
{
  forevery_unit_intro pre;
  forevery_iso Enumerable.bij_unit _;
  let e' = launch_kernel_n_async 1sz (kernel_1_as_n k);
  ghost
  fn aux ()
    requires forevery (natlt 1) (fun _ -> post)
    ensures  post
  {
    forevery_iso (Bijection.bij_sym Enumerable.bij_unit) _;
    forevery_unit_elim post;
  };
  rewrite_pledge
    (forevery (natlt 1) (fun _ -> post))
    post
    aux;
  e'
}

inline_for_extraction noextract
fn launch_kernel_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  requires cpu ** pre
  ensures  cpu ** post
{
  get_epoch ();
  launch_kernel_1_async #pre #post k;
  sync ();
  with e'. assert (epoch_done e');
  redeem_pledge emp_inames (epoch_done e') post;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

let lemma_mul_lt (a b: nat) (c: nat { a < c }) (d: nat { b <= d /\ d > 0 }): Lemma (a * b < c * d) = ()

inline_for_extraction noextract
fn thread_idx_all () (#n: tid_t)
  preserves
    thread_id n
  requires
    emp
  returns
    id : SZ.t
  ensures
    pure (SZ.v id == thread_index n /\ SZ.v id < max_blocks * max_threads)
{
  assert (pure (bidx_x n < 1024 * 1024 * 1024 /\ tidx_x n < 1024 /\ bdim_x n <= 1024));
  lemma_mul_lt (bidx_x n) (bdim_x n) (1024 * 1024 * 1024) 1024;
  assert (pure (bidx_x n * tidx_x n < 1024 * 1024 * 1024 * 1024 /\ bdim_x n <= 1024));
  let bid = block_idx_x ();
  let bdim = block_dim_x ();
  let tid = thread_idx_x ();
  open FStar.SizeT;
  let r = (bid *^ bdim) +^ tid;
  r
}
