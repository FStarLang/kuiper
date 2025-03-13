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
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                shmem_tok ar **
                shared_pre ar ebid etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                // shmem_tok ar **
                shared_post ar ebid etid **
                post ebid etid)
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

let norpm (n:nat) : rpm_t n = fun _ _ _ -> emp

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
  open Pulse.Lib.BigStar;
  drop_ (exists* v. gpu_pts_to_array #u32 #0 ar #1.0R v);
  mk_mbarrier nthr p;
  bigstar_eta ();
  with p.
    rewrite
      bigstar 0 nthr p
    as
      bigstar 0 (Enumerable.cardinal (natlt nthr) #_) p;
  forevery_fromstar #(natlt nthr) _;
}

inline_for_extraction noextract
fn kernel_no_shmem
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                mbarrier_tok nthr p 0 etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                (exists* it. mbarrier_tok nthr p it etid) **
                post ebid etid)
  )
  (ar : erased (gpu_array u32 0))
  (ebid : enatlt nblk)
  (etid : enatlt nthr)
  requires
    gpu **
    block_id nblk ebid **
    thread_id nthr etid **
    shmem_tok ar **
    barrier_shared_pre nblk nthr p ar ebid etid **
    pre ebid etid
  ensures
    gpu **
    block_id nblk ebid **
    thread_id nthr etid **
    barrier_shared_post nblk nthr p ar ebid etid **
    post ebid etid
{
  drop_ (shmem_tok ar);
  unfold barrier_shared_pre;
  k ebid etid;
  fold barrier_shared_post nblk nthr p ar ebid etid;
}

inline_for_extraction noextract
fn launch_kernel_n_m_barrier_async
  (nblk : szp { nblk <= max_blocks })
  (nthr : szp { nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (#p : rpm_t nthr)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                mbarrier_tok nthr p 0 etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                (exists* it. mbarrier_tok nthr p it etid) **
                post ebid etid)
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
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                mbarrier_tok nthr p 0 etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                (exists* it. mbarrier_tok nthr p it etid) **
                post ebid etid)
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

inline_for_extraction noextract
fn no_barrier
  (#nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k0 :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                post ebid etid)
  )
  (ebid : enatlt nblk)
  (etid : enatlt nthr)
  requires
    gpu **
    block_id nblk ebid **
    thread_id nthr etid **
    mbarrier_tok nthr (norpm nthr) 0 etid **
    pre ebid etid
  ensures
    gpu **
    block_id nblk ebid **
    thread_id nthr etid **
    (exists* it. mbarrier_tok nthr (norpm nthr) it etid) **
    post ebid etid
{
  k0 ebid etid;
}

inline_for_extraction noextract
fn launch_kernel_n_m_async
  (nblk : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (nthr : SZ.t { 0 < nthr /\ nthr <= max_threads })
  (#pre #post : natlt nblk -> natlt nthr -> slprop)
  (k :
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                post ebid etid)
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
    (ebid : enatlt nblk) ->
    (etid : enatlt nthr) ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                pre ebid etid)
      (fun _ -> gpu **
                block_id nblk ebid **
                thread_id nthr etid **
                post ebid etid)
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
  (k0 :
    (ebid : enatlt nblk ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                pre ebid)
      (fun _ -> gpu **
                block_id nblk ebid **
                post ebid)
  ))
  (ebid : enatlt nblk)
  (etid : enatlt 1) (* == 0 *)
  requires
    gpu **
    block_id nblk ebid **
    thread_id 1 etid **
    mbarrier_tok 1 (norpm 1) 0 etid **
    pre ebid
  ensures
    gpu **
    block_id nblk ebid **
    thread_id 1 etid **
    (exists* it. mbarrier_tok 1 (norpm 1) it etid) **
    post ebid
{
  k0 ebid;
}

inline_for_extraction noextract
fn launch_kernel_n_blocks_async
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : natlt nblk -> slprop)
  (k :
    (ebid : enatlt nblk ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                pre ebid)
      (fun _ -> gpu **
                block_id nblk ebid **
                post ebid)
  ))
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
  forevery_ext_2
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
    forevery_ext_2
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
fn launch_kernel_n_blocks
  (nblk  : SZ.t { 0 < nblk /\ nblk <= max_blocks })
  (#pre #post : (natlt nblk -> slprop))
  (k :
    (ebid : enatlt nblk ->
    stt unit
      (         gpu **
                block_id nblk ebid **
                pre ebid)
      (fun _ -> gpu **
                block_id nblk ebid **
                post ebid)
  ))
  requires
    cpu **
    (forall+ (b : natlt nblk). pre b)
  ensures
    cpu **
    (forall+ (b : natlt nblk). post b)
{
  get_epoch ();
  let e' = launch_kernel_n_blocks_async nblk #pre #post k;
  sync ();
  redeem_pledge emp_inames (epoch_done e') _;
  drop_ (epoch_done e');
  drop_ (epoch_live _);
}

(* f<<<1, 1>>>(...); *)
// Private
inline_for_extraction noextract
fn kernel_1_as_n_blocks
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  (ebid : enatlt 1)
  requires gpu ** block_id 1 ebid ** pre
  ensures  gpu ** block_id 1 ebid ** post
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
  let e' = launch_kernel_n_blocks_async 1sz (kernel_1_as_n_blocks k);
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

(* f<<<1, 1>>>(...); *)

inline_for_extraction noextract
fn frame_right1 (fr1 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1
  ensures  q ** fr1
  { f (); }

inline_for_extraction noextract
fn frame_right2 (fr1 fr2 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2
  ensures  q ** fr1 ** fr2
  { f (); }

inline_for_extraction noextract
fn frame_right3 (fr1 fr2 fr3 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2 ** fr3
  ensures  q ** fr1 ** fr2 ** fr3
  { f (); }

inline_for_extraction noextract
fn frame_right4 (fr1 fr2 fr3 fr4 : slprop) (#p #q :slprop)
  (f : unit -> stt unit p (fun _ -> q))
  requires p ** fr1 ** fr2 ** fr3 ** fr4
  ensures  q ** fr1 ** fr2 ** fr3 ** fr4
  { f (); }

inline_for_extraction noextract
let mk_desc_1
  (#pre #post : slprop)
  (k : unit ->
    stt unit (gpu ** pre) (fun _ -> gpu ** post)
  )
  : kernel_desc pre post
  = { Kernel.Nop.nop_desc
      with
      kpre = (fun _ _ -> pre);
      kpost = (fun _ _ -> post);
      f = (fun _shmem _ebid _etid -> frame_right4 _ _ _ _ k);

      setup = magic();
      teardown = magic();
  }

(* f<<<1, 1>>>(...); *)

inline_for_extraction noextract
fn launch_kernel_sync
  (#full_pre : slprop)
  (#full_post : slprop)
  (k : kernel_desc full_pre full_post)
  requires
    cpu **
    full_pre
  ensures
    cpu **
    full_post
{
  get_epoch ();
  launch_kernel k;
  sync ();
  redeem_pledge emp_inames (epoch_done _) full_post;
  drop_ (epoch_done _);
  drop_ (epoch_live _);
}

inline_for_extraction noextract
let launch_kernel_1 #pre #post kf = launch_kernel_sync (mk_desc_1 kf)
