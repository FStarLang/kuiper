module KPlay.Proof.Base.KernelSend

#lang-pulse

#set-options "--fuel 2 --ifuel 1 --z3rlimit 40"

open Kuiper
open Kuiper.Tensor
open Pulse.Lib.Array
open Pulse.Lib.Send
open FStar.Tactics.Typeclasses { solve }

module A = Pulse.Lib.Array
module Seq = FStar.Seq
module T = Kuiper.Tensor
module C = Kuiper.Chest
module Mem = Kuiper.Array.Core
module SH = Kuiper.SHMem
module Loc = Kuiper.Locs
module BF = Kuiper.BFloat16
module FP = Kuiper.Float32
module I64 = FStar.Int64
module D = KPlay.Proof.Shared.Selection
module Model = KPlay.Proof.Shared.Model
module Entry = KPlay.Proof.Base.Entry
module K = KPlay.Proof.Base.KernelState
module Owned = KPlay.Proof.Base.KernelOwnership
module Controls = KPlay.Proof.Base.KernelControls
module Book = KPlay.Proof.Base.WarpBook
module Head = KPlay.Proof.Base.Head
module Start = KPlay.Proof.Base.KernelStart
module Result = KPlay.Proof.Base.KernelResult
module Layout = KPlay.Proof.Base.KernelSetup
module Native = KPlay.Proof.Base.NativeProjection
module Hidden = KPlay.Proof.Base.WarpHidden
module Cache = KPlay.Proof.Base.CacheNative
module Init = KPlay.Proof.Base.Initialization
module Global = KPlay.Proof.Base.GlobalCache

inline_for_extraction noextract
let join (#p #q:slprop) (left:is_send_across Loc.block_of p) (right:is_send_across Loc.block_of q)
  : is_send_across Loc.block_of (p ** q) =
  is_send_across_star p q #left #right

inline_for_extraction noextract
let quantify (#a:Type0) (#p:a -> slprop) (move:(x:a -> is_send_across Loc.block_of (p x)))
  : is_send_across Loc.block_of (exists* (x:a). p x) =
  is_send_across_exists p #move

inline_for_extraction noextract
let join_gpu (#p #q:slprop) (left:is_send_across Loc.gpu_of p) (right:is_send_across Loc.gpu_of q)
  : is_send_across Loc.gpu_of (p ** q) =
  is_send_across_star p q #left #right

inline_for_extraction noextract
let quantify_gpu (#a:Type0) (#p:a -> slprop) (move:(x:a -> is_send_across Loc.gpu_of (p x)))
  : is_send_across Loc.gpu_of (exists* (x:a). p x) =
  is_send_across_exists p #move

inline_for_extraction noextract
instance global_array (#a:Type0) (array:A.array a{Mem.is_global_array array}) (#fraction:perm) (values:Seq.seq a)
  : is_send_across Loc.gpu_of (A.pts_to array #fraction values) =
  Mem.is_send_pts_to array #fraction values

inline_for_extraction noextract
instance global_array_block (#a:Type0) (array:A.array a{Mem.is_global_array array}) (#fraction:perm) (values:Seq.seq a)
  : is_send_across Loc.block_of (A.pts_to array #fraction values) =
  Loc.send_across_if_send_across_gpu (A.pts_to array #fraction values) (global_array array #fraction values)

inline_for_extraction noextract
instance global_mask_block (#a:Type0) (array:A.array a{Mem.is_global_array array})
  (#fraction:perm) (values:Seq.seq (option a)) (mask:nat -> prop)
  : is_send_across Loc.block_of (A.pts_to_mask array #fraction values mask) =
  Loc.send_across_if_send_across_gpu (A.pts_to_mask array #fraction values mask)
    (Mem.is_send_across_global_array_mask array #fraction values mask)

inline_for_extraction noextract
let shared_locations (sh:Layout.shared)
  : Lemma
      (requires SH.c_shmems_inv sh)
      (ensures
        Mem.visibility_of (T.core (Layout.hidden sh)) == Loc.block_of /\
        Mem.visibility_of (Layout.best sh) == Loc.block_of /\
        Mem.visibility_of (Layout.next sh) == Loc.block_of) =
  norm_spec [zeta; iota; delta_only [
    "Kuiper.SHMem.c_shmems_inv"; "Kuiper.SHMem.c_shmems_block_inv"; "Kuiper.SHMem.c_shmem_inv";
    "Kuiper.SHMem.is_block_array"; "KPlay.Proof.Base.KernelSetup.declarations";
    "KPlay.Proof.Base.KernelSetup.hidden"; "KPlay.Proof.Base.KernelSetup.best"; "KPlay.Proof.Base.KernelSetup.next"]]
    (SH.c_shmems_inv sh);
  T.lem_from_array_core #BF.t #2 #_ #Native.hidden_layout sh._1

inline_for_extraction noextract
instance shared_hidden (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  (#fraction:perm) (values:Hidden.snapshot)
  : is_send_across Loc.block_of (Layout.hidden sh |-> Frac fraction values) =
  shared_locations sh;
  T.is_send_across_tensor (Layout.hidden sh) Loc.block_of #() #fraction values

inline_for_extraction noextract
instance shared_best (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  (#fraction:perm) (values:Seq.seq FP.t)
  : is_send_across Loc.block_of (A.pts_to (Layout.best sh) #fraction values) =
  shared_locations sh;
  Mem.is_send_pts_to (Layout.best sh) #fraction values

inline_for_extraction noextract
instance shared_next (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  (#fraction:perm) (values:Seq.seq I64.t)
  : is_send_across Loc.block_of (A.pts_to (Layout.next sh) #fraction values) =
  shared_locations sh;
  Mem.is_send_pts_to (Layout.next sh) #fraction values

inline_for_extraction noextract
let readers (call:Model.invocation) (args:Entry.arguments call{Layout.device_arguments call args})
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length) (fraction:perm)
  : is_send_across Loc.gpu_of (Layout.readers call args vw sp fraction) =
  T.lem_from_array_core #BF.t #2 #_ #Native.weight_layout args.Entry.weight;
  T.lem_is_global_iff_core (Layout.weight call args);
  let weight = T.is_send_across_global_tensor (Layout.weight call args) #fraction vw in
  let prompt = global_array args.Entry.prompt #fraction sp in
  solve

inline_for_extraction noextract
let pre (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  : is_send_across Loc.gpu_of (Layout.pre call refs args vw sp prompt fraction) =
  let input = readers call args vw sp fraction in
  solve

inline_for_extraction noextract
let result (call:Model.invocation) (args:Entry.arguments call{Layout.device_arguments call args})
  (values:KPlay.Proof.Base.Transfer.packed) (prompt:D.vec (D.token 248320) call.Model.prompt_length)
  : is_send_across Loc.gpu_of (Result.post call args values prompt) =
  quantify_gpu (fun (m:K.metadata call) ->
    quantify_gpu (fun (kv:Cache.snapshot) ->
      join_gpu (global_array args.Entry.state #1.0R m.K.memory.K.state)
        (join_gpu (global_array args.Entry.convolution #1.0R m.K.memory.K.convolution)
        (join_gpu (Mem.is_send_across_global_array_mask args.Entry.cache #1.0R kv Init.all_cells)
        (join_gpu (Mem.is_send_across_global_array_mask args.Entry.tokens #1.0R m.K.memory.K.tokens Init.all_cells)
        (join_gpu (Mem.is_send_across_global_array_mask args.Entry.hidden #1.0R m.K.memory.K.hidden Init.all_cells)
          (solve #(is_send_across Loc.gpu_of (pure (Result.complete call values prompt m kv))))))))))

inline_for_extraction noextract
let post (call:Model.invocation) (args:Entry.arguments call{Layout.device_arguments call args})
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  : is_send_across Loc.gpu_of (Layout.post call args vw sp prompt fraction) =
  join_gpu (readers call args vw sp fraction)
    (join_gpu (result call args (Native.packed_values vw) prompt)
      (solve #(is_send_across Loc.gpu_of
        (pure (forall (i:KPlay.Proof.Shared.Projection.index call.Model.prompt_length).
          I64.v (Seq.index sp i) == Seq.index prompt i)))))

inline_for_extraction noextract
let pending_witness (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  (state:lseq (option FP.t) 4718592) (conv:lseq (option BF.t) 442368) (kv:Cache.snapshot)
  (tokens:lseq (option I64.t) call.Model.decode_steps)
  (history:lseq (option BF.t) (call.Model.decode_steps * 1024))
  (h:Hidden.snapshot) (b:lseq FP.t 1) (n:lseq I64.t 1)
  : is_send_across Loc.block_of (
      A.pts_to_mask args.Entry.state state Init.all_cells **
      A.pts_to_mask args.Entry.convolution conv Init.all_cells **
      A.pts_to_mask args.Entry.cache kv Init.all_cells **
      A.pts_to_mask args.Entry.tokens tokens Init.all_cells **
      A.pts_to_mask args.Entry.hidden history Init.all_cells **
      Layout.hidden sh |-> h ** Layout.best sh |-> b ** Layout.next sh |-> n **
      Owned.parked refs.Owned.global ** Owned.parked refs.Owned.linear ** Owned.parked refs.Owned.attention **
      pure (Global.available_contents kv)) =
  let parked = join (solve #(is_send_across Loc.block_of (Owned.parked refs.Owned.global)))
    (join (solve #(is_send_across Loc.block_of (Owned.parked refs.Owned.linear)))
    (join (solve #(is_send_across Loc.block_of (Owned.parked refs.Owned.attention)))
      (solve #(is_send_across Loc.block_of (pure (Global.available_contents kv)))))) in
  let shared = join (shared_hidden sh #proof #1.0R h)
    (join (shared_best sh #proof #1.0R b)
    (join (shared_next sh #proof #1.0R n) parked)) in
  join (global_mask_block args.Entry.state #1.0R state Init.all_cells)
    (join (global_mask_block args.Entry.convolution #1.0R conv Init.all_cells)
    (join (global_mask_block args.Entry.cache #1.0R kv Init.all_cells)
    (join (global_mask_block args.Entry.tokens #1.0R tokens Init.all_cells)
    (join (global_mask_block args.Entry.hidden #1.0R history Init.all_cells) shared))))

inline_for_extraction noextract
let pending (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  : is_send_across Loc.block_of (Start.pending call refs args (Layout.hidden sh) (Layout.best sh) (Layout.next sh)) =
  quantify (fun (state:lseq (option FP.t) 4718592) ->
    quantify (fun (conv:lseq (option BF.t) 442368) ->
    quantify (fun (kv:Cache.snapshot) ->
    quantify (fun (tokens:lseq (option I64.t) call.Model.decode_steps) ->
    quantify (fun (history:lseq (option BF.t) (call.Model.decode_steps * 1024)) ->
    quantify (fun (h:Hidden.snapshot) ->
    quantify (fun (b:lseq FP.t 1) ->
    quantify (fun (n:lseq I64.t 1) ->
      pending_witness call refs args sh #proof state conv kv tokens history h b n))))))))

inline_for_extraction noextract
let local_owners (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh)) (m:K.metadata call)
  : is_send_across Loc.block_of (Owned.local_owners call refs args (Layout.best sh) (Layout.next sh) m) =
  let other = join (global_mask_block args.Entry.tokens #1.0R m.K.memory.K.tokens Init.all_cells)
    (join (global_mask_block args.Entry.hidden #1.0R m.K.memory.K.hidden Init.all_cells)
    (join (shared_best sh #proof #1.0R (Seq.create 1 m.K.memory.K.best))
      (shared_next sh #proof #1.0R (Seq.create 1 m.K.memory.K.next)))) in
  let memory = join (global_array_block args.Entry.state #1.0R m.K.memory.K.state)
    (join (global_array_block args.Entry.convolution #1.0R m.K.memory.K.convolution) other) in
  join memory
    (join (solve #(is_send_across Loc.block_of (Owned.parked refs.Owned.linear)))
      (solve #(is_send_across Loc.block_of (Owned.parked refs.Owned.attention))))

inline_for_extraction noextract
let book (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh))
  (valid:K.metadata call -> Hidden.snapshot -> Cache.snapshot -> prop)
  : is_send_across Loc.block_of
      (Book.full refs.Owned.global (Layout.hidden sh) args.Entry.cache
        (Owned.local_owners call refs args (Layout.best sh) (Layout.next sh)) valid) =
  quantify (fun (m:K.metadata call) ->
    quantify (fun (h:Hidden.snapshot) ->
    quantify (fun (kv:Cache.snapshot) ->
      join (solve #(is_send_across Loc.block_of (Pulse.Lib.GhostReference.pts_to refs.Owned.global m)))
        (join (shared_hidden sh #proof #1.0R h)
        (join (global_mask_block args.Entry.cache #1.0R kv Init.all_cells)
        (join (local_owners call refs args sh #proof m)
          (solve #(is_send_across Loc.block_of (pure (valid m h kv))))))))))

inline_for_extraction noextract
let lane (i:Head.lane) (p:slprop) (move:is_send_across Loc.block_of p)
  : is_send_across Loc.block_of (Head.lane_only i p) =
  if i = 0 then move else solve

inline_for_extraction noextract
let kpre (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh)) (i:Head.lane)
  : is_send_across Loc.block_of (Layout.kpre call refs args vw sp prompt fraction sh i) =
  let input = Loc.send_across_if_send_across_gpu (Layout.readers call args vw sp (fraction /. 32.0R))
    (readers call args vw sp (fraction /. 32.0R)) in
  let owned = lane i (Start.pending call refs args (Layout.hidden sh) (Layout.best sh) (Layout.next sh))
    (pending call refs args sh #proof) in
  join input (join owned (solve #(is_send_across Loc.block_of
    (pure (forall (j:KPlay.Proof.Shared.Projection.index call.Model.prompt_length).
      I64.v (Seq.index sp j) == Seq.index prompt j)))))

inline_for_extraction noextract
let kpost (call:Model.invocation) (refs:Owned.controls call) (args:Entry.arguments call{Layout.device_arguments call args})
  (vw:C.chest2 BF.t 752393024 1) (sp:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (fraction:perm)
  (sh:Layout.shared) (#proof:squash (SH.c_shmems_inv sh)) (i:Head.lane)
  : is_send_across Loc.block_of (Layout.kpost call refs args vw sp prompt fraction sh i) =
  let input = Loc.send_across_if_send_across_gpu (Layout.readers call args vw sp (fraction /. 32.0R))
    (readers call args vw sp (fraction /. 32.0R)) in
  let owned = lane i
    (Book.full refs.Owned.global (Layout.hidden sh) args.Entry.cache
      (Owned.local_owners call refs args (Layout.best sh) (Layout.next sh))
      (K.ready call (Native.packed_values vw) prompt (Model.calls call)))
    (book call refs args sh #proof (K.ready call (Native.packed_values vw) prompt (Model.calls call))) in
  join input (join owned (solve #(is_send_across Loc.block_of
    (pure (forall (j:KPlay.Proof.Shared.Projection.index call.Model.prompt_length).
      I64.v (Seq.index sp j) == Seq.index prompt j)))))
