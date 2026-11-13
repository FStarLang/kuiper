module KPlay.Proof.BaseNative

#lang-pulse

open Kuiper
open Pulse.Lib.Send
open Kuiper.Tensor.Runtime.Helpers

module PA = Pulse.Lib.Array
module KA = Kuiper.Array
module S = FStar.Seq
module C = Kuiper.Chest
module RT = Kuiper.Tensor.Runtime
module Input = Kuiper.Tensor.Input
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module Kernel = Kuiper.Kernel
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module Shared = KPlay.SharedSpec
module Native = KPlay.Proof.Shared.Native
module Capacity = KPlay.Proof.Shared.Capacity
module BC = KPlay.Proof.BaseCapacity
module History = KPlay.Proof.BaseHistory
module Caller = KPlay.Proof.BaseCaller

inline_for_extraction noextract
let token_values = lseq bf16 (256 * 512)

inline_for_extraction noextract
let weight_values = lseq bf16 (8 * 512)

inline_for_extraction noextract
let token_model (xs:token_values) : Shared.token_matrix =
  C.mk2 (fun t h -> S.index xs (t * 512 + h))

inline_for_extraction noextract
let weight_model (ws:weight_values) : Shared.weight_matrix =
  C.mk2 (fun h e -> S.index ws (e * 512 + h))

inline_for_extraction noextract
let token_argument = value:Input.argument bf16{Input.dimensions value == [256; 512]}

inline_for_extraction noextract
let weight_argument = value:Input.argument bf16{Input.dimensions value == [8; 512]}

inline_for_extraction noextract
let garray (a:Type0) (n:nat) = ar:larray a n{KA.is_global_array ar}

inline_for_extraction noextract
instance global_pts_to (#a:Type0)
  (ar:PA.array a{KA.is_global_array ar}) (#f:perm) (#values:S.seq a)
  : is_send_across KL.gpu_of (PA.pts_to ar #f values) =
  let send:is_send_across (KA.visibility_of ar) (PA.pts_to ar #f values) =
    FStar.Tactics.Typeclasses.solve in
  send

inline_for_extraction noextract
let device_pre (xs:token_values) (ws:weight_values) (fp fw:perm)
  (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (demand:garray i32 8) (indices:garray i64 256)
  (probabilities:garray f32 256) (survival:garray i32 2048) : slprop =
  PA.pts_to tokens #fp xs ** PA.pts_to weights #fw ws **
  (exists* (ds:lseq i32 8) (is:lseq i64 256) (ps:lseq f32 256) (ss:lseq i32 2048).
    PA.pts_to demand ds ** PA.pts_to indices is **
    PA.pts_to probabilities ps ** PA.pts_to survival ss) **
  pure (Shared.input_domain (token_model xs) (weight_model ws))

inline_for_extraction noextract
let device_post (xs:token_values) (ws:weight_values) (fp fw:perm)
  (tokens:garray bf16 131072) (weights:garray bf16 4096)
  (demand:garray i32 8) (indices:garray i64 256)
  (probabilities:garray f32 256) (survival:garray i32 2048) : slprop =
  PA.pts_to tokens #fp xs ** PA.pts_to weights #fw ws **
  (exists* (ds:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256).
    PA.pts_to demand ds ** PA.pts_to indices st.indices **
    PA.pts_to probabilities st.weights ** PA.pts_to survival st.flags **
    pure (Shared.task_relation (token_model xs) (weight_model ws)
      (token_model xs) (weight_model ws) (History.output ds st)))

inline_for_extraction noextract
let source_body (xs:token_values) (ws:weight_values) (fp fw:perm) =
  tokens:garray bf16 131072 -> weights:garray bf16 4096 ->
  demand:garray i32 8 -> indices:garray i64 256 ->
  probabilities:garray f32 256 -> survival:garray i32 2048 ->
  unit -> stt unit
    (gpu ** device_pre xs ws fp fw tokens weights demand indices probabilities survival)
    (fun _ -> gpu **
      device_post xs ws fp fw tokens weights demand indices probabilities survival)

inline_for_extraction noextract
let result_values = values:Native.output_values{
  S.length values.demand_values == 8 /\
  S.length values.index_values == 256 /\
  S.length values.probability_values == 256 /\
  S.length values.survival_values == 2048}

inline_for_extraction noextract
let public_view (values:result_values) : Shared.public_outputs = {
  Capacity.expert_demand = (fun e -> S.index values.demand_values e);
  Capacity.token_indices = (fun e c -> S.index values.index_values (e * 32 + c));
  Capacity.probabilities = (fun e c -> S.index values.probability_values (e * 32 + c));
  Capacity.survival = (fun t e -> S.index values.survival_values (t * 8 + e))
}

inline_for_extraction noextract
let output_values (ds:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256)
  : result_values = {
  Native.demand_values = ds;
  Native.index_values = st.indices;
  Native.probability_values = st.weights;
  Native.survival_values = st.flags
}

inline_for_extraction noextract
let public_view_matches (ds:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256)
  : Lemma (public_view (output_values ds st) == History.output ds st)
  = assert_norm (public_view (output_values ds st) == History.output ds st)

inline_for_extraction noextract
let output_post (handles:Native.output_handles) (xs:token_values) (ws:weight_values)
  : slprop =
  exists* (values:result_values).
    Native.output_cells handles values **
    pure (Native.output_binding handles values (public_view values) /\
      Shared.task_relation (token_model xs) (weight_model ws)
        (token_model xs) (weight_model ws) (public_view values))

inline_for_extraction noextract
let payload_pre (tokens:token_argument) (weights:weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm) (handles:Native.output_handles)
  : slprop =
  Native.input_cells tokens weights fp fw xs ws ** Caller.contents handles

inline_for_extraction noextract
let payload_post (tokens:token_argument) (weights:weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm) (handles:Native.output_handles)
  : slprop =
  Native.input_cells tokens weights fp fw xs ws ** output_post handles xs ws

inline_for_extraction noextract
ghost
fn rebind_cells (#a:Type0) (before after:PA.array a) (#f:perm) (#values:S.seq a)
  requires on gpu_loc (PA.pts_to before #f values) ** pure (before == after)
  ensures on gpu_loc (PA.pts_to after #f values)
{
  rewrite on gpu_loc (PA.pts_to before #f values)
    as on gpu_loc (PA.pts_to after #f values);
}

inline_for_extraction noextract
ghost
fn group_output
  (handles:Native.output_handles{Caller.metadata handles})
  (xs:token_values) (ws:weight_values)
  (ds:lseq i32 8) (st:BC.physical_state Capacity.Smoke 256)
  requires
    on gpu_loc (
      PA.pts_to (RT.data handles.demand) ds **
      PA.pts_to (RT.data handles.indices) st.indices **
      PA.pts_to (RT.data handles.probabilities) st.weights **
      PA.pts_to (RT.data handles.survival) st.flags) **
    pure (Shared.task_relation (token_model xs) (weight_model ws)
      (token_model xs) (weight_model ws) (History.output ds st))
  ensures output_post handles xs ws
{
  fold Native.output_cells handles (output_values ds st);
  public_view_matches ds st;
  Pulse.Lib.Core.intro_exists
    (fun (values:result_values) -> Native.output_cells handles values **
      pure (Native.output_binding handles values (public_view values) /\
        Shared.task_relation (token_model xs) (weight_model ws)
          (token_model xs) (weight_model ws) (public_view values))) (output_values ds st);
  fold output_post handles xs ws;
}

inline_for_extraction noextract
fn device_call
  (tokens:token_argument) (weights:weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (handles:Native.output_handles{Caller.metadata handles})
  (body:source_body xs ws fp fw)
  (domain:squash (Shared.input_domain (token_model xs) (weight_model ws)))
  (token_ptr:garray bf16 131072{token_ptr == Input.data tokens})
  (weight_ptr:garray bf16 4096{weight_ptr == Input.data weights})
  (demand_ptr:garray i32 8{demand_ptr == RT.data handles.demand})
  (index_ptr:garray i64 256{index_ptr == RT.data handles.indices})
  (probability_ptr:garray f32 256{probability_ptr == RT.data handles.probabilities})
  (survival_ptr:garray i32 2048{survival_ptr == RT.data handles.survival})
  (frame:slprop) ()
  preserves gpu
  requires payload_pre tokens weights xs ws fp fw handles ** frame
  ensures payload_post tokens weights xs ws fp fw handles ** frame
{
  unfold payload_pre tokens weights xs ws fp fw handles;
  unfold Native.input_cells tokens weights fp fw xs ws;
  unfold Caller.contents handles;
  unfold RT.contents handles.demand;
  unfold RT.contents handles.indices;
  unfold RT.contents handles.probabilities;
  unfold RT.contents handles.survival;
  with ds is ps ss.
    assert
      on gpu_loc (PA.pts_to (RT.data handles.demand) ds) **
      on gpu_loc (PA.pts_to (RT.data handles.indices) is) **
      on gpu_loc (PA.pts_to (RT.data handles.probabilities) ps) **
      on gpu_loc (PA.pts_to (RT.data handles.survival) ss);
  rebind_cells (Input.data tokens) token_ptr #fp #xs;
  rebind_cells (Input.data weights) weight_ptr #fw #ws;
  rebind_cells (RT.data handles.demand) demand_ptr #1.0R #ds;
  rebind_cells (RT.data handles.indices) index_ptr #1.0R #is;
  rebind_cells (RT.data handles.probabilities) probability_ptr #1.0R #ps;
  rebind_cells (RT.data handles.survival) survival_ptr #1.0R #ss;
  KL.elim_gpu (PA.pts_to token_ptr #fp xs ** PA.pts_to weight_ptr #fw ws **
    PA.pts_to demand_ptr ds ** PA.pts_to index_ptr is **
    PA.pts_to probability_ptr ps ** PA.pts_to survival_ptr ss) ();
  PA.pts_to_len demand_ptr;
  PA.pts_to_len index_ptr;
  PA.pts_to_len probability_ptr;
  PA.pts_to_len survival_ptr;
  fold device_pre xs ws fp fw token_ptr weight_ptr demand_ptr index_ptr probability_ptr survival_ptr;
  body token_ptr weight_ptr demand_ptr index_ptr probability_ptr survival_ptr ();
  unfold device_post xs ws fp fw token_ptr weight_ptr demand_ptr index_ptr probability_ptr survival_ptr;
  with result state.
    assert PA.pts_to demand_ptr result ** PA.pts_to index_ptr state.BC.indices **
      PA.pts_to probability_ptr state.BC.weights ** PA.pts_to survival_ptr state.BC.flags;
  KL.intro_gpu (PA.pts_to token_ptr #fp xs ** PA.pts_to weight_ptr #fw ws **
    PA.pts_to demand_ptr result ** PA.pts_to index_ptr state.BC.indices **
    PA.pts_to probability_ptr state.BC.weights ** PA.pts_to survival_ptr state.BC.flags) ();
  rebind_cells token_ptr (Input.data tokens) #fp #xs;
  rebind_cells weight_ptr (Input.data weights) #fw #ws;
  rebind_cells demand_ptr (RT.data handles.demand) #1.0R #result;
  rebind_cells index_ptr (RT.data handles.indices) #1.0R #state.BC.indices;
  rebind_cells probability_ptr (RT.data handles.probabilities) #1.0R #state.BC.weights;
  rebind_cells survival_ptr (RT.data handles.survival) #1.0R #state.BC.flags;
  group_output handles xs ws result state;
  fold Native.input_cells tokens weights fp fw xs ws;
  fold payload_post tokens weights xs ws fp fw handles;
}

inline_for_extraction noextract
ghost
fn prepare_payload
  (tokens:token_argument) (weights:weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (handles:Native.output_handles) (frame:slprop) ()
  requires Caller.contents handles **
    (Native.input_cells tokens weights fp fw xs ws ** frame)
  ensures payload_pre tokens weights xs ws fp fw handles ** frame
{
  fold payload_pre tokens weights xs ws fp fw handles;
}

inline_for_extraction noextract
let same_process (host:loc_id) ()
  : stt_ghost unit emp_inames
      (loc host ** pure (process_of host == process_of host))
      (fun _ -> in_same_process host) =
  Pulse.Lib.Core.intro_exists
    (fun l -> loc l ** pure (process_of l == process_of host)) host

inline_for_extraction noextract
ghost
fn locate_payload (payload frame:slprop) {| placeless payload |} (host:loc_id) ()
  requires payload ** on host frame
  ensures on gpu_loc (payload ** on host frame)
{
  placeless_on_intro payload gpu_loc;
}

inline_for_extraction noextract
ghost
fn unlocate_payload (payload frame:slprop) {| placeless payload |} (host:loc_id) ()
  requires on gpu_loc (payload ** on host frame)
  ensures payload ** on host frame
{
  placeless_on_elim payload gpu_loc;
}

inline_for_extraction noextract
fn launch_source
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (handles:Native.output_handles{Caller.metadata handles})
  (body:source_body xs ws fp fw)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream **
    Input.reference tokens ** Input.reference weights ** Native.output_owners stream handles
  requires Epoch.epoch_live stream epoch **
    Pledge.pledge0 (Epoch.epoch_done stream epoch)
      (Caller.contents handles ** (Native.input_cells tokens weights fp fw xs ws ** frame)) **
    pure (Shared.input_domain (token_model xs) (weight_model ws))
  ensures Epoch.epoch_live stream (Epoch.epoch_next epoch) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
      (payload_post tokens weights xs ws fp fw handles ** frame)
{
  Input.view_metadata tokens;
  Input.view_metadata weights;
  let token_ptr = Input.data_ptr tokens;
  let weight_ptr = Input.data_ptr weights;
  unfold Native.output_owners stream handles;
  let demand_ptr = RT.data_ptr stream handles.demand;
  let index_ptr = RT.data_ptr stream handles.indices;
  let probability_ptr = RT.data_ptr stream handles.probabilities;
  let survival_ptr = RT.data_ptr stream handles.survival;
  fold Native.output_owners stream handles;
  Pledge.rewrite_pledge
    (Caller.contents handles ** (Native.input_cells tokens weights fp fw xs ws ** frame))
    (payload_pre tokens weights xs ws fp fw handles ** frame)
    (prepare_payload tokens weights xs ws fp fw handles frame);
  unfold KL.cpu;
  with host. assert loc host;
  loc_dup host;
  fold KL.cpu;
  loc_dup host;
  same_process host ();
  Caller.locate_frame (Epoch.epoch_done stream epoch) host
    (payload_pre tokens weights xs ws fp fw handles) frame ();
  Pledge.rewrite_pledge
    (payload_pre tokens weights xs ws fp fw handles ** on host frame)
    (on gpu_loc (payload_pre tokens weights xs ws fp fw handles ** on host frame))
    (locate_payload (payload_pre tokens weights xs ws fp fw handles) frame host);
  let kernel:Kernel.kernel_desc_1_1
      (payload_pre tokens weights xs ws fp fw handles ** on host frame)
      (payload_post tokens weights xs ws fp fw handles ** on host frame) = {
    Kernel.f = device_call tokens weights #xs #ws #fp #fw handles body ()
      token_ptr weight_ptr demand_ptr index_ptr probability_ptr survival_ptr (on host frame);
    Kernel.full_pre_sendable = FStar.Tactics.Typeclasses.solve;
    Kernel.full_post_sendable = FStar.Tactics.Typeclasses.solve
  };
  Kernel.launch_kernel_full (Kernel.k11_as_kfull kernel) stream #epoch;
  Pledge.rewrite_pledge
    (on gpu_loc (payload_post tokens weights xs ws fp fw handles ** on host frame))
    (payload_post tokens weights xs ws fp fw handles ** on host frame)
    (unlocate_payload (payload_post tokens weights xs ws fp fw handles) frame host);
  loc_dup host;
  same_process host ();
  Caller.restore_frame (Epoch.epoch_done stream (Epoch.epoch_next epoch)) host
    (payload_post tokens weights xs ws fp fw handles) frame ();
  drop_ (loc host);
}

inline_for_extraction noextract
let before_call
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm) (frame:slprop) : slprop =
  Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch **
  Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (Native.input_cells tokens weights fp fw xs ws ** frame) **
  pure (Shared.input_domain (token_model xs) (weight_model ws))

inline_for_extraction noextract
let after_call
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm) (frame:slprop)
  (result:Caller.allocation) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
   | Caller.Failed _ ->
     Epoch.epoch_live stream epoch **
     Pledge.pledge0 (Epoch.epoch_done stream epoch)
       (Native.input_cells tokens weights fp fw xs ws ** frame)
   | Caller.Allocated handles ->
     Native.output_owners stream handles ** pure (Caller.metadata handles) **
     Epoch.epoch_live stream (Epoch.epoch_next epoch) **
     Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch))
       (payload_post tokens weights xs ws fp fw handles ** frame))

inline_for_extraction noextract
fn source_parameter_body
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (body:source_body xs ws fp fw) (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference weights ** Input.reference tokens **
    before_call stream epoch tokens weights xs ws fp fw frame
  returns result:Caller.allocation
  ensures cpu ** Input.reference weights ** Input.reference tokens **
    after_call stream epoch tokens weights xs ws fp fw frame result
{
  unfold before_call stream epoch tokens weights xs ws fp fw frame;
  let result = Caller.allocate_outputs stream #epoch
    (Native.input_cells tokens weights fp fw xs ws ** frame) ();
  match result {
    Caller.Failed error -> {
      fold after_call stream epoch tokens weights xs ws fp fw frame (Caller.Failed error);
      Caller.Failed error
    }
    Caller.Allocated handles -> {
      launch_source stream #epoch tokens weights #xs #ws #fp #fw handles body frame ();
      fold after_call stream epoch tokens weights xs ws fp fw frame (Caller.Allocated handles);
      Caller.Allocated handles
    }
  }
}

inline_for_extraction noextract
fn source_weight_parameter
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (body:source_body xs ws fp fw) (frame:slprop) {| is_send frame |} ()
  requires cpu ** Input.reference tokens **
    before_call stream epoch tokens weights xs ws fp fw frame
  returns result:Caller.allocation
  ensures cpu ** Input.reference tokens **
    after_call stream epoch tokens weights xs ws fp fw frame result
{
  Input.with_parameter weights
    (Input.reference tokens ** before_call stream epoch tokens weights xs ws fp fw frame)
    (fun result -> Input.reference tokens **
      after_call stream epoch tokens weights xs ws fp fw frame result)
    (source_parameter_body stream #epoch tokens weights #xs #ws #fp #fw body frame)
}

inline_for_extraction noextract
fn source_parameter_scopes
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (tokens:token_argument) (weights:weight_argument)
  (#xs:erased token_values) (#ws:erased weight_values) (#fp #fw:perm)
  (body:source_body xs ws fp fw) (frame:slprop) {| is_send frame |} ()
  requires cpu ** before_call stream epoch tokens weights xs ws fp fw frame
  returns result:Caller.allocation
  ensures cpu ** after_call stream epoch tokens weights xs ws fp fw frame result
{
  Input.with_parameter tokens
    (before_call stream epoch tokens weights xs ws fp fw frame)
    (after_call stream epoch tokens weights xs ws fp fw frame)
    (source_weight_parameter stream #epoch tokens weights #xs #ws #fp #fw body frame)
}
