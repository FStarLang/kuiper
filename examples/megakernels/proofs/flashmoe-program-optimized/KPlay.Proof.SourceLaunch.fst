module KPlay.Proof.SourceLaunch
#lang-pulse

open Kuiper
open Pulse.Lib.Send

module Kernel = Kuiper.Kernel
module Epoch = Kuiper.Epoch
module Stream = Kuiper.Kernel.Stream
module KL = Kuiper.Locs
module Pledge = Pulse.Lib.Pledge
module Frames = KPlay.Proof.OutputAllocations

inline_for_extraction noextract
ghost
fn setup_frame (#pre #post:slprop) (k:Kernel.kernel_desc pre post) (frame:slprop) ()
  norewrite
  requires pre ** frame
  ensures (forall+ (bid:natlt k.nblk). k.block_pre bid) ** (k.frame ** frame)
{
  let setup = k.setup;
  setup ();
}

inline_for_extraction noextract
ghost
fn teardown_frame (#pre #post:slprop) (k:Kernel.kernel_desc pre post) (frame:slprop) ()
  norewrite
  requires (forall+ (bid:natlt k.nblk). k.block_post bid) ** (k.frame ** frame)
  ensures post ** frame
{
  let teardown = k.teardown;
  teardown ();
}

inline_for_extraction noextract
let frame_kernel (#pre #post:slprop) (k:Kernel.kernel_desc pre post) (frame:slprop)
  : Kernel.kernel_desc (pre ** frame) (post ** frame) = {
  k with
    frame = k.frame ** frame;
    setup = setup_frame k frame;
    teardown = teardown_frame k frame
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
fn locate (payload frame:slprop) (host:loc_id) ()
  requires on gpu_loc payload ** on host frame
  ensures on gpu_loc (payload ** on host frame)
{
  placeless_on_intro (on host frame) gpu_loc;
}

inline_for_extraction noextract
ghost
fn unlocate (payload frame:slprop) (host:loc_id) ()
  requires on gpu_loc (payload ** on host frame)
  ensures on gpu_loc payload ** on host frame
{
  on_star_elim payload (on host frame);
}

inline_for_extraction noextract
fn launch (#pre #post:slprop) (k:Kernel.kernel_desc pre post)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (frame:slprop) {| is_send frame |} ()
  preserves cpu ** Stream.stream_live stream
  requires Epoch.epoch_live stream epoch **
    Pledge.pledge0 (Epoch.epoch_done stream epoch) (on gpu_loc pre ** frame)
  ensures Epoch.epoch_live stream (Epoch.epoch_next epoch) **
    Pledge.pledge0 (Epoch.epoch_done stream (Epoch.epoch_next epoch)) (on gpu_loc post ** frame)
{
  unfold KL.cpu;
  with host. assert loc host;
  loc_dup host;
  fold KL.cpu;
  loc_dup host;
  same_process host ();
  Frames.locate_frame (Epoch.epoch_done stream epoch) host (on gpu_loc pre) frame ();
  Pledge.rewrite_pledge (on gpu_loc pre ** on host frame)
    (on gpu_loc (pre ** on host frame)) (locate pre frame host);
  Kernel.launch_kernel_full (frame_kernel k (on host frame)) stream #epoch;
  Pledge.rewrite_pledge (on gpu_loc (post ** on host frame))
    (on gpu_loc post ** on host frame) (unlocate post frame host);
  loc_dup host;
  same_process host ();
  Frames.restore_frame (Epoch.epoch_done stream (Epoch.epoch_next epoch)) host
    (on gpu_loc post) frame ();
  drop_ (loc host);
}
